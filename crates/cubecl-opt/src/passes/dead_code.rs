use std::{
    mem::transmute,
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};

use cubecl_core::ir::{ConstantScalarValue, Variable};
use petgraph::{graph::NodeIndex, visit::EdgeRef};

use crate::{visit_noop, AtomicCounter, BasicBlock, BlockUse, ControlFlow, Optimizer};

use super::OptimizerPass;

/// Eliminate non-output variables that are never read in the program.
pub struct EliminateUnusedVariables;

impl OptimizerPass for EliminateUnusedVariables {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        while search_loop(opt) {
            changes.inc();
        }
    }
}

fn search_loop(opt: &mut Optimizer) -> bool {
    for node in opt.program.node_indices().collect::<Vec<_>>() {
        let ops = opt.program[node].ops.borrow().indices().collect::<Vec<_>>();

        for idx in ops {
            let mut op = opt.program[node].ops.borrow()[idx].clone();
            let mut out = None;
            let used = Rc::new(AtomicBool::new(false));
            opt.visit_operation(&mut op, visit_noop, |_, var| {
                // Exclude outputs
                if !matches!(
                    var,
                    Variable::GlobalOutputArray { .. } | Variable::Slice { .. }
                ) {
                    out = Some(*var);
                }
            });
            if let Some(out) = out {
                let used = used.clone();
                opt.visit_all(
                    |_, var| {
                        if *var == out {
                            used.store(true, Ordering::Release);
                        }
                    },
                    visit_noop,
                );
                if !used.load(Ordering::Acquire) {
                    opt.program[node].ops.borrow_mut().remove(idx);
                    return true;
                }
            }
        }
    }

    false
}

/// Eliminates branches that can be evaluated at compile time.
pub struct EliminateConstBranches;

impl OptimizerPass for EliminateConstBranches {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for block in opt.node_ids() {
            let control_flow = opt.program[block].control_flow.clone();
            let current = control_flow.borrow().clone();
            match current {
                ControlFlow::IfElse {
                    cond,
                    then,
                    or_else,
                    merge,
                } if cond.as_const().is_some() => {
                    let cond = cond.as_const().unwrap().as_bool();
                    let mut edges = opt.program.edges(block);
                    if cond {
                        let edge = edges.find(|it| it.target() == or_else).unwrap().id();
                        opt.program.remove_edge(edge);
                    } else {
                        let edge = edges.find(|it| it.target() == then).unwrap().id();
                        opt.program.remove_edge(edge);
                    }
                    if let Some(merge) = merge {
                        opt.program[merge]
                            .block_use
                            .retain(|it| *it != BlockUse::Merge);
                    }

                    *control_flow.borrow_mut() = ControlFlow::None;
                    changes.inc();
                }
                ControlFlow::Switch {
                    value,
                    default,
                    branches,
                    ..
                } if value.as_const().is_some() => {
                    let value = match value.as_const().unwrap() {
                        ConstantScalarValue::Int(val, _) => unsafe {
                            transmute::<i32, u32>(val as i32)
                        },
                        ConstantScalarValue::UInt(val) => val as u32,
                        _ => unreachable!("Switch cases must be integer"),
                    };
                    let branch = branches.into_iter().find(|(val, _)| *val == value);
                    let branch = branch.map(|it| it.1).unwrap_or(default);
                    let edges = opt.program.edges(block).filter(|it| it.target() != branch);
                    let edges: Vec<_> = edges.map(|it| it.id()).collect();
                    for edge in edges {
                        opt.program.remove_edge(edge);
                    }
                    *control_flow.borrow_mut() = ControlFlow::None;
                    changes.inc();
                }
                _ => {}
            }
        }
    }
}

/// Eliminates dead code blocks left over from other optimizations like branch elimination.
pub struct EliminateDeadBlocks;

impl OptimizerPass for EliminateDeadBlocks {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        while search_dead_blocks(opt) {
            changes.inc();
        }
    }
}

fn search_dead_blocks(opt: &mut Optimizer) -> bool {
    for block in opt.node_ids() {
        if block != opt.entry() && opt.predecessors(block).is_empty() {
            let edges: Vec<_> = opt.program.edges(block).map(|it| it.id()).collect();
            for edge in edges {
                opt.program.remove_edge(edge);
            }
            opt.program.remove_node(block);
            return true;
        }
    }

    false
}

/// Merges unnecessary basic blocks left over from constant branch evaluation and dead code
/// elimination
pub struct MergeBlocks;

impl OptimizerPass for MergeBlocks {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        // Need to cancel analysis and restart when changes occur, because node ids get invalidated
        while merge_blocks(opt) {
            changes.inc();
        }
    }
}

fn merge_blocks(opt: &mut Optimizer) -> bool {
    for block_idx in opt.post_order.clone().into_iter().rev() {
        let successors = opt.sucessors(block_idx);
        if successors.len() == 1 && can_merge(opt, block_idx, successors[0]) {
            let mut new_block = BasicBlock::default();
            let block = opt.program[block_idx].clone();
            let successor = opt.program[successors[0]].clone();
            let b_phi = block.phi_nodes.borrow().clone();
            let s_phi = successor.phi_nodes.borrow().clone();
            let b_ops = block.ops.borrow().values().cloned().collect::<Vec<_>>();
            let s_ops = successor.ops.borrow().values().cloned().collect::<Vec<_>>();

            new_block.live_vars.extend(block.live_vars.clone());
            new_block.live_vars.extend(successor.live_vars.clone());
            new_block.phi_nodes.borrow_mut().extend(b_phi);
            new_block.phi_nodes.borrow_mut().extend(s_phi);
            new_block.ops.borrow_mut().extend(b_ops);
            new_block.ops.borrow_mut().extend(s_ops);
            *new_block.control_flow.borrow_mut() = successor.control_flow.borrow().clone();
            new_block.block_use.extend(block.block_use);
            new_block.block_use.extend(successor.block_use);

            if successors[0] == opt.ret {
                opt.ret = block_idx;
            }
            for incoming in opt.predecessors(successors[0]) {
                if incoming != block_idx {
                    opt.program.add_edge(incoming, block_idx, ());
                }
            }
            for outgoing in opt.sucessors(successors[0]) {
                opt.program.add_edge(block_idx, outgoing, ());
            }
            *opt.program.node_weight_mut(block_idx).unwrap() = new_block;
            opt.program.remove_node(successors[0]);
            opt.post_order.retain(|it| *it != successors[0]);
            update_references(opt, successors[0], block_idx);
            return true;
        }
    }

    false
}

fn can_merge(opt: &mut Optimizer, block: NodeIndex, successor: NodeIndex) -> bool {
    let has_multiple_entries = opt.predecessors(successor).len() > 1;
    let block = &opt.program[block];
    let successor = &opt.program[successor];
    let b_has_control_flow = !matches!(*block.control_flow.borrow(), ControlFlow::None);
    let b_is_continue = block.block_use.contains(&BlockUse::ContinueTarget);
    let s_is_continue = successor.block_use.contains(&BlockUse::ContinueTarget);

    let is_continue = b_is_continue || s_is_continue;
    let s_is_header = matches!(*block.control_flow.borrow(), ControlFlow::Loop { .. });
    let b_is_merge = block
        .block_use
        .iter()
        .any(|it| matches!(it, BlockUse::Merge));
    let s_is_merge = successor
        .block_use
        .iter()
        .any(|it| matches!(it, BlockUse::Merge));
    let both_merge = b_is_merge && s_is_merge;
    !has_multiple_entries && !b_has_control_flow && !s_is_header && !is_continue && !both_merge
}

fn update_references(opt: &mut Optimizer, from: NodeIndex, to: NodeIndex) {
    let update = |id: &mut NodeIndex| {
        if *id == from {
            *id = to
        }
    };

    for node in opt.node_ids() {
        for phi in opt.program[node].phi_nodes.borrow_mut().iter_mut() {
            for entry in phi.entries.iter_mut() {
                update(&mut entry.block);
            }
        }

        match &mut *opt.program[node].control_flow.borrow_mut() {
            ControlFlow::IfElse {
                then,
                or_else,
                merge,
                ..
            } => {
                update(then);
                update(or_else);
                if let Some(it) = merge.as_mut() {
                    update(it);
                }
            }
            ControlFlow::Switch {
                default,
                branches,
                merge,
                ..
            } => {
                update(default);
                if let Some(it) = merge.as_mut() {
                    update(it);
                }

                for branch in branches {
                    update(&mut branch.1);
                }
            }
            ControlFlow::Loop {
                body,
                continue_target,
                merge,
                ..
            } => {
                update(body);
                update(continue_target);
                update(merge);
            }
            _ => {}
        }
    }
}