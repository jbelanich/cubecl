pub mod complex;
pub mod memory;

use complex::{add, make_complex, mul, mul_i, sub, unit_complex_from_angle};
use memory::{
    array_read, array_to_shared, array_write, shared_read, shared_to_array, shared_write,
    ComplexBufferAccess, ComplexNumberQuery, InlineComplexAccess, SplitComplexAccess,
};

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

#[cube]
fn make_query(elem_idx: u32, signal_idx: u32, #[comptime] plan: FFTPlan) -> ComplexNumberQuery {
    ComplexNumberQuery {
        elem_idx,
        signal_idx,
        signal_length: plan.config.signal_length,
        num_signals: plan.config.num_signals,
    }
}

#[cube]
fn make_shared_query(elem_idx: u32, #[comptime] plan: FFTPlan) -> ComplexNumberQuery {
    ComplexNumberQuery {
        elem_idx,
        signal_idx: 0,
        signal_length: plan.config.signal_length,
        num_signals: plan.config.num_signals,
    }
}

#[cube]
fn stockham_radix_4_step_large<A: ComplexBufferAccess>(
    l: u32,
    source: &Array<f32>,
    dest: &mut Array<f32>,
    #[comptime] plan: FFTPlan,
) {
    let l_star = l / 4;

    let r = plan.config.signal_length / l;

    let mut signal_idx = UNIT_POS_X;
    while signal_idx < (plan.config.signal_length / 4) {
        let k = signal_idx / l_star;
        let j = signal_idx % l_star;

        let w1 = unit_complex_from_angle(2.0 * 3.14159265 * (j as f32) / (l as f32));
        let w2 = mul(w1, w1);
        let w3 = mul(w1, w2);

        let y1 = array_read::<A>(source, make_query(k * l_star + j, CUBE_POS_X, plan));
        let y2 = array_read::<A>(source, make_query((k + r) * l_star + j, CUBE_POS_X, plan));
        let y3 = array_read::<A>(
            source,
            make_query((k + 2 * r) * l_star + j, CUBE_POS_X, plan),
        );
        let y4 = array_read::<A>(
            source,
            make_query((k + 3 * r) * l_star + j, CUBE_POS_X, plan),
        );

        let alpha = y1;
        let beta = mul(w1, y2);
        let gamma = mul(w2, y3);
        let delta = mul(w3, y4);

        let tao_0 = add(alpha, gamma);
        let tao_1 = sub(alpha, gamma);
        let tao_2 = add(beta, delta);
        let tao_3 = sub(beta, delta);

        array_write::<A>(
            dest,
            make_query(k * l + j, CUBE_POS_X, plan),
            add(tao_0, tao_2),
        );

        array_write::<A>(
            dest,
            make_query(k * l + l_star + j, CUBE_POS_X, plan),
            sub(tao_1, mul_i(tao_3)),
        );

        array_write::<A>(
            dest,
            make_query(k * l + 2 * l_star + j, CUBE_POS_X, plan),
            sub(tao_0, tao_2),
        );

        array_write::<A>(
            dest,
            make_query(k * l + 3 * l_star + j, CUBE_POS_X, plan),
            add(tao_1, mul_i(tao_3)),
        );

        signal_idx += CUBE_DIM_X;
    }
}

#[cube]
fn stockham_radix_4_step<A: ComplexBufferAccess>(
    l: u32,
    source: &Array<f32>,
    dest: &mut Array<f32>,
    #[comptime] plan: FFTPlan,
) {
    let l_star = l / 4;

    let r = plan.config.signal_length / l;
    let r_star = 4 * r;

    let k = UNIT_POS_X / l_star;
    let j = UNIT_POS_X % l_star;

    let w1 = unit_complex_from_angle(2.0 * 3.14159265 * (j as f32) / (l as f32));
    let w2 = mul(w1, w1);
    let w3 = mul(w1, w2);

    let y1 = array_read::<A>(source, make_query(j * r_star + k, CUBE_POS_X, plan));
    let y2 = array_read::<A>(source, make_query(j * r_star + r + k, CUBE_POS_X, plan));
    let y3 = array_read::<A>(source, make_query(j * r_star + 2 * r + k, CUBE_POS_X, plan));
    let y4 = array_read::<A>(source, make_query(j * r_star + 3 * r + k, CUBE_POS_X, plan));

    let alpha = y1;
    let beta = mul(w1, y2);
    let gamma = mul(w2, y3);
    let delta = mul(w3, y4);

    let tao_0 = add(alpha, gamma);
    let tao_1 = sub(alpha, gamma);
    let tao_2 = add(beta, delta);
    let tao_3 = sub(beta, delta);

    array_write::<A>(
        dest,
        make_query(j * r + k, CUBE_POS_X, plan),
        add(tao_0, tao_2),
    );

    array_write::<A>(
        dest,
        make_query((j + l_star) * r + k, CUBE_POS_X, plan),
        sub(tao_1, mul(make_complex(0f32, 1f32), tao_3)),
    );

    array_write::<A>(
        dest,
        make_query((j + 2 * l_star) * r + k, CUBE_POS_X, plan),
        sub(tao_0, tao_2),
    );

    array_write::<A>(
        dest,
        make_query((j + 3 * l_star) * r + k, CUBE_POS_X, plan),
        add(tao_1, mul(make_complex(0f32, 1f32), tao_3)),
    );
}

#[cube]
fn stockham_radix_4_shared_step<A: ComplexBufferAccess>(
    l: u32,
    source: &SharedMemory<f32>,
    dest: &mut SharedMemory<f32>,
    #[comptime] plan: FFTPlan,
) {
    let l_star = l / 4;

    let r = plan.config.signal_length / l;

    let k = UNIT_POS_X / l_star;
    let j = UNIT_POS_X % l_star;

    let w1 = unit_complex_from_angle(2.0 * 3.14159265 * (j as f32) / (l as f32));
    let w2 = mul(w1, w1);
    let w3 = mul(w1, w2);

    // Good coalescing early, bad later.o
    let y1 = shared_read::<A>(source, make_shared_query(k * l_star + j, plan));
    let y2 = shared_read::<A>(source, make_shared_query((k + r) * l_star + j, plan));
    let y3 = shared_read::<A>(source, make_shared_query((k + 2 * r) * l_star + j, plan));
    let y4 = shared_read::<A>(source, make_shared_query((k + 3 * r) * l_star + j, plan));

    let alpha = y1;
    let beta = mul(w1, y2);
    let gamma = mul(w2, y3);
    let delta = mul(w3, y4);

    let tao_0 = add(alpha, gamma);
    let tao_1 = sub(alpha, gamma);
    let tao_2 = add(beta, delta);
    let tao_3 = sub(beta, delta);

    shared_write::<A>(dest, make_shared_query(k * l + j, plan), add(tao_0, tao_2));

    shared_write::<A>(
        dest,
        make_shared_query(k * l + l_star + j, plan),
        sub(tao_1, mul(make_complex(0f32, 1f32), tao_3)),
    );

    shared_write::<A>(
        dest,
        make_shared_query(k * l + 2 * l_star + j, plan),
        sub(tao_0, tao_2),
    );

    shared_write::<A>(
        dest,
        make_shared_query(k * l + 3 * l_star + j, plan),
        add(tao_1, mul(make_complex(0f32, 1f32), tao_3)),
    );
}

#[cube(launch_unchecked)]
fn stockham_radix_4_full<A: ComplexBufferAccess>(
    a_buffer: &mut Array<f32>,
    b_buffer: &mut Array<f32>,
    #[comptime] plan: FFTPlan,
) {
    let mut l = 1u32;

    #[unroll]
    for q in 0..plan.num_steps {
        l *= 4u32;

        if q % 2 == 0 {
            stockham_radix_4_step_large::<A>(l, a_buffer, b_buffer, plan);
        } else {
            stockham_radix_4_step_large::<A>(l, b_buffer, a_buffer, plan);
        }

        sync_units();
    }
}

#[cube]
// Run stockham update on large array where the Y dimension
// of cubes correspond to chunking the signal into groups.
//
// We do a grid-stride loop along this dimension.
fn stockham_step_large<A: ComplexBufferAccess>(
    l: u32,
    source: &Array<f32>,
    dest: &mut Array<f32>,
    #[comptime] plan: FFTPlan,
) {
    let l_star = l / 2;

    // r tracks the number of times the signal can be chunked into
    // length L chunks.
    let r = plan.config.signal_length / l;

    let mut signal_idx = UNIT_POS_X;
    while signal_idx < plan.config.signal_length / 2 {
        let k = signal_idx / l_star;
        let j = signal_idx % l_star;

        let w = unit_complex_from_angle(2.0 * 3.14159265 * (j as f32) / (l as f32));
        let tao = mul(
            array_read::<A>(source, make_query((k + r) * l_star + j, CUBE_POS_X, plan)),
            w,
        );
        let y_left = array_read::<A>(source, make_query(k * l_star + j, CUBE_POS_X, plan));

        array_write::<A>(
            dest,
            make_query(k * l + j, CUBE_POS_X, plan),
            add(y_left, tao),
        );
        array_write::<A>(
            dest,
            make_query(k * l + l_star + j, CUBE_POS_X, plan),
            sub(y_left, tao),
        );

        signal_idx += CUBE_DIM_X;
    }
}

#[cube]

// It is expected we call this with N / 2 threads in one block. Each thread will
// write to two locations in the array.
fn stockham_step<A: ComplexBufferAccess>(
    l: u32,
    source: &Array<f32>,
    dest: &mut Array<f32>,
    #[comptime] plan: FFTPlan,
) {
    let l_star = l / 2;

    // r tracks the number of times the signal can be chunked into
    // length L chunks.
    let r = plan.config.signal_length / l;

    let k = UNIT_POS_X / l_star;
    let j = UNIT_POS_X % l_star;

    let w = unit_complex_from_angle(2.0 * 3.14159265 * (j as f32) / (l as f32));
    let tao = mul(
        array_read::<A>(source, make_query((k + r) * l_star + j, CUBE_POS_X, plan)),
        w,
    );
    let y_left = array_read::<A>(source, make_query(k * l_star + j, CUBE_POS_X, plan));

    array_write::<A>(
        dest,
        make_query(k * l + j, CUBE_POS_X, plan),
        add(y_left, tao),
    );
    array_write::<A>(
        dest,
        make_query(k * l + l_star + j, CUBE_POS_X, plan),
        sub(y_left, tao),
    );
}

#[cube]
// It is expected we call this with N / 2 threads in one block. Each thread will
// write to two locations in the array.
fn stockham_step_shared<A: ComplexBufferAccess>(
    l: u32,
    source: &SharedMemory<f32>,
    dest: &mut SharedMemory<f32>,
    #[comptime] plan: FFTPlan,
) {
    let l_star = l / 2;

    // r tracks the number of times the signal can be chunked into
    // length L chunks.
    let r = plan.config.signal_length / l;

    let k = UNIT_POS_X / l_star;
    let j = UNIT_POS_X % l_star;

    let w = unit_complex_from_angle(2.0 * 3.14159265 * (j as f32) / (l as f32));
    let tao = mul(
        shared_read::<A>(source, make_shared_query((k + r) * l_star + j, plan)),
        w,
    );
    let y_left = shared_read::<A>(source, make_shared_query(k * l_star + j, plan));

    shared_write::<A>(dest, make_shared_query(k * l + j, plan), add(y_left, tao));
    shared_write::<A>(
        dest,
        make_shared_query(k * l + l_star + j, plan),
        sub(y_left, tao),
    );
}

#[cube]
fn load_buffer_to_shared<A: ComplexBufferAccess>(
    shared: &mut SharedMemory<f32>,
    global: &Array<f32>,
    radix: u32,
    #[comptime] plan: FFTPlan,
) {
    for i in 0..radix {
        let offset = i * (plan.config.signal_length / radix);
        array_to_shared::<A>(
            shared,
            global,
            make_query(offset + UNIT_POS_X, CUBE_POS_X, plan),
            make_shared_query(offset + UNIT_POS_X, plan),
        );
    }
}

#[cube]
fn load_shared_to_buffer<A: ComplexBufferAccess>(
    global: &mut Array<f32>,
    shared: &SharedMemory<f32>,
    radix: u32,
    #[comptime] plan: FFTPlan,
) {
    for i in 0..radix {
        let offset = i * (plan.config.signal_length / radix);
        shared_to_array::<A>(
            global,
            shared,
            make_shared_query(offset + UNIT_POS_X, plan),
            make_query(offset + UNIT_POS_X, CUBE_POS_X, plan),
        );
    }
}

// Try replacing shared memory with new arrays and see if the copy logic
// is correct.
#[cube(launch_unchecked)]
fn stockham_full_shared<A: ComplexBufferAccess>(
    a_buffer: &mut Array<f32>,
    b_buffer: &mut Array<f32>,
    #[comptime] plan: FFTPlan,
) {
    let mut a_buffer_shared = SharedMemory::<f32>::new(plan.config.signal_length * 2);
    let mut b_buffer_shared = SharedMemory::<f32>::new(plan.config.signal_length * 2);

    // Only load a buffer -- we don't care what the contents of the memory of
    // the b buffer is.
    load_buffer_to_shared::<A>(&mut a_buffer_shared, a_buffer, 2, plan);

    sync_units();

    let mut l = 1u32;

    #[unroll]
    for q in 0..plan.num_steps {
        l *= 2u32;

        if q % 2 == 0 {
            stockham_step_shared::<A>(l, &a_buffer_shared, &mut b_buffer_shared, plan);
        } else {
            stockham_step_shared::<A>(l, &b_buffer_shared, &mut a_buffer_shared, plan);
        }
        sync_units();
    }

    load_shared_to_buffer::<A>(a_buffer, &a_buffer_shared, 2, plan);
    load_shared_to_buffer::<A>(b_buffer, &b_buffer_shared, 2, plan);
}

#[cube(launch_unchecked)]
fn stockham_radix_4_full_shared<A: ComplexBufferAccess>(
    a_buffer: &mut Array<f32>,
    b_buffer: &mut Array<f32>,
    #[comptime] plan: FFTPlan,
) {
    let mut a_buffer_shared = SharedMemory::<f32>::new(plan.config.signal_length * 2);
    let mut b_buffer_shared = SharedMemory::<f32>::new(plan.config.signal_length * 2);

    load_buffer_to_shared::<A>(&mut a_buffer_shared, a_buffer, 4, plan);
    load_buffer_to_shared::<A>(&mut b_buffer_shared, b_buffer, 4, plan);

    sync_units();

    let mut l = 1u32;

    #[unroll]
    for q in 0..plan.num_steps {
        l *= 4u32;

        if q % 2 == 0 {
            stockham_radix_4_shared_step::<A>(l, &a_buffer_shared, &mut b_buffer_shared, plan);
        } else {
            stockham_radix_4_shared_step::<A>(l, &b_buffer_shared, &mut a_buffer_shared, plan);
        }
        sync_units();
    }

    load_shared_to_buffer::<A>(a_buffer, &a_buffer_shared, 4, plan);
    load_shared_to_buffer::<A>(b_buffer, &b_buffer_shared, 4, plan);
}

#[cube(launch_unchecked)]
fn stockham_full<A: ComplexBufferAccess>(
    a_buffer: &mut Array<f32>,
    b_buffer: &mut Array<f32>,
    #[comptime] plan: FFTPlan,
) {
    let mut l = 1u32;

    #[unroll]
    for q in 0..plan.num_steps {
        l *= 2u32;

        if q % 2 == 0 {
            stockham_step_large::<A>(l, a_buffer, b_buffer, plan);
        } else {
            stockham_step_large::<A>(l, b_buffer, a_buffer, plan);
        }

        sync_units();
    }
}

#[derive(CubeType, Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub struct FFTConfig {
    pub stockham_strategy: StockhamStrategy,
    pub memory_strategy: MemoryStrategy,
    pub signal_length: u32,
    pub num_signals: u32,
}

impl FFTConfig {
    fn buffer_size(&self) -> usize {
        (2 * self.signal_length * self.num_signals) as usize
    }
}

#[derive(CubeType, Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub struct FFTPlan {
    pub config: FFTConfig,
    pub num_steps: u32,
}

#[derive(CubeType, Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub enum StockhamStrategy {
    GlobalMemory,
    SharedMemory,
}

#[derive(CubeType, Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub enum MemoryStrategy {
    InlineComplex,
    SplitComplex,
}

pub fn plan_and_launch_fft<R: Runtime>(
    a_handle: Handle,
    b_handle: Handle,
    client: &ComputeClient<R::Server, R::Channel>,
    fft_config: FFTConfig,
) -> Handle {
    let num_steps = fft_config.signal_length.ilog2() as u32;
    let cube_count = CubeCount::Static(fft_config.num_signals, 1, 1);
    let cube_dim = CubeDim::new(
        std::cmp::min(fft_config.signal_length / 2, 1024) as u32,
        1,
        1,
    );
    unsafe {
        let a_array_arg = ArrayArg::from_raw_parts(&a_handle, fft_config.buffer_size(), 1u8);
        let b_array_arg = ArrayArg::from_raw_parts(&b_handle, fft_config.buffer_size(), 1u8);
        match fft_config.memory_strategy {
            MemoryStrategy::InlineComplex => match fft_config.stockham_strategy {
                StockhamStrategy::GlobalMemory => {
                    stockham_full::launch_unchecked::<InlineComplexAccess, R>(
                        &client,
                        cube_count,
                        cube_dim,
                        a_array_arg,
                        b_array_arg,
                        FFTPlan {
                            config: fft_config,
                            num_steps,
                        },
                    )
                }
                StockhamStrategy::SharedMemory => {
                    stockham_full_shared::launch_unchecked::<InlineComplexAccess, R>(
                        &client,
                        cube_count,
                        cube_dim,
                        a_array_arg,
                        b_array_arg,
                        FFTPlan {
                            config: fft_config,
                            num_steps,
                        },
                    )
                }
            },
            MemoryStrategy::SplitComplex => match fft_config.stockham_strategy {
                StockhamStrategy::GlobalMemory => {
                    stockham_full::launch_unchecked::<SplitComplexAccess, R>(
                        &client,
                        cube_count,
                        cube_dim,
                        a_array_arg,
                        b_array_arg,
                        FFTPlan {
                            config: fft_config,
                            num_steps,
                        },
                    )
                }
                StockhamStrategy::SharedMemory => {
                    stockham_full_shared::launch_unchecked::<SplitComplexAccess, R>(
                        &client,
                        cube_count,
                        cube_dim,
                        a_array_arg,
                        b_array_arg,
                        FFTPlan {
                            config: fft_config,
                            num_steps,
                        },
                    )
                }
            },
        }
    };

    let output_handle = if (num_steps - 1) % 2 == 0 {
        b_handle
    } else {
        a_handle
    };

    output_handle
}

pub fn plan_and_launch_fft_radix_4<R: Runtime>(
    a_handle: Handle,
    b_handle: Handle,
    client: &ComputeClient<R::Server, R::Channel>,
    fft_config: FFTConfig,
) -> Handle {
    let num_steps = (fft_config.signal_length.ilog2() / 2) as u32;
    let cube_count = CubeCount::Static(fft_config.num_signals, 1, 1);
    let cube_dim = CubeDim::new(
        std::cmp::min(fft_config.signal_length / 4, 1024) as u32,
        1,
        1,
    );
    let plan = FFTPlan {
        config: fft_config,
        num_steps,
    };
    unsafe {
        let a_array_arg = ArrayArg::from_raw_parts(&a_handle, fft_config.buffer_size(), 1u8);
        let b_array_arg = ArrayArg::from_raw_parts(&b_handle, fft_config.buffer_size(), 1u8);
        match fft_config.memory_strategy {
            MemoryStrategy::InlineComplex => match fft_config.stockham_strategy {
                StockhamStrategy::GlobalMemory => {
                    stockham_radix_4_full::launch_unchecked::<InlineComplexAccess, R>(
                        &client,
                        cube_count,
                        cube_dim,
                        a_array_arg,
                        b_array_arg,
                        plan,
                    )
                }
                StockhamStrategy::SharedMemory => {
                    stockham_radix_4_full_shared::launch_unchecked::<InlineComplexAccess, R>(
                        &client,
                        cube_count,
                        cube_dim,
                        a_array_arg,
                        b_array_arg,
                        plan,
                    )
                }
            },
            MemoryStrategy::SplitComplex => match fft_config.stockham_strategy {
                StockhamStrategy::GlobalMemory => {
                    stockham_radix_4_full::launch_unchecked::<SplitComplexAccess, R>(
                        &client,
                        cube_count,
                        cube_dim,
                        a_array_arg,
                        b_array_arg,
                        plan,
                    )
                }
                StockhamStrategy::SharedMemory => {
                    stockham_radix_4_full_shared::launch_unchecked::<SplitComplexAccess, R>(
                        &client,
                        cube_count,
                        cube_dim,
                        a_array_arg,
                        b_array_arg,
                        plan,
                    )
                }
            },
        }
    };

    let output_handle = if (num_steps - 1) % 2 == 0 {
        b_handle
    } else {
        a_handle
    };

    output_handle
}

pub fn make_data(fft_config: &FFTConfig) -> Vec<f32> {
    match fft_config.memory_strategy {
        MemoryStrategy::SplitComplex => memory::make_data::<SplitComplexAccess>(
            fft_config.signal_length as usize,
            fft_config.num_signals as usize,
        ),
        MemoryStrategy::InlineComplex => memory::make_data::<InlineComplexAccess>(
            fft_config.signal_length as usize,
            fft_config.num_signals as usize,
        ),
    }
}
