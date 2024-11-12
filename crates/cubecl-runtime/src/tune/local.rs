use super::{AutotuneKey, AutotuneOperationSet, Tuner};
use crate::{
    channel::ComputeChannel, client::ComputeClient, server::ComputeServer, tune::TuneCacheResult,
};
use core::{fmt::Display, hash::Hash};
use hashbrown::HashMap;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::ToString};

/// A local tuner allows to create a tuner for a specific key that can be different from the server
/// key.
pub struct LocalTuner<AK: AutotuneKey, ID> {
    state: spin::RwLock<Option<HashMap<ID, Tuner<AK>>>>,
    name: &'static str,
}

/// Create a local tuner with the provided name.
#[macro_export]
macro_rules! local_tuner {
    ($name:expr) => {
        LocalTuner::new(concat!(module_path!(), "-", $name));
    };
    () => {
        LocalTuner::new(module_path!());
    };
}

pub use local_tuner;

impl<AK: AutotuneKey + 'static, ID: Hash + PartialEq + Eq + Clone + Display> LocalTuner<AK, ID> {
    /// Create a new local tuner.
    pub const fn new(name: &'static str) -> Self {
        Self {
            state: spin::RwLock::new(None),
            name,
        }
    }

    /// Clear the autotune state.
    pub fn clear(&self) {
        let mut state = self.state.write();
        *state = None;
    }

    /// Execute the best operation in the provided [autotune operation set](AutotuneOperationSet)
    pub fn execute<S, C, Out: Send + 'static>(
        &self,
        id: &ID,
        client: &ComputeClient<S, C>,
        autotune_operation_set: Box<dyn AutotuneOperationSet<AK, Out>>,
    ) -> Out
    where
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    {
        let key = autotune_operation_set.key();

        // If this is cached and ready, use the operation.
        if let Some(map) = self.state.read().as_ref() {
            if let Some(tuner) = map.get(id) {
                if let TuneCacheResult::Hit { fastest_index } = tuner.fastest(&key) {
                    let op = autotune_operation_set.fastest(fastest_index);
                    return op.execute();
                }
            }
        }

        // Create the tuner if needed, and update some state like
        // checksums if need be.
        let fastest = {
            let mut state = self.state.write();
            let map = state.get_or_insert_with(Default::default);
            let tuner = map.entry(id.clone()).or_insert_with(move || {
                let name = self.name.replace("::", "-");
                Tuner::new(&name, &id.to_string())
            });

            #[allow(unused_mut)]
            let mut fastest = tuner.fastest(&key);

            // If the cache checksum hasn't been checked, do so now, and retry.
            #[cfg(autotune_persistent_cache)]
            if matches!(fastest, TuneCacheResult::Unchecked) {
                let checksum = autotune_operation_set.compute_checksum();
                tuner.validate_checksum(&key, &checksum);
                fastest = tuner.fastest(&key);
            }
            fastest
        };

        match fastest {
            TuneCacheResult::Hit { fastest_index } => {
                return autotune_operation_set.fastest(fastest_index).execute();
            }
            TuneCacheResult::Miss => {
                // We don't know the results yet, start autotuning.
                //
                // Running benchmarks shound't lock the tuner, since an autotune operation can recursively use the
                // same tuner.
                //
                // # Example
                //
                // ```
                // - tune_1 start
                //   - tune_2 start
                //   - tune_2 save
                // - tune_1 save
                // ```
                let state = self.state.read();
                let state = state.as_ref().expect("Should be initialized");
                let tuner = state.get(id).expect("Should be initialized");

                tuner.execute_autotune(autotune_operation_set.as_ref(), client);
            }
            TuneCacheResult::Pending => {
                // We're waiting for results to come in.
            }
            TuneCacheResult::Unchecked => {
                panic!("Should have checked the cache already.")
            }
        };

        let fastest = {
            let mut state = self.state.write();
            let state = state.as_mut().expect("Should be initialized");
            let tuner = state.get_mut(id).expect("Should be initialized");
            // Now read all results that have come in since.
            tuner.resolve();

            // Check again what the fastest option is, new results might have come in.
            match tuner.fastest(&key) {
                TuneCacheResult::Hit { fastest_index } => {
                    // Theres a known good value - just run it.
                    fastest_index
                }
                TuneCacheResult::Pending => {
                    // If we still don't know, just execute a default index.
                    0
                }
                TuneCacheResult::Miss => {
                    panic!("Should have at least started autotuning");
                }
                TuneCacheResult::Unchecked => {
                    panic!("Should have checked the cache.")
                }
            }
        };

        autotune_operation_set.fastest(fastest).execute()
    }
}
