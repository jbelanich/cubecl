use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::frontend::Float;
use cubecl::future;
use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use std::marker::PhantomData;

impl<R: Runtime, E: Float> Benchmark for FFTBench<'_, R, E> {
    type Args = (Handle, Handle);

    fn prepare(&self) -> Self::Args {
        let signal = fft::make_data(&self.fft_config);

        let a_handle = self.client.create(f32::as_bytes(signal.as_slice()));
        let b_handle = self
            .client
            .empty(signal.len() * core::mem::size_of::<f32>());

        (a_handle, b_handle)
    }

    fn execute(&self, (a_handle, b_handle): Self::Args) {
        let _ = match self.radix {
            Radix::Radix2 => {
                fft::plan_and_launch_fft::<R>(a_handle, b_handle, &self.client, self.fft_config)
            }
            Radix::Radix4 => fft::plan_and_launch_fft_radix_4::<R>(
                a_handle,
                b_handle,
                &self.client,
                self.fft_config,
            ),
        };
    }

    fn num_samples(&self) -> usize {
        100
    }

    fn name(&self) -> String {
        format!(
            "fft-{:?}-{}-{}-{:?}-{:?}",
            self.radix,
            R::name(),
            E::as_elem(),
            self.fft_config.memory_strategy,
            self.fft_config.stockham_strategy
        )
        .to_lowercase()
    }

    fn sync(&self) -> std::time::Duration {
        future::block_on(self.client.sync())
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Radix {
    Radix2,
    Radix4,
}

#[allow(dead_code)]
struct FFTBench<'a, R: Runtime, E> {
    fft_config: fft::FFTConfig,
    radix: Radix,
    client: &'a ComputeClient<R::Server, R::Channel>,
    _e: PhantomData<E>,
}

#[allow(dead_code)]
fn run<R: Runtime, E: Float>(device: R::Device) {
    let client = R::client(&device);

    let signal_length = 1024;
    let num_signals = 1000;

    for radix in [Radix::Radix2, Radix::Radix4] {
        for stockham_strategy in [
            fft::StockhamStrategy::GlobalMemory,
            fft::StockhamStrategy::SharedMemory,
        ] {
            for memory_strategy in [
                fft::MemoryStrategy::InlineComplex,
                fft::MemoryStrategy::SplitComplex,
            ] {
                let bench = FFTBench::<R, E> {
                    fft_config: fft::FFTConfig {
                        stockham_strategy,
                        memory_strategy,
                        signal_length,
                        num_signals,
                    },
                    radix,
                    client: &client,
                    _e: PhantomData,
                };
                println!("{}", bench.name());
                println!("{}", bench.run(TimingMethod::DeviceOnly));
            }
        }
    }
}

fn main() {
    run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default());
}
