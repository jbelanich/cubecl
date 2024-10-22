#[cfg(feature = "cuda")]
use cubecl::cuda;
use cubecl::prelude::*;
#[cfg(feature = "wgpu")]
use cubecl::wgpu;
use cubecl::Runtime;

#[cfg(feature = "wgpu")]
fn execute_wgpu(fft_config: fft::FFTConfig) {
    let buffer = fft::make_data(&fft_config);

    let device = wgpu::WgpuDevice::default();
    let client = wgpu::WgpuRuntime::client(&device);

    let a_handle = client.create(f32::as_bytes(buffer.as_slice()));
    let b_handle = client.empty(buffer.len() * core::mem::size_of::<f32>());

    let output_handle = fft::plan_and_launch_fft_radix_4::<wgpu::WgpuRuntime>(
        a_handle, b_handle, &client, fft_config,
    );

    let bytes = client.read(output_handle.binding());
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed stockham_step with runtime {:?} => {output:?}",
        wgpu::WgpuRuntime::name()
    );
}

#[cfg(feature = "cuda")]
fn execute_cuda(fft_config: fft::FFTConfig) {
    let buffer = fft::make_data(&fft_config);

    let device = cuda::CudaDevice::default();
    let client = cuda::CudaRuntime::client(&device);

    let a_handle = client.create(f32::as_bytes(buffer.as_slice()));
    let b_handle = client.empty(buffer.len() * core::mem::size_of::<f32>());

    let output_handle =
        fft::plan_and_launch_fft::<cuda::CudaRuntime>(a_handle, b_handle, &client, fft_config);

    let bytes = client.read(output_handle.binding());
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed stockham_step with runtime {:?} => {output:?}",
        cuda::CudaRuntime::name()
    );
}

fn main() {
    let fft_config = fft::FFTConfig {
        stockham_strategy: fft::StockhamStrategy::SharedMemory,
        memory_strategy: fft::MemoryStrategy::InlineComplex,
        signal_length: 1024,
        num_signals: 1,
    };

    #[cfg(feature = "wgpu")]
    execute_wgpu(fft_config);

    #[cfg(feature = "cuda")]
    execute_cuda(fft_config);
}
