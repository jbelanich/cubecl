use crate::complex::Complex;
use cubecl::prelude::*;

#[derive(CubeType, Debug, Clone, Copy)]
pub struct ComplexNumberQuery {
    pub elem_idx: u32,
    pub signal_idx: u32,
    pub signal_length: u32,
    pub num_signals: u32,
}

#[cube]
pub trait ComplexBufferAccess: 'static + Send + Sync {
    fn real_idx(q: ComplexNumberQuery) -> u32;
    fn imag_idx(q: ComplexNumberQuery) -> u32;
}

pub struct InlineComplexAccess;

#[cube]
impl ComplexBufferAccess for InlineComplexAccess {
    fn real_idx(q: ComplexNumberQuery) -> u32 {
        (q.signal_idx * q.signal_length * 2) + 2 * q.elem_idx
    }
    fn imag_idx(q: ComplexNumberQuery) -> u32 {
        (q.signal_idx * q.signal_length * 2) + 2 * q.elem_idx + 1
    }
}

pub struct SplitComplexAccess;

#[cube]
impl ComplexBufferAccess for SplitComplexAccess {
    fn real_idx(q: ComplexNumberQuery) -> u32 {
        (q.signal_idx * q.signal_length * 2) + q.elem_idx
    }
    fn imag_idx(q: ComplexNumberQuery) -> u32 {
        (q.signal_idx * q.signal_length * 2) + q.signal_length + q.elem_idx
    }
}

#[cube]
pub fn array_read<A: ComplexBufferAccess>(buffer: &Array<f32>, q: ComplexNumberQuery) -> Complex {
    Complex {
        real: buffer[A::real_idx(q)],
        imag: buffer[A::imag_idx(q)],
    }
}

#[cube]
pub fn array_write<A: ComplexBufferAccess>(
    buffer: &mut Array<f32>,
    q: ComplexNumberQuery,
    val: Complex,
) {
    buffer[A::real_idx(q)] = val.real;
    buffer[A::imag_idx(q)] = val.imag;
}

#[cube]
pub fn shared_read<A: ComplexBufferAccess>(
    buffer: &SharedMemory<f32>,
    q: ComplexNumberQuery,
) -> Complex {
    Complex {
        real: buffer[A::real_idx(q)],
        imag: buffer[A::imag_idx(q)],
    }
}

#[cube]
pub fn shared_write<A: ComplexBufferAccess>(
    buffer: &mut SharedMemory<f32>,
    q: ComplexNumberQuery,
    val: Complex,
) {
    buffer[A::real_idx(q)] = val.real;
    buffer[A::imag_idx(q)] = val.imag;
}

#[cube]
pub fn shared_to_array<A: ComplexBufferAccess>(
    array_buffer: &mut Array<f32>,
    shared_buffer: &SharedMemory<f32>,
    read_q: ComplexNumberQuery,
    write_q: ComplexNumberQuery,
) {
    array_write::<A>(
        array_buffer,
        write_q,
        shared_read::<A>(shared_buffer, read_q),
    );
}

#[cube]
pub fn array_to_shared<A: ComplexBufferAccess>(
    shared_buffer: &mut SharedMemory<f32>,
    array_buffer: &Array<f32>,
    read_q: ComplexNumberQuery,
    write_q: ComplexNumberQuery,
) {
    shared_write::<A>(
        shared_buffer,
        write_q,
        array_read::<A>(array_buffer, read_q),
    );
}

pub fn make_data<A: ComplexBufferAccess>(n: usize, num_signals: usize) -> Vec<f32> {
    let mut v: Vec<f32> = vec![0.0; 2 * n * num_signals];
    for sig_idx in 0..num_signals {
        for i in 0..n {
            v[A::real_idx(ComplexNumberQuery {
                elem_idx: i as u32,
                signal_idx: sig_idx as u32,
                signal_length: n as u32,
                num_signals: num_signals as u32,
            }) as usize] = f32::sin(2.0 * (i as f32) / (n as f32)) as f32;
            v[A::imag_idx(ComplexNumberQuery {
                elem_idx: i as u32,
                signal_idx: sig_idx as u32,
                signal_length: n as u32,
                num_signals: num_signals as u32,
            }) as usize] = 0.0;
        }
    }
    v
}
