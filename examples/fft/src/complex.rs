use cubecl::prelude::*;

#[derive(CubeType, Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub real: f32,
    pub imag: f32,
}

#[cube]
pub fn unit_complex_from_angle(angle_radians: f32) -> Complex {
    Complex {
        real: f32::cos(angle_radians),
        imag: -f32::sin(angle_radians),
    }
}

#[cube]
pub fn make_complex(real: f32, imag: f32) -> Complex {
    Complex { real, imag }
}

#[cube]
pub fn add(x: Complex, y: Complex) -> Complex {
    Complex {
        real: x.real + y.real,
        imag: x.imag + y.imag,
    }
}

#[cube]
pub fn sub(x: Complex, y: Complex) -> Complex {
    Complex {
        real: x.real - y.real,
        imag: x.imag - y.imag,
    }
}

#[cube]
pub fn mul(x: Complex, y: Complex) -> Complex {
    let k1 = y.real * (x.real + x.imag);
    let k2 = x.real * (y.imag - y.real);
    let k3 = x.imag * (y.real + y.imag);
    Complex {
        real: k1 - k3,
        imag: k1 + k2,
    }
}

#[cube]
pub fn mul_i(x: Complex) -> Complex {
    Complex {
        real: -x.imag,
        imag: x.real,
    }
}

#[cube]
pub fn complement(x: Complex) -> Complex {
    Complex {
        real: x.real,
        imag: -x.imag,
    }
}
