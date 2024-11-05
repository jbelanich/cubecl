use bytemuck::{Pod, Zeroable};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use half::f16;
use num_traits::{NumCast, ToPrimitive};
use serde::Serialize;
use std::{mem::transmute, num::NonZero};

use crate::{
    ir::{Elem, FloatKind, Item},
    prelude::Numeric,
    unexpanded,
};

use super::{
    init_expand_element, CubeContext, CubePrimitive, CubeType, ExpandElement,
    ExpandElementBaseInit, ExpandElementTyped, Float, Init, IntoRuntime, KernelBuilder,
    KernelLauncher, LaunchArgExpand, Runtime, ScalarArgSettings, Vectorized,
};

/// A 19-bit floating point type implementing the [`tfloat32`] format.
///
/// The [`tfloat32`] floating point format is a truncated 19-bit version of the IEEE 754 standard
/// `binary32`, a.k.a [`f32`]. [`bf16`] has approximately the same dynamic range as [`f32`] but a
/// a lower precision equal to [`f16`][half::f16].
///
/// [`tfloat32`]: https://en.wikipedia.org/wiki/TensorFloat-32
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Clone, Copy, Default, Serialize, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct tf32(f32);

impl tf32 {
    /// Constructs a [`tf32`] value from the raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u32) -> tf32 {
        tf32(unsafe { transmute::<u32, f32>(bits) })
    }

    /// Constructs a [`tf32`] value from a 32-bit floating point value.
    ///
    /// This operation is lossy. If the 32-bit value is too large to fit, ±∞ will result. NaN values
    /// are preserved. Subnormal values that are too tiny to be represented will result in ±0. All
    /// other values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f32(value: f32) -> tf32 {
        tf32(value)
    }

    /// Constructs a [`tf32`] value from a 64-bit floating point value.
    ///
    /// This operation is lossy. If the 64-bit value is to large to fit, ±∞ will result. NaN values
    /// are preserved. 64-bit subnormal values are too tiny to be represented and result in ±0.
    /// Exponents that underflow the minimum exponent will result in subnormals or ±0. All other
    /// values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f64(value: f64) -> tf32 {
        tf32(value as f32)
    }

    /// Converts a [`tf32`] into the underlying bit representation.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u32 {
        unsafe { transmute(self.0) }
    }

    /// Converts a [`tf32`] value into an [`f32`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f32`].
    #[inline]
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        self.0
    }

    /// Converts a [`tf32`] value into an [`f64`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f64`].
    #[inline]
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        self.0 as f64
    }
}

impl Mul for tf32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulAssign for tf32 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for tf32 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl DivAssign for tf32 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Add for tf32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddAssign for tf32 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for tf32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl SubAssign for tf32 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ToPrimitive for tf32 {
    fn to_i64(&self) -> Option<i64> {
        Some(tf32::to_f32(*self) as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(tf32::to_f64(*self) as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(tf32::to_f32(*self))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(tf32::to_f64(*self))
    }
}

impl NumCast for tf32 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(Self::from_f32(n.to_f32()?))
    }
}

impl CubeType for tf32 {
    type ExpandType = ExpandElementTyped<tf32>;
}

impl CubePrimitive for tf32 {
    /// Return the element type to use on GPU
    fn as_elem() -> Elem {
        Elem::Float(FloatKind::TF32)
    }
}

impl IntoRuntime for tf32 {
    fn __expand_runtime_method(self, context: &mut CubeContext) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = self.into();
        Init::init(expand, context)
    }
}

impl Numeric for tf32 {
    const MAX: Self = tf32::from_f32(f32::MAX);
    const MIN: Self = tf32::from_f32(f32::MIN);
}

impl Vectorized for tf32 {
    fn vectorization_factor(&self) -> u32 {
        1
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl ExpandElementBaseInit for tf32 {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl Float for tf32 {
    const DIGITS: u32 = 32;

    const EPSILON: Self = tf32::from_f32(half::f16::EPSILON.to_f32_const());

    const INFINITY: Self = tf32::from_f32(f32::INFINITY);

    const MANTISSA_DIGITS: u32 = 10;

    /// Maximum possible [`tf32`] power of 10 exponent
    const MAX_10_EXP: i32 = 38;
    /// Maximum possible [`tf32`] power of 2 exponent
    const MAX_EXP: i32 = 128;

    /// Minimum possible normal [`tf32`] power of 10 exponent
    const MIN_10_EXP: i32 = -37;
    /// One greater than the minimum possible normal [`v`] power of 2 exponent
    const MIN_EXP: i32 = -125;

    /// `MIN_POSITIVE` is defined by precision, so use `f16` as reference
    const MIN_POSITIVE: Self = tf32(f16::MIN_POSITIVE.to_f32_const());

    const NAN: Self = tf32::from_f32(f32::NAN);

    const NEG_INFINITY: Self = tf32::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        tf32(val)
    }

    fn vectorized(_val: f32, _vectorization: u32) -> Self {
        unexpanded!()
    }

    fn vectorized_empty(_vectorization: u32) -> Self {
        unexpanded!()
    }

    fn __expand_vectorized_empty(
        context: &mut super::CubeContext,
        vectorization: u32,
    ) -> <Self as super::CubeType>::ExpandType {
        context
            .create_local_variable(Item::vectorized(
                Self::as_elem(),
                NonZero::new(vectorization as u8),
            ))
            .into()
    }
}

impl ScalarArgSettings for tf32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32((*self).to_f32());
    }
}

impl LaunchArgExpand for tf32 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder.scalar(tf32::as_elem()).into()
    }
}