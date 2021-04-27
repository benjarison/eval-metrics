use std::ops::{Add, Sub, Mul, Div, AddAssign};

///
/// Represents a floating point number which can either be single (f32) or double (f64) precision
///
pub trait Float:
    private::Sealed +
    Copy +
    Add<Self, Output=Self> +
    Sub<Self, Output=Self> +
    Mul<Self, Output=Self> +
    Div<Self, Output=Self> +
    AddAssign +
    PartialOrd {

    ///
    /// Computes the absolute value
    ///
    fn abs(self) -> Self;

    ///
    /// Compute the square root
    ///
    fn sqrt(self) -> Self;

    ///
    /// Indicates whether or not the value is finite
    ///
    fn is_finite(self) -> bool;

    ///
    /// Provides a representation of the number zero
    ///
    fn zero() -> Self;

    ///
    /// Provides a representation of the number one
    ///
    fn one() -> Self;

    ///
    /// Constructs a float from an f32 value
    ///
    fn from_f32(x: f32) -> Self;

    ///
    /// Constructs a float from an f64 value
    ///
    fn from_f64(x: f64) -> Self;

    ///
    /// Constructs a float from a usize value
    ///
    fn from_usize(x: usize) -> Self;
}

///
/// Implementation for f32 single-precision values
///
impl Float for f32 {
    fn abs(self) -> Self {self.abs()}
    fn sqrt(self) -> Self {self.sqrt()}
    fn is_finite(self) -> bool {self.is_finite()}
    fn zero() -> Self {0.0_f32}
    fn one() -> Self {1.0_f32}
    fn from_f32(x: f32) -> Self {x}
    fn from_f64(x: f64) -> Self {x as f32}
    fn from_usize(x: usize) -> Self {x as f32}
}

///
/// Implementation for f64 double-precision values
///
impl Float for f64 {
    fn abs(self) -> Self {self.abs()}
    fn sqrt(self) -> Self {self.sqrt()}
    fn is_finite(self) -> bool {self.is_finite()}
    fn zero() -> Self {0.0}
    fn one() -> Self {1.0}
    fn from_f32(x: f32) -> Self {x as f64}
    fn from_f64(x: f64) -> Self {x}
    fn from_usize(x: usize) -> Self {x as f64}
}

mod private {

    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
