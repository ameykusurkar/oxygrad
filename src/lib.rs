use std::ops::{Add, Mul};

pub use crate::backward::Backward;
use crate::product::Product;
use crate::sum::Sum;
pub use crate::var::Var;

mod backward;
mod product;
mod sum;
mod var;

macro_rules! impl_bin_add {
    ($t:ident) => {
        impl<R, A, B> Add<R> for $t<A, B>
        where
            R: Backward + Clone,
            A: Backward + Clone,
            B: Backward + Clone,
        {
            type Output = Sum<Self, R>;

            fn add(self, rhs: R) -> Self::Output {
                Sum::new(self, rhs)
            }
        }
    };
}
impl_bin_add!(Sum);
impl_bin_add!(Product);

macro_rules! impl_bin_mul {
    ($t:ident) => {
        impl<R, A, B> Mul<R> for $t<A, B>
        where
            R: Backward + Clone,
            A: Backward + Clone,
            B: Backward + Clone,
        {
            type Output = Product<Self, R>;

            fn mul(self, rhs: R) -> Self::Output {
                Product::new(self, rhs)
            }
        }
    };
}
impl_bin_mul!(Sum);
impl_bin_mul!(Product);

#[macro_export]
macro_rules! var {
    ($value:expr) => {
        Var::new($value)
    };
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lots_add_mul_grad() {
        let a = var!(3.0);
        let b = var!(2.0);
        let c = var!(4.0);

        // d = a^2 + bc
        let mut d = a.clone() * a.clone() + b.clone() * c.clone();

        assert_eq!(d.data(), 17.0);

        d.backward(1.0);
        assert_eq!(d.grad(), 1.0);
        assert_eq!(a.grad(), 6.0);
        assert_eq!(b.grad(), 4.0);
        assert_eq!(c.grad(), 2.0);
    }
}
