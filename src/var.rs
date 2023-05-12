use std::cell::RefCell;
use std::ops::{Add, Mul};
use std::rc::Rc;

use crate::{Backward, Product, Sum};

pub struct Value {
    data: f32,
    grad: f32,
}

impl Value {
    fn new(data: f32) -> Self {
        Self { data, grad: 0.0 }
    }
}

impl Backward for Value {
    fn backward(&mut self, grad: f32) {
        self.grad += grad;
    }

    fn data(&self) -> f32 {
        self.data
    }

    fn grad(&self) -> f32 {
        self.grad
    }
}

#[derive(Clone)]
pub struct Var {
    val: Rc<RefCell<Value>>,
}

impl Var {
    pub fn new(data: f32) -> Self {
        Self {
            val: Rc::new(RefCell::new(Value::new(data))),
        }
    }
}

impl Backward for Var {
    fn backward(&mut self, grad: f32) {
        self.val.backward(grad);
    }

    fn data(&self) -> f32 {
        self.val.data()
    }

    fn grad(&self) -> f32 {
        self.val.grad()
    }
}

impl<R> Add<R> for Var
where
    R: Backward + Clone,
{
    type Output = Sum<Self, R>;

    fn add(self, rhs: R) -> Self::Output {
        Sum::new(self, rhs)
    }
}

impl<R> Mul<R> for Var
where
    R: Backward + Clone,
{
    type Output = Product<Self, R>;

    fn mul(self, rhs: R) -> Self::Output {
        Product::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_grad() {
        let mut val = Var::new(3.0);
        assert_eq!(val.data(), 3.0);
        assert_eq!(val.grad(), 0.0);

        val.backward(1.0);
        assert_eq!(val.grad(), 1.0);
    }
}
