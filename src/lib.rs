use std::cell::RefCell;
use std::rc::Rc;

pub trait Backward {
    fn backward(&mut self, grad: f32);
    fn data(&self) -> f32;
    fn grad(&self) -> f32;
}

struct Value {
    data: f32,
    grad: f32,
}

impl Value {
    pub fn new(data: f32) -> Self {
        Self { data, grad: 0.0 }
    }
}

impl<T: Backward> Backward for Rc<RefCell<T>> {
    fn backward(&mut self, grad: f32) {
        self.borrow_mut().backward(grad)
    }

    fn data(&self) -> f32 {
        self.borrow().data()
    }

    fn grad(&self) -> f32 {
        self.borrow().grad()
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

#[derive(Clone)]
pub struct Sum<L, R>
where
    L: Backward + Clone,
    R: Backward + Clone,
{
    var: Var,
    left: L,
    right: R,
}

impl<L, R> Sum<L, R>
where
    L: Backward + Clone,
    R: Backward + Clone,
{
    pub fn new(left: L, right: R) -> Self {
        let data = left.data() + right.data();
        Self {
            var: Var::new(data),
            left,
            right,
        }
    }
}

impl<L, R> Backward for Sum<L, R>
where
    L: Backward + Clone,
    R: Backward + Clone,
{
    fn backward(&mut self, grad: f32) {
        self.var.backward(grad);
        self.left.backward(grad);
        self.right.backward(grad);
    }

    fn data(&self) -> f32 {
        self.var.data()
    }

    fn grad(&self) -> f32 {
        self.var.grad()
    }
}

#[derive(Clone)]
pub struct Product<L, R>
where
    L: Backward + Clone,
    R: Backward + Clone,
{
    var: Var,
    left: L,
    right: R,
}

impl<L, R> Product<L, R>
where
    L: Backward + Clone,
    R: Backward + Clone,
{
    pub fn new(left: L, right: R) -> Self {
        let data = left.data() * right.data();
        Self {
            var: Var::new(data),
            left,
            right,
        }
    }
}

impl<L, R> Backward for Product<L, R>
where
    L: Backward + Clone,
    R: Backward + Clone,
{
    fn backward(&mut self, grad: f32) {
        self.var.backward(grad);
        self.left.backward(self.right.data() * grad);
        self.right.backward(self.left.data() * grad);
    }

    fn data(&self) -> f32 {
        self.var.data()
    }

    fn grad(&self) -> f32 {
        self.var.grad()
    }
}

#[macro_export]
macro_rules! var {
    ($value:expr) => {
        Var::new($value)
    };
}

#[macro_export]
macro_rules! add {
    ($a:expr, $b:expr) => {
        Sum::new($a.clone(), $b.clone())
    };
}

#[macro_export]
macro_rules! mul {
    ($a:expr, $b:expr) => {
        Product::new($a.clone(), $b.clone())
    };
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

    #[test]
    fn add_grad() {
        let a = var!(3.0);
        let b = var!(2.0);
        let mut c = add!(a, b);

        assert_eq!(c.data(), 5.0);

        c.backward(1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn twice_add_grad() {
        let a = var!(3.0);
        let mut c = add!(a, a);

        assert_eq!(c.data(), 6.0);

        c.backward(1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 2.0);
    }

    #[test]
    fn mul_grad() {
        let a = var!(3.0);
        let b = var!(2.0);
        let mut c = mul!(a, b);

        assert_eq!(c.data(), 6.0);

        c.backward(1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 2.0);
        assert_eq!(b.grad(), 3.0);
    }

    #[test]
    fn twice_mul_grad() {
        let a = var!(3.0);
        let mut c = mul!(a, a);

        assert_eq!(c.data(), 9.0);

        c.backward(1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 6.0);
    }

    #[test]
    fn lots_add_mul_grad() {
        let a = var!(3.0);
        let b = var!(2.0);
        let c = var!(4.0);
        let aa = mul!(a, a);
        let bc = mul!(b, c);

        // d = a^2 + bc
        let mut d = add!(aa, bc);

        assert_eq!(d.data(), 17.0);

        d.backward(1.0);
        assert_eq!(d.grad(), 1.0);
        assert_eq!(a.grad(), 6.0);
        assert_eq!(b.grad(), 4.0);
        assert_eq!(c.grad(), 2.0);
    }
}
