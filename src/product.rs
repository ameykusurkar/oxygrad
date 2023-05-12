use crate::{Backward, Var};

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
            var: Var::from(data),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_grad() {
        let a = Var::from(3.0);
        let b = Var::from(2.0);
        let mut c = Product::new(a.clone(), b.clone());

        assert_eq!(c.data(), 6.0);

        c.backward(1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 2.0);
        assert_eq!(b.grad(), 3.0);
    }

    #[test]
    fn twice_mul_grad() {
        let a = Var::from(3.0);
        let mut c = Product::new(a.clone(), a.clone());

        assert_eq!(c.data(), 9.0);

        c.backward(1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 6.0);
    }
}
