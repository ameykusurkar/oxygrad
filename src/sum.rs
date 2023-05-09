use crate::{Backward, Var};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_grad() {
        let a = Var::new(3.0);
        let b = Var::new(2.0);
        let mut c = a.clone() + b.clone();

        assert_eq!(c.data(), 5.0);

        c.backward(1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn twice_add_grad() {
        let a = Var::new(3.0);
        let mut c = a.clone() + a.clone();

        assert_eq!(c.data(), 6.0);

        c.backward(1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.grad(), 2.0);
    }
}
