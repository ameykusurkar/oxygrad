use std::cell::RefCell;
use std::rc::Rc;

pub trait Backward {
    fn backward(&mut self, grad: f32);
    fn data(&self) -> f32;
    fn grad(&self) -> f32;
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
