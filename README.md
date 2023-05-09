# oxygrad

A simple library for automatic gradient.

```rust
let a = var!(3.0);
let b = var!(2.0);
let c = var!(4.0);

// d = a^2 + bc
let mut d = a.clone() * a.clone() + b.clone() * c.clone();

assert_eq!(d.data(), 17.0);

d.backward(1.0); // backpropagation
assert_eq!(d.grad(), 1.0); // dd/dd
assert_eq!(a.grad(), 6.0); // dd/da
assert_eq!(b.grad(), 4.0); // dd/db
assert_eq!(c.grad(), 2.0); // dd/dc
```
