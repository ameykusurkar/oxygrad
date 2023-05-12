# oxygrad

A simple library for automatic gradient.

```rust
use oxygrad::{var, Backward, Var};

let x = var!(3.0);
let x2 = x.clone() * x.clone();

// y = 6x^2 + 4x + 5;
let mut y = var!(6.0) * x2.clone() + var!(4.0) * x.clone() + var!(5.0);

assert_eq!(y.data(), 71.0);

y.backward(1.0);
assert_eq!(y.grad(), 1.0); // dy/dy
assert_eq!(x.grad(), 40.0); // dy/dx
```
