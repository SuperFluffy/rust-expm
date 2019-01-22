# Matrix exponentiation in Rust

This crate contains `expm`, an implementation of Algorithm 6.1 by [Al-Mohy, Higham] in the Rust
programming language. It calculates the exponential of a matrix. See the linked paper for more
information.

[Al-Mohy, Higham]: http://eprints.ma.man.ac.uk/1300/1/covered/MIMS_ep2009_9.pdf 

It uses the excellent [`rust-ndarray`] crate for matrix storage.

[`rust-ndarray`]: https://github.com/rust-ndarray/ndarray

## Example usage

The example below calculates the exponential of the unit matrix.

**Important:** You need to explicitly link to a BLAS + LAPACK provider such as `openblas_src`.
See the explanations given at the [`blas-lapack-rs` organization].

[`blas-lapack-rs` organization]: https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki

```rust
extern crate openblas_src;
use approx::assert_ulps_eq;
#[test]
fn exp_of_unit() {
    let n = 5;
    let a = ndarray::Array2::eye(n);
    let mut b = unsafe { ndarray::Array2::<f64>::uninitialized((n, n)) };

    crate::expm(&a, &mut b);

    for &elem in &b.diag() {
        assert_ulps_eq!(elem, 1f64.exp(), max_ulps=1);
    }
}
```

## TODO

Care was taken to implement the algorithm with performance in mind. As such, no extra allocations
after the initial setup of the `Expm` struct are done, and `Expm` can be reused to repeatedly
calculate the exponential of matrices with the same dimension.

However, profiling the code might reveal ways to improve it.

+ [ ] Profile code and optimize;
+ [ ] Write more tests to verify the goodness of the algorithm for difficult matrices;
+ [ ] Tune the parameters `t` and `itmax` when calculating the 1-norms (right now, `t=2`, `itmax=5`, which is what Scipy and Matlab are doing);
+ [ ] Ensure that the compiler performs const propagation when calculating the Padé coefficients;
+ [ ] Test the Padé coefficient against their hard-coded results;
+ [ ] Evaluate, whether the unsafe blocks are really necessary in all instances or could be removed.

