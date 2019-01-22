/// This crate contains `expm`, an implementation of Algorithm 6.1 by [Al-Mohy, Higham] in the Rust
/// programming language. It calculates the exponential of a matrix. See the linked paper for more
/// information.
///
/// An important ingredient is `normest1`, Algorithm 2.4 in [Higham, Tisseur], which estimates
/// the 1-norm of a matrix.
///
/// Furthermore, to fully understand the algorithm as described in the original paper, one has to
/// understand that the factor $\lvert C_{2m+1} \rvert$ arises during the Padé approximation of the
/// exponential function. The derivation is described in [Gautschi 2012], pp. 363--365, and the
/// factor reads:
///
/// \begin{equation}
/// C_{n,m} = (-1)^n \frac{n!m!}{(n+m)!(n+m+1)!},
/// \end{equation}
///
/// or using only the diagonal elements, $m=n$:
///
/// \begin{equation}
/// C_m = (-1)^m \frac{m!m!}{(2m)!(2m+1)!}
/// \end{equation}
/// 
///
/// [Al-Mohy, Higham]: http://eprints.ma.man.ac.uk/1300/1/covered/MIMS_ep2009_9.pdf 
/// [Higham, Tisseur]: http://eprints.ma.man.ac.uk/321/1/covered/MIMS_ep2006_145.pdf
/// [Gautschi 2012]: https://doi.org/10.1007/978-0-8176-8259-0

use condest::Normest1;
use ndarray::{
    self,
    prelude::*,
    Data,
    DataMut,
    Dimension,
    Zip
};

// Can we calculate these at compile time?
const THETA_3: f64 = 1.495585217958292e-2;
const THETA_5: f64 = 2.539398330063230e-1;
const THETA_7: f64 = 9.504178996162932e-1;
const THETA_9: f64 = 2.097847961257068e0;
// const THETA_13: f64 = 5.371920351148152e0 // Alg 3.1
const THETA_13: f64 = 4.25; // Alg 5.1

/// Calculates the i-th coefficient arising in the [m/m] Padé approximant of the exponential
/// function.
fn pade_coefficient(i: u64, m: u64) -> f64 {
    use statrs::function::factorial::factorial;

    assert!(i <= m, "The i-th coefficient for a [m/m] Padé approximant is undefined for i > m.");

    // TODO: Check if the order of multiplications and divisions should be adapted to always
    // multiply quantities of similar magnitude, i.e. to not lose precision to floating point
    // arithmetic
    factorial(2*m - i) * factorial(m) / factorial(2*m) / factorial(m-i) / factorial(i)
}

/// Calculates the of leading terms in the backward error function for the [m/m] Padé approximant
/// to the exponential function, i.e. it calculates:
///
/// \begin{align}
///     C_{2m+1} &= \frac{(m!)^2}{(2m)!(2m+1)!} \\
///              &= \frac{1}{\binom{2m}{m} (2m+1)!}
/// \end{align}
///
/// NOTE: Depending on the notation used in the scientific papers, the coefficient `C` is,
/// confusingly, sometimes indexed `C_i` and sometimes `C_{2m+1}`. These essentially mean the same
/// thing and is due to the power series expansion of the backward error function:
///
/// \begin{equation}
///     h(x) = \sum^\infty_{i=2m+1} C_i x^i
/// \end{equation}
fn pade_error_coefficient(m: u64) -> f64 {
    use statrs::function::factorial::{binomial, factorial};

    return 1.0 / ( binomial(2*m, m) * factorial(2*m + 1) )
}

#[allow(non_camel_case_types)]
struct PadeOrder_3;
#[allow(non_camel_case_types)]
struct PadeOrder_5;
#[allow(non_camel_case_types)]
struct PadeOrder_7;
#[allow(non_camel_case_types)]
struct PadeOrder_9;
#[allow(non_camel_case_types)]
struct PadeOrder_13;

enum PadeOrders {
    _3,
    _5,
    _7,
    _9,
    _13,
}

trait PadeOrder {
    const ORDER: u64;

    /// Return the coefficients arising in both the numerator as well as in the denominator of the
    /// Padé approximant (they are the same, due to $p(x) = q(-x)$.
    ///
    /// TODO: This is a great usecase for const generics, returning &'static [u64; Self::ORDER],
    /// once RFC 2000 lands. See the PR https://github.com/rust-lang/rust/pull/53645
    unsafe fn coefficients() -> &'static [f64];

    fn calculate_pade_sums<S1, S2, S3>(a: &ArrayBase<S1, Ix2>, a_powers: &[&ArrayBase<S1, Ix2>], u: &mut ArrayBase<S2, Ix2>, v: &mut ArrayBase<S3, Ix2>, work: &mut ArrayBase<S2, Ix2>)
        where S1: Data<Elem=f64>,
              S2: DataMut<Elem=f64>,
              S3: DataMut<Elem=f64>;
}

macro_rules! impl_padeorder {
    ($($ty:ty, $m:literal, $coeff_slice:ident),+) => {

$(

static mut $coeff_slice: [f64; $m+1] = [0.0; $m + 1];

impl PadeOrder for $ty {
    const ORDER: u64 = $m;

    // TODO: Check if the compiler performs const-propagation, i.e. calculates the
    // coefficients at compile time. Potential ...
    // ... FIXME: If the compiler does not perform const-propagation, replace the
    // coefficients by their hardcoded values.
    unsafe fn coefficients() -> &'static [f64] {
        assert!($m > 0);
        {
            let mut coeff_iter = $coeff_slice.iter_mut().enumerate().rev();

            let highest_order_coeff;
            {
                // NOTE: Guaranteed to work due to assert! above.
                let (i, coeff) = coeff_iter.next().unwrap();
                highest_order_coeff = pade_coefficient(i as u64, $m);
                *coeff = 1.0;
            }

            for (i, elem) in coeff_iter {
                *elem = pade_coefficient(i as u64, $m) / highest_order_coeff;
            }
        }

        &$coeff_slice
    }

    fn calculate_pade_sums<S1, S2, S3>(
        a: &ArrayBase<S1, Ix2>,
        a_powers: &[&ArrayBase<S1, Ix2>],
        u: &mut ArrayBase<S2, Ix2>,
        v: &mut ArrayBase<S3, Ix2>,
        work: &mut ArrayBase<S2, Ix2>,
    )
        where S1: Data<Elem=f64>,
              S2: DataMut<Elem=f64>,
              S3: DataMut<Elem=f64>,
    {
        assert_eq!(a_powers.len(), ($m - 1)/2 + 1);

        let (n_rows, n_cols) = a.dim();
        assert_eq!(n_rows, n_cols, "Pade sum only defined for square matrices.");
        let n = n_rows as i32;

        // Iterator to get 2 coefficients, c_{2i} and c_{2i+1}, and 1 matrix power at a time.
        let mut iterator = unsafe { Self::coefficients().chunks_exact(2).zip(a_powers.iter()) };

        // First element from the iterator.
        //
        // NOTE: The unwrap() and unreachable!() are permissable because the assertion above
        // ensures the validity.
        //
        // TODO: An optimization is probably to just set u and v to zero and only assign the
        // coefficients to its diagonal, given that A_0 = A^0 = 1.
        let (c_0, c_1, a_pow) = match iterator.next().unwrap() {
            (&[c_0, c_1], a_pow) => (c_0, c_1, a_pow),
            _ => unreachable!()
        };

        work.zip_mut_with(a_pow, |x, &y| *x = c_1 * y);
        v.zip_mut_with(a_pow, |x, &y| *x = c_0 * y);

        // Rest of the iterator
        while let Some(item) = iterator.next() {
            let (c_2k, c_2k1, a_pow) = match item {
                (&[c_2k, c_2k1], a_pow) => (c_2k, c_2k1, a_pow),
                _ => unreachable!()
            };

            work.zip_mut_with(a_pow, |x, &y| *x = *x + c_2k1 * y);
            v.zip_mut_with(a_pow, |x, &y| *x = *x + c_2k * y);
        }

        let (a_slice, a_layout) = as_slice_with_layout(a).expect("Matrix `a` not contiguous.");
        let (work_slice, _) = as_slice_with_layout(work).expect("Matrix `work` not contiguous.");
        let (u_slice, u_layout) = as_slice_with_layout_mut(u).expect("Matrix `u` not contiguous.");
        assert_eq!(a_layout, u_layout, "Memory layout mismatch between matrices; currently only row major matrices are supported.");
        let layout = a_layout;
        unsafe {
            cblas::dgemm(
                layout,
                cblas::Transpose::None,
                cblas::Transpose::None,
                n,
                n,
                n,
                1.0,
                a_slice,
                n,
                work_slice,
                n,
                0.0,
                u_slice,
                n,
            )
        }
    }
}

)+
}
}

impl_padeorder!(
    PadeOrder_3, 3, PADE_COEFFICIENTS_3,
    PadeOrder_5, 5, PADE_COEFFICIENTS_5,
    PadeOrder_7, 7, PADE_COEFFICIENTS_7,
    PadeOrder_9, 9, PADE_COEFFICIENTS_9
);

static mut PADE_COEFFICIENTS_13: [f64; 13 + 1] = [1.0; 13 + 1];

impl PadeOrder for PadeOrder_13 {
    const ORDER: u64 = 13;

    // TODO: Check if the compiler performs const-propagation, i.e. calculates the
    // coefficients at compile time. Potential ...
    // ... FIXME: If the compiler does not perform const-propagation, replace the
    // coefficients by their hardcoded values.
    unsafe fn coefficients() -> &'static [f64] {
        assert!(13 > 0);
        let mut coeff_iter = PADE_COEFFICIENTS_13.iter_mut().enumerate().rev();

        let highest_order_coeff;
        {
            // NOTE: Guaranteed to work due to assert! above.
            let (i, coeff) = coeff_iter.next().unwrap();
            highest_order_coeff = pade_coefficient(i as u64, 13);
            *coeff = 1.0;
        }

        for (i, elem) in coeff_iter {
            *elem = pade_coefficient(i as u64, 13) / highest_order_coeff;
        }

        &PADE_COEFFICIENTS_13
    }

    fn calculate_pade_sums<S1, S2, S3>(
        a: &ArrayBase<S1, Ix2>,
        a_powers: &[&ArrayBase<S1, Ix2>],
        u: &mut ArrayBase<S2, Ix2>,
        v: &mut ArrayBase<S3, Ix2>,
        work: &mut ArrayBase<S2, Ix2>,
    )
        where S1: Data<Elem=f64>,
              S2: DataMut<Elem=f64>,
              S3: DataMut<Elem=f64>,
    {
        assert_eq!(a_powers.len(), (13 - 1)/2 + 1);

        let (n_rows, n_cols) = a.dim();
        assert_eq!(n_rows, n_cols, "Pade sum only defined for square matrices.");
        let n = n_rows;

        let coefficients = unsafe { Self::coefficients() };

        Zip::from(&mut *work)
            .and(a_powers[0])
            .and(a_powers[1])
            .and(a_powers[2])
            .and(a_powers[3])
            .apply(|x, &a0, &a2, &a4, &a6| {
                *x = *x + coefficients[1] * a0 + coefficients[3] * a2 + coefficients[5] * a4 + coefficients[7] * a6;
        });

        {
            let (a_slice, a_layout) = as_slice_with_layout(a).expect("Matrix `a` not contiguous.");
            let (work_slice, _) = as_slice_with_layout(work).expect("Matrix `work` not contiguous.");
            let (u_slice, u_layout) = as_slice_with_layout_mut(u).expect("Matrix `u` not contiguous.");
            assert_eq!(a_layout, u_layout, "Memory layout mismatch between matrices; currently only row major matrices are supported.");
            let layout = a_layout;
            unsafe {
                cblas::dgemm(
                    layout,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    n as i32,
                    n as i32,
                    n as i32,
                    1.0,
                    a_slice,
                    n as i32,
                    work_slice,
                    n as i32,
                    0.0,
                    u_slice,
                    n as i32,
                )
            }
        }

        Zip::from(&mut *work)
            .and(a_powers[1])
            .and(a_powers[2])
            .and(a_powers[3])
            .apply(|x, &a2, &a4, &a6| {
                *x = coefficients[8] * a2 + coefficients[10] * a4 + coefficients[12] * a6;
        });

        {
            let (a6_slice, a6_layout) = as_slice_with_layout(a_powers[3]).expect("Matrix `a6` not contiguous.");
            let (work_slice, _) = as_slice_with_layout(work).expect("Matrix `work` not contiguous.");
            let (v_slice, v_layout) = as_slice_with_layout_mut(v).expect("Matrix `v` not contiguous.");
            assert_eq!(a6_layout, v_layout, "Memory layout mismatch between matrices; currently only row major matrices are supported.");
            let layout = a6_layout;
            unsafe {
                cblas::dgemm(
                    layout,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    n as i32,
                    n as i32,
                    n as i32,
                    1.0,
                    a6_slice,
                    n as i32,
                    work_slice,
                    n as i32,
                    0.0,
                    v_slice,
                    n as i32,
                )
            }
        }

        Zip::from(v)
            .and(a_powers[0])
            .and(a_powers[1])
            .and(a_powers[2])
            .and(a_powers[3])
            .apply(|x, &a0, &a2, &a4, &a6| {
                *x = *x + coefficients[0] * a0 + coefficients[2] * a2 + coefficients[4] * a4 + coefficients[6] * a6;
        })
    }
}

/// Storage for calculating the matrix exponential.
pub struct Expm {
    n: usize,
    itmax: usize,
    eye: Array2<f64>,
    a1: Array2<f64>,
    a2: Array2<f64>,
    a4: Array2<f64>,
    a6: Array2<f64>,
    a8: Array2<f64>,
    a_abs: Array2<f64>,
    u: Array2<f64>,
    work: Array2<f64>,
    pivot: Array1<i32>,
    normest1: Normest1,
    layout: cblas::Layout,
}

impl Expm {
    /// Allocates all space to calculate the matrix exponential for a square matrix of dimension
    /// n×n.
    pub fn new(n: usize) -> Self {
        let eye = Array2::<f64>::eye(n);
        let a1 = unsafe { Array2::<f64>::uninitialized((n, n)) };
        let a2 = unsafe { Array2::<f64>::uninitialized((n, n)) };
        let a4 = unsafe { Array2::<f64>::uninitialized((n, n)) };
        let a6 = unsafe { Array2::<f64>::uninitialized((n, n)) };
        let a8 = unsafe { Array2::<f64>::uninitialized((n, n)) };
        let a_abs = unsafe { Array2::<f64>::uninitialized((n, n)) };
        let u = unsafe { Array2::<f64>::uninitialized((n, n)) };
        let work = unsafe { Array2::<f64>::uninitialized((n, n)) };
        let pivot = unsafe { Array1::<i32>::uninitialized(n) };
        let layout = cblas::Layout::RowMajor;

        // TODO: Investigate what an optimal value for t is when estimating the 1-norm.
        // Python's SciPY uses t=2. Why?
        let t = 2;
        let itmax = 5;

        let normest1 = Normest1::new(n, t);

        Expm {
            n,
            itmax,
            eye,
            a1,
            a2,
            a4,
            a6,
            a8,
            a_abs,
            u,
            work,
            pivot,
            normest1,
            layout,
        }
    }

    /// Calculate the matrix exponential of the n×n matrix `a` storing the result in matrix `b`.
    ///
    /// NOTE: Panics if input matrices `a` and `b` don't have matching dimensions, are not square,
    /// not in row-major order, or don't have the same dimension as the `Expm` object `expm` is
    /// called on.
    pub fn expm<S1, S2>(&mut self, a: &ArrayBase<S1, Ix2>, b: &mut ArrayBase<S2, Ix2>)
        where S1: Data<Elem=f64>,
              S2: DataMut<Elem=f64>,
    {
        assert_eq!(a.dim(), b.dim(), "Input matrices `a` and `b` have to have matching dimensions.");
        let (n_rows, n_cols) = a.dim();
        assert_eq!(n_rows, n_cols, "expm is only implemented for square matrices.");
        assert_eq!(n_rows, self.n, "Dimension mismatch between matrix `a` and preconfigured `Expm` struct.");

        // Rename b to v to be in line with the nomenclature of the original paper.
        let v = b;

        self.a1.assign(a);

        let n = self.n as i32;

        {
            let (a_slice, a_layout) = as_slice_with_layout(&self.a1).expect("Matrix `a` not contiguous.");
            let (a2_slice, _) = as_slice_with_layout_mut(&mut self.a2).expect("Matrix `a2` not contiguous.");
            assert_eq!(a_layout, self.layout, "Memory layout mismatch between matrices; currently only row major matrices are supported.");
            unsafe {
                cblas::dgemm(
                    self.layout,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    n,
                    n,
                    n,
                    1.0,
                    a_slice,
                    n,
                    a_slice,
                    n,
                    0.0,
                    a2_slice,
                    n as i32,
                )
            }
        }

        let d4_estimated = self.normest1.normest1_pow(&self.a2, 2, self.itmax).powf(1.0/4.0);
        let d6_estimated = self.normest1.normest1_pow(&self.a2, 3, self.itmax).powf(1.0/6.0);
        let eta_1 = d4_estimated.max(d6_estimated);

        if eta_1 <= THETA_3 && self.ell(3) == 0 {
            println!("eta_1 condition");
            self.solve_via_pade(PadeOrders::_3, v);
            return;
        }

        {
            let (a2_slice, _) = as_slice_with_layout(&self.a2).expect("Matrix `a2` not contiguous.");
            let (a4_slice, _) = as_slice_with_layout_mut(&mut self.a4).expect("Matrix `a4` not contiguous.");
            unsafe {
                cblas::dgemm(
                    self.layout,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    self.n as i32,
                    self.n as i32,
                    self.n as i32,
                    1.0,
                    a2_slice,
                    n as i32,
                    a2_slice,
                    n as i32,
                    0.0,
                    a4_slice,
                    n as i32,
                )
            }
        }

        let d4_precise = self.normest1.normest1(&self.a4, self.itmax).powf(1.0/4.0);
        let eta_2 = d4_precise.max(d6_estimated);

        if eta_2 <= THETA_5 && self.ell(5) == 0 {
            println!("eta_2 condition");
            self.solve_via_pade(PadeOrders::_5, v);
            return;
        }

        {
            let (a2_slice, _) = as_slice_with_layout(&self.a2).expect("Matrix `a2` not contiguous.");
            let (a4_slice, _) = as_slice_with_layout(&self.a4).expect("Matrix `a4` not contiguous.");
            let (a6_slice, _) = as_slice_with_layout_mut(&mut self.a6).expect("Matrix `a6` not contiguous.");
            unsafe {
                cblas::dgemm(
                    self.layout,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    self.n as i32,
                    self.n as i32,
                    self.n as i32,
                    1.0,
                    a2_slice,
                    n as i32,
                    a4_slice,
                    n as i32,
                    0.0,
                    a6_slice,
                    n as i32,
                )
            }
        }

        let d6_precise = self.normest1.normest1(&self.a6, self.itmax).powf(1.0/6.0);
        let d8_estimated = self.normest1.normest1_pow(&self.a4, 2, self.itmax);
        let eta_3 = d6_precise.max(d8_estimated);

        if eta_3 <= THETA_7 && self.ell(7) == 0 {
            println!("eta_3 (first) condition");
            self.solve_via_pade(PadeOrders::_7, v);
            return;
        }

        {
            let (a4_slice, _) = as_slice_with_layout(&self.a4).expect("Matrix `a4` not contiguous.");
            let (a8_slice, _) = as_slice_with_layout_mut(&mut self.a8).expect("Matrix `a8` not contiguous.");
            unsafe {
                cblas::dgemm(
                    self.layout,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    self.n as i32,
                    self.n as i32,
                    self.n as i32,
                    1.0,
                    a4_slice,
                    n as i32,
                    a4_slice,
                    n as i32,
                    0.0,
                    a8_slice,
                    n as i32,
                )
            }
        }

        if eta_3 <= THETA_9 && self.ell(9) == 0 {
            println!("eta_3 (second) condition");
            self.solve_via_pade(PadeOrders::_9, v);
            return;
        }

        let eta_4 = d8_estimated.max(self.normest1.normest1_prod(&[&self.a4, &self.a6], self.itmax).powf(1.0/10.0));
        let eta_5 = eta_3.min(eta_4);

        use std::f64;
        use std::cmp;
        let mut s = cmp::max(f64::ceil(f64::log2(eta_5/THETA_13)) as i32, 0);
        self.a1.mapv_inplace(|x| x / 2f64.powi(s));
        s = s + self.ell(13);
        self.a1.zip_mut_with(a, |x, &y| *x = y / 2f64.powi(s));
        self.a2.mapv_inplace(|x| x / 2f64.powi(2*s));
        self.a4.mapv_inplace(|x| x / 2f64.powi(4*s));
        self.a6.mapv_inplace(|x| x / 2f64.powi(6*s));

        self.solve_via_pade(PadeOrders::_13, v);

        // TODO: Call code fragment 2.1 in the paper if `a` is triangular, instead of the code below.
        //
        // NOTE: it's guaranteed that s >= 0 by its definition.
        let (u_slice, _) = as_slice_with_layout_mut(&mut self.u).expect("Matrix `u` not contiguous.");

        // NOTE: v initially contains r after `solve_via_pade`.
        let (v_slice, _) = as_slice_with_layout_mut(v).expect("Matrix `v` not contiguous.");
        for _ in 0..s {
            unsafe {
                cblas::dgemm(
                    self.layout,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    self.n as i32,
                    self.n as i32,
                    self.n as i32,
                    1.0,
                    v_slice,
                    n as i32,
                    v_slice,
                    n as i32,
                    0.0,
                    u_slice,
                    n as i32,
                )
            }

            u_slice.swap_with_slice(v_slice);
        }
    }

    /// A helper function (as it is called in the original paper) returning the
    /// $\max(\lceil \log_2(\alpha/u) / 2m \rceil, 0)$, where
    /// $\alpha = \lvert c_{2m+1}\rvert \texttt{normest}(\lvert A\rvert^{2m+1})/\lVertA\rVert_1$.
    fn ell(&mut self, m: usize) -> i32 {
        Zip::from(&mut self.a_abs)
            .and(&self.a1)
            .apply(|x, &y| *x = y.abs());

        let c2m1 = pade_error_coefficient(m as u64);

        let norm_abs_a_2m1 = self.normest1.normest1_pow(&self.a_abs, 2*m + 1, self.itmax);
        let norm_a = self.normest1.normest1(&self.a1, self.itmax);
        let alpha = c2m1.abs() * norm_abs_a_2m1 / norm_a;

        // The unit roundoff, defined as half the machine epsilon.
        let u = std::f64::EPSILON / 2.0;

        use std::f64;
        use std::cmp;

        cmp::max(0, f64::ceil( f64::log2(alpha/u) / (2 * m) as f64 ) as i32)
    }

    fn solve_via_pade<S>(&mut self, pade_order: PadeOrders, v: &mut ArrayBase<S, Ix2>)
        where S: DataMut<Elem=f64>,
    {
        use PadeOrders::*;

        macro_rules! pade {
            ($order:ty, [$(&$apow:expr),+]) => {
                <$order as PadeOrder>::calculate_pade_sums(&self.a1, &[$(&$apow),+], &mut self.u, v, &mut self.work);
            }
        }

        match pade_order {
            _3  => pade!(PadeOrder_3, [&self.eye, &self.a2]),
            _5  => pade!(PadeOrder_5, [&self.eye, &self.a2, &self.a4]),
            _7  => pade!(PadeOrder_7, [&self.eye, &self.a2, &self.a4, &self.a6]),
            _9  => pade!(PadeOrder_9, [&self.eye, &self.a2, &self.a4, &self.a6, &self.a8]),
            _13 => pade!(PadeOrder_13, [&self.eye, &self.a2, &self.a4, &self.a6]),
        };

        // Here we set v = p <- u + v and u = q <- -u + v, overwriting u and v via work.
        self.work.assign(v);

        Zip::from(&mut *v)
            .and(&self.u)
            .apply(|x, &y| {
                *x = *x + y;
        });

        Zip::from(&mut self.u)
            .and(&self.work)
            .apply(|x, &y| {
                *x = -*x + y;
        });

        let (u_slice, _) = as_slice_with_layout_mut(&mut self.u).expect("Matrix `u` not contiguous.");
        let (v_slice, _) = as_slice_with_layout_mut(v).expect("Matrix `v` not contiguous.");
        let (pivot_slice, _) = as_slice_with_layout_mut(&mut self.pivot).expect("Vector `pivot` not contiguous.");

        let n = self.n as i32;

        let layout = {
            match self.layout {
                cblas::Layout::ColumnMajor => lapacke::Layout::ColumnMajor,
                cblas::Layout::RowMajor => lapacke::Layout::RowMajor,
            }
        };

        // FIXME: Handle the info for error management.
        let _ = unsafe {
            lapacke::dgesv(
                layout,
                n,
                n,
                u_slice,
                n,
                pivot_slice,
                v_slice,
                n,
            )
        };
    }
}

/// Calculate the matrix exponential of the n×n matrix `a` storing the result in matrix `b`.
///
/// NOTE: Panics if input matrices `a` and `b` don't have matching dimensions, are not square,
/// not in row-major order, or don't have the same dimension as the `Expm` object `expm` is
/// called on.
pub fn expm<S1, S2>(a: &ArrayBase<S1, Ix2>, b: &mut ArrayBase<S2, Ix2>)
    where S1: Data<Elem=f64>,
          S2: DataMut<Elem=f64>,
{
    let (n, _) = a.dim();

    let mut expm = Expm::new(n);
    expm.expm(a, b);
}

/// Returns slice and layout underlying an array `a`.
fn as_slice_with_layout<S, T, D>(a: &ArrayBase<S, D>) -> Option<(&[T], cblas::Layout)>
    where S: Data<Elem=T>,
          D: Dimension
{
    if let Some(a_slice) = a.as_slice() {
        Some((a_slice, cblas::Layout::RowMajor))
    } else if let Some(a_slice) = a.as_slice_memory_order() {
        Some((a_slice, cblas::Layout::ColumnMajor))
    } else {
        None
    }
}

/// Returns mutable slice and layout underlying an array `a`.
fn as_slice_with_layout_mut<S, T, D>(a: &mut ArrayBase<S, D>) -> Option<(&mut [T], cblas::Layout)>
    where S: DataMut<Elem=T>,
          D: Dimension
{
    if a.as_slice_mut().is_some() {
        Some((a.as_slice_mut().unwrap(), cblas::Layout::RowMajor))
    } else if a.as_slice_memory_order_mut().is_some() {
        Some((a.as_slice_memory_order_mut().unwrap(), cblas::Layout::ColumnMajor))
    } else {
        None
    }
    // XXX: The above is a workaround for Rust not having non-lexical lifetimes yet.
    // More information here:
    // http://smallcultfollowing.com/babysteps/blog/2016/04/27/non-lexical-lifetimes-introduction/#problem-case-3-conditional-control-flow-across-functions
    //
    // if let Some(slice) = a.as_slice_mut() {
    //     Some((slice, cblas::Layout::RowMajor))
    // } else if let Some(slice) = a.as_slice_memory_order_mut() {
    //     Some((slice, cblas::Layout::ColumnMajor))
    // } else {
    //     None
    // }
}

#[cfg(test)]
mod tests {
    extern crate openblas_src;
    use ndarray::prelude::*;
    use approx::assert_ulps_eq;
    #[test]
    fn exp_of_unit() {
        let n = 5;
        let a = Array2::eye(n);
        let mut b = unsafe { Array2::<f64>::uninitialized((n, n)) };

        crate::expm(&a, &mut b);

        for &elem in &b.diag() {
            assert_ulps_eq!(elem, 1f64.exp(), max_ulps=1);
        }
    }
}
