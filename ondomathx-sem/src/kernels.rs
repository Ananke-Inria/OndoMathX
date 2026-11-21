//! Point kernels of the v1 physics (second-order, one field).
//!
//! - Acoustics, pressure `p`: `(1/κ) p̈ − div((1/ρ) ∇p) = f`. Stiffness
//!   [`AcousticStiffness`], mass coefficient `1/κ` ([`crate::fields::Inverse`]).
//! - Elastodynamics, displacement `u`: `ρ ü − div σ(u) = f`, isotropic
//!   `σ = λ tr(ε) I + 2μ ε`. Stiffness [`IsotropicElastic`], mass coefficient `ρ`.
//!
//! [`Mass`] applies a weighted mass `∫ c u·v` to any number of components; time
//! stepping uses [`crate::Space::c_inverse_mass`] instead, which is diagonal.
//!
//! Mixed kernels ([`crate::MixedKernel`], for [`crate::mlt_add_mixed`]): the
//! divergence pair of the displacement–pressure coupling,
//! `b(u, q) = ∫ c q div u` ([`Divergence`], `D` → 1 components) and its
//! transpose `bᵀ(p, v) = ∫ c p div v` ([`DivergenceTranspose`], 1 → `D`).
//!
//! Error kernels ([`L2Error`], [`H1Error`], `C` → 1 components): the source is
//! the pointwise squared error against an exact field. Tested against any
//! scalar space, the sum of all entries of the (element-local) output is
//! `∫ S` (the basis is a partition of unity), so with an over-integration
//! quadrature `Q`:
//! ```ignore
//! let mut s = vec![0.0; space.n_dup()];
//! mlt_add_mixed::<Q, C, 1, _, _, _, _>(&space, &space, &L2Error { exact }, 1.0, &u, 0.0, &mut s);
//! let err_l2 = s.iter().sum::<f64>().sqrt();
//! ```

use nalgebra::SMatrix;

use crate::engine::{MixedKernel, PointIn, PointKernel, PointOut};
use crate::fields::Field;

/// `a(p, q) = ∫ (1/ρ) ∇p·∇q`.
#[derive(Clone, Debug)]
pub struct AcousticStiffness<R> {
    /// Density `ρ`.
    pub rho: R,
}

impl<const D: usize, R: Field<D>> PointKernel<D, 1> for AcousticStiffness<R> {
    const USES_GRAD: bool = true;
    const USES_XYZ: bool = R::NEEDS_XYZ;
    const USES_LABEL: bool = R::NEEDS_LABEL;
    const USES_DOF: bool = R::NEEDS_DOF;
    const OUTPUTS_FLUX: bool = true;

    #[inline]
    fn apply(&self, i: &PointIn<D, 1>, o: &mut PointOut<D, 1>) {
        o.flux = i.grad / self.rho.eval(i);
    }
}

/// `a(u, v) = ∫ σ(u) : ε(v)`, `σ = λ tr(ε) I + 2μ ε`, `ε = (∇u + ∇uᵀ)/2`.
///
/// `σ` is symmetric, so testing it against `∇v` equals testing against `ε(v)`.
#[derive(Clone, Debug)]
pub struct IsotropicElastic<L, M> {
    /// First Lamé parameter `λ`.
    pub lambda: L,
    /// Shear modulus `μ`.
    pub mu: M,
}

impl<const D: usize, L: Field<D>, M: Field<D>> PointKernel<D, D> for IsotropicElastic<L, M> {
    const USES_GRAD: bool = true;
    const USES_XYZ: bool = L::NEEDS_XYZ || M::NEEDS_XYZ;
    const USES_LABEL: bool = L::NEEDS_LABEL || M::NEEDS_LABEL;
    const USES_DOF: bool = L::NEEDS_DOF || M::NEEDS_DOF;
    const OUTPUTS_FLUX: bool = true;

    #[inline]
    fn apply(&self, i: &PointIn<D, D>, o: &mut PointOut<D, D>) {
        let (lambda, mu) = (self.lambda.eval(i), self.mu.eval(i));
        let eps = (i.grad + i.grad.transpose()) * 0.5;
        o.flux = eps * (2.0 * mu) + SMatrix::<f64, D, D>::identity() * (lambda * eps.trace());
    }
}

/// `a(p, q) = ∫ (A ∇p)·∇q` with a constant tensor `A` (anisotropic
/// acoustics, the C++ `Field::Matrix` "velocity"): flux `A ∇p`.
#[derive(Clone, Debug)]
pub struct AnisotropicAcoustic<const D: usize> {
    /// The tensor `A`, `a[i][j]` (symmetric positive definite).
    pub a: [[f64; D]; D],
}

impl<const D: usize> PointKernel<D, 1> for AnisotropicAcoustic<D> {
    const USES_GRAD: bool = true;
    const OUTPUTS_FLUX: bool = true;

    #[inline]
    fn apply(&self, i: &PointIn<D, 1>, o: &mut PointOut<D, 1>) {
        for r in 0..D {
            let mut s = 0.0;
            for (c, a) in self.a[r].iter().enumerate() {
                s += a * i.grad[(0, c)];
            }
            o.flux[(0, r)] = s;
        }
    }
}

/// `a(u, v) = ∫ c u·v` for `C` components (no flux: no derivative kernel runs).
#[derive(Clone, Debug)]
pub struct Mass<F> {
    /// Coefficient `c`.
    pub coef: F,
}

impl<const D: usize, const C: usize, F: Field<D>> PointKernel<D, C> for Mass<F> {
    const USES_VALUE: bool = true;
    const USES_XYZ: bool = F::NEEDS_XYZ;
    const USES_LABEL: bool = F::NEEDS_LABEL;
    const USES_DOF: bool = F::NEEDS_DOF;
    const OUTPUTS_SOURCE: bool = true;

    #[inline]
    fn apply(&self, i: &PointIn<D, C>, o: &mut PointOut<D, C>) {
        o.source = i.value * self.coef.eval(i);
    }
}

/// `b(u, q) = ∫ c q div u`: the divergence of the trial field (`D`
/// components), tested with the values of the test field (1 component).
#[derive(Clone, Debug)]
pub struct Divergence<F> {
    /// Coefficient `c`.
    pub coef: F,
}

impl<const D: usize, F: Field<D>> MixedKernel<D, D, 1> for Divergence<F> {
    const USES_GRAD: bool = true;
    const USES_XYZ: bool = F::NEEDS_XYZ;
    const USES_LABEL: bool = F::NEEDS_LABEL;
    const USES_DOF: bool = F::NEEDS_DOF;
    const OUTPUTS_SOURCE: bool = true;

    #[inline]
    fn apply(&self, i: &PointIn<D, D>, o: &mut PointOut<D, 1>) {
        o.source[0] = self.coef.eval(i) * i.grad.trace();
    }
}

/// `bᵀ(p, v) = ∫ c p div v = ∫ (c p I) : ∇v`: the value of the trial field
/// (1 component), tested with the divergence of the test field (`D`
/// components), the transpose of [`Divergence`].
#[derive(Clone, Debug)]
pub struct DivergenceTranspose<F> {
    /// Coefficient `c`.
    pub coef: F,
}

impl<const D: usize, F: Field<D>> MixedKernel<D, 1, D> for DivergenceTranspose<F> {
    const USES_VALUE: bool = true;
    const USES_XYZ: bool = F::NEEDS_XYZ;
    const USES_LABEL: bool = F::NEEDS_LABEL;
    const USES_DOF: bool = F::NEEDS_DOF;
    const OUTPUTS_FLUX: bool = true;

    #[inline]
    fn apply(&self, i: &PointIn<D, 1>, o: &mut PointOut<D, D>) {
        o.flux = SMatrix::<f64, D, D>::identity() * (self.coef.eval(i) * i.value[0]);
    }
}

/// Pointwise `|u − u_ex(x)|²` (source), `C` → 1 components: see the module
/// documentation for the error integral.
#[derive(Clone, Debug)]
pub struct L2Error<F> {
    /// The exact field `x ↦ u_ex(x)`.
    pub exact: F,
}

impl<const D: usize, const C: usize, F: Fn([f64; D]) -> [f64; C] + Sync> MixedKernel<D, C, 1> for L2Error<F> {
    const USES_VALUE: bool = true;
    const USES_XYZ: bool = true;
    const OUTPUTS_SOURCE: bool = true;

    #[inline]
    fn apply(&self, i: &PointIn<D, C>, o: &mut PointOut<D, 1>) {
        let ex = (self.exact)(core::array::from_fn(|a| i.xyz[a]));
        o.source[0] = (0..C).map(|c| (i.value[c] - ex[c]).powi(2)).sum();
    }
}

/// Pointwise `|∇u − ∇u_ex(x)|²` (source, Frobenius norm), `C` → 1
/// components: the `H¹` semi-norm error, see the module documentation.
#[derive(Clone, Debug)]
pub struct H1Error<F> {
    /// The exact gradient `x ↦ ∇u_ex(x)`, `[c][d] = ∂u_c/∂x_d`.
    pub exact_grad: F,
}

impl<const D: usize, const C: usize, F: Fn([f64; D]) -> [[f64; D]; C] + Sync> MixedKernel<D, C, 1> for H1Error<F> {
    const USES_GRAD: bool = true;
    const USES_XYZ: bool = true;
    const OUTPUTS_SOURCE: bool = true;

    #[inline]
    fn apply(&self, i: &PointIn<D, C>, o: &mut PointOut<D, 1>) {
        let ex = (self.exact_grad)(core::array::from_fn(|a| i.xyz[a]));
        let mut s = 0.0;
        for (c, row) in ex.iter().enumerate() {
            for (d, e) in row.iter().enumerate() {
                s += (i.grad[(c, d)] - e).powi(2);
            }
        }
        o.source[0] = s;
    }
}
