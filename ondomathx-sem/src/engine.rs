//! Matrix-free operator application: `v ← β·v + α·A·u`, element by element.
//!
//! An operator is given by a [`PointKernel`]: at every GLL point it maps the
//! value and physical gradient of `u` to a **flux** `G` (tested against `∇v`)
//! and a **source** `S` (tested against `v`):
//! $$ a(u, v) = \int_\Omega G(u) : \nabla v + S(u) \cdot v , $$
//! integrated with the GLL rule (collocation: quadrature points = nodes).
//!
//! The same-space product [`Space::mlt_dg`] is the mixed product on the
//! space's own node rule `S::Nodes` ([`crate::Pointwise`] run by
//! [`Space::mlt_element_dg`](crate::Space::mlt_element_dg)). Per batch of `W` elements of the same
//! geometric type it
//! 1. gathers `u` into lobatto's batched layout (node `g`, lane `l` at `g·W + l`),
//! 2. applies the reference derivatives: on the node rule the interpolations
//!    are identities (skipped) and each derivative is a single `D_b` pass,
//!    hand-coded lobatto kernels for isotropic N = 4..7, generic ones otherwise
//!    ([`crate::Sweep::gradient_interpolation`]),
//! 3. at each point and lane: evaluates the geometry, pushes the gradient
//!    forward, `∇ₓu = ∇ᵤu · cof(J)ᵀ / det J`, calls the kernel, and pulls the
//!    flux back, `H = sign(det J)·w·G·cof(J)`, so that `Σ_b D_bᵀ H_b` tests it,
//! 4. applies the transposed derivatives and scatters `β·v + α·A·u`.
//!
//! The output is element-local (no shared-DOF sum), so products can be chained
//! before [`Space::sum_shared`].

use nalgebra::{SMatrix, SVector};

use crate::element::mlt_add_element_batched;
use crate::mixed::Pointwise;
use crate::shape::Shape;
use crate::space::Space;

/// Input of a point kernel. Fields whose `USES_*` flag is false are zero.
#[derive(Clone, Debug)]
pub struct PointIn<const D: usize, const C: usize> {
    /// `u_c` at the point (`USES_VALUE`).
    pub value: SVector<f64, C>,
    /// `grad[(c, d)] = ∂u_c/∂x_d` (`USES_GRAD`).
    pub grad: SMatrix<f64, C, D>,
    /// Physical coordinates of the point (`USES_XYZ`).
    pub xyz: SVector<f64, D>,
    /// Mesh label of the element (`USES_LABEL`).
    pub label: usize,
    /// Duplicated node index `e·NLOC + node` (`USES_DOF`), e.g. to read
    /// one-value-per-node data.
    pub dof: usize,
}

impl<const D: usize, const C: usize> PointIn<D, C> {
    /// All fields zero.
    pub(crate) fn zeros() -> Self {
        PointIn {
            value: SVector::zeros(),
            grad: SMatrix::zeros(),
            xyz: SVector::zeros(),
            label: 0,
            dof: 0,
        }
    }
}

/// Output of a point kernel; the engine zeroes it before each call.
#[derive(Clone, Debug)]
pub struct PointOut<const D: usize, const C: usize> {
    /// Flux `G`, tested against `∂v_c/∂x_d` (`OUTPUTS_FLUX`).
    pub flux: SMatrix<f64, C, D>,
    /// Source `S`, tested against `v_c` (`OUTPUTS_SOURCE`).
    pub source: SVector<f64, C>,
}

impl<const D: usize, const C: usize> PointOut<D, C> {
    /// All fields zero.
    pub(crate) fn zeros() -> Self {
        PointOut {
            flux: SMatrix::zeros(),
            source: SVector::zeros(),
        }
    }
}

/// Pointwise physics of an operator `a(u, v) = ∫ G(u) : ∇v + S(u)·v` in
/// dimension `D`, with `u` of `CU` components (trial space) and `v` of `CV`
/// components (test space). `CV` defaults to `CU`: a same-space kernel is a
/// `PointKernel<D, C>` (applied by [`Space::mlt_dg`] or
/// [`Space::mlt_mixed_dg`](crate::Space::mlt_mixed_dg)), a mixed one e.g. a `PointKernel<D, D, 1>`
/// (applied by [`Space::mlt_mixed_dg`](crate::Space::mlt_mixed_dg) between two spaces).
///
/// The flags are compile-time switches: unused inputs are neither computed nor
/// copied, and an operator without flux skips all derivative kernels. All
/// default to `false`; a kernel states what it uses and produces.
///
/// `PointIn::dof` is the point index `e·NQ + q` of the quadrature, which is
/// the duplicated node index on the node rule (`Space::mlt_dg`): so
/// one-value-per-node data (`PointData`) only applies there.
pub trait PointKernel<const D: usize, const CU: usize, const CV: usize = CU>: Sync {
    const USES_VALUE: bool = false;
    const USES_GRAD: bool = false;
    const USES_XYZ: bool = false;
    const USES_LABEL: bool = false;
    const USES_DOF: bool = false;
    const OUTPUTS_FLUX: bool = false;
    const OUTPUTS_SOURCE: bool = false;

    fn apply(&self, input: &PointIn<D, CU>, output: &mut PointOut<D, CV>);
}

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// `v ← α·A·u` with the operator of `kernel`, element by element, batches
    /// of 4 elements (`v` is overwritten, its previous content never read).
    ///
    /// `u` and `v` are `C`-component DG vectors (`num_dg_dofs · C` entries).
    /// The result is element-local: [`Space::sum_shared`] assembles it, or
    /// [`Space::mlt_assembled`] does both.
    pub fn mlt_dg<const C: usize>(&self, kernel: &impl PointKernel<D, C>, u: &[f64], v: &mut [f64], alpha: f64) {
        self.mlt_batched_dg::<C, 4>(kernel, u, v, alpha);
    }

    /// `v ← v + α·A·u`, element by element (as [`Space::mlt_dg`]).
    pub fn mlt_add_dg<const C: usize>(&self, kernel: &impl PointKernel<D, C>, u: &[f64], v: &mut [f64], alpha: f64) {
        self.mlt_add_batched_dg::<C, 4>(kernel, u, v, alpha);
    }

    /// [`Space::mlt_dg`] with an explicit batch width `W`.
    pub fn mlt_batched_dg<const C: usize, const W: usize>(
        &self,
        kernel: &impl PointKernel<D, C>,
        u: &[f64],
        v: &mut [f64],
        alpha: f64,
    ) {
        let op = Pointwise::<D, _, S::Nodes, S, S>::on_set(self, self, kernel, self.node_points());
        mlt_add_element_batched::<D, S, S, C, C, _, W>(self, self, &op, alpha, u, 0.0, v);
    }

    /// [`Space::mlt_add_dg`] with an explicit batch width `W`.
    pub fn mlt_add_batched_dg<const C: usize, const W: usize>(
        &self,
        kernel: &impl PointKernel<D, C>,
        u: &[f64],
        v: &mut [f64],
        alpha: f64,
    ) {
        let op = Pointwise::<D, _, S::Nodes, S, S>::on_set(self, self, kernel, self.node_points());
        mlt_add_element_batched::<D, S, S, C, C, _, W>(self, self, &op, alpha, u, 1.0, v);
    }
}
