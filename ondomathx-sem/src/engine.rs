//! Matrix-free operator application: `v ← β·v + α·A·u`, element by element.
//!
//! An operator is given by a [`PointKernel`]: at every GLL point it maps the
//! value and physical gradient of `u` to a **flux** `G` (tested against `∇v`)
//! and a **source** `S` (tested against `v`):
//! $$ a(u, v) = \int_\Omega G(u) : \nabla v + S(u) \cdot v , $$
//! integrated with the GLL rule (collocation: quadrature points = nodes).
//!
//! The same-space product [`Space::mlt_add`] is the mixed product on the
//! space's own node rule `S::Nodes` ([`crate::Pointwise`] run by
//! [`Space::mlt_add_element`](crate::Space::mlt_add_element)). Per batch of `W` elements of the same
//! geometric type it
//! 1. gathers `u` into lobatto's batched layout (node `g`, lane `l` at `g·W + l`),
//! 2. applies the reference derivatives: on the node rule the interpolations
//!    are identities (skipped) and each derivative is a single `D_b` pass,
//!    hand-coded lobatto kernels for isotropic N = 4..7, generic ones otherwise
//!    ([`crate::Sweep::gradient`]),
//! 3. at each point and lane: evaluates the geometry, pushes the gradient
//!    forward, `∇ₓu = ∇ᵤu · cof(J)ᵀ / det J`, calls the kernel, and pulls the
//!    flux back, `H = sign(det J)·w·G·cof(J)`, so that `Σ_b D_bᵀ H_b` tests it,
//! 4. applies the transposed derivatives and scatters `β·v + α·A·u`.
//!
//! The output is element-local (no shared-DOF sum), so products can be chained
//! before [`Space::c_sum_shared`].

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
/// `PointKernel<D, C>` (applied by [`Space::mlt_add`] or
/// [`Space::mlt_add_mixed`](crate::Space::mlt_add_mixed)), a mixed one e.g. a `PointKernel<D, D, 1>`
/// (applied by [`Space::mlt_add_mixed`](crate::Space::mlt_add_mixed) between two spaces).
///
/// The flags are compile-time switches: unused inputs are neither computed nor
/// copied, and an operator without flux skips all derivative kernels. All
/// default to `false`; a kernel states what it uses and produces.
///
/// `PointIn::dof` is the point index `e·NQ + q` of the quadrature, which is
/// the duplicated node index on the node rule (`Space::mlt_add`): so
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
    /// `v ← β·v + α·A·u` with the operator of `kernel`, batches of 4 elements.
    ///
    /// `u` and `v` are `C`-component vectors (`n_elem · C · NLOC` entries). The
    /// result is element-local: call [`Space::c_sum_shared`] to assemble it.
    /// With `β = 0`, `v` is overwritten (its previous content is never read).
    pub fn mlt_add<const C: usize>(
        &self,
        kernel: &impl PointKernel<D, C>,
        alpha: f64,
        u: &[f64],
        beta: f64,
        v: &mut [f64],
    ) {
        self.mlt_add_batched::<C, 4>(kernel, alpha, u, beta, v);
    }

    /// [`Space::mlt_add`] with an explicit batch width `W`.
    pub fn mlt_add_batched<const C: usize, const W: usize>(
        &self,
        kernel: &impl PointKernel<D, C>,
        alpha: f64,
        u: &[f64],
        beta: f64,
        v: &mut [f64],
    ) {
        let op = Pointwise::<D, _, S::Nodes, S, S>::on_set(self, self, kernel, self.node_points());
        mlt_add_element_batched::<D, S, S, C, C, _, W>(self, self, &op, alpha, u, beta, v);
    }
}
