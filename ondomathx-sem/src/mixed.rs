//! Mixed products: `v ← β·v + α·A·u` with `u` in one space (trial, `CU`
//! components), `v` in another (test, `CV` components), integrated with a
//! quadrature `Q` of either space (declared, or its node rule `S::Nodes`):
//! $$ a(u, v) = \sum_q w_q |\det J_q| \big( G(u)(x_q) : \nabla v(x_q) + S(u)(x_q) \cdot v(x_q) \big). $$
//!
//! It is an [`ElementOperator`] ([`Pointwise`]) run by [`Space::mlt_element_dg`],
//! which gathers `u` and scatters the result; per batch of `W` elements:
//! 1. (gather, by the driver),
//! 2. values and reference gradients of `u` at the points of `Q`
//!    ([`Sweep::interpolation`], [`Sweep::gradient_interpolation`] on the trial shape),
//! 3. at each point and lane: geometry at the point (stored or recomputed,
//!    from `Q`'s entry), push-forward, kernel, pull-back,
//! 4. test against `v`: [`Sweep::transpose_interpolation`] of the source and
//!    [`Sweep::transpose_gradient_interpolation`] of the pulled-back flux on the test shape,
//! 5. scatter `β·v + α·y`.
//!
//! The same-space product [`Space::mlt_dg`] is this one on the node rule
//! `S::Nodes`. The point loop is monomorphized per geometric kind; the 1D
//! sweep tables are borrowed from the spaces (node rule, declared
//! quadratures) and only built for the call when a space does not store them.
//!
//! The two spaces must come from the same mesh (same element storage,
//! checked by a fingerprint); the output is element-local (no shared-DOF sum).

use nalgebra::SMatrix;

use crate::element::{mlt_add_element_batched, Batch, ElementOperator, PointSet};
use crate::engine::{PointIn, PointKernel, PointOut};
use crate::geometry::PointGeo;
use crate::quadrature::{Quadrature, Sweep};
use crate::shape::Shape;
use crate::space::{GeoKind, Space};

/// The generic element operator of [`Space::mlt_mixed_dg`]: the pointwise `kernel`
/// at the points of the quadrature `Q`, node ↔ point sweeps on both shapes.
pub struct Pointwise<'a, const D: usize, K, Q, SU, SV>
where
    SU: Shape<D>,
    SV: Shape<D>,
    Q: Sweep<D, SU> + Sweep<D, SV>,
{
    kernel: &'a K,
    set: PointSet<'a, D>,
    tu: Tables<'a, <Q as Sweep<D, SU>>::Tables>,
    tv: Tables<'a, <Q as Sweep<D, SV>>::Tables>,
}

/// Sweep tables borrowed from a space (node rule or declared quadrature), or
/// built for the call when the space does not store them.
enum Tables<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> core::ops::Deref for Tables<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        match self {
            Tables::Borrowed(t) => t,
            Tables::Owned(t) => t,
        }
    }
}

/// The tables of `Q` on `space`'s shape: stored ones if any, else built.
fn tables_of<const D: usize, Q: Sweep<D, S>, S: Shape<D>>(
    space: &Space<D, S>,
) -> Tables<'_, Q::Tables> {
    match space.sweep_tables::<Q>() {
        Some(t) => Tables::Borrowed(t),
        None => Tables::Owned(Q::tables()),
    }
}

impl<'a, const D: usize, K, Q, SU, SV> Pointwise<'a, D, K, Q, SU, SV>
where
    SU: Shape<D>,
    SV: Shape<D>,
    Q: Sweep<D, SU> + Sweep<D, SV>,
{
    /// The operator of `kernel` with `Q` declared on either space (panics
    /// otherwise).
    pub fn new(u_space: &'a Space<D, SU>, v_space: &'a Space<D, SV>, kernel: &'a K) -> Self {
        let set = u_space
            .quadrature_points::<Q>()
            .or_else(|| v_space.quadrature_points::<Q>())
            .unwrap_or_else(|| {
                panic!(
                    "quadrature {} is declared on neither space",
                    std::any::type_name::<Q>()
                )
            });
        Self::on_set(u_space, v_space, kernel, set)
    }

    /// The operator of `kernel` on the point set `set`, which must be the
    /// points of `Q` (e.g. the nodes, `Q = S::Nodes`).
    pub(crate) fn on_set(
        u_space: &'a Space<D, SU>,
        v_space: &'a Space<D, SV>,
        kernel: &'a K,
        set: PointSet<'a, D>,
    ) -> Self {
        Pointwise {
            kernel,
            set,
            tu: tables_of::<D, Q, SU>(u_space),
            tv: tables_of::<D, Q, SV>(v_space),
        }
    }
}

/// Per-task scratch of [`Pointwise`].
pub struct PointwiseScratch {
    /// Values at the points, `[c][q][lane]`.
    uq: Vec<f64>,
    /// Reference gradients at the points, overwritten point by point with the
    /// pulled-back fluxes, `[c][dir][q][lane]` (`max(CU, CV)` components).
    gq: Vec<f64>,
    /// Weighted sources at the points, `[c][q][lane]`.
    sq: Vec<f64>,
    /// One transposed sweep on the test shape.
    yt: Vec<f64>,
    /// Sweep scratch.
    tmp: Vec<f64>,
}

/// Geometric kinds as const generic arguments, so that each group gets its
/// own monomorphized point loop (indices of [`GeoKind::ALL`]).
const AFFINE: usize = 0;
const Q1: usize = 1;
const Q2: usize = 2;
const MAPPED: usize = 3;

impl<const D: usize, K, Q, SU, SV, const CU: usize, const CV: usize>
    ElementOperator<D, SU, SV, CU, CV> for Pointwise<'_, D, K, Q, SU, SV>
where
    SU: Shape<D>,
    SV: Shape<D>,
    Q: Sweep<D, SU> + Sweep<D, SV>,
    K: PointKernel<D, CU, CV>,
    <Q as Sweep<D, SU>>::Tables: Sync,
    <Q as Sweep<D, SV>>::Tables: Sync,
{
    type Scratch = PointwiseScratch;
    const READS_INPUT: bool = K::USES_VALUE || K::USES_GRAD;

    fn scratch<const W: usize>(&self) -> PointwiseScratch {
        let nqw = <Q as Quadrature<D>>::NQ * W;
        PointwiseScratch {
            // Buffers the kernel does not use are empty (the scratch is
            // allocated and zeroed once per rayon job).
            uq: vec![0.0; if K::USES_VALUE { CU * nqw } else { 0 }],
            gq: vec![
                0.0;
                if K::USES_GRAD || K::OUTPUTS_FLUX {
                    CU.max(CV) * D * nqw
                } else {
                    0
                }
            ],
            sq: vec![0.0; if K::OUTPUTS_SOURCE { CV * nqw } else { 0 }],
            yt: vec![0.0; SV::NLOC * W],
            tmp: vec![0.0; W * <Q as Sweep<D, SU>>::TMP.max(<Q as Sweep<D, SV>>::TMP)],
        }
    }

    fn apply<const W: usize>(
        &self,
        batch: &Batch<'_, D, SU, SV>,
        u: &[f64],
        y: &mut [f64],
        s: &mut PointwiseScratch,
    ) {
        let (sp, geo, w, k) = (batch.u_space, self.set.geo, self.set.weights, self.kernel);
        match batch.kind {
            GeoKind::Affine => self.apply_kind::<CU, CV, W, AFFINE>(sp, geo, w, k, batch, u, y, s),
            GeoKind::Q1 => self.apply_kind::<CU, CV, W, Q1>(sp, geo, w, k, batch, u, y, s),
            GeoKind::Q2 => self.apply_kind::<CU, CV, W, Q2>(sp, geo, w, k, batch, u, y, s),
            GeoKind::Mapped => self.apply_kind::<CU, CV, W, MAPPED>(sp, geo, w, k, batch, u, y, s),
        }
    }
}

impl<const D: usize, K, Q, SU, SV> Pointwise<'_, D, K, Q, SU, SV>
where
    SU: Shape<D>,
    SV: Shape<D>,
    Q: Sweep<D, SU> + Sweep<D, SV>,
{
    /// [`ElementOperator::apply`] for a batch of geometric kind
    /// `GeoKind::ALL[G]`, in three phases: nodes → points, point loop,
    /// points → nodes. The space, point geometry, weights and kernel are
    /// arguments (not read through `batch` and `self`): the compiler then
    /// knows that the stores to the scratch do not change them, and keeps
    /// them (and the kernel's constants) in registers.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn apply_kind<const CU: usize, const CV: usize, const W: usize, const G: usize>(
        &self,
        space: &Space<D, SU>,
        geo: &PointGeo<D>,
        weights: &[f64],
        kernel: &K,
        batch: &Batch<'_, D, SU, SV>,
        u: &[f64],
        y: &mut [f64],
        s: &mut PointwiseScratch,
    ) where
        K: PointKernel<D, CU, CV>,
    {
        let nqw = <Q as Quadrature<D>>::NQ * W;
        // Every buffer re-sliced to its exact (constant) length once: the
        // compiler can then prove the indices in bounds and drop the checks
        // (the scratch is allocated in another function).
        let u = &u[..CU * SU::NLOC * W];
        let y = &mut y[..CV * SV::NLOC * W];
        let weights = &weights[..<Q as Quadrature<D>>::NQ];
        let uq = &mut s.uq[..if K::USES_VALUE { CU * nqw } else { 0 }];
        let gq = &mut s.gq[..if K::USES_GRAD || K::OUTPUTS_FLUX {
            CU.max(CV) * D * nqw
        } else {
            0
        }];
        let sq = &mut s.sq[..if K::OUTPUTS_SOURCE { CV * nqw } else { 0 }];
        let yt = &mut s.yt[..SV::NLOC * W];
        let tmp = &mut s.tmp[..];

        self.to_points::<CU, CV, W>(u, uq, gq, tmp);
        Self::point_loop::<CU, CV, W, G>(space, geo, weights, kernel, batch, uq, gq, sq);
        self.to_nodes::<CU, CV, W>(gq, sq, yt, y, tmp);
    }

    /// Nodes → points: the values `uq` and reference gradients `gq` of the
    /// trial field `u` at the points of `Q` (`[c][q][lane]`,
    /// `[c][dir][q][lane]`).
    #[inline(always)]
    fn to_points<const CU: usize, const CV: usize, const W: usize>(
        &self,
        u: &[f64],
        uq: &mut [f64],
        gq: &mut [f64],
        tmp: &mut [f64],
    ) where
        K: PointKernel<D, CU, CV>,
    {
        let (nuw, nqw) = (SU::NLOC * W, <Q as Quadrature<D>>::NQ * W);
        for c in 0..CU {
            let uc = &u[c * nuw..(c + 1) * nuw];
            if K::USES_VALUE {
                <Q as Sweep<D, SU>>::interpolation::<W>(
                    &self.tu,
                    uc,
                    &mut uq[c * nqw..(c + 1) * nqw],
                    tmp,
                );
            }
            if K::USES_GRAD {
                for d in 0..D {
                    let out = &mut gq[(c * D + d) * nqw..(c * D + d + 1) * nqw];
                    <Q as Sweep<D, SU>>::gradient_interpolation::<W>(&self.tu, d, uc, out, tmp);
                }
            }
        }
    }

    /// At every point and lane: geometry, push-forward of the gradient,
    /// kernel, pull-back of the flux (written over the gradient of the same
    /// point and lane, already read) and weighting of the source.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn point_loop<const CU: usize, const CV: usize, const W: usize, const G: usize>(
        space: &Space<D, SU>,
        geo: &PointGeo<D>,
        weights: &[f64],
        kernel: &K,
        batch: &Batch<'_, D, SU, SV>,
        uq: &[f64],
        gq: &mut [f64],
        sq: &mut [f64],
    ) where
        K: PointKernel<D, CU, CV>,
    {
        let (nq, nqw) = (<Q as Quadrature<D>>::NQ, <Q as Quadrature<D>>::NQ * W);
        let kind = match G {
            AFFINE => GeoKind::Affine,
            Q1 => GeoKind::Q1,
            Q2 => GeoKind::Q2,
            _ => GeoKind::Mapped,
        };
        // Affine elements: one Jacobian per lane, hoisted out of the loop.
        let affine: [(f64, SMatrix<f64, D, D>); W] = core::array::from_fn(|lane| {
            if G == AFFINE {
                let (_, dj, cof) = space.geo_at::<false>(geo, kind, batch.element(lane), 0);
                (dj, cof)
            } else {
                (0.0, SMatrix::zeros())
            }
        });
        let mut input = PointIn::<D, CU>::zeros();
        let mut output = PointOut::<D, CV>::zeros();
        for (q, &w) in weights.iter().enumerate() {
            for (lane, aff) in affine.iter().enumerate() {
                // Padded lanes reuse the last valid element (finite values);
                // their results are never scattered.
                let e = batch.element(lane);
                let (det_j, cof) = if G == AFFINE && !K::USES_XYZ {
                    *aff
                } else {
                    let (x, dj, cof) = space.geo_at::<true>(geo, kind, e, q);
                    if K::USES_XYZ {
                        input.xyz = x;
                    }
                    (dj, cof)
                };
                if K::USES_VALUE {
                    for c in 0..CU {
                        input.value[c] = uq[c * nqw + q * W + lane];
                    }
                }
                if K::USES_GRAD {
                    let mut gu = SMatrix::<f64, CU, D>::zeros();
                    for c in 0..CU {
                        for d in 0..D {
                            gu[(c, d)] = gq[(c * D + d) * nqw + q * W + lane];
                        }
                    }
                    input.grad = gu * cof.transpose() / det_j;
                }
                if K::USES_LABEL {
                    input.label = batch.label(lane);
                }
                if K::USES_DOF {
                    input.dof = e * nq + q;
                }
                output.flux.fill(0.0);
                output.source.fill(0.0);
                kernel.apply(&input, &mut output);
                if K::OUTPUTS_FLUX {
                    let h = output.flux * cof * (w * det_j.signum());
                    for c in 0..CV {
                        for d in 0..D {
                            gq[(c * D + d) * nqw + q * W + lane] = h[(c, d)];
                        }
                    }
                }
                if K::OUTPUTS_SOURCE {
                    let ws = w * det_j.abs();
                    for c in 0..CV {
                        sq[c * nqw + q * W + lane] = ws * output.source[c];
                    }
                }
            }
        }
    }

    /// Points → nodes: tests the weighted source `sq` and the pulled-back
    /// flux `gq` against the test basis, `y = Iᵀ sq + Σ_d G_dᵀ gq_d`.
    #[inline(always)]
    fn to_nodes<const CU: usize, const CV: usize, const W: usize>(
        &self,
        gq: &[f64],
        sq: &[f64],
        yt: &mut [f64],
        y: &mut [f64],
        tmp: &mut [f64],
    ) where
        K: PointKernel<D, CU, CV>,
    {
        let (nvw, nqw) = (SV::NLOC * W, <Q as Quadrature<D>>::NQ * W);
        for c in 0..CV {
            let yc = &mut y[c * nvw..(c + 1) * nvw];
            if K::OUTPUTS_SOURCE {
                <Q as Sweep<D, SV>>::transpose_interpolation::<W>(
                    &self.tv,
                    &sq[c * nqw..(c + 1) * nqw],
                    yc,
                    tmp,
                );
            } else if !K::OUTPUTS_FLUX {
                yc.fill(0.0);
            }
            if K::OUTPUTS_FLUX {
                for d in 0..D {
                    let h = &gq[(c * D + d) * nqw..(c * D + d + 1) * nqw];
                    <Q as Sweep<D, SV>>::transpose_gradient_interpolation::<W>(&self.tv, d, h, yt, tmp);
                    if d == 0 && !K::OUTPUTS_SOURCE {
                        // First term without a source: `0 + t`, not `t`, so
                        // that a `−0` sums to `+0` as with a zeroed `y`.
                        for (yv, t) in yc.iter_mut().zip(yt.iter()) {
                            *yv = 0.0 + t;
                        }
                    } else {
                        for (yv, t) in yc.iter_mut().zip(yt.iter()) {
                            *yv += t;
                        }
                    }
                }
            }
        }
    }
}

impl<const D: usize, SU: Shape<D>> Space<D, SU> {
    /// `v ← β·v + α·A·u` for the mixed operator of `kernel` from this space
    /// (trial, `CU` components) to `v_space` (test shape `SV`, `CV`
    /// components), integrated with the quadrature `Q` of either space
    /// (declared, or its node rule `S::Nodes`); batches of 4 elements
    /// ([`Pointwise`] run by [`Space::mlt_element_dg`]).
    ///
    /// ```ignore
    /// // b(u, q) = ∫ q div u: displacement → pressure, Gauss–Lobatto rule of the displacement
    /// let div = Divergence { coef: Constant(1.0) };
    /// disp.mlt_mixed_dg::<Quad2<Gll<5>, Gll<5>>, 2, 1, _>(&pres, &div, &u, &mut p, 1.0);
    /// ```
    ///
    /// The `_` is the test shape (Rust infers no generic argument once one
    /// is written). `v` is overwritten (never read); [`Space::mlt_add_mixed_dg`]
    /// accumulates. Panics if the spaces
    /// come from different meshes, if `Q` is a quadrature of neither space,
    /// or on wrong vector lengths.
    pub fn mlt_mixed_dg<Q, const CU: usize, const CV: usize, SV: Shape<D>>(
        &self,
        v_space: &Space<D, SV>,
        kernel: &impl PointKernel<D, CU, CV>,
        u: &[f64],
        v: &mut [f64],
        alpha: f64,
    ) where
        Q: Sweep<D, SU> + Sweep<D, SV>,
    {
        self.mixed::<Q, CU, CV, SV>(v_space, kernel, alpha, u, 0.0, v);
    }

    /// `v ← v + α·A·u` (as [`Space::mlt_mixed_dg`], accumulating).
    pub fn mlt_add_mixed_dg<Q, const CU: usize, const CV: usize, SV: Shape<D>>(
        &self,
        v_space: &Space<D, SV>,
        kernel: &impl PointKernel<D, CU, CV>,
        u: &[f64],
        v: &mut [f64],
        alpha: f64,
    ) where
        Q: Sweep<D, SU> + Sweep<D, SV>,
    {
        self.mixed::<Q, CU, CV, SV>(v_space, kernel, alpha, u, 1.0, v);
    }

    fn mixed<Q, const CU: usize, const CV: usize, SV: Shape<D>>(
        &self,
        v_space: &Space<D, SV>,
        kernel: &impl PointKernel<D, CU, CV>,
        alpha: f64,
        u: &[f64],
        beta: f64,
        v: &mut [f64],
    ) where
        Q: Sweep<D, SU> + Sweep<D, SV>,
    {
        assert_eq!(
            self.fingerprint, v_space.fingerprint,
            "mixed product between spaces built from different meshes"
        );
        let op = Pointwise::<D, _, Q, SU, SV>::new(self, v_space, kernel);
        mlt_add_element_batched::<D, SU, SV, CU, CV, _, 4>(self, v_space, &op, alpha, u, beta, v);
    }
}
