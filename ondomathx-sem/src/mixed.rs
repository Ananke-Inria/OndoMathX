//! Mixed products: `v ← β·v + α·A·u` with `u` in one space (trial, `CU`
//! components), `v` in another (test, `CV` components), integrated with a
//! declared quadrature `Q` of either space:
//! $$ a(u, v) = \sum_q w_q |\det J_q| \big( G(u)(x_q) : \nabla v(x_q) + S(u)(x_q) \cdot v(x_q) \big). $$
//!
//! It is an [`ElementOperator`] ([`Pointwise`]) run by [`mlt_add_element`],
//! which gathers `u` and scatters the result; per batch of `W` elements:
//! 1. (gather, by the driver),
//! 2. values and reference gradients of `u` at the points of `Q`
//!    ([`Sweep::interpolate`], [`Sweep::gradient`] on the trial shape),
//! 3. at each point and lane: geometry at the point (stored or recomputed,
//!    from `Q`'s entry), push-forward, kernel, pull-back (same formulas as
//!    [`Space::mlt_add`]),
//! 4. test against `v`: [`Sweep::interpolate_t`] of the source and
//!    [`Sweep::gradient_t`] of the pulled-back flux on the test shape,
//! 5. scatter `β·v + α·y`.
//!
//! The two spaces must come from the same mesh (same element storage,
//! checked by a fingerprint); the output is element-local (no shared-DOF sum).

use nalgebra::{SMatrix, SVector};

use crate::element::{mlt_add_element, Batch, ElementOperator, PointSet};
use crate::engine::{MixedKernel, PointIn, PointOut};
use crate::quadrature::{Quadrature, Sweep};
use crate::shape::Shape;
use crate::space::{GeoKind, Space};

/// The generic element operator of [`mlt_add_mixed`]: the pointwise `kernel`
/// at the points of the quadrature `Q`, node ↔ point sweeps on both shapes.
pub struct Pointwise<'a, const D: usize, K, Q, SU, SV>
where
    SU: Shape<D>,
    SV: Shape<D>,
    Q: Sweep<D, SU> + Sweep<D, SV>,
{
    kernel: &'a K,
    set: PointSet<'a, D>,
    tu: <Q as Sweep<D, SU>>::Tables,
    tv: <Q as Sweep<D, SV>>::Tables,
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
            .unwrap_or_else(|| panic!("quadrature {} is declared on neither space", std::any::type_name::<Q>()));
        Pointwise {
            kernel,
            set,
            tu: <Q as Sweep<D, SU>>::tables(),
            tv: <Q as Sweep<D, SV>>::tables(),
        }
    }
}

/// Per-task scratch of [`Pointwise`].
pub struct PointwiseScratch {
    /// Values at the points, `[c][q][lane]`.
    uq: Vec<f64>,
    /// Reference gradients at the points.
    gq: Vec<f64>,
    /// Pulled-back fluxes at the points.
    hq: Vec<f64>,
    /// Weighted sources at the points, `[c][q][lane]`.
    sq: Vec<f64>,
    /// One transposed sweep on the test shape.
    yt: Vec<f64>,
    /// Sweep scratch.
    tmp: Vec<f64>,
}

impl<const D: usize, K, Q, SU, SV, const CU: usize, const CV: usize> ElementOperator<D, SU, SV, CU, CV>
    for Pointwise<'_, D, K, Q, SU, SV>
where
    SU: Shape<D>,
    SV: Shape<D>,
    Q: Sweep<D, SU> + Sweep<D, SV>,
    K: MixedKernel<D, CU, CV>,
    <Q as Sweep<D, SU>>::Tables: Sync,
    <Q as Sweep<D, SV>>::Tables: Sync,
{
    type Scratch = PointwiseScratch;
    const READS_INPUT: bool = K::USES_VALUE || K::USES_GRAD;

    fn scratch<const W: usize>(&self) -> PointwiseScratch {
        let nqw = <Q as Quadrature<D>>::NQ * W;
        PointwiseScratch {
            uq: vec![0.0; CU * nqw],
            gq: vec![0.0; CU * D * nqw],
            hq: vec![0.0; CV * D * nqw],
            sq: vec![0.0; CV * nqw],
            yt: vec![0.0; SV::NLOC * W],
            tmp: vec![0.0; W * <Q as Sweep<D, SU>>::TMP.max(<Q as Sweep<D, SV>>::TMP)],
        }
    }

    fn apply<const W: usize>(&self, batch: &Batch<'_, D, SU, SV>, u: &[f64], y: &mut [f64], s: &mut PointwiseScratch) {
        let (nq, nuw, nqw, nvw) = (<Q as Quadrature<D>>::NQ, SU::NLOC * W, <Q as Quadrature<D>>::NQ * W, SV::NLOC * W);
        let (tu, tv, kernel, set) = (&self.tu, &self.tv, self.kernel, &self.set);

        // 2. Values and reference gradients at the points.
        for c in 0..CU {
            let uc = &u[c * nuw..(c + 1) * nuw];
            if K::USES_VALUE {
                <Q as Sweep<D, SU>>::interpolate::<W>(tu, uc, &mut s.uq[c * nqw..(c + 1) * nqw], &mut s.tmp);
            }
            if K::USES_GRAD {
                for d in 0..D {
                    let out = &mut s.gq[(c * D + d) * nqw..(c * D + d + 1) * nqw];
                    <Q as Sweep<D, SU>>::gradient::<W>(tu, d, uc, out, &mut s.tmp);
                }
            }
        }
        // Affine elements: one Jacobian per lane.
        let affine: [(f64, SMatrix<f64, D, D>); W] = core::array::from_fn(|lane| {
            if batch.kind == GeoKind::Affine {
                let (_, dj, cof) = batch.geometry(set, lane, 0);
                (dj, cof)
            } else {
                (0.0, SMatrix::zeros())
            }
        });

        // 3. Point loop.
        let mut input = PointIn::<D, CU> {
            value: SVector::zeros(),
            grad: SMatrix::zeros(),
            xyz: SVector::zeros(),
            label: 0,
            dof: 0,
        };
        let mut output = PointOut::<D, CV> {
            flux: SMatrix::zeros(),
            source: SVector::zeros(),
        };
        for q in 0..nq {
            let w = set.weight(q);
            for (lane, aff) in affine.iter().enumerate() {
                let e = batch.element(lane);
                let (det_j, cof) = if batch.kind == GeoKind::Affine && !K::USES_XYZ {
                    *aff
                } else {
                    let (x, dj, cof) = batch.geometry(set, lane, q);
                    input.xyz = x;
                    (dj, cof)
                };
                if K::USES_VALUE {
                    for c in 0..CU {
                        input.value[c] = s.uq[c * nqw + q * W + lane];
                    }
                }
                if K::USES_GRAD {
                    let mut gu = SMatrix::<f64, CU, D>::zeros();
                    for c in 0..CU {
                        for d in 0..D {
                            gu[(c, d)] = s.gq[(c * D + d) * nqw + q * W + lane];
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
                            s.hq[(c * D + d) * nqw + q * W + lane] = h[(c, d)];
                        }
                    }
                }
                if K::OUTPUTS_SOURCE {
                    let ws = w * det_j.abs();
                    for c in 0..CV {
                        s.sq[c * nqw + q * W + lane] = ws * output.source[c];
                    }
                }
            }
        }

        // 4. Test against v.
        for c in 0..CV {
            let yc = &mut y[c * nvw..(c + 1) * nvw];
            if K::OUTPUTS_SOURCE {
                <Q as Sweep<D, SV>>::interpolate_t::<W>(tv, &s.sq[c * nqw..(c + 1) * nqw], yc, &mut s.tmp);
            } else {
                yc.fill(0.0);
            }
            if K::OUTPUTS_FLUX {
                for d in 0..D {
                    let h = &s.hq[(c * D + d) * nqw..(c * D + d + 1) * nqw];
                    <Q as Sweep<D, SV>>::gradient_t::<W>(tv, d, h, &mut s.yt, &mut s.tmp);
                    for (yv, t) in yc.iter_mut().zip(&s.yt) {
                        *yv += t;
                    }
                }
            }
        }
    }
}

/// `v ← β·v + α·A·u` for the mixed operator of `kernel` between `u_space`
/// (trial) and `v_space` (test), integrated with the quadrature `Q` declared
/// on either space; batches of 4 elements ([`Pointwise`] run by
/// [`mlt_add_element`]).
///
/// ```ignore
/// // b(u, q) = ∫ q div u: displacement → pressure, Gauss–Lobatto rule of the displacement
/// mlt_add_mixed::<Quad2<Gll<5>, Gll<5>>, 2, 1, _, _, _, _>(&disp, &pres, &Divergence { coef: Constant(1.0) }, 1.0, &u, 0.0, &mut p);
/// ```
///
/// With `β = 0`, `v` is overwritten. Panics if the spaces come from different
/// meshes, if `Q` is declared on neither space, or on wrong vector lengths.
#[allow(clippy::too_many_arguments)]
pub fn mlt_add_mixed<Q, const CU: usize, const CV: usize, const D: usize, SU, SV, K>(
    u_space: &Space<D, SU>,
    v_space: &Space<D, SV>,
    kernel: &K,
    alpha: f64,
    u: &[f64],
    beta: f64,
    v: &mut [f64],
) where
    SU: Shape<D>,
    SV: Shape<D>,
    Q: Sweep<D, SU> + Sweep<D, SV>,
    K: MixedKernel<D, CU, CV>,
{
    assert_eq!(
        u_space.fingerprint, v_space.fingerprint,
        "mixed product between spaces built from different meshes"
    );
    let op = Pointwise::<D, K, Q, SU, SV>::new(u_space, v_space, kernel);
    mlt_add_element::<D, SU, SV, CU, CV, _>(u_space, v_space, &op, alpha, u, beta, v);
}
