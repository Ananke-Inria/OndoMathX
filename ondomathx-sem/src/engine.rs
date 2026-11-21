//! Matrix-free operator application: `v ← β·v + α·A·u`, element by element.
//!
//! An operator is given by a [`PointKernel`]: at every GLL point it maps the
//! value and physical gradient of `u` to a **flux** `G` (tested against `∇v`)
//! and a **source** `S` (tested against `v`):
//! $$ a(u, v) = \int_\Omega G(u) : \nabla v + S(u) \cdot v , $$
//! integrated with the GLL rule (collocation: quadrature points = nodes).
//!
//! Per batch of `W` elements of the same geometric type the engine
//! 1. gathers `u` into lobatto's batched layout (node `g`, lane `l` at `g·W + l`),
//! 2. applies the reference derivatives ([`Shape::ref_derivative`]: hardcoded
//!    lobatto kernels for isotropic N = 4..7, generic ones otherwise),
//! 3. at each point and lane: evaluates the geometry, pushes the gradient
//!    forward, `∇ₓu = ∇ᵤu · cof(J)ᵀ / det J`, calls the kernel, and pulls the
//!    flux back, `H = sign(det J)·w·G·cof(J)`, so that `Σ_b D_bᵀ H_b` tests it,
//! 4. applies the transposed derivatives and scatters `β·v + α·A·u`.
//!
//! The output is element-local (no shared-DOF sum), so products can be chained
//! before [`Space::c_sum_shared`].

use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::geometry::{cofactor, det};
use crate::shape::Shape;
use crate::space::{GeoKind, Space};

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

/// Output of a point kernel; the engine zeroes it before each call.
#[derive(Clone, Debug)]
pub struct PointOut<const D: usize, const C: usize> {
    /// Flux `G`, tested against `∂v_c/∂x_d` (`OUTPUTS_FLUX`).
    pub flux: SMatrix<f64, C, D>,
    /// Source `S`, tested against `v_c` (`OUTPUTS_SOURCE`).
    pub source: SVector<f64, C>,
}

/// Pointwise physics of an operator with `C` components in dimension `D`.
///
/// The flags are compile-time switches: unused inputs are neither computed nor
/// copied, and an operator without flux skips all derivative kernels. All
/// default to `false`; a kernel states what it uses and produces.
pub trait PointKernel<const D: usize, const C: usize>: Sync {
    const USES_VALUE: bool = false;
    const USES_GRAD: bool = false;
    const USES_XYZ: bool = false;
    const USES_LABEL: bool = false;
    const USES_DOF: bool = false;
    const OUTPUTS_FLUX: bool = false;
    const OUTPUTS_SOURCE: bool = false;

    fn apply(&self, input: &PointIn<D, C>, output: &mut PointOut<D, C>);
}

/// Pointwise physics of a mixed operator `a(u, v)` with `u` of `CU`
/// components (trial space) and `v` of `CV` components (test space):
/// `a(u, v) = ∫ G(u) : ∇v + S(u)·v`, applied by [`crate::mlt_add_mixed`] at
/// the points of a quadrature. Same flags as [`PointKernel`]; every
/// `PointKernel<D, C>` is a `MixedKernel<D, C, C>`.
///
/// In a mixed product `PointIn::dof` is the quadrature-point index
/// `e·NQ + q`, so one-value-per-node data (`PointData`) does not apply there.
pub trait MixedKernel<const D: usize, const CU: usize, const CV: usize>: Sync {
    const USES_VALUE: bool = false;
    const USES_GRAD: bool = false;
    const USES_XYZ: bool = false;
    const USES_LABEL: bool = false;
    const USES_DOF: bool = false;
    const OUTPUTS_FLUX: bool = false;
    const OUTPUTS_SOURCE: bool = false;

    fn apply(&self, input: &PointIn<D, CU>, output: &mut PointOut<D, CV>);
}

impl<const D: usize, const C: usize, K: PointKernel<D, C>> MixedKernel<D, C, C> for K {
    const USES_VALUE: bool = <K as PointKernel<D, C>>::USES_VALUE;
    const USES_GRAD: bool = <K as PointKernel<D, C>>::USES_GRAD;
    const USES_XYZ: bool = <K as PointKernel<D, C>>::USES_XYZ;
    const USES_LABEL: bool = <K as PointKernel<D, C>>::USES_LABEL;
    const USES_DOF: bool = <K as PointKernel<D, C>>::USES_DOF;
    const OUTPUTS_FLUX: bool = <K as PointKernel<D, C>>::OUTPUTS_FLUX;
    const OUTPUTS_SOURCE: bool = <K as PointKernel<D, C>>::OUTPUTS_SOURCE;

    #[inline]
    fn apply(&self, input: &PointIn<D, C>, output: &mut PointOut<D, C>) {
        PointKernel::apply(self, input, output)
    }
}

/// Per-task scratch, allocated once per rayon task.
struct Scratch {
    /// Gathered input, `[c][node][lane]`.
    u: Vec<f64>,
    /// Reference gradients, then pulled-back fluxes, `[c][dir][node][lane]`.
    g: Vec<f64>,
    /// Element-local result, `[c][node][lane]`.
    y: Vec<f64>,
    /// One transposed derivative, `[node][lane]`.
    tmp: Vec<f64>,
}

/// Geometry kinds as const generic arguments, so each group gets its own loop.
const AFFINE: usize = 0;
const Q1: usize = 1;
const Q2: usize = 2;
const MAPPED: usize = 3;

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// `v ← β·v + α·A·u` with the operator of `kernel`, batches of 4 elements.
    ///
    /// `u` and `v` are `C`-component vectors (`n_elem · C · NLOC` entries). The
    /// result is element-local: call [`Space::c_sum_shared`] to assemble it.
    /// With `β = 0`, `v` is overwritten (its previous content is never read).
    pub fn mlt_add<const C: usize, K: PointKernel<D, C>>(
        &self,
        kernel: &K,
        alpha: f64,
        u: &[f64],
        beta: f64,
        v: &mut [f64],
    ) {
        self.mlt_add_batched::<C, 4, K>(kernel, alpha, u, beta, v);
    }

    /// [`Space::mlt_add`] with an explicit batch width `W`.
    pub fn mlt_add_batched<const C: usize, const W: usize, K: PointKernel<D, C>>(
        &self,
        kernel: &K,
        alpha: f64,
        u: &[f64],
        beta: f64,
        v: &mut [f64],
    ) {
        assert!(W >= 1, "batch width must be positive");
        let block = C * Self::NLOC;
        assert_eq!(u.len(), self.n_elem * block, "u length");
        assert_eq!(v.len(), self.n_elem * block, "v length");
        let weights: Vec<f64> = (0..Self::NLOC).map(|n| self.weight(n)).collect();

        for kind in GeoKind::ALL {
            let range = self.elements(kind);
            if range.is_empty() {
                continue;
            }
            let u_group = &u[range.start * block..range.end * block];
            let v_group = &mut v[range.start * block..range.end * block];
            let ctx = GroupCtx {
                kernel,
                alpha,
                beta,
                weights: &weights,
                start: range.start,
            };
            match kind {
                GeoKind::Affine => self.apply_group::<C, W, K, AFFINE>(&ctx, u_group, v_group),
                GeoKind::Q1 => self.apply_group::<C, W, K, Q1>(&ctx, u_group, v_group),
                GeoKind::Q2 => self.apply_group::<C, W, K, Q2>(&ctx, u_group, v_group),
                GeoKind::Mapped => self.apply_group::<C, W, K, MAPPED>(&ctx, u_group, v_group),
            }
        }
    }

    fn apply_group<const C: usize, const W: usize, K: PointKernel<D, C>, const G: usize>(
        &self,
        ctx: &GroupCtx<'_, K>,
        u_group: &[f64],
        v_group: &mut [f64],
    ) {
        let nloc = Self::NLOC;
        let block = C * nloc;
        v_group.par_chunks_mut(W * block).enumerate().for_each_init(
            || Scratch {
                u: vec![0.0; C * nloc * W],
                g: vec![0.0; C * D * nloc * W],
                y: vec![0.0; C * nloc * W],
                tmp: vec![0.0; nloc * W],
            },
            |s, (b, v_batch)| {
                let n_valid = v_batch.len() / block;
                let u_batch = &u_group[b * W * block..b * W * block + v_batch.len()];
                self.apply_batch::<C, W, K, G>(
                    ctx,
                    ctx.start + b * W,
                    n_valid,
                    u_batch,
                    v_batch,
                    s,
                );
            },
        );
    }

    /// `(x, det J, cof J)` at local node `node` of storage element `e`, of
    /// kind `G` (stored or recomputed, see `GeoStorage`).
    #[inline]
    fn geometry<const G: usize>(&self, e: usize, node: usize) -> (SVector<f64, D>, f64, SMatrix<f64, D, D>) {
        let kind = match G {
            AFFINE => GeoKind::Affine,
            Q1 => GeoKind::Q1,
            Q2 => GeoKind::Q2,
            _ => GeoKind::Mapped,
        };
        self.geo_at::<true>(&self.nodes, kind, e, node)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_batch<const C: usize, const W: usize, K: PointKernel<D, C>, const G: usize>(
        &self,
        ctx: &GroupCtx<'_, K>,
        e0: usize,
        n_valid: usize,
        u_batch: &[f64],
        v_batch: &mut [f64],
        s: &mut Scratch,
    ) {
        let nloc = Self::NLOC;
        let nw = nloc * W;

        // 1. Gather u into [c][node][lane]; padded lanes are zero.
        if K::USES_VALUE || K::USES_GRAD {
            for lane in 0..W {
                for c in 0..C {
                    for node in 0..nloc {
                        s.u[c * nw + node * W + lane] = if lane < n_valid {
                            u_batch[lane * C * nloc + c * nloc + node]
                        } else {
                            0.0
                        };
                    }
                }
            }
        }

        // 2. Reference gradients ∂u_c/∂ξ_b → s.g[(c·D + b)·nw ..].
        if K::USES_GRAD {
            for c in 0..C {
                for b in 0..D {
                    let out = &mut s.g[(c * D + b) * nw..(c * D + b + 1) * nw];
                    S::ref_derivative::<W>(&self.tables, b, false, &s.u[c * nw..(c + 1) * nw], out);
                }
            }
        }

        // Affine elements: one Jacobian per lane, hoisted out of the point loop.
        let affine_geo: [(f64, SMatrix<f64, D, D>); W] = core::array::from_fn(|lane| {
            if G == AFFINE {
                let j = self.affine[e0 + lane.min(n_valid - 1) - self.ranges[0].start].1;
                (det(&j), cofactor(&j))
            } else {
                (0.0, SMatrix::zeros())
            }
        });

        // 3. Point loop: push forward, kernel, pull back.
        let mut input = PointIn::<D, C> {
            value: SVector::zeros(),
            grad: SMatrix::zeros(),
            xyz: SVector::zeros(),
            label: 0,
            dof: 0,
        };
        let mut output = PointOut::<D, C> {
            flux: SMatrix::zeros(),
            source: SVector::zeros(),
        };
        for node in 0..nloc {
            let w = ctx.weights[node];
            for (lane, affine_lane) in affine_geo.iter().enumerate() {
                // Padded lanes reuse the last valid element's geometry (finite
                // values); their results are never scattered.
                let e = e0 + lane.min(n_valid - 1);
                let (det_j, cof) = if G == AFFINE {
                    *affine_lane
                } else {
                    let (x, det_j, cof) = self.geometry::<G>(e, node);
                    if K::USES_XYZ {
                        input.xyz = x;
                    }
                    (det_j, cof)
                };
                if G == AFFINE && K::USES_XYZ {
                    input.xyz = self.geometry::<G>(e, node).0;
                }
                if K::USES_VALUE {
                    for c in 0..C {
                        input.value[c] = s.u[c * nw + node * W + lane];
                    }
                }
                if K::USES_GRAD {
                    let mut gu = SMatrix::<f64, C, D>::zeros();
                    for c in 0..C {
                        for b in 0..D {
                            gu[(c, b)] = s.g[(c * D + b) * nw + node * W + lane];
                        }
                    }
                    input.grad = gu * cof.transpose() / det_j;
                }
                if K::USES_LABEL {
                    input.label = self.labels[e];
                }
                if K::USES_DOF {
                    input.dof = e * nloc + node;
                }

                output.flux.fill(0.0);
                output.source.fill(0.0);
                ctx.kernel.apply(&input, &mut output);

                if K::OUTPUTS_FLUX {
                    let h = output.flux * cof * (w * det_j.signum());
                    for c in 0..C {
                        for b in 0..D {
                            s.g[(c * D + b) * nw + node * W + lane] = h[(c, b)];
                        }
                    }
                }
                let ws = w * det_j.abs();
                for c in 0..C {
                    s.y[c * nw + node * W + lane] = if K::OUTPUTS_SOURCE {
                        ws * output.source[c]
                    } else {
                        0.0
                    };
                }
            }
        }

        // 4. Transposed derivatives of the pulled-back flux: y_c += Σ_b D_bᵀ H_cb.
        if K::OUTPUTS_FLUX {
            for c in 0..C {
                for b in 0..D {
                    S::ref_derivative::<W>(
                        &self.tables,
                        b,
                        true,
                        &s.g[(c * D + b) * nw..(c * D + b + 1) * nw],
                        &mut s.tmp,
                    );
                    for (y, t) in s.y[c * nw..(c + 1) * nw].iter_mut().zip(&s.tmp) {
                        *y += t;
                    }
                }
            }
        }

        // Scatter: v ← β·v + α·y on the valid lanes.
        for lane in 0..n_valid {
            for c in 0..C {
                for node in 0..nloc {
                    let y = s.y[c * nw + node * W + lane];
                    let out = &mut v_batch[lane * C * nloc + c * nloc + node];
                    *out = if ctx.beta == 0.0 {
                        ctx.alpha * y
                    } else {
                        ctx.beta * *out + ctx.alpha * y
                    };
                }
            }
        }
    }
}

/// Per-call constants shared by all batches of a group.
struct GroupCtx<'a, K> {
    kernel: &'a K,
    alpha: f64,
    beta: f64,
    weights: &'a [f64],
    start: usize,
}
