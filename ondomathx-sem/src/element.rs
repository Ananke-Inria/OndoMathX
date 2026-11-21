//! Hand-written element operators: the shared driver of the matrix-free
//! products and the hook to plug a local operator into it.
//!
//! [`mlt_add_element`] does the plumbing common to every operator: loop over
//! the geometric groups, batches of `W` elements, parallel chunks, per-task
//! scratch, gather of `u` into lobatto's batched layout `[c][node][lane]`,
//! scatter of `β·v + α·y`, size and same-mesh checks. An
//! [`ElementOperator`] only writes the local result `y` of a batch from its
//! gathered input, with the [`Batch`] giving the element indices, the labels
//! and the geometry at the nodes of either space or at a declared quadrature
//! ([`PointSet`]). An operator struct can carry any precomputed per-element
//! data (indexed by [`Batch::element`]), e.g. local inverse masses or
//! boundary matrices.
//!
//! [`crate::mlt_add_mixed`] is the generic implementation (pointwise kernel
//! at the points of a quadrature).

use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::geometry::PointGeo;
use crate::shape::Shape;
use crate::space::{GeoKind, Space};

/// A set of reference points of the elements with its weights and geometry:
/// the nodes of a space ([`Space::node_points`]) or a declared quadrature
/// ([`Space::quadrature_points`]).
#[derive(Clone, Copy)]
pub struct PointSet<'a, const D: usize> {
    pub(crate) geo: &'a PointGeo<D>,
    pub(crate) weights: &'a [f64],
}

impl<const D: usize> PointSet<'_, D> {
    /// Number of points per element.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the set is empty (never, for a built space).
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Tensor weight of point `p` on the reference element (they sum to 1).
    pub fn weight(&self, p: usize) -> f64 {
        self.weights[p]
    }

    /// Reference coordinates of point `p` in `[0, 1]^D`.
    pub fn ref_point(&self, p: usize) -> [f64; D] {
        core::array::from_fn(|d| self.geo.ref_points[p][d])
    }
}

/// One batch of at most `W` consecutive elements of the same geometric kind,
/// as seen by an [`ElementOperator`]. Lane `l < n_valid` is element
/// `first + l`; padded lanes (`l ≥ n_valid`) repeat the last valid element
/// (their input is zero and their output is not scattered).
pub struct Batch<'a, const D: usize, SU: Shape<D>, SV: Shape<D>> {
    /// Trial space (input `u`).
    pub u_space: &'a Space<D, SU>,
    /// Test space (output `v`).
    pub v_space: &'a Space<D, SV>,
    /// Geometric kind of all the elements of the batch.
    pub kind: GeoKind,
    /// Storage index of lane 0.
    pub first: usize,
    /// Number of valid lanes (`1..=W`).
    pub n_valid: usize,
}

impl<const D: usize, SU: Shape<D>, SV: Shape<D>> Batch<'_, D, SU, SV> {
    /// Storage element of lane `lane` (padded lanes: the last valid one).
    #[inline]
    pub fn element(&self, lane: usize) -> usize {
        self.first + lane.min(self.n_valid - 1)
    }

    /// Mesh label of lane `lane`.
    #[inline]
    pub fn label(&self, lane: usize) -> usize {
        self.u_space.label(self.element(lane))
    }

    /// `(x, det J, cof J)` at point `p` of `set` for lane `lane` (stored or
    /// recomputed, as declared).
    #[inline]
    pub fn geometry(&self, set: &PointSet<'_, D>, lane: usize, p: usize) -> (SVector<f64, D>, f64, SMatrix<f64, D, D>) {
        self.u_space.geo_at::<true>(set.geo, self.kind, self.element(lane), p)
    }

    /// The declared quadrature `Q` of either space; panics if it is declared
    /// on neither.
    pub fn quadrature<Q: 'static>(&self) -> PointSet<'_, D> {
        self.u_space
            .quadrature_points::<Q>()
            .or_else(|| self.v_space.quadrature_points::<Q>())
            .unwrap_or_else(|| panic!("quadrature {} is declared on neither space", std::any::type_name::<Q>()))
    }
}

/// A local operator `v_e ← A_e u_e`, applied by [`mlt_add_element`] to
/// batches of `W` elements: `u` holds the gathered input `[c][node][lane]`
/// (`CU · SU::NLOC · W`), `v` receives the local result `[c][node][lane]`
/// (`CV · SV::NLOC · W`, to be overwritten entirely).
pub trait ElementOperator<const D: usize, SU: Shape<D>, SV: Shape<D>, const CU: usize, const CV: usize>: Sync {
    /// Per-task scratch.
    type Scratch: Send;
    /// Whether the input is read (otherwise it is not gathered).
    const READS_INPUT: bool = true;

    /// Allocates the scratch of one task, for batches of `W` elements.
    fn scratch<const W: usize>(&self) -> Self::Scratch;

    /// Writes the local result of `batch` into `v`.
    fn apply<const W: usize>(&self, batch: &Batch<'_, D, SU, SV>, u: &[f64], v: &mut [f64], s: &mut Self::Scratch);
}

/// `v ← β·v + α·A·u` for the element operator `op` between `u_space` (trial,
/// `CU` components) and `v_space` (test, `CV` components), batches of 4
/// elements. With `β = 0`, `v` is overwritten. The output is element-local.
///
/// Panics if the spaces come from different meshes or on wrong lengths.
#[allow(clippy::too_many_arguments)]
pub fn mlt_add_element<const D: usize, SU, SV, const CU: usize, const CV: usize, Op>(
    u_space: &Space<D, SU>,
    v_space: &Space<D, SV>,
    op: &Op,
    alpha: f64,
    u: &[f64],
    beta: f64,
    v: &mut [f64],
) where
    SU: Shape<D>,
    SV: Shape<D>,
    Op: ElementOperator<D, SU, SV, CU, CV>,
{
    const W: usize = 4;
    assert_eq!(
        u_space.fingerprint, v_space.fingerprint,
        "element operator between spaces built from different meshes"
    );
    let (nu, nv) = (SU::NLOC, SV::NLOC);
    let n_elem = u_space.n_elem();
    assert_eq!(u.len(), n_elem * CU * nu, "u length");
    assert_eq!(v.len(), n_elem * CV * nv, "v length");
    let (block_u, block_v) = (CU * nu, CV * nv);
    let (nuw, nvw) = (nu * W, nv * W);

    for kind in GeoKind::ALL {
        let range = u_space.elements(kind);
        if range.is_empty() {
            continue;
        }
        let u_group = &u[range.start * block_u..range.end * block_u];
        let v_group = &mut v[range.start * block_v..range.end * block_v];
        v_group.par_chunks_mut(W * block_v).enumerate().for_each_init(
            || (vec![0.0; CU * nuw], vec![0.0; CV * nvw], op.scratch::<W>()),
            |(su, sy, s), (b, v_batch)| {
                let n_valid = v_batch.len() / block_v;
                let batch = Batch {
                    u_space,
                    v_space,
                    kind,
                    first: range.start + b * W,
                    n_valid,
                };
                // Gather u; padded lanes are zero.
                if Op::READS_INPUT {
                    let u_batch = &u_group[b * W * block_u..(b * W + n_valid) * block_u];
                    for lane in 0..W {
                        for c in 0..CU {
                            for node in 0..nu {
                                su[c * nuw + node * W + lane] =
                                    if lane < n_valid { u_batch[lane * block_u + c * nu + node] } else { 0.0 };
                            }
                        }
                    }
                }
                op.apply::<W>(&batch, su, sy, s);
                // Scatter.
                for lane in 0..n_valid {
                    for c in 0..CV {
                        for node in 0..nv {
                            let y = sy[c * nvw + node * W + lane];
                            let out = &mut v_batch[lane * block_v + c * nv + node];
                            *out = if beta == 0.0 { alpha * y } else { beta * *out + alpha * y };
                        }
                    }
                }
            },
        );
    }
}
