//! The discrete space: GLL Lagrange elements on duplicated `[elem][comp][node]` storage.
//!
//! - `build`: construction ([`SpaceBuilder`]), classification of the elements,
//!   geometry at a point set, declared quadratures;
//! - `geo`: geometry evaluation at a point (`geo_at`, [`Space::map_and_jacobian`]);
//! - `points`: point sets and quadratures (nodes, declared rules, sweep tables);
//! - `mass`: masses and loads ([`InverseMass`]);
//! - `continuous`: the continuous numbering and the `c_` functions that use it.

mod build;
mod continuous;
mod geo;
mod mass;
mod points;

pub(crate) use build::QuadEntry;
pub use build::SpaceBuilder;
pub use mass::InverseMass;

use std::marker::PhantomData;
use std::ops::Range;

use anamesh::mesh::Mesh;
use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::geometry::{GeoStorage, PointGeo};
use crate::quadrature::Sweep;
use crate::shape::Shape;

/// Geometric type of an element; storage holds one contiguous group per kind,
/// in the order [`GeoKind::ALL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeoKind {
    /// `x(ξ) = x0 + J ξ`: a single Jacobian per element.
    Affine,
    /// General bilinear/trilinear element (`2^D` nodes).
    Q1,
    /// General biquadratic/triquadratic element (`3^D` nodes).
    Q2,
    /// Element whose map is given by the mesh (analytic map or attached
    /// deformation, `Mesh::is_mapped_element`) and is not affine: its geometry
    /// is stored at every point set at construction.
    Mapped,
}

impl GeoKind {
    /// Storage order of the groups.
    pub const ALL: [GeoKind; 4] = [GeoKind::Affine, GeoKind::Q1, GeoKind::Q2, GeoKind::Mapped];
}

/// Storage ranges of the four geometric groups (contiguous, in the order
/// [`GeoKind::ALL`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct GroupRanges {
    pub(crate) affine: Range<usize>,
    pub(crate) q1: Range<usize>,
    pub(crate) q2: Range<usize>,
    pub(crate) mapped: Range<usize>,
}

impl GroupRanges {
    /// The range of the group of `kind`.
    pub(crate) fn of(&self, kind: GeoKind) -> &Range<usize> {
        match kind {
            GeoKind::Affine => &self.affine,
            GeoKind::Q1 => &self.q1,
            GeoKind::Q2 => &self.q2,
            GeoKind::Mapped => &self.mapped,
        }
    }

    fn of_mut(&mut self, kind: GeoKind) -> &mut Range<usize> {
        match kind {
            GeoKind::Affine => &mut self.affine,
            GeoKind::Q1 => &mut self.q1,
            GeoKind::Q2 => &mut self.q2,
            GeoKind::Mapped => &mut self.mapped,
        }
    }
}

/// Numbering built by [`Space::new`] on top of the discontinuous storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numbering {
    /// No numbering: a discontinuous space. The `c_` functions panic.
    None,
    /// Continuous everywhere: copies of a node shared by several elements are
    /// one global DOF.
    Continuous,
}

/// Shared-DOF data of the continuous numbering.
pub(crate) struct Shared {
    pub(crate) n_global: usize,
    /// CSR over the global DOFs shared by ≥ 2 copies; entries are `e·NLOC + node`.
    pub(crate) offsets: Vec<usize>,
    pub(crate) indices: Vec<usize>,
}

/// GLL Lagrange space on a quadrangle (`D = 2`) or hexahedron (`D = 3`) mesh,
/// with the point counts per direction of the shape `S` ([`crate::Gll2`],
/// [`crate::Gll3`], [`crate::Iso2`], [`crate::Iso3`]).
///
/// Every space is discontinuous: vectors with `C` components have
/// `n_elem · C · NLOC` entries, element `e` owns `e·C·NLOC .. (e+1)·C·NLOC`, one
/// contiguous slice per component, nodes in tensor order
/// `i + N0·j + N0·N1·k`.
/// An optional continuous numbering ([`Numbering`]) identifies the copies of a
/// node shared by several elements; [`Space::c_sum_shared`] then restores
/// continuity.
///
/// Naming rule: functions prefixed `c_` use the continuous numbering and panic
/// on a space built with [`Numbering::None`]; the others are element-local and
/// never touch it.
///
/// Elements are stored grouped by [`GeoKind`]; [`Space::mesh_element`] maps a
/// storage index back to the mesh. The space does not borrow the mesh.
pub struct Space<const D: usize, S: Shape<D>> {
    /// Per-direction 1D data of the shape.
    pub(crate) tables: S::Tables,
    /// Geometry at the nodes (entry 0 of the point sets).
    pub(crate) nodes: PointGeo<D>,
    /// Tensor GLL weights of the nodes.
    pub(crate) node_weights: Vec<f64>,
    /// Sweep tables of the node rule (built once: the same-space products
    /// borrow them at every call).
    pub(crate) node_sweep: <S::Nodes as Sweep<D, S>>::Tables,
    /// Declared quadratures (fixed at construction).
    pub(crate) quads: Vec<QuadEntry<D>>,
    pub(crate) n_elem: usize,
    /// Storage range of each geometric group.
    pub(crate) ranges: GroupRanges,
    pub(crate) mesh_elem: Vec<usize>,
    pub(crate) labels: Vec<usize>,
    /// Affine group, indexed by `e − ranges.affine.start`.
    pub(crate) affine: Vec<(SVector<f64, D>, SMatrix<f64, D, D>)>,
    /// Q1 group, `2^D` tensor-ordered nodes per element.
    pub(crate) q1_nodes: Vec<[f64; D]>,
    /// Q2 group, `3^D` tensor-ordered nodes per element.
    pub(crate) q2_nodes: Vec<[f64; D]>,
    /// Continuous numbering, if requested.
    pub(crate) shared: Option<Shared>,
    /// Hash of the element storage (order and groups): spaces built from the
    /// same mesh share it, mixed products check it.
    pub(crate) fingerprint: u64,
}

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// Nodes per element and component, `N0·N1(·N2)`.
    pub const NLOC: usize = S::NLOC;

    /// Builds the space on the volume elements of `mesh` with the requested
    /// numbering, no declared quadrature and stored node geometry: the
    /// shorthand of [`Space::builder`].
    pub fn new<M: Mesh>(mesh: &M, numbering: Numbering) -> Self {
        Self::builder(mesh).numbering(numbering).build()
    }

    /// Builder: numbering, node geometry storage and quadratures.
    pub fn builder<M: Mesh>(mesh: &M) -> SpaceBuilder<'_, D, S, M> {
        SpaceBuilder {
            mesh,
            numbering: Numbering::None,
            node_storage: GeoStorage::Store,
            quads: Vec::new(),
            _shape: PhantomData,
        }
    }

    /// Whether `other` was built from the same mesh (same element storage):
    /// the condition of the mixed and element products.
    pub fn same_mesh<T: Shape<D>>(&self, other: &Space<D, T>) -> bool {
        self.fingerprint == other.fingerprint
    }

    /// Whether the space was built with a continuous numbering.
    pub fn has_numbering(&self) -> bool {
        self.shared.is_some()
    }

    /// Multi-index `(i, j[, k])` of local node `node` (tensor order, `i` fastest).
    #[inline]
    pub fn multi_index(mut node: usize) -> [usize; D] {
        core::array::from_fn(|d| {
            let i = node % S::N[d];
            node /= S::N[d];
            i
        })
    }

    /// GLL points on `[0, 1]` of direction `d`.
    pub fn points_1d(&self, d: usize) -> &[f64] {
        S::points_1d(&self.tables, d)
    }

    /// GLL weights of direction `d`.
    pub fn weights_1d(&self, d: usize) -> &[f64] {
        S::weights_1d(&self.tables, d)
    }

    /// Reference point of local node `node` in `[0, 1]^D`.
    pub fn ref_point(&self, node: usize) -> [f64; D] {
        core::array::from_fn(|d| self.nodes.ref_points[node][d])
    }

    /// Number of (volume) elements.
    pub fn n_elem(&self) -> usize {
        self.n_elem
    }

    /// Number of duplicated nodes, `n_elem · NLOC` (entries of a 1-component vector).
    pub fn n_dup(&self) -> usize {
        self.n_elem * Self::NLOC
    }

    /// Storage range of the elements of geometric type `kind`.
    pub fn elements(&self, kind: GeoKind) -> Range<usize> {
        self.ranges.of(kind).clone()
    }

    /// Geometric type of storage element `e`.
    pub fn kind(&self, e: usize) -> GeoKind {
        GeoKind::ALL
            .into_iter()
            .find(|&kind| self.ranges.of(kind).contains(&e))
            .expect("element index out of range")
    }

    /// Mesh index of storage element `e`.
    pub fn mesh_element(&self, e: usize) -> usize {
        self.mesh_elem[e]
    }

    /// Mesh label of storage element `e`.
    pub fn label(&self, e: usize) -> usize {
        self.labels[e]
    }

    /// Tensor GLL weight of local node `node`, `w_i · w_j (· w_k)`.
    pub fn weight(&self, node: usize) -> f64 {
        self.node_weights[node]
    }

    /// Samples `f` at every duplicated node: a `C`-component vector.
    pub fn sample<const C: usize>(&self, f: impl Fn([f64; D]) -> [f64; C] + Sync) -> Vec<f64> {
        let nloc = Self::NLOC;
        let mut v = vec![0.0; self.n_dup() * C];
        v.par_chunks_mut(C * nloc)
            .enumerate()
            .for_each(|(e, block)| {
                for node in 0..nloc {
                    let x = self.map_and_jacobian(e, node).0;
                    let val = f(core::array::from_fn(|a| x[a]));
                    for c in 0..C {
                        block[c * nloc + node] = val[c];
                    }
                }
            });
        v
    }
}
