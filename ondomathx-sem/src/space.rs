//! The discrete space: GLL Lagrange elements on duplicated `[elem][comp][node]` storage.

use std::any::{type_name, Any, TypeId};
use std::marker::PhantomData;
use std::ops::Range;

use anamesh::geometric_elements::GeometricElement;
use anamesh::mesh::Mesh;
use lobatto::tables::points_weights_dyn;
use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::element::PointSet;
use crate::engine::PointIn;
use crate::fields::Field;
use crate::geometry::{
    classify, cofactor, det, digits, jacobian_from_cofactor, tensor_permutation, ElementMap, GeoStorage, GeoTab,
    PointGeo, StoredGeo,
};
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

    pub(crate) fn index(self) -> usize {
        self as usize
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
    /// Declared quadratures (fixed at construction).
    pub(crate) quads: Vec<QuadEntry<D>>,
    pub(crate) n_elem: usize,
    pub(crate) ranges: [Range<usize>; 4],
    pub(crate) mesh_elem: Vec<usize>,
    pub(crate) labels: Vec<usize>,
    /// Affine group, indexed by `e − ranges[Affine].start`.
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

    /// Q1 and Q2 geometric elements of dimension `D` with their tensor permutations.
    fn geometric_elements() -> [(&'static GeometricElement, Vec<usize>); 2] {
        let (q1, q2) = match D {
            2 => (&anamesh::geometric_elements::QUADRANGLE4, &anamesh::geometric_elements::QUADRANGLE9),
            _ => (&anamesh::geometric_elements::HEXAHEDRON8, &anamesh::geometric_elements::HEXAHEDRON27),
        };
        [(q1, tensor_permutation::<D>(q1, 2)), (q2, tensor_permutation::<D>(q2, 3))]
    }

    /// Tensor reference points `[0, 1]^D` (trailing zeros) of `n` per direction
    /// with the 1D points `x1[d]`, x fastest.
    fn tensor_points(n: [usize; 3], x1: &[Vec<f64>]) -> Vec<[f64; 3]> {
        let nq: usize = n[..D].iter().product();
        (0..nq)
            .map(|q| {
                let mut p = [0.0; 3];
                let mut k = q;
                for d in 0..D {
                    p[d] = x1[d][k % n[d]];
                    k /= n[d];
                }
                p
            })
            .collect()
    }

    /// Construction (see [`SpaceBuilder::build`]).
    fn construct<M: Mesh>(mesh: &M, numbering: Numbering, node_storage: GeoStorage) -> Self {
        assert!(D == 2 || D == 3, "D must be 2 or 3 (D = {D})");
        assert!(
            mesh.get_topologic_dimension() == D && mesh.get_geometric_dimension() == D,
            "mesh dimensions (topologic {}, geometric {}) differ from D = {D}",
            mesh.get_topologic_dimension(),
            mesh.get_geometric_dimension()
        );
        assert!(
            (0..D).all(|d| S::N[d] >= 2),
            "every direction needs at least 2 GLL points (N = {:?})",
            &S::N[..D]
        );
        let tables = S::tables();
        let x1: Vec<Vec<f64>> = (0..D).map(|d| S::points_1d(&tables, d).to_vec()).collect();
        let ref_points = Self::tensor_points(S::N, &x1);

        let volume = mesh.get_elements_by_dimension(D);
        assert!(!volume.is_empty(), "the mesh has no volume elements");

        // Classify, then order the storage by kind (stable within a kind).
        let [(q1_ge, q1_perm), (_, q2_perm)] = Self::geometric_elements();
        // 3^D tensor reference nodes of a Q2 element: Q1 test of the
        // classification, affine test of mapped elements.
        let q2_ref: Vec<[f64; 3]> = (0..3usize.pow(D as u32))
            .map(|k| {
                let kd = digits::<D>(k, 3);
                let mut p = [0.0; 3];
                for d in 0..D {
                    p[d] = kd[d] as f64 / 2.0;
                }
                p
            })
            .collect();
        let q1_at_q2 = GeoTab::<D>::new(q1_ge, &q1_perm, &q2_ref);
        let maps: Vec<(usize, ElementMap<D>)> = volume
            .iter()
            .map(|&m| {
                let el = mesh.get_element(m);
                let ge = el.geometric_element;
                if mesh.is_mapped_element(m) {
                    assert_eq!(ge.reference_element.get_dim(), D, "element {m} is not a volume element");
                    // Affine test on the images of the 3^D reference nodes.
                    let images: Vec<[f64; D]> = q2_ref
                        .iter()
                        .map(|uvw| {
                            let mut x = [0.0; 3];
                            mesh.get_element_def(m, uvw, &mut x);
                            core::array::from_fn(|a| x[a])
                        })
                        .collect();
                    return match classify(images, 3, &q1_at_q2) {
                        ElementMap::Affine(x0, j) => (m, ElementMap::Affine(x0, j)),
                        _ => (m, ElementMap::Mapped),
                    };
                }
                let (n1, perm) = match (D, ge.msh_id) {
                    (2, 3) | (3, 5) => (2, &q1_perm),
                    (2, 10) | (3, 12) => (3, &q2_perm),
                    _ => panic!(
                        "element {m} (gmsh type {}) is not a Q1/Q2 {} element",
                        ge.msh_id,
                        if D == 2 { "quadrangle" } else { "hexahedron" }
                    ),
                };
                let tensor_nodes: Vec<[f64; D]> = perm
                    .iter()
                    .map(|&i| {
                        let x = mesh.get_node(el.indexes[i]);
                        core::array::from_fn(|a| x[a])
                    })
                    .collect();
                (m, classify(tensor_nodes, n1, &q1_at_q2))
            })
            .collect();

        let mut mesh_elem = Vec::with_capacity(volume.len());
        let mut affine = Vec::new();
        let mut q1_nodes = Vec::new();
        let mut q2_nodes = Vec::new();
        let mut ranges: [Range<usize>; 4] = [0..0, 0..0, 0..0, 0..0];
        for kind in GeoKind::ALL {
            let start = mesh_elem.len();
            for (m, map) in &maps {
                match (kind, map) {
                    (GeoKind::Affine, ElementMap::Affine(x0, j)) => affine.push((*x0, *j)),
                    (GeoKind::Q1, ElementMap::Q1(n)) => q1_nodes.extend_from_slice(n),
                    (GeoKind::Q2, ElementMap::Q2(n)) => q2_nodes.extend_from_slice(n),
                    (GeoKind::Mapped, ElementMap::Mapped) => {}
                    _ => continue,
                }
                mesh_elem.push(*m);
            }
            ranges[kind.index()] = start..mesh_elem.len();
        }
        let n_elem = mesh_elem.len();
        let labels = mesh_elem.iter().map(|&m| mesh.get_element(m).label).collect();
        let shared = match numbering {
            Numbering::None => None,
            Numbering::Continuous => Some(Self::continuous_numbering(mesh, &mesh_elem)),
        };

        // FNV-1a over the storage order and the group sizes.
        let mut fingerprint: u64 = 0xcbf2_9ce4_8422_2325;
        for v in mesh_elem.iter().chain(ranges.iter().map(|r| &r.end)) {
            for byte in v.to_le_bytes() {
                fingerprint = (fingerprint ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
            }
        }
        let mut space = Space {
            fingerprint,
            node_weights: Vec::new(),
            tables,
            nodes: PointGeo {
                ref_points: Vec::new(),
                tabs: [None, None],
                stored: [None, None, None],
            },
            quads: Vec::new(),
            n_elem,
            ranges,
            mesh_elem,
            labels,
            affine,
            q1_nodes,
            q2_nodes,
            shared,
        };
        space.nodes = space.point_geo(mesh, ref_points, node_storage);
        space.node_weights = (0..S::NLOC).map(|n| space.weight(n)).collect();
        space
    }

    /// Geometry at the reference points `ref_points` of every element: tables
    /// or stored values for the Q1/Q2 groups (`storage`), stored values for
    /// the mapped group (from the mesh's own map).
    fn point_geo<M: Mesh>(&self, mesh: &M, ref_points: Vec<[f64; 3]>, storage: GeoStorage) -> PointGeo<D> {
        let np = ref_points.len();
        let [(q1_ge, q1_perm), (q2_ge, q2_perm)] = Self::geometric_elements();
        let tabs = [
            (!self.ranges[1].is_empty()).then(|| GeoTab::new(q1_ge, &q1_perm, &ref_points)),
            (!self.ranges[2].is_empty()).then(|| GeoTab::new(q2_ge, &q2_perm, &ref_points)),
        ];
        // (x, J) at point p of storage element e → stored (x, det J, cof J).
        let store = |range: &Range<usize>, f: &dyn Fn(usize, usize) -> (SVector<f64, D>, SMatrix<f64, D, D>)| {
            let mut st = StoredGeo {
                x: Vec::with_capacity(range.len() * np),
                det: Vec::with_capacity(range.len() * np),
                cof: Vec::with_capacity(range.len() * np),
            };
            for e in range.clone() {
                for p in 0..np {
                    let (x, j) = f(e, p);
                    st.x.push(x);
                    st.det.push(det(&j));
                    st.cof.push(cofactor(&j));
                }
            }
            st
        };
        let mut stored = [None, None, None];
        if storage == GeoStorage::Store {
            for (g, n1) in [(1usize, 2usize), (2, 3)] {
                if let Some(tab) = &tabs[g - 1] {
                    let n = n1.pow(D as u32);
                    let nodes = if g == 1 { &self.q1_nodes } else { &self.q2_nodes };
                    let base = self.ranges[g].start;
                    stored[g - 1] = Some(store(&self.ranges[g], &|e, p| {
                        let k = e - base;
                        tab.eval(&nodes[k * n..(k + 1) * n], p)
                    }));
                }
            }
        }
        if !self.ranges[3].is_empty() {
            stored[2] = Some(store(&self.ranges[3], &|e, p| {
                let m = self.mesh_elem[e];
                let (mut x, mut g) = ([0.0; 3], [0.0; 9]);
                mesh.get_element_def(m, &ref_points[p], &mut x);
                mesh.get_element_grad_def(m, &ref_points[p], &mut g);
                (
                    SVector::<f64, D>::from_fn(|a, _| x[a]),
                    SMatrix::<f64, D, D>::from_fn(|a, b| g[a * D + b]),
                )
            }));
        }
        let tabs = if storage == GeoStorage::Store { [None, None] } else { tabs };
        PointGeo {
            ref_points,
            tabs,
            stored,
        }
    }

    /// Shared-DOF groups of the continuous numbering
    /// (`Mesh::get_tensor_numbering`, local index in tensor order).
    fn continuous_numbering<M: Mesh>(mesh: &M, mesh_elem: &[usize]) -> Shared {
        let nloc = Self::NLOC;
        let (numbering, n_global) = mesh.get_tensor_numbering(S::N);

        // Bucket every copy by global DOF, keep buckets ≥ 2.
        let mut count = vec![0usize; n_global];
        for &m in mesh_elem {
            assert_eq!(numbering[m].len(), nloc, "numbering of element {m}");
            for &g in &numbering[m] {
                count[g] += 1;
            }
        }
        let mut start_of = vec![usize::MAX; n_global];
        let mut offsets = vec![0];
        for (g, &c) in count.iter().enumerate() {
            if c >= 2 {
                start_of[g] = *offsets.last().unwrap();
                offsets.push(start_of[g] + c);
            }
        }
        let mut indices = vec![0; *offsets.last().unwrap()];
        let mut fill = start_of;
        for (e, &m) in mesh_elem.iter().enumerate() {
            for (n, &g) in numbering[m].iter().enumerate() {
                if count[g] >= 2 {
                    indices[fill[g]] = e * nloc + n;
                    fill[g] += 1;
                }
            }
        }
        Shared {
            n_global,
            offsets,
            indices,
        }
    }

    /// The continuous numbering; panics if the space has none.
    fn shared(&self) -> &Shared {
        self.shared
            .as_ref()
            .expect("this function needs the continuous numbering: build the space with Numbering::Continuous")
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

    /// Number of global DOFs of the continuous numbering (distinct nodes).
    pub fn c_n_global(&self) -> usize {
        self.shared().n_global
    }

    /// Storage range of the elements of geometric type `kind`.
    pub fn elements(&self, kind: GeoKind) -> Range<usize> {
        self.ranges[kind.index()].clone()
    }

    /// Geometric type of storage element `e`.
    pub fn kind(&self, e: usize) -> GeoKind {
        GeoKind::ALL
            .into_iter()
            .find(|k| self.ranges[k.index()].contains(&e))
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

    /// Image `x` and Jacobian `J[a][b] = ∂x_a/∂u_b` at local node `node` of
    /// storage element `e` (from the stored cofactor when the node geometry
    /// is stored).
    pub fn map_and_jacobian(&self, e: usize, node: usize) -> (SVector<f64, D>, SMatrix<f64, D, D>) {
        let kind = self.kind(e);
        let g = kind.index();
        if g >= 1 {
            if let Some(st) = &self.nodes.stored[g - 1] {
                let i = (e - self.ranges[g].start) * Self::NLOC + node;
                return (st.x[i], jacobian_from_cofactor(&st.cof[i], st.det[i]));
            }
        }
        match kind {
            GeoKind::Affine => {
                let (x0, j) = &self.affine[e - self.ranges[0].start];
                (x0 + j * SVector::<f64, D>::from(self.ref_point(node)), *j)
            }
            _ => self.tab_eval(&self.nodes, kind, e, node),
        }
    }

    /// `(x, J)` of a Q1/Q2 element from the tables of a point set.
    #[inline]
    fn tab_eval(&self, pg: &PointGeo<D>, kind: GeoKind, e: usize, p: usize) -> (SVector<f64, D>, SMatrix<f64, D, D>) {
        let (n, nodes, tab) = match kind {
            GeoKind::Q1 => (1 << D, &self.q1_nodes, &pg.tabs[0]),
            GeoKind::Q2 => (3usize.pow(D as u32), &self.q2_nodes, &pg.tabs[1]),
            _ => unreachable!("only Q1/Q2 elements are tabulated"),
        };
        let k = e - self.ranges[kind.index()].start;
        tab.as_ref().expect("geometry tables").eval(&nodes[k * n..(k + 1) * n], p)
    }

    /// `(x, det J, cof J)` at point `p` of the point set `pg`, element `e` of
    /// kind `kind` (the image only if `XYZ`, for affine elements).
    #[inline]
    pub(crate) fn geo_at<const XYZ: bool>(
        &self,
        pg: &PointGeo<D>,
        kind: GeoKind,
        e: usize,
        p: usize,
    ) -> (SVector<f64, D>, f64, SMatrix<f64, D, D>) {
        let g = kind.index();
        if g == 0 {
            let (x0, j) = &self.affine[e - self.ranges[0].start];
            let x = if XYZ {
                x0 + j * SVector::<f64, D>::from_fn(|d, _| pg.ref_points[p][d])
            } else {
                SVector::zeros()
            };
            return (x, det(j), cofactor(j));
        }
        if let Some(st) = &pg.stored[g - 1] {
            let i = (e - self.ranges[g].start) * pg.ref_points.len() + p;
            return (st.x[i], st.det[i], st.cof[i]);
        }
        let (x, j) = self.tab_eval(pg, kind, e, p);
        (x, det(&j), cofactor(&j))
    }

    /// Tensor GLL weight of local node `node`, `w_i · w_j (· w_k)`.
    pub fn weight(&self, node: usize) -> f64 {
        Self::multi_index(node)
            .iter()
            .enumerate()
            .map(|(d, &i)| self.weights_1d(d)[i])
            .product()
    }

    /// Index of component `c` of shared entry `i = e·NLOC + node` in a
    /// `C`-component vector.
    #[inline]
    fn component_index<const C: usize>(i: usize, c: usize) -> usize {
        let nloc = Self::NLOC;
        (i / nloc) * C * nloc + c * nloc + i % nloc
    }

    /// Shared-DOF sum: every copy of a shared node receives the sum over its
    /// copies, component by component. Interior nodes are left unchanged.
    ///
    /// All copies receive bitwise-identical values (same summation order).
    /// `v` has `n_elem · C · NLOC` entries.
    pub fn c_sum_shared<const C: usize>(&self, v: &mut [f64]) {
        let shared = self.shared();
        assert_eq!(v.len(), self.n_dup() * C, "vector length");
        let base = v.as_mut_ptr() as usize;
        let n_groups = shared.offsets.len() - 1;
        (0..n_groups).into_par_iter().for_each(|g| {
            let ptr = base as *mut f64;
            let group = &shared.indices[shared.offsets[g]..shared.offsets[g + 1]];
            for c in 0..C {
                // SAFETY: the groups partition the shared duplicated indices
                // (each copy belongs to exactly one global DOF), so different
                // `g` touch disjoint entries; every index is < n_dup·C = v.len()
                // (checked above), and `v` is exclusively borrowed for the call.
                unsafe {
                    let mut s = 0.0;
                    for &i in group {
                        s += *ptr.add(Self::component_index::<C>(i, c));
                    }
                    for &i in group {
                        *ptr.add(Self::component_index::<C>(i, c)) = s;
                    }
                }
            }
        });
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

    /// Element-local lumped mass `c · w · |det J|`, one value per duplicated
    /// node (before the shared-DOF sum).
    ///
    /// For `C`-component vectors `a`, `b`, [`Space::mass_inner`]`(m, a, b)` with
    /// this `m` is the GLL mass product `⟨a, b⟩_M` of the discontinuous space,
    /// and of the continuous space when `a`, `b` are continuous; no assembly.
    pub fn local_mass<F: Field<D>>(&self, coef: &F) -> Vec<f64> {
        let nloc = Self::NLOC;
        let mut m = vec![0.0; self.n_dup()];
        m.par_chunks_mut(nloc).enumerate().for_each(|(e, block)| {
            let mut p = PointIn::<D, 1> {
                value: SVector::zeros(),
                grad: SMatrix::zeros(),
                xyz: SVector::zeros(),
                label: self.labels[e],
                dof: 0,
            };
            let kind = self.kind(e);
            for (node, out) in block.iter_mut().enumerate() {
                let (x, det_j, _) = self.geo_at::<true>(&self.nodes, kind, e, node);
                p.xyz = x;
                p.dof = e * nloc + node;
                *out = coef.eval(&p) * self.weight(node) * det_j.abs();
            }
        });
        m
    }

    /// Element-local load vector of a volume source, `F_i = ∫ f·φ_i` with the
    /// GLL rule (the nodes): `w_i·|det J_i|·f(x_i)`, `C` components, one entry
    /// per duplicated DoF (before the shared-DOF sum).
    ///
    /// For a source `f(x)·g(t)`, `F` is computed once; with the lumped mass
    /// `M⁻¹ Σ_shared F` is the nodal value of `f` divided by the mass
    /// coefficient, and a leapfrog step adds `dt²·g(tⁿ)·M⁻¹F`.
    pub fn local_load<const C: usize>(&self, f: impl Fn([f64; D]) -> [f64; C] + Sync) -> Vec<f64> {
        let nloc = Self::NLOC;
        let mut v = vec![0.0; self.n_dup() * C];
        v.par_chunks_mut(C * nloc).enumerate().for_each(|(e, block)| {
            let kind = self.kind(e);
            for node in 0..nloc {
                let (x, det_j, _) = self.geo_at::<true>(&self.nodes, kind, e, node);
                let ws = self.node_weights[node] * det_j.abs();
                let val = f(core::array::from_fn(|a| x[a]));
                for c in 0..C {
                    block[c * nloc + node] = ws * val[c];
                }
            }
        });
        v
    }

    /// Element-local load vector `F_i = ∫ f·φ_i` integrated with the declared
    /// quadrature `Q` (e.g. over-integrated, or the `L²` projection load of a
    /// non-polynomial source): `Σ_q w_q·|det J_q|·f(x_q)·φ_i(x_q)`.
    pub fn local_load_with<Q: Sweep<D, S>, const C: usize>(
        &self,
        f: impl Fn([f64; D]) -> [f64; C] + Sync,
    ) -> Vec<f64> {
        let nloc = Self::NLOC;
        let q = self.entry::<Q>();
        let t = q.sweep.downcast_ref::<Q::Tables>().expect("sweep tables of the declared quadrature");
        let mut v = vec![0.0; self.n_dup() * C];
        v.par_chunks_mut(C * nloc).enumerate().for_each_init(
            || (vec![0.0; C * Q::NQ], vec![0.0; Q::TMP]),
            |(fq, tmp), (e, block)| {
                let kind = self.kind(e);
                for p in 0..Q::NQ {
                    let (x, det_j, _) = self.geo_at::<true>(&q.geo, kind, e, p);
                    let ws = q.weights[p] * det_j.abs();
                    let val = f(core::array::from_fn(|a| x[a]));
                    for c in 0..C {
                        fq[c * Q::NQ + p] = ws * val[c];
                    }
                }
                for c in 0..C {
                    Q::interpolate_t::<1>(t, &fq[c * Q::NQ..(c + 1) * Q::NQ], &mut block[c * nloc..(c + 1) * nloc], tmp);
                }
            },
        );
        v
    }

    /// Inverse of the element-local lumped mass `c · w · |det J|`: the mass of
    /// the discontinuous space (block diagonal, here diagonal). Never uses the
    /// numbering; see [`Space::c_inverse_mass`] for the continuous mass.
    ///
    /// Panics if a mass coefficient is not positive.
    pub fn inverse_mass<F: Field<D>>(&self, coef: &F) -> InverseMass<D, S> {
        InverseMass::from_mass(self.local_mass(coef))
    }

    /// Inverse of the assembled lumped mass `Σ_shared c · w · |det J|` of the
    /// continuous space (equal across the copies of a node, shared by all
    /// components): `ρ` for elastodynamics, `Inverse(κ)` for acoustics.
    ///
    /// Panics if a mass coefficient is not positive, or without the continuous
    /// numbering.
    pub fn c_inverse_mass<F: Field<D>>(&self, coef: &F) -> InverseMass<D, S> {
        let mut m = self.local_mass(coef);
        self.c_sum_shared::<1>(&mut m);
        InverseMass::from_mass(m)
    }

    /// Homogeneous Dirichlet mask: one value per duplicated node, `0` on the
    /// Dirichlet boundary, `1` elsewhere (all copies of a boundary node).
    ///
    /// `labels = None`: the whole boundary (facets with a single volume
    /// neighbour, `UnstructuredMesh::get_boundary_faces`); `Some(labels)`: the
    /// facets carrying a boundary element (dimension `D − 1`) with one of these
    /// labels (`UnstructuredMesh::get_parent_face`). `mesh` is the one the
    /// space was built from (its unstructured view gives the topology).
    ///
    /// With the diagonal mass, the scheme on `V_h ∩ H¹₀` is the full scheme
    /// with the update multiplied by the mask (and the same in the power
    /// iteration of the CFL step). Panics without the continuous numbering.
    pub fn c_boundary_mask<M: Mesh>(&self, mesh: &M, labels: Option<&[usize]>) -> Vec<f64> {
        let nloc = Self::NLOC;
        let um = mesh.get_unstructured_mesh();
        let facets: Vec<(usize, usize)> = match labels {
            None => um.get_boundary_faces(),
            Some(labels) => labels
                .iter()
                .flat_map(|&l| um.get_elements_by_dimension_and_label(l, D - 1))
                .map(|b| um.get_parent_face(b))
                .collect(),
        };
        let mut storage = vec![usize::MAX; um.get_nelements()];
        for (e, &m) in self.mesh_elem.iter().enumerate() {
            storage[m] = e;
        }
        let mut on = vec![0.0; self.n_dup()];
        for (m, f) in facets {
            let e = storage[m];
            assert!(e != usize::MAX, "boundary facet of element {m}, not in the space");
            let reference = um.get_element(m).geometric_element.reference_element;
            for (n, uvw) in self.nodes.ref_points.iter().enumerate() {
                let hit = if D == 2 { reference.is_on_edge(uvw, f) } else { reference.is_on_face(uvw, f) };
                if hit {
                    on[e * nloc + n] = 1.0;
                }
            }
        }
        // Copies of boundary nodes in elements touching the boundary only at
        // an edge or a vertex.
        self.c_sum_shared::<1>(&mut on);
        on.iter().map(|&v| if v > 0.0 { 0.0 } else { 1.0 }).collect()
    }

    /// Number of copies of each duplicated node in the continuous numbering
    /// (`1` for interior nodes).
    pub fn c_multiplicity(&self) -> Vec<f64> {
        let mut k = vec![1.0; self.n_dup()];
        self.c_sum_shared::<1>(&mut k);
        k
    }
}

/// Element-local tensor application of `M_K⁻¹ = B W⁻¹ Bᵀ` to one component
/// block: `(1/W at the points of the element, block, scratch)`.
type TensorApply = Box<dyn Fn(&[f64], &mut [f64], &mut [f64]) + Send + Sync>;

enum InverseMassKind {
    /// One value per duplicated node.
    Diagonal(Vec<f64>),
    /// `B W⁻¹ Bᵀ` per element ([`Space::inverse_mass_with`]).
    Tensor {
        inv_w: Vec<f64>,
        nq: usize,
        scratch: usize,
        apply: TensorApply,
    },
}

/// Inverse mass operator of a [`Space`], built by [`Space::inverse_mass`]
/// (discontinuous GLL lumped mass), [`Space::c_inverse_mass`] (continuous
/// lumped mass) or [`Space::inverse_mass_with`] (discontinuous mass of a
/// quadrature, tensor form), applied with [`InverseMass::apply`].
pub struct InverseMass<const D: usize, S: Shape<D>> {
    kind: InverseMassKind,
    _shape: PhantomData<S>,
}

impl<const D: usize, S: Shape<D>> InverseMass<D, S> {
    /// Inverts a one-value-per-node mass. Panics on a non-positive value.
    fn from_mass(mut m: Vec<f64>) -> Self {
        m.par_iter_mut().for_each(|x| {
            assert!(*x > 0.0, "non-positive mass coefficient {x}");
            *x = 1.0 / *x;
        });
        InverseMass {
            kind: InverseMassKind::Diagonal(m),
            _shape: PhantomData,
        }
    }

    /// `v ← M⁻¹ v` for a `C`-component vector (`n_elem · C · NLOC` entries).
    pub fn apply<const C: usize>(&self, v: &mut [f64]) {
        let nloc = S::NLOC;
        match &self.kind {
            InverseMassKind::Diagonal(diag) => {
                assert_eq!(v.len(), diag.len() * C, "vector length");
                v.par_chunks_mut(C * nloc)
                    .zip(diag.par_chunks(nloc))
                    .for_each(|(block, d)| {
                        for comp in block.chunks_exact_mut(nloc) {
                            for (x, s) in comp.iter_mut().zip(d) {
                                *x *= s;
                            }
                        }
                    });
            }
            InverseMassKind::Tensor {
                inv_w,
                nq,
                scratch,
                apply,
            } => {
                assert_eq!(v.len() / (C * nloc), inv_w.len() / nq, "vector length");
                v.par_chunks_mut(C * nloc).enumerate().for_each_init(
                    || vec![0.0; *scratch],
                    |tmp, (e, block)| {
                        for comp in block.chunks_exact_mut(nloc) {
                            apply(&inv_w[e * nq..(e + 1) * nq], comp, tmp);
                        }
                    },
                );
            }
        }
    }

    /// The diagonal of `M⁻¹`, one value per duplicated node, when `M⁻¹` is
    /// diagonal (`None` for [`Space::inverse_mass_with`]).
    pub fn diagonal(&self) -> Option<&[f64]> {
        match &self.kind {
            InverseMassKind::Diagonal(d) => Some(d),
            InverseMassKind::Tensor { .. } => None,
        }
    }
}

// ── Builder and quadratures ────────────────────────────────────────────

/// A declared quadrature: points, weights, geometry at the points and the
/// node → point interpolation tables of its [`Sweep`].
pub(crate) struct QuadEntry<const D: usize> {
    pub(crate) id: TypeId,
    /// Tensor weights `w_q` (sum 1 on the reference element).
    pub(crate) weights: Vec<f64>,
    pub(crate) geo: PointGeo<D>,
    /// `<Q as Sweep<D, S>>::Tables`.
    pub(crate) sweep: Box<dyn Any + Send + Sync>,
}

type EntryBuilder<const D: usize, S, M> = fn(&Space<D, S>, &M, GeoStorage) -> QuadEntry<D>;

/// Builder of a [`Space`] ([`Space::builder`]): numbering (default
/// [`Numbering::None`]), storage of the node geometry (default
/// [`GeoStorage::Store`]: 2–4× faster on Q1/Q2 elements than `Recompute`, at
/// `D + D² + 1` values per node) and the fixed list of quadratures.
///
/// ```ignore
/// type PressureMass = Quad3<Gl<3>, Gl<3>, Gl<3>>;
/// let space = Space::<3, Iso3<5>>::builder(&mesh)
///     .numbering(Numbering::Continuous)
///     .quadrature::<PressureMass>(GeoStorage::Store)
///     .build();
/// let w = space.quadrature_weights::<PressureMass>();
/// ```
pub struct SpaceBuilder<'m, const D: usize, S: Shape<D>, M: Mesh> {
    mesh: &'m M,
    numbering: Numbering,
    node_storage: GeoStorage,
    quads: Vec<(TypeId, &'static str, GeoStorage, EntryBuilder<D, S, M>)>,
    _shape: PhantomData<S>,
}

impl<const D: usize, S: Shape<D>, M: Mesh> SpaceBuilder<'_, D, S, M> {
    /// Numbering of the space (`Mesh::get_tensor_numbering` for
    /// [`Numbering::Continuous`]).
    pub fn numbering(mut self, numbering: Numbering) -> Self {
        self.numbering = numbering;
        self
    }

    /// Storage of the geometry at the nodes (affine/Q1/Q2 groups; mapped
    /// elements are always stored).
    pub fn node_geometry(mut self, storage: GeoStorage) -> Self {
        self.node_storage = storage;
        self
    }

    /// Declares the quadrature `Q`, with the storage of its geometry. Panics if
    /// `Q` is declared twice.
    pub fn quadrature<Q: Sweep<D, S>>(mut self, storage: GeoStorage) -> Self {
        assert!(
            self.quads.iter().all(|q| q.0 != TypeId::of::<Q>()),
            "quadrature {} declared twice",
            type_name::<Q>()
        );
        self.quads
            .push((TypeId::of::<Q>(), type_name::<Q>(), storage, Space::<D, S>::quad_entry::<Q, M>));
        self
    }

    /// Builds the space on the volume elements of the mesh.
    ///
    /// Elements for which `Mesh::is_mapped_element` holds take their geometry
    /// from the mesh's own map (`get_element_def`, `get_element_grad_def`),
    /// stored at every point set, and are affine if that map is; the others
    /// must be QUADRANGLE4/9 (`D = 2`) or HEXAHEDRON8/27 (`D = 3`) and their
    /// map is the interpolation of their geometric nodes.
    ///
    /// Panics if `D` is not 2 or 3, if the mesh dimensions differ from `D`, on
    /// another element type, or if the mesh cannot number the shape
    /// continuously (e.g. anisotropic counts on an `UnstructuredMesh`).
    pub fn build(self) -> Space<D, S> {
        let mut space = Space::construct(self.mesh, self.numbering, self.node_storage);
        for (_, _, storage, make) in self.quads {
            let entry = make(&space, self.mesh, storage);
            space.quads.push(entry);
        }
        space
    }
}

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// Entry of the quadrature `Q` (points and weights from `lobatto::tables`).
    fn quad_entry<Q: Sweep<D, S>, M: Mesh>(&self, mesh: &M, storage: GeoStorage) -> QuadEntry<D> {
        let families = Q::families();
        let (x1, w1): (Vec<Vec<f64>>, Vec<Vec<f64>>) =
            (0..D).map(|d| points_weights_dyn(Q::M[d], families[d])).unzip();
        let ref_points = Self::tensor_points(Q::M, &x1);
        let weights = (0..Q::NQ)
            .map(|q| {
                let mut k = q;
                let mut w = 1.0;
                for (d, wd) in w1.iter().enumerate() {
                    w *= wd[k % Q::M[d]];
                    k /= Q::M[d];
                }
                w
            })
            .collect();
        QuadEntry {
            id: TypeId::of::<Q>(),
            weights,
            geo: self.point_geo(mesh, ref_points, storage),
            sweep: Box::new(Q::tables()),
        }
    }

    /// The declared quadrature `Q`, if any.
    pub(crate) fn find_entry<Q: 'static>(&self) -> Option<&QuadEntry<D>> {
        self.quads.iter().find(|q| q.id == TypeId::of::<Q>())
    }

    /// The nodes as a point set (GLL weights, node geometry), for
    /// [`crate::ElementOperator`]s.
    pub fn node_points(&self) -> PointSet<'_, D> {
        PointSet {
            geo: &self.nodes,
            weights: &self.node_weights,
        }
    }

    /// The declared quadrature `Q` as a point set, if declared.
    pub fn quadrature_points<Q: 'static>(&self) -> Option<PointSet<'_, D>> {
        self.find_entry::<Q>().map(|q| PointSet {
            geo: &q.geo,
            weights: &q.weights,
        })
    }

    /// The declared quadrature `Q`; panics if it was not declared.
    fn entry<Q: Sweep<D, S>>(&self) -> &QuadEntry<D> {
        self.find_entry::<Q>()
            .unwrap_or_else(|| panic!("quadrature {} was not declared (SpaceBuilder::quadrature)", type_name::<Q>()))
    }

    /// Inverse of the element-local mass `∫ c u v` integrated with the
    /// declared quadrature `Q` (the discontinuous mass, e.g. the pressure mass
    /// `M^P` with a Gauss–Legendre rule), in tensor form: with `I` the
    /// interpolation nodes → points (square and invertible: `Q` has the
    /// space's counts per direction) and `B = I⁻¹` the interpolation back,
    /// `M_K = Iᵀ W_K I` and `M_K⁻¹ = B W_K⁻¹ Bᵀ`, `W_K = c·w·|det J|` at the
    /// points. Two sum-factorized passes and a diagonal per element and
    /// component, also on non-affine elements.
    ///
    /// Panics if `Q` was not declared, if its counts differ from the space's,
    /// or on a non-positive `c·w·|det J|`.
    pub fn inverse_mass_with<Q: Sweep<D, S>, F: Field<D>>(&self, coef: &F) -> InverseMass<D, S> {
        assert!(
            (0..D).all(|d| Q::M[d] == S::N[d]),
            "inverse_mass_with: {} must have the space's counts {:?} per direction",
            type_name::<Q>(),
            &S::N[..D]
        );
        let q = self.entry::<Q>();
        let nq = Q::NQ;
        let mut inv_w = vec![0.0; self.n_elem * nq];
        inv_w.par_chunks_mut(nq).enumerate().for_each(|(e, block)| {
            let kind = self.kind(e);
            let mut p = PointIn::<D, 1> {
                value: SVector::zeros(),
                grad: SMatrix::zeros(),
                xyz: SVector::zeros(),
                label: self.labels[e],
                dof: 0,
            };
            for (k, out) in block.iter_mut().enumerate() {
                let (x, det_j, _) = self.geo_at::<true>(&q.geo, kind, e, k);
                p.xyz = x;
                p.dof = e * nq + k;
                let m = coef.eval(&p) * q.weights[k] * det_j.abs();
                assert!(m > 0.0, "non-positive mass coefficient {m}");
                *out = 1.0 / m;
            }
        });
        let t = Q::tables();
        let apply: TensorApply = Box::new(move |iw: &[f64], v: &mut [f64], tmp: &mut [f64]| {
            let (a, rest) = tmp.split_at_mut(iw.len());
            Q::back_t::<1>(&t, v, a, rest);
            for (x, w) in a.iter_mut().zip(iw) {
                *x *= w;
            }
            Q::back::<1>(&t, a, v, rest);
        });
        InverseMass {
            kind: InverseMassKind::Tensor {
                inv_w,
                nq,
                scratch: nq + Q::TMP,
                apply,
            },
            _shape: PhantomData,
        }
    }

    /// `w_q · |det J|` at every point of the quadrature `Q`, `[elem][q]`
    /// (`n_elem · NQ` entries): `∫ f = Σ w f(x_q)`.
    pub fn quadrature_weights<Q: Sweep<D, S>>(&self) -> Vec<f64> {
        let q = self.entry::<Q>();
        let mut out = vec![0.0; self.n_elem * Q::NQ];
        out.par_chunks_mut(Q::NQ).enumerate().for_each(|(e, block)| {
            let kind = self.kind(e);
            for (p, w) in block.iter_mut().enumerate() {
                *w = q.weights[p] * self.geo_at::<false>(&q.geo, kind, e, p).1.abs();
            }
        });
        out
    }

    /// Physical points of the quadrature `Q`, `[elem][q]`.
    pub fn quadrature_xyz<Q: Sweep<D, S>>(&self) -> Vec<[f64; D]> {
        let q = self.entry::<Q>();
        let mut out = vec![[0.0; D]; self.n_elem * Q::NQ];
        out.par_chunks_mut(Q::NQ).enumerate().for_each(|(e, block)| {
            let kind = self.kind(e);
            for (p, x) in block.iter_mut().enumerate() {
                let xp = self.geo_at::<true>(&q.geo, kind, e, p).0;
                *x = core::array::from_fn(|a| xp[a]);
            }
        });
        out
    }

    /// Values of the `C`-component vector `u` at the points of the quadrature
    /// `Q` (Lagrange interpolation from the GLL nodes, by sum factorization),
    /// `[elem][comp][q]` (`n_elem · C · NQ` entries).
    pub fn quadrature_values<Q: Sweep<D, S>, const C: usize>(&self, u: &[f64]) -> Vec<f64> {
        let nloc = Self::NLOC;
        assert_eq!(u.len(), self.n_dup() * C, "vector length");
        let t = self
            .entry::<Q>()
            .sweep
            .downcast_ref::<Q::Tables>()
            .expect("sweep tables of the declared quadrature");
        let mut out = vec![0.0; self.n_elem * C * Q::NQ];
        out.par_chunks_mut(C * Q::NQ).enumerate().for_each_init(
            || vec![0.0; Q::TMP],
            |tmp, (e, block)| {
                for c in 0..C {
                    let input = &u[(e * C + c) * nloc..(e * C + c + 1) * nloc];
                    Q::interpolate::<1>(t, input, &mut block[c * Q::NQ..(c + 1) * Q::NQ], tmp);
                }
            },
        );
        out
    }
}
