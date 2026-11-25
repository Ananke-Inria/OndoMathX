//! Construction of a [`Space`]: the builder, the classification of the
//! elements by geometric kind, the geometry at a point set and the entries of
//! the declared quadratures.

use std::any::{type_name, Any, TypeId};
use std::marker::PhantomData;
use std::ops::Range;

use anamesh::geometric_elements::GeometricElement;
use anamesh::mesh::Mesh;
use lobatto::tables::points_weights_dyn;
use nalgebra::{SMatrix, SVector};

use crate::geometry::{
    classify, cofactor, det, digits, tensor_permutation, ElementMap, GeoStorage, GeoTab, GroupGeo,
    PointGeo, StoredGeo,
};
use crate::quadrature::Sweep;
use crate::shape::Shape;

use super::{GeoKind, GroupRanges, Numbering, Space};

/// The elements in storage order with their geometric data (the result of
/// [`Space::group_by_kind`] during construction).
struct Grouped<const D: usize> {
    mesh_elem: Vec<usize>,
    ranges: GroupRanges,
    affine: Vec<(SVector<f64, D>, SMatrix<f64, D, D>)>,
    q1_nodes: Vec<[f64; D]>,
    q2_nodes: Vec<[f64; D]>,
}

/// FNV-1a hash of the storage order and the group sizes: spaces built from
/// the same mesh have the same one.
fn storage_fingerprint(mesh_elem: &[usize], ranges: &GroupRanges) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let ends = GeoKind::ALL.map(|kind| ranges.of(kind).end);
    for v in mesh_elem.iter().chain(&ends) {
        for byte in v.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

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
    pub(super) mesh: &'m M,
    pub(super) numbering: Numbering,
    pub(super) node_storage: GeoStorage,
    pub(super) quads: Vec<(TypeId, &'static str, GeoStorage, EntryBuilder<D, S, M>)>,
    pub(super) _shape: PhantomData<S>,
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
        self.quads.push((
            TypeId::of::<Q>(),
            type_name::<Q>(),
            storage,
            Space::<D, S>::quad_entry::<Q, M>,
        ));
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
    /// Q1 and Q2 geometric elements of dimension `D` with their tensor permutations.
    fn geometric_elements() -> [(&'static GeometricElement, Vec<usize>); 2] {
        let (q1, q2) = match D {
            2 => (
                &anamesh::geometric_elements::QUADRANGLE4,
                &anamesh::geometric_elements::QUADRANGLE9,
            ),
            _ => (
                &anamesh::geometric_elements::HEXAHEDRON8,
                &anamesh::geometric_elements::HEXAHEDRON27,
            ),
        };
        [
            (q1, tensor_permutation::<D>(q1, 2)),
            (q2, tensor_permutation::<D>(q2, 3)),
        ]
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

    /// Construction (see [`SpaceBuilder::build`]): classify the volume
    /// elements, store them grouped by geometric kind, then build the
    /// numbering and the geometry at the nodes.
    fn construct<M: Mesh>(mesh: &M, numbering: Numbering, node_storage: GeoStorage) -> Self {
        Self::check_mesh(mesh);
        let Grouped {
            mesh_elem,
            ranges,
            affine,
            q1_nodes,
            q2_nodes,
        } = Self::group_by_kind(Self::classify_elements(mesh));
        let labels = mesh_elem
            .iter()
            .map(|&m| mesh.get_element(m).label)
            .collect();
        let shared = match numbering {
            Numbering::None => None,
            Numbering::Continuous => Some(Self::continuous_numbering(mesh, &mesh_elem)),
        };
        let tables = S::tables();
        let x1: Vec<Vec<f64>> = (0..D).map(|d| S::points_1d(&tables, d).to_vec()).collect();
        let node_points = Self::tensor_points(S::N, &x1);
        let mut space = Space {
            fingerprint: storage_fingerprint(&mesh_elem, &ranges),
            node_weights: Self::tensor_weights(&tables),
            node_sweep: <S::Nodes as Sweep<D, S>>::tables(),
            tables,
            nodes: PointGeo {
                ref_points: Vec::new(),
                q1: None,
                q2: None,
                mapped: None,
            },
            quads: Vec::new(),
            n_elem: mesh_elem.len(),
            ranges,
            mesh_elem,
            labels,
            affine,
            q1_nodes,
            q2_nodes,
            shared,
        };
        space.nodes = space.point_geo(mesh, node_points, node_storage);
        space
    }

    /// Panics unless `D` is 2 or 3, the mesh has dimension `D` and the shape
    /// has at least 2 points per direction.
    fn check_mesh<M: Mesh>(mesh: &M) {
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
    }

    /// The geometric map of every volume element, in mesh order.
    fn classify_elements<M: Mesh>(mesh: &M) -> Vec<(usize, ElementMap<D>)> {
        let volume = mesh.get_elements_by_dimension(D);
        assert!(!volume.is_empty(), "the mesh has no volume elements");
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
        volume
            .iter()
            .map(|&m| {
                let map = if mesh.is_mapped_element(m) {
                    Self::classify_mapped(mesh, m, &q2_ref, &q1_at_q2)
                } else {
                    Self::classify_nodal(mesh, m, [&q1_perm, &q2_perm], &q1_at_q2)
                };
                (m, map)
            })
            .collect()
    }

    /// An element with the mesh's own map: affine if that map is (tested on
    /// the images of the `3^D` reference nodes `q2_ref`), else mapped.
    fn classify_mapped<M: Mesh>(
        mesh: &M,
        m: usize,
        q2_ref: &[[f64; 3]],
        q1_at_q2: &GeoTab<D>,
    ) -> ElementMap<D> {
        assert_eq!(
            mesh.get_element(m)
                .geometric_element
                .reference_element
                .get_dim(),
            D,
            "element {m} is not a volume element"
        );
        let images: Vec<[f64; D]> = q2_ref
            .iter()
            .map(|uvw| {
                let mut x = [0.0; 3];
                mesh.get_element_def(m, uvw, &mut x);
                core::array::from_fn(|a| x[a])
            })
            .collect();
        match classify(images, 3, q1_at_q2) {
            ElementMap::Affine(x0, j) => ElementMap::Affine(x0, j),
            _ => ElementMap::Mapped,
        }
    }

    /// A QUADRANGLE4/9 or HEXAHEDRON8/27 element: affine, Q1 or Q2 from its
    /// geometric nodes (`perms`: the Q1 and Q2 tensor permutations).
    fn classify_nodal<M: Mesh>(
        mesh: &M,
        m: usize,
        perms: [&[usize]; 2],
        q1_at_q2: &GeoTab<D>,
    ) -> ElementMap<D> {
        let el = mesh.get_element(m);
        let gmsh_type = el.geometric_element.msh_id;
        let (n1, perm) = match (D, gmsh_type) {
            (2, 3) | (3, 5) => (2, perms[0]),
            (2, 10) | (3, 12) => (3, perms[1]),
            _ => panic!(
                "element {m} (gmsh type {gmsh_type}) is not a Q1/Q2 {} element",
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
        classify(tensor_nodes, n1, q1_at_q2)
    }

    /// The storage order: the elements grouped by kind, in the order
    /// [`GeoKind::ALL`], and in mesh order within a kind.
    fn group_by_kind(maps: Vec<(usize, ElementMap<D>)>) -> Grouped<D> {
        let mut g = Grouped {
            mesh_elem: Vec::with_capacity(maps.len()),
            ranges: GroupRanges::default(),
            affine: Vec::new(),
            q1_nodes: Vec::new(),
            q2_nodes: Vec::new(),
        };
        for kind in GeoKind::ALL {
            let start = g.mesh_elem.len();
            for (m, map) in &maps {
                match (kind, map) {
                    (GeoKind::Affine, ElementMap::Affine(x0, j)) => g.affine.push((*x0, *j)),
                    (GeoKind::Q1, ElementMap::Q1(n)) => g.q1_nodes.extend_from_slice(n),
                    (GeoKind::Q2, ElementMap::Q2(n)) => g.q2_nodes.extend_from_slice(n),
                    (GeoKind::Mapped, ElementMap::Mapped) => {}
                    _ => continue,
                }
                g.mesh_elem.push(*m);
            }
            *g.ranges.of_mut(kind) = start..g.mesh_elem.len();
        }
        g
    }

    /// Tensor GLL weights of the nodes, `w_i · w_j (· w_k)`.
    fn tensor_weights(tables: &S::Tables) -> Vec<f64> {
        (0..S::NLOC)
            .map(|n| {
                Self::multi_index(n)
                    .iter()
                    .enumerate()
                    .map(|(d, &i)| S::weights_1d(tables, d)[i])
                    .product()
            })
            .collect()
    }

    /// Geometry at the reference points `ref_points` of every element: tables
    /// or stored values for the Q1/Q2 groups (`storage`), stored values for
    /// the mapped group (from the mesh's own map).
    fn point_geo<M: Mesh>(
        &self,
        mesh: &M,
        ref_points: Vec<[f64; 3]>,
        storage: GeoStorage,
    ) -> PointGeo<D> {
        let np = ref_points.len();
        let [(q1_ge, q1_perm), (q2_ge, q2_perm)] = Self::geometric_elements();
        // (x, J) at point p of storage element e → stored (x, det J, cof J).
        let store =
            |range: &Range<usize>,
             f: &dyn Fn(usize, usize) -> (SVector<f64, D>, SMatrix<f64, D, D>)| {
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
        // A Q1/Q2 group: its shape functions tabulated at the points, kept
        // as tables (Recompute) or evaluated once per element and stored.
        let group = |kind: GeoKind, ge: &GeometricElement, perm: &[usize]| {
            let range = self.ranges.of(kind);
            if range.is_empty() {
                return None;
            }
            let tab = GeoTab::new(ge, perm, &ref_points);
            Some(match storage {
                GeoStorage::Recompute => GroupGeo::Tabulated(tab),
                GeoStorage::Store => GroupGeo::Stored(store(range, &|e, p| {
                    tab.eval(self.geometric_nodes(kind, e), p)
                })),
            })
        };
        let q1 = group(GeoKind::Q1, q1_ge, &q1_perm);
        let q2 = group(GeoKind::Q2, q2_ge, &q2_perm);
        // Mapped group: always stored, from the mesh's own map.
        let mapped = (!self.ranges.mapped.is_empty()).then(|| {
            store(&self.ranges.mapped, &|e, p| {
                let m = self.mesh_elem[e];
                let (mut x, mut g) = ([0.0; 3], [0.0; 9]);
                mesh.get_element_def(m, &ref_points[p], &mut x);
                mesh.get_element_grad_def(m, &ref_points[p], &mut g);
                (
                    SVector::<f64, D>::from_fn(|a, _| x[a]),
                    SMatrix::<f64, D, D>::from_fn(|a, b| g[a * D + b]),
                )
            })
        });
        PointGeo {
            ref_points,
            q1,
            q2,
            mapped,
        }
    }

    /// Entry of the quadrature `Q` (points and weights from `lobatto::tables`).
    fn quad_entry<Q: Sweep<D, S>, M: Mesh>(&self, mesh: &M, storage: GeoStorage) -> QuadEntry<D> {
        let families = Q::families();
        let (x1, w1): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (0..D)
            .map(|d| points_weights_dyn(Q::M[d], families[d]))
            .unzip();
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
}
