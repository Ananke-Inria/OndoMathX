//! The continuous numbering: its construction (shared-DOF groups) and the
//! `c_` functions that use it (except `c_inverse_mass`, in `mass`).

use anamesh::mesh::Mesh;
use rayon::prelude::*;

use crate::shape::Shape;

use super::{Shared, Space};

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// Shared-DOF groups of the continuous numbering
    /// (`Mesh::get_tensor_numbering`, local index in tensor order).
    pub(super) fn continuous_numbering<M: Mesh>(mesh: &M, mesh_elem: &[usize]) -> Shared {
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

    /// Number of global DOFs of the continuous numbering (distinct nodes).
    pub fn c_n_global(&self) -> usize {
        self.shared().n_global
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
            assert!(
                e != usize::MAX,
                "boundary facet of element {m}, not in the space"
            );
            let reference = um.get_element(m).geometric_element.reference_element;
            for (n, uvw) in self.nodes.ref_points.iter().enumerate() {
                let hit = if D == 2 {
                    reference.is_on_edge(uvw, f)
                } else {
                    reference.is_on_face(uvw, f)
                };
                if hit {
                    on[e * nloc + n] = 1.0;
                }
            }
        }
        // Copies of boundary nodes in elements touching the boundary only at
        // an edge or a vertex.
        self.c_sum_shared::<1>(&mut on);
        on.iter()
            .map(|&v| if v > 0.0 { 0.0 } else { 1.0 })
            .collect()
    }

    /// Number of copies of each duplicated node in the continuous numbering
    /// (`1` for interior nodes).
    pub fn c_multiplicity(&self) -> Vec<f64> {
        let mut k = vec![1.0; self.n_dup()];
        self.c_sum_shared::<1>(&mut k);
        k
    }
}
