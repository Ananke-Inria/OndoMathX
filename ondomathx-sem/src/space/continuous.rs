//! The continuous numbering: its construction (an
//! `ondomathx_toolbox::numbering::DgMap`), the `c_` functions and the DG ↔
//! global transfers that use it (except `inverse_mass`, in `mass`).

use ondomathx_mesh::mesh::Mesh;
use ondomathx_toolbox::numbering::DgMap;

use crate::element::shape::Shape;

use super::Space;

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// The continuous numbering (`Mesh::get_tensor_numbering`, local index in
    /// tensor order) as a `DgMap` over the storage elements `mesh_elem`.
    pub(super) fn continuous_numbering<M: Mesh>(mesh: &M, mesh_elem: &[usize]) -> DgMap {
        let (numbering, n_global) = mesh.get_tensor_numbering(S::N);
        for &m in mesh_elem {
            assert_eq!(numbering[m].len(), Self::NLOC, "numbering of element {m}");
        }
        DgMap::from_numbering(&numbering, n_global, mesh_elem)
    }

    /// The continuous numbering; panics if the space has none.
    fn dg_map(&self) -> &DgMap {
        self.dg_map
            .as_ref()
            .expect("this function needs the continuous numbering: build the space with Numbering::Continuous")
    }

    /// Number of global DOFs of the continuous numbering (distinct nodes).
    pub fn num_dofs(&self) -> usize {
        self.dg_map().num_dofs()
    }

    /// Shared-DOF sum: every copy of a shared node receives the sum over its
    /// copies, component by component. Interior nodes are left unchanged.
    ///
    /// All copies receive bitwise-identical values (same summation order).
    /// `v` has `num_elements · C · NLOC` entries.
    pub fn sum_shared<const C: usize>(&self, v: &mut [f64]) {
        assert_eq!(v.len(), self.num_dg_dofs() * C, "vector length");
        self.dg_map().sum_shared::<C>(v);
    }

    /// DG → global (the assembly): `cg[g·C + c]` is the sum of the copies of
    /// global DoF `g`, component `c` (`cg` has `num_dofs · C` entries,
    /// `[dof][comp]`).
    pub fn collect_dg<const C: usize>(&self, dg: &[f64], cg: &mut [f64]) {
        self.dg_map().collect_dg::<C>(dg, cg);
    }

    /// DG → global: `cg[g·C + c]` is the average of the copies of `g`.
    pub fn collect_avg_dg<const C: usize>(&self, dg: &[f64], cg: &mut [f64]) {
        self.dg_map().collect_avg_dg::<C>(dg, cg);
    }

    /// Global → DG: every copy of `g` receives `cg[g·C + c]`.
    pub fn scatter_cg<const C: usize>(&self, cg: &[f64], dg: &mut [f64]) {
        self.dg_map().scatter_cg::<C>(cg, dg);
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
    pub fn boundary_mask<M: Mesh>(&self, mesh: &M, labels: Option<&[usize]>) -> Vec<f64> {
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
        let mut on = vec![0.0; self.num_dg_dofs()];
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
        self.sum_shared::<1>(&mut on);
        on.iter()
            .map(|&v| if v > 0.0 { 0.0 } else { 1.0 })
            .collect()
    }

    /// Number of copies of global DoF `i_glob` (the elements sharing it).
    pub fn multiplicity(&self, i_glob: usize) -> usize {
        self.dg_map().multiplicity(i_glob)
    }

    /// Number of copies of each duplicated node in the continuous numbering
    /// (`1` for interior nodes).
    pub fn multiplicity_dg(&self) -> Vec<f64> {
        let mut k = vec![1.0; self.num_dg_dofs()];
        self.sum_shared::<1>(&mut k);
        k
    }
}
