# TODO

## P2 / Higher-order finite elements

- [ ] **Triangle6** — P2 Lagrange triangle (6 nodes: 3 vertices + 3 edge midpoints)
- [ ] **Tetrahedron10** — P2 Lagrange tetrahedron (10 nodes: 4 vertices + 6 edge midpoints)
- [ ] **Quadrangle9** —  full Q2 quadrilateral
- [ ] **Hexahedron27** —  full Q2 hexahedron
- [ ] Wire up P2 elements in `unstructured_mesh.rs` section (currently only P1 elements are parsed: Triangle3, Quadrangle4, Tetrahedron4, Hexahedron8)
- [ ] Add `.mesh` format support for P2 elements (Gmsh type IDs 9, 11, 16, 17, …)

## Parametrized meshes

- [ ] **deformation_fields** — define a clear strategy to use the deformation fields

- [ ] **Circle** — structured mesh of a 2D disk (center, radius, refinement)
    The idea could be to first create a mesh of a square that it adapted to the future deformation. Then to construct the mesh of the circle by applying the deformation 


- [ ] **Cylinder** — 3D cylinder mesh (center, radius, height, refinement)
- [ ] **Ball / Sphere** — 3D ball mesh (center, radius, refinement)
- [ ] **Triangle domain** — structured mesh of a 2D triangle
- [ ] **Annulus** — 2D ring mesh (inner radius, outer radius) ? or use deformation of square with periodicity
- [ ] **Cube** — 3D structured hexahedral mesh (extends `Square` to 3D)
- [ ] **Ventricle** — 3D unstructured hexahedral mesh of a ventricle
- [ ] **Square and cube** — mesh with tri and tet

 

 ## Subdomain and boundaries of mesh

 - [ ] **politic** — define a clear strategy to define the subdomains and the boundaries.
        Add get_parent_face (and get_parent_edge for 2D meshes) to UnstructuredMesh — this is the only structural gap.

    /// For a boundary element (dim = topologic_dimension - 1),
    /// returns (volume_element_index, local_face_index_in_that_element).
    pub fn get_parent_face(&self, boundary_el: usize) -> (usize, usize) {
        let i_face = self.neighbour_faces[boundary_el][0];
        let [vol_el, local_face] = self.faces_neighbour[i_face][0];
        (vol_el, local_face)
    }

## Renaming

 - [ ] **Change the name ?** - Anamesh ? (analytic mesh) because some meshes are described by analytic functions, Paramesh (parametric mesh)

 ## Algorithms

 - [ ] **Refine** refine mesh by refining every element uniformly
- [ ] **Split** split mesh of triangle or tet into quadrangle or hexhedral mesh
- [ ] **Renumbering** renumbers the dof to optimize vertex proximity

 ## Numbering

 - [ ] **Skeleton** - deal with the numbering of skeleton of meshes

