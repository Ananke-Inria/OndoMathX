# TODO

## P2 / Higher-order finite elements

- [x] **Triangle6** — P2 Lagrange triangle (6 nodes: 3 vertices + 3 edge midpoints)
- [x] **Tetrahedron10** — P2 Lagrange tetrahedron (10 nodes: 4 vertices + 6 edge midpoints)
- [x] **Hexahedron27** —  full Q2 hexahedron
- [x] Wire up P2 elements in `unstructured_mesh.rs` (reading: Gmsh types 8, 9, 10, 11, 12 with Gmsh→internal node permutations)
- [x] Add `.mesh` format support for P2 elements (EdgesP2, TrianglesP2, QuadrilateralsQ2, TetrahedraP2, HexahedraQ2)
- [x] `write_msh`: apply the internal→Gmsh node permutation for Tetrahedron10/Hexahedron27
- [x] Add a constructor in Circle and Square that outputs Q2 mesh when using write_msh or get_unstructured_mesh (optional `q2` flag on both constructors)


## Parametrized meshes

- [x] **deformation_fields** — define a clear strategy to use the deformation fields

- [x] **Circle** — structured mesh of a 2D disk (center, radius, refinement)
    The idea could be to first create a mesh of a square that it adapted to the future deformation. Then to construct the mesh of the circle by applying the deformation 


- [ ] **Cylinder** — 3D cylinder mesh (center, radius, height, refinement)
- [ ] **Ball / Sphere** — 3D ball mesh (center, radius, refinement)
- [ ] **Triangle domain** — structured mesh of a 2D triangle
- [ ] **Annulus** — 2D ring mesh (inner radius, outer radius)  
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

 - [x] **Change the name** - Anamesh (analytic mesh) because some meshes are described by analytic functions

 ## Algorithms

 - [ ] **Refine** refine mesh by refining every element uniformly
- [ ] **Split** split mesh of triangle or tet into quadrangle or hexhedral mesh
- [ ] **Renumbering** renumbers the dof to optimize vertex proximity

 ## Numbering

 - [ ] **Skeleton** - deal with the numbering of skeleton of meshes

