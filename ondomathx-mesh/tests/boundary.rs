extern crate anamesh;

//Validation of UnstructuredMesh::get_parent_face and get_boundary_faces:
//the parent volume element of a boundary element and the local facet (edge
//in 2D, face in 3D) it occupies, and the topological boundary.
#[cfg(test)]
mod tests {
    use anamesh::mesh::cube::Cube;
    use anamesh::mesh::square::Square;
    use anamesh::mesh::{Mesh, UnstructuredMesh};

    //The boundary element and the parent's local facet are the same edge
    //(2D) or face (3D): Edge/Face compare by their sorted vertex indices.
    fn check_parents(mesh: &UnstructuredMesh) {
        let dim = mesh.get_topologic_dimension();
        let boundary = mesh.get_elements_by_dimension(dim - 1);
        assert!(!boundary.is_empty());
        for b in boundary {
            let (vol, f) = mesh.get_parent_face(b);
            let (pe, be) = (mesh.get_element(vol), mesh.get_element(b));
            assert_eq!(pe.geometric_element.reference_element.get_dim(), dim);
            if dim == 2 {
                assert!(pe.get_edge(f) == be.get_edge(0), "boundary edge {b}");
            } else {
                assert!(pe.get_face(f) == be.get_face(0), "boundary face {b}");
            }
        }
    }

    #[test]
    fn parent_faces_square_and_cube() {
        for q2 in [None, Some(true)] {
            check_parents(&Square::new([0.0, 0.0], 2.0, 1.0, 3, 2, None, q2).get_unstructured_mesh());
            check_parents(&Cube::new([0.0; 3], 1.0, 2.0, 1.0, 2, 3, 2, None, q2).get_unstructured_mesh());
        }
    }

    #[test]
    fn boundary_faces_are_the_labelled_boundary() {
        //Square 3×2: 2·(3 + 2) = 10 boundary edges; Cube 2×3×2: 2·(6 + 4 + 6) = 32 faces.
        let square = Square::new([0.0, 0.0], 2.0, 1.0, 3, 2, None, None).get_unstructured_mesh();
        let cube = Cube::new([0.0; 3], 1.0, 2.0, 1.0, 2, 3, 2, None, None).get_unstructured_mesh();
        for (mesh, n) in [(&square, 10), (&cube, 32)] {
            let dim = mesh.get_topologic_dimension();
            let mut topo = mesh.get_boundary_faces();
            let mut labelled: Vec<(usize, usize)> = mesh
                .get_elements_by_dimension(dim - 1)
                .into_iter()
                .map(|b| mesh.get_parent_face(b))
                .collect();
            topo.sort();
            labelled.sort();
            assert_eq!(topo.len(), n);
            assert_eq!(topo, labelled);
        }
    }

    #[test]
    #[should_panic(expected = "is not of dimension")]
    fn parent_face_of_a_volume_element_panics() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None, None).get_unstructured_mesh();
        let vol = square.get_elements_by_dimension(2)[0];
        let _ = square.get_parent_face(vol);
    }
}
