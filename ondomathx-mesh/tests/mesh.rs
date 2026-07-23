extern crate anamesh;

#[cfg(test)]
mod tests {
    use anamesh::mesh::square::*;
    use anamesh::mesh::Mesh;

    #[test]
    fn coloring_algorithm() {
        let mut mesh = UnstructuredMesh::new_from_msh("./assets/Potatoe.msh").unwrap();

        // Precompute the color of each volume element, then relabel through
        // set_labels (boundary elements keep their label).
        let vec_elem = mesh.get_elements_by_dimension(2);
        let mut labels: Vec<Option<usize>> = vec![None; mesh.get_nelements()];
        for &e in &vec_elem {
            labels[e] = Some(mesh.get_element_color(e));
        }
        mesh.set_labels(|el, elem| labels[el].unwrap_or(elem.label));

        mesh.write_msh("./tests/results/Potatoe_with_colors.msh")
            .unwrap();
    }

    #[test]
    fn mesh_write_and_read_roundtrip() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 3, None, None);
        let mesh1 = square.get_unstructured_mesh();
        mesh1
            .write_msh("./tests/results/roundtrip_test.msh")
            .unwrap();

        let mesh2 = UnstructuredMesh::new_from_msh("./tests/results/roundtrip_test.msh").unwrap();

        assert_eq!(mesh1.get_nnodes(), mesh2.get_nnodes());
        assert_eq!(mesh1.get_nelements(), mesh2.get_nelements());
    }
}
