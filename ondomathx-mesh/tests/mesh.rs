extern crate rouage;

#[cfg(test)]
mod tests {
    use rouage::mesh::square::*;
    use rouage::mesh::Mesh;

    #[test]
    fn coloring_algorithm() {
        let mut mesh = UnstructuredMesh::new_from_msh("./assets/Potatoe.msh").unwrap();

        let vec_elem = mesh.get_elements_by_dimension(2);
        for i in 0..vec_elem.len() {
            let label = mesh.get_element_color(i);
            let elem = mesh.get_element_as_mut(vec_elem[i]);
            elem.label = label;
        }

        mesh.write_msh("./tests/results/Potatoe_with_colors.msh")
            .unwrap();
    }

    #[test]
    fn mesh_write_and_read_roundtrip() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 3, None);
        let mesh1 = square.get_unstructured_mesh();
        mesh1
            .write_msh("./tests/results/roundtrip_test.msh")
            .unwrap();

        let mesh2 = UnstructuredMesh::new_from_msh("./tests/results/roundtrip_test.msh").unwrap();

        assert_eq!(mesh1.get_nnodes(), mesh2.get_nnodes());
        assert_eq!(mesh1.get_nelements(), mesh2.get_nelements());
    }
}
