extern crate rouage;

#[cfg(test)]
mod tests {
    use rouage::deformation_field::*;
    use rouage::mesh::square::*;
    use rouage::mesh::unstructured_mesh::*;
    use rouage::mesh::Mesh;

    #[test]
    fn square_mesh() {
        let square_mesh = Square::new([-0.5, -0.5], 1.0, 1.0, 10, 10, None);

        let mut mesh = square_mesh.get_unstructured_mesh();

        let deformation_field_x = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = 3.0 * xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        let deformation_field_y = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0];
                f_xyz[1] = 3.0 * xyz[1];
            },
            grad_diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        mesh.add_deformation(deformation_field_x);
        //mesh.add_deformation(deformation_field_y);

        mesh.write_msh("./tests/results/square_mesh.msh").unwrap();
    }

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
    fn reference_element_triangle() {
        use ::rouage::reference_elements::TRIANGLE;

        assert!(TRIANGLE.is_inside(&[0.0, 0.0, 0.0]));
        assert!(TRIANGLE.is_inside(&[1.0, 0.0, 0.0]));
        assert!(TRIANGLE.is_inside(&[0.0, 1.0, 0.0]));
        assert!(TRIANGLE.is_inside(&[0.0, 0.5, 0.0]));
        assert!(TRIANGLE.is_inside(&[0.5, 0.0, 0.0]));
        assert!(TRIANGLE.is_inside(&[0.5, 0.5, 0.0]));

        assert!(!TRIANGLE.is_inside(&[0.0, 0.0, 0.1]));
        assert!(!TRIANGLE.is_inside(&[1.0001, 0.0, 0.0]));
        assert!(!TRIANGLE.is_inside(&[0.0001, 1.0, 0.0]));
        assert!(!TRIANGLE.is_inside(&[-0.001, 0.5, 0.0]));
        assert!(!TRIANGLE.is_inside(&[0.5001, 0.5001, 0.0]));

        assert_eq!(
            TRIANGLE.uvw_edge_transform(&[0.0, 0.0, 0.0], 0, 1),
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            TRIANGLE.uvw_edge_transform(&[0.25, 0.75, 0.0], 1, 0),
            [0.25, 0.75, 0.0]
        );
        assert_eq!(
            TRIANGLE.uvw_edge_transform(&[0.25, 0.75, 0.0], 1, 1),
            [0.75, 0.25, 0.0]
        );
        assert_eq!(
            TRIANGLE.uvw_edge_transform(&[0.0, 0.9, 0.0], 2, 1),
            [0.0, 1.0 - 0.9, 0.0]
        );
    }

    #[test]
    fn reference_element_hexahedron() {
        use ::rouage::reference_elements::HEXAHEDRON;

        assert!(HEXAHEDRON.is_inside(&[0.0, 0.0, 0.5]));
        assert!(HEXAHEDRON.is_inside(&[1.0, 1.0, 1.0]));
        assert!(HEXAHEDRON.is_inside(&[0.0, 1.0, 0.0]));
        assert!(HEXAHEDRON.is_inside(&[0.0, 0.5, 0.0]));
        assert!(HEXAHEDRON.is_inside(&[0.5, 0.0, 1.0]));
        assert!(HEXAHEDRON.is_inside(&[0.5, 0.5, 0.0]));

        assert!(!HEXAHEDRON.is_inside(&[0.0, 0.0, -0.1]));
        assert!(!HEXAHEDRON.is_inside(&[1.0001, 0.0, 0.0]));
        assert!(!HEXAHEDRON.is_inside(&[0.0001, 1.0001, 0.0]));
        assert!(!HEXAHEDRON.is_inside(&[-0.001, 0.5, 0.0]));
        assert!(!HEXAHEDRON.is_inside(&[0.5001, 0.5001, 1.0001]));

        assert_eq!(
            HEXAHEDRON.uvw_edge_transform(&[0.0, 0.0, 0.0], 0, 1),
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            HEXAHEDRON.uvw_edge_transform(&[0.0, 0.9, 0.0], 3, 1),
            [0.0, 1.0 - 0.9, 0.0]
        );

        assert_eq!(
            HEXAHEDRON.uvw_face_transform(&[0.3, 0.4, 0.0], 0, 0),
            [0.3, 0.4, 0.0]
        );
        assert_eq!(
            HEXAHEDRON.uvw_face_transform(&[0.25, 0.25, 0.0], 0, 1),
            [0.75, 0.25, 0.0]
        );
        assert_eq!(
            HEXAHEDRON.uvw_face_transform(&[0.25, 0.25, 0.0], 0, 5),
            [0.25, 0.75, 0.0]
        );
    }

    #[test]
    fn reference_element_tetrahedron() {
        use ::rouage::reference_elements::TETRAHEDRON;

        assert_eq!(
            TETRAHEDRON.uvw_face_transform(&[0.5, 0.5, 0.0], 0, 0),
            [0.5, 0.5, 0.0]
        );
        assert_eq!(
            TETRAHEDRON.uvw_face_transform(&[0.5, 0.5, 0.0], 0, 1),
            [0.0, 0.5, 0.0]
        );
        assert_eq!(
            TETRAHEDRON.uvw_face_transform(&[0.5, 0.5, 0.0], 0, 2),
            [0.5, 0.0, 0.0]
        );
        assert_eq!(
            TETRAHEDRON.uvw_face_transform(&[0.5, 0.5, 0.0], 0, 3),
            [0.5, 0.5, 0.0]
        );
        assert_eq!(
            TETRAHEDRON.uvw_face_transform(&[0.5, 0.5, 0.0], 0, 4),
            [0.5, 0.0, 0.0]
        );
        assert_eq!(
            TETRAHEDRON.uvw_face_transform(&[0.5, 0.5, 0.0], 0, 5),
            [0.0, 0.5, 0.0]
        );

        // assert!(crate::vec_eq!(
        //     TETRAHEDRON.uvw_face_transform(&[0.2, 0.2, 0.6], 3, 0),
        //     [0.2, 0.2, 0.6]
        // ));

        // assert!(crate::vec_eq!(
        //     TETRAHEDRON.uvw_face_transform(&[0.2, 0.2, 0.6], 3, 1),
        //     [0.6, 0.2, 0.2]
        // ));
    }
}
