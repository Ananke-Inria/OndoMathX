extern crate rouage;

#[cfg(test)]
mod tests {
    use rouage::deformation_field::*;
    use rouage::geometric_elements::*;
    use rouage::mesh::square::*;
    use rouage::mesh::Mesh;

    // === Construction ===

    #[test]
    fn construction() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 3, None);
        assert_eq!(square.nx, 3);
        assert_eq!(square.ny, 3);
        assert!((square.lx - 1.0).abs() < 1e-12);
        assert!((square.ly - 1.0).abs() < 1e-12);
    }

    #[test]
    fn construction_with_origin() {
        let square = Square::new([-1.0, 2.0], 3.0, 4.0, 2, 2, None);
        let n0 = square.get_node(0);
        assert!((n0[0] - (-1.0)).abs() < 1e-12);
        assert!((n0[1] - 2.0).abs() < 1e-12);
    }
    // === Dimensions ===

    #[test]
    fn geometric_and_topologic_dimension() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None);
        assert_eq!(square.get_geometric_dimension(), 2);
        assert_eq!(square.get_topologic_dimension(), 2);
    }

    // === Nodes ===

    #[test]
    fn nnodes() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 5, 3, None);
        assert_eq!(square.get_nnodes(), 6 * 4); // (nx+1)*(ny+1)
    }

    #[test]
    fn node_positions_unit_square() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None);
        // 3x3 grid of nodes
        let expected = [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ];
        for (i, exp) in expected.iter().enumerate() {
            let n = square.get_node(i);
            assert!(
                (n[0] - exp[0]).abs() < 1e-12 && (n[1] - exp[1]).abs() < 1e-12,
                "Node {}: expected ({}, {}), got ({}, {})",
                i,
                exp[0],
                exp[1],
                n[0],
                n[1]
            );
            assert!((n[2] - 0.0).abs() < 1e-12); // z = 0
        }
    }

    #[test]
    fn node_positions_with_origin_and_size() {
        let square = Square::new([1.0, 2.0], 4.0, 6.0, 2, 3, None);
        // First node = origin
        let n0 = square.get_node(0);
        assert!((n0[0] - 1.0).abs() < 1e-12);
        assert!((n0[1] - 2.0).abs() < 1e-12);

        // Last node = origin + (lx, ly)
        let last = square.get_node(square.get_nnodes() - 1);
        assert!((last[0] - 5.0).abs() < 1e-12); // 1.0 + 4.0
        assert!((last[1] - 8.0).abs() < 1e-12); // 2.0 + 6.0
    }

    // === Elements ===

    #[test]
    fn nelements() {
        let nx = 4;
        let ny = 3;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None);
        // nx*ny quads + 2*nx boundary bottom/top + 2*ny boundary left/right
        assert_eq!(square.get_nelements(), nx * ny + 2 * nx + 2 * ny);
    }

    #[test]
    fn interior_elements_are_quadrangles() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None);
        for i in 0..(nx * ny) {
            let elem = square.get_element(i);
            assert_eq!(elem.geometric_element.msh_id, QUADRANGLE4.msh_id);
            assert_eq!(elem.indexes.len(), 4);
            assert_eq!(elem.label, 0);
        }
    }

    #[test]
    fn boundary_elements_are_edges() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None);
        for i in (nx * ny)..square.get_nelements() {
            let elem = square.get_element(i);
            assert_eq!(elem.geometric_element.msh_id, EDGE2.msh_id);
            assert_eq!(elem.indexes.len(), 2);
        }
    }

    #[test]
    fn boundary_labels() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None);
        let offset = nx * ny;

        // Bottom edges: label 0
        for i in 0..nx {
            assert_eq!(square.get_element(offset + i).label, 0);
        }
        // Right edges: label 1
        for i in 0..ny {
            assert_eq!(square.get_element(offset + nx + i).label, 1);
        }
        // Top edges: label 2
        for i in 0..nx {
            assert_eq!(square.get_element(offset + nx + ny + i).label, 2);
        }
        // Left edges: label 3
        for i in 0..ny {
            assert_eq!(square.get_element(offset + 2 * nx + ny + i).label, 3);
        }
    }

    #[test]
    fn boundary_edge_count() {
        let nx = 4;
        let ny = 3;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None);
        let n_boundary = square.get_nelements() - nx * ny;
        assert_eq!(n_boundary, 2 * nx + 2 * ny);
    }

    #[test]
    fn quad_element_node_indices_valid() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None);
        let nnodes = square.get_nnodes();
        for i in 0..(nx * ny) {
            let elem = square.get_element(i);
            for &idx in &elem.indexes {
                assert!(
                    idx < nnodes,
                    "Node index {} out of bounds for element {}",
                    idx,
                    i
                );
            }
        }
    }

    #[test]
    fn first_quad_element() {
        // Element 0 at (ix=0, iy=0), should reference corners of the first cell
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 2, None);
        let elem = square.get_element(0);
        // nx+1 = 4 nodes per row
        // Nodes: bottom-left=0, bottom-right=1, top-right=5, top-left=4
        assert_eq!(elem.indexes, vec![0, 1, 5, 4]);
    }

    // === Coloring ===

    #[test]
    fn ncolors() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 4, 4, None);
        assert_eq!(square.get_ncolors(), 4);
    }

    #[test]
    fn coloring_covers_all_quads() {
        for &(nx, ny) in &[(3, 3), (4, 4), (5, 3), (2, 7)] {
            let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None);
            let total: usize = (0..4).map(|c| square.get_nelements_by_color(c)).sum();
            assert_eq!(
                total,
                nx * ny,
                "Color partition mismatch for {}x{}: total={}, expected={}",
                nx,
                ny,
                total,
                nx * ny
            );
        }
    }

    #[test]
    fn coloring_is_checkerboard() {
        let nx = 4;
        let ny = 4;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None);
        for i in 0..(nx * ny) {
            let ix = i % nx;
            let iy = i / nx;
            let color = square.get_element_color(i);
            let expected = (iy % 2) + 2 * (ix % 2);
            assert_eq!(
                color, expected,
                "Element ({}, {}): color={}, expected={}",
                ix, iy, color, expected
            );
        }
    }

    // === Deformation ===

    #[test]
    fn no_deformation_by_default() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None);
        assert!(square.get_deformation().is_none());
    }

    #[test]
    fn with_deformation() {
        let def = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], _f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        let square = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, Some(def));
        assert!(square.get_deformation().is_some());
    }

    #[test]
    fn write_msh() {
        let square = Square::new([-0.5, -0.5], 1.0, 1.0, 10, 10, None);
        let mut mesh = square.get_unstructured_mesh();

        let def = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = 3.0 * xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], _f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        mesh.add_deformation(def);
        mesh.write_msh("./tests/results/square_mesh.msh").unwrap();
    }

    // === Conversion to unstructured ===

    #[test]
    fn to_unstructured_preserves_counts() {
        let nx = 4;
        let ny = 3;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None);
        let mesh = square.get_unstructured_mesh();
        assert_eq!(mesh.get_nnodes(), square.get_nnodes());
        assert_eq!(mesh.get_nelements(), square.get_nelements());
    }

    // === 1x1 minimal mesh ===

    #[test]
    fn minimal_1x1_mesh() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 1, 1, None);
        assert_eq!(square.get_nnodes(), 4);
        assert_eq!(square.get_nelements(), 5); // 1 quad + 4 boundary edges

        // The single quad
        let elem = square.get_element(0);
        assert_eq!(elem.geometric_element.msh_id, QUADRANGLE4.msh_id);
        assert_eq!(elem.indexes, vec![0, 1, 3, 2]);

        // 4 boundary edges
        for i in 1..5 {
            let elem = square.get_element(i);
            assert_eq!(elem.geometric_element.msh_id, EDGE2.msh_id);
        }
    }
}
