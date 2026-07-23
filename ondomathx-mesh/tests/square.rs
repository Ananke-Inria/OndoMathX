extern crate anamesh;

#[cfg(test)]
mod tests {
    use anamesh::diffeomorphism::*;
    use anamesh::geometric_elements::*;
    use anamesh::mesh::square::*;
    use anamesh::mesh::Mesh;

    // === Construction ===

    #[test]
    fn construction() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 3, None, None);
        assert_eq!(square.nx, 3);
        assert_eq!(square.ny, 3);
        assert!((square.lx - 1.0).abs() < 1e-12);
        assert!((square.ly - 1.0).abs() < 1e-12);
    }

    #[test]
    fn construction_with_origin() {
        let square = Square::new([-1.0, 2.0], 3.0, 4.0, 2, 2, None, None);
        let n0 = square.get_node(0);
        assert!((n0[0] - (-1.0)).abs() < 1e-12);
        assert!((n0[1] - 2.0).abs() < 1e-12);
    }
    // === Dimensions ===

    #[test]
    fn geometric_and_topologic_dimension() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None, None);
        assert_eq!(square.get_geometric_dimension(), 2);
        assert_eq!(square.get_topologic_dimension(), 2);
    }

    // === Nodes ===

    #[test]
    fn nnodes() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 5, 3, None, None);
        assert_eq!(square.get_nnodes(), 6 * 4); // (nx+1)*(ny+1)
    }

    #[test]
    fn node_positions_unit_square() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None, None);
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
        let square = Square::new([1.0, 2.0], 4.0, 6.0, 2, 3, None, None);
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
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        // nx*ny quads + 2*nx boundary bottom/top + 2*ny boundary left/right
        assert_eq!(square.get_nelements(), nx * ny + 2 * nx + 2 * ny);
    }

    #[test]
    fn interior_elements_are_quadrangles() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
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
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
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
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
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
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let n_boundary = square.get_nelements() - nx * ny;
        assert_eq!(n_boundary, 2 * nx + 2 * ny);
    }

    #[test]
    fn quad_element_node_indices_valid() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
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
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 2, None, None);
        let elem = square.get_element(0);
        // nx+1 = 4 nodes per row
        // Nodes: bottom-left=0, bottom-right=1, top-right=5, top-left=4
        assert_eq!(elem.indexes, vec![0, 1, 5, 4]);
    }

    // === Coloring ===

    #[test]
    fn ncolors() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 4, 4, None, None);
        assert_eq!(square.get_ncolors(), 4);
    }

    #[test]
    fn coloring_covers_all_quads() {
        for &(nx, ny) in &[(3, 3), (4, 4), (5, 3), (2, 7)] {
            let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
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
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
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
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None, None);
        assert!(square.get_deformation().is_none());
    }

    #[test]
    fn with_deformation() {
        let def = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_p, _xyz, _f_xyz| {},
            inner_deformation: None,
        };

        let square = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, Some(def), None);
        assert!(square.get_deformation().is_some());
    }

    #[test]
    fn write_msh() {
        let square = Square::new([-0.5, -0.5], 1.0, 1.0, 10, 10, None, None);
        let mut mesh = square.get_unstructured_mesh();

        let def = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = 3.0 * xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_p, _xyz, _f_xyz| {},
            inner_deformation: None,
        };

        mesh.compose_deformation(def);
        mesh.write_msh("./tests/results/square_mesh.msh").unwrap();
    }

    // === Conversion to unstructured ===

    #[test]
    fn to_unstructured_preserves_counts() {
        let nx = 4;
        let ny = 3;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let mesh = square.get_unstructured_mesh();
        assert_eq!(mesh.get_nnodes(), square.get_nnodes());
        assert_eq!(mesh.get_nelements(), square.get_nelements());
    }

    // === get_elements_by_dimension ===

    #[test]
    fn elements_by_dim2_returns_all_quads() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let elems = square.get_elements_by_dimension(2);
        assert_eq!(elems.len(), nx * ny);
        assert_eq!(elems, (0..nx * ny).collect::<Vec<_>>());
    }

    #[test]
    fn elements_by_dim1_returns_all_boundary_edges() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let elems = square.get_elements_by_dimension(1);
        assert_eq!(elems.len(), 2 * nx + 2 * ny);
        assert_eq!(
            elems,
            (nx * ny..nx * ny + 2 * nx + 2 * ny).collect::<Vec<_>>()
        );
    }

    #[test]
    fn elements_by_dim_other_returns_empty() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 2, None, None);
        assert!(square.get_elements_by_dimension(0).is_empty());
        assert!(square.get_elements_by_dimension(3).is_empty());
    }

    #[test]
    fn elements_by_dim_have_correct_type() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        for i in square.get_elements_by_dimension(2) {
            assert_eq!(
                square.get_element(i).geometric_element.msh_id,
                QUADRANGLE4.msh_id,
                "Element {} should be a quadrangle",
                i
            );
        }
        for i in square.get_elements_by_dimension(1) {
            assert_eq!(
                square.get_element(i).geometric_element.msh_id,
                EDGE2.msh_id,
                "Element {} should be an edge",
                i
            );
        }
    }

    // === get_elements_by_dimension_and_label ===

    #[test]
    fn elements_by_dim_and_label_quads() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let elems = square.get_elements_by_dimension_and_label(0, 2);
        assert_eq!(elems.len(), nx * ny);
        assert_eq!(elems, (0..nx * ny).collect::<Vec<_>>());
        for i in &elems {
            assert_eq!(square.get_element(*i).label, 0);
        }
    }

    #[test]
    fn elements_by_dim_and_label_bottom_edges() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let elems = square.get_elements_by_dimension_and_label(0, 1);
        assert_eq!(elems.len(), nx);
        for i in &elems {
            assert_eq!(square.get_element(*i).label, 0);
            assert_eq!(
                square.get_element(*i).geometric_element.msh_id,
                EDGE2.msh_id
            );
        }
    }

    #[test]
    fn elements_by_dim_and_label_right_edges() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let elems = square.get_elements_by_dimension_and_label(1, 1);
        assert_eq!(elems.len(), ny);
        for i in &elems {
            assert_eq!(square.get_element(*i).label, 1);
        }
    }

    #[test]
    fn elements_by_dim_and_label_top_edges() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let elems = square.get_elements_by_dimension_and_label(2, 1);
        assert_eq!(elems.len(), nx);
        for i in &elems {
            assert_eq!(square.get_element(*i).label, 2);
        }
    }

    #[test]
    fn elements_by_dim_and_label_left_edges() {
        let nx = 3;
        let ny = 2;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let elems = square.get_elements_by_dimension_and_label(3, 1);
        assert_eq!(elems.len(), ny);
        for i in &elems {
            assert_eq!(square.get_element(*i).label, 3);
        }
    }

    #[test]
    fn elements_by_dim_and_label_invalid_returns_empty() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 2, None, None);
        assert!(square.get_elements_by_dimension_and_label(99, 1).is_empty());
        assert!(square.get_elements_by_dimension_and_label(0, 3).is_empty());
        assert!(square.get_elements_by_dimension_and_label(1, 2).is_empty());
    }

    #[test]
    fn elements_by_dim_and_label_partition_covers_all_edges() {
        let nx = 4;
        let ny = 3;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let mut all_boundary: Vec<usize> = (0..4)
            .flat_map(|label| square.get_elements_by_dimension_and_label(label, 1))
            .collect();
        all_boundary.sort();
        assert_eq!(all_boundary, square.get_elements_by_dimension(1));
    }

    #[test]
    fn elements_by_dim_and_label_partition_covers_all_elements() {
        let nx = 4;
        let ny = 3;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let quads = square.get_elements_by_dimension_and_label(0, 2);
        let mut edges: Vec<usize> = (0..4)
            .flat_map(|label| square.get_elements_by_dimension_and_label(label, 1))
            .collect();
        edges.sort();
        let mut all = quads;
        all.extend(edges);
        all.sort();
        let expected: Vec<usize> = (0..square.get_nelements()).collect();
        assert_eq!(all, expected);
    }

    // === Coloring round-trip and visual output ===

    #[test]
    fn color_to_element_roundtrip() {
        // For every (color, index-within-color), the returned global element
        // must have that color according to get_element_color.
        for &(nx, ny) in &[(3, 3), (4, 4), (5, 3), (2, 7)] {
            let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
            for c in 0..square.get_ncolors() {
                for e in 0..square.get_nelements_by_color(c) {
                    let global = square.get_element_index_by_color_to_element_index(c, e);
                    assert_eq!(
                        square.get_element_color(global),
                        c,
                        "Color round-trip failed for {}x{}: color={}, e={}, global={}",
                        nx,
                        ny,
                        c,
                        e,
                        global
                    );
                }
            }
        }
    }

    #[test]
    fn color_to_element_is_bijection() {
        // The union of all per-color element lists must cover {0..nx*ny} exactly once.
        for &(nx, ny) in &[(3, 3), (4, 4), (5, 3), (2, 7)] {
            let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
            let mut seen = vec![false; nx * ny];
            for c in 0..square.get_ncolors() {
                for e in 0..square.get_nelements_by_color(c) {
                    let global = square.get_element_index_by_color_to_element_index(c, e);
                    assert!(
                        global < nx * ny,
                        "Global index {} out of range for {}x{} mesh",
                        global,
                        nx,
                        ny
                    );
                    assert!(
                        !seen[global],
                        "Element {} visited twice in {}x{} mesh",
                        global, nx, ny
                    );
                    seen[global] = true;
                }
            }
            assert!(
                seen.iter().all(|&v| v),
                "Not all elements covered for {}x{} mesh",
                nx,
                ny
            );
        }
    }

    #[test]
    fn write_msh_with_color_labels() {
        // Same pattern as coloring_algorithm() in mesh.rs:
        // convert to UnstructuredMesh, assign each quad's label = its color, write.
        let nx = 6;
        let ny = 4;
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, None);
        let mut mesh = square.get_unstructured_mesh();
        let vec_elem = mesh.get_elements_by_dimension(2);
        mesh.set_labels(|el, elem| {
            // Volume elements come first when converting a Square.
            if el < vec_elem.len() {
                square.get_element_color(el)
            } else {
                elem.label
            }
        });
        mesh.write_msh("./tests/results/square_with_colors.msh")
            .unwrap();
    }

    // === 1x1 minimal mesh ===

    #[test]
    fn minimal_1x1_mesh() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 1, 1, None, None);
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

    // === Q2 geometric nodes ===

    #[test]
    fn q2_node_count_and_q1_prefix() {
        let (nx, ny) = (3, 2);
        let q1 = Square::new([0.5, -1.0], 2.0, 1.0, nx, ny, None, None);
        let q2 = Square::new([0.5, -1.0], 2.0, 1.0, nx, ny, None, Some(true));

        assert_eq!(q2.get_nnodes(), (2 * nx + 1) * (2 * ny + 1));
        assert_eq!(q1.get_nelements(), q2.get_nelements());

        // The Q1 vertices keep their numbering and positions.
        for i in 0..q1.get_nnodes() {
            assert_eq!(q1.get_node(i), q2.get_node(i));
        }
    }

    #[test]
    fn q2_elements_types_and_node_positions() {
        let (nx, ny) = (3, 2);
        let (lx, ly) = (3.0, 2.0);
        let square = Square::new([0.0, 0.0], lx, ly, nx, ny, None, Some(true));
        let hx = lx / nx as f64;
        let hy = ly / ny as f64;

        // Cells: QUADRANGLE9 with nodes at the vertices, the edge midpoints
        // (bottom, right, top, left) and the center of the cell.
        for e in 0..nx * ny {
            let elem = square.get_element(e);
            assert_eq!(elem.geometric_element.msh_id, QUADRANGLE9.msh_id);
            let i = (e % nx) as f64;
            let j = (e / nx) as f64;
            let expected = [
                [i, j],
                [i + 1.0, j],
                [i + 1.0, j + 1.0],
                [i, j + 1.0],
                [i + 0.5, j],
                [i + 1.0, j + 0.5],
                [i + 0.5, j + 1.0],
                [i, j + 0.5],
                [i + 0.5, j + 0.5],
            ];
            for (k, exp) in expected.iter().enumerate() {
                let node = square.get_node(elem.indexes[k]);
                assert!(
                    (node[0] - exp[0] * hx).abs() < 1e-12 && (node[1] - exp[1] * hy).abs() < 1e-12,
                    "element {} node {}: {:?}, expected {:?}",
                    e,
                    k,
                    node,
                    exp
                );
            }
        }

        // Boundary: EDGE3 whose midpoint is the average of the two vertices.
        for e in square.get_elements_by_dimension(1) {
            let elem = square.get_element(e);
            assert_eq!(elem.geometric_element.msh_id, EDGE3.msh_id);
            let a = square.get_node(elem.indexes[0]);
            let b = square.get_node(elem.indexes[1]);
            let m = square.get_node(elem.indexes[2]);
            assert!((m[0] - 0.5 * (a[0] + b[0])).abs() < 1e-12);
            assert!((m[1] - 0.5 * (a[1] + b[1])).abs() < 1e-12);
        }
    }

    #[test]
    fn q2_unstructured_conversion_and_write_roundtrip() {
        let (nx, ny) = (3, 2);
        let square = Square::new([0.0, 0.0], 1.0, 1.0, nx, ny, None, Some(true));
        let mesh = square.get_unstructured_mesh();

        // The topology is that of the Q1 grid: the Q2 geometric nodes are
        // not topological vertices.
        assert_eq!(mesh.get_nvertexes(), (nx + 1) * (ny + 1));
        assert_eq!(mesh.get_nedges(), nx * (ny + 1) + (nx + 1) * ny);

        // Write + reread: the Q2 elements survive the roundtrip.
        mesh.write_msh("./tests/results/square_q2.msh").unwrap();
        let reread = UnstructuredMesh::new_from_msh("./tests/results/square_q2.msh").unwrap();
        assert_eq!(reread.get_nnodes(), square.get_nnodes());
        for e in 0..reread.get_nelements() {
            let a = square.get_element(e);
            let b = reread.get_element(e);
            assert_eq!(a.geometric_element.msh_id, b.geometric_element.msh_id);
            assert_eq!(a.indexes, b.indexes);
            assert_eq!(a.label, b.label);
        }
    }
}
