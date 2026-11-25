extern crate ondomathx_mesh;

#[cfg(test)]
mod tests {
    use ondomathx_mesh::geometric_elements::*;
    use ondomathx_mesh::mesh::circle::*;
    use ondomathx_mesh::mesh::Mesh;

    // === Construction ===

    #[test]
    fn construction() {
        let c = Circle::new([0.0, 0.0], 1.0, 3, None, None);
        assert_eq!(c.n, 3);
        assert!((c.radius - 1.0).abs() < 1e-12);
    }

    // === Counts ===

    #[test]
    fn nnodes() {
        // nnodes = (n_x+1)² + n_rho·n_theta,  n_theta = 4·n_x
        for n in [3_usize, 6, 9] {
            let n_rho = n / 3;
            let n_x = n - 2 * n_rho;
            let n_theta = 4 * n_x;
            let expected = (n_x + 1) * (n_x + 1) + n_rho * n_theta;
            let c = Circle::new([0.0, 0.0], 1.0, n, None, None);
            assert_eq!(c.get_nnodes(), expected, "n={}", n);
        }
    }

    #[test]
    fn nelements() {
        // nelements = n_x² + n_rho·n_theta + n_theta
        for n in [3_usize, 6, 9] {
            let n_rho = n / 3;
            let n_x = n - 2 * n_rho;
            let n_theta = 4 * n_x;
            let expected = n_x * n_x + n_rho * n_theta + n_theta;
            let c = Circle::new([0.0, 0.0], 1.0, n, None, None);
            assert_eq!(c.get_nelements(), expected, "n={}", n);
        }
    }

    // === Element types ===

    #[test]
    fn volume_elements_are_quads() {
        let c = Circle::new([0.0, 0.0], 1.0, 6, None, None);
        for i in c.get_elements_by_dimension(2) {
            assert_eq!(
                c.get_element(i).geometric_element.msh_id,
                QUADRANGLE4.msh_id,
                "Element {} should be a quad",
                i
            );
        }
    }

    #[test]
    fn boundary_elements_are_edges() {
        let c = Circle::new([0.0, 0.0], 1.0, 3, None, None);
        for i in c.get_elements_by_dimension(1) {
            assert_eq!(
                c.get_element(i).geometric_element.msh_id,
                EDGE2.msh_id,
                "Element {} should be an edge",
                i
            );
        }
    }

    #[test]
    fn boundary_edge_count() {
        for n in [3_usize, 6, 9] {
            let n_x = n - 2 * (n / 3);
            let n_theta = 4 * n_x;
            let c = Circle::new([0.0, 0.0], 1.0, n, None, None);
            assert_eq!(c.get_elements_by_dimension(1).len(), n_theta, "n={}", n);
        }
    }

    // === Node indices in range ===

    #[test]
    fn all_element_node_indices_valid() {
        let c = Circle::new([0.0, 0.0], 1.0, 6, None, None);
        let nnodes = c.get_nnodes();
        for i in 0..c.get_nelements() {
            for &idx in &c.get_element(i).indexes {
                assert!(
                    idx < nnodes,
                    "Element {}: node index {} out of range (nnodes={})",
                    i,
                    idx,
                    nnodes
                );
            }
        }
    }

    // === Reference-domain geometry ===

    // Inner-square nodes must lie inside [-a, a]².
    #[test]
    fn inner_square_nodes_in_inner_square() {
        let radius = 2.0;
        let center = [1.0, -3.0];
        let c = Circle::new(center, radius, 6, None, None);
        let r_ref = radius / 2f64.sqrt();
        let a = c.alpha * r_ref;
        let n_vert_square = (c.n_x + 1) * (c.n_x + 1);
        for i in 0..n_vert_square {
            let [x, y, _] = c.get_node(i);
            let lx = x - center[0];
            let ly = y - center[1];
            assert!(
                lx.abs() <= a + 1e-12 && ly.abs() <= a + 1e-12,
                "Inner node {} at ({},{}) outside inner square (a={})",
                i,
                x,
                y,
                a
            );
        }
    }

    // Boundary edge nodes must lie on the circle (distance from center = radius).
    #[test]
    fn outer_boundary_nodes_on_circle() {
        let radius = 1.5;
        let center = [1.0, -2.0];
        let c = Circle::new(center, radius, 6, None, None);
        for i in c.get_elements_by_dimension(1) {
            let elem = c.get_element(i);
            for &idx in &elem.indexes {
                let [x, y, _] = c.get_node(idx);
                let r = ((x - center[0]).powi(2) + (y - center[1]).powi(2)).sqrt();
                assert!(
                    (r - radius).abs() < 1e-9,
                    "Boundary node {} at ({:.6},{:.6}): distance {:.9} ≠ radius {}",
                    idx,
                    x,
                    y,
                    r,
                    radius
                );
            }
        }
    }

    // Outermost arm nodes (t_rho=1) lie on the circle boundary.
    #[test]
    fn outermost_arm_nodes_on_circle() {
        let radius = 2.0;
        let center = [0.0, 0.0];
        let c = Circle::new(center, radius, 6, None, None);
        let n_vert_square = (c.n_x + 1) * (c.n_x + 1);
        for i_theta in 0..c.n_theta {
            let idx = n_vert_square + (c.n_rho - 1) * c.n_theta + i_theta;
            let [x, y, _] = c.get_node(idx);
            let r = ((x - center[0]).powi(2) + (y - center[1]).powi(2)).sqrt();
            assert!(
                (r - radius).abs() < 1e-12,
                "Outermost node {} (i_theta={}) at ({:.6},{:.6}): distance {:.12} ≠ radius {}",
                idx,
                i_theta,
                x,
                y,
                r,
                radius
            );
        }
    }

    // The four corners (±r_ref, ±r_ref) must appear as nodes.
    #[test]
    fn outer_square_corners_present() {
        let radius = 3.0;
        let center = [0.5, -1.0];
        let c = Circle::new(center, radius, 3, None, None);
        let r_ref = radius / 2f64.sqrt();
        let (cx, cy) = (center[0], center[1]);
        let corners = [
            (cx - r_ref, cy - r_ref),
            (cx + r_ref, cy - r_ref),
            (cx + r_ref, cy + r_ref),
            (cx - r_ref, cy + r_ref),
        ];
        for &(ex, ey) in &corners {
            let found = (0..c.get_nnodes()).any(|i| {
                let [x, y, _] = c.get_node(i);
                (x - ex).abs() < 1e-12 && (y - ey).abs() < 1e-12
            });
            assert!(found, "Corner ({},{}) not found in mesh", ex, ey);
        }
    }

    // For n=6 (n_x=2) the center node is at index (n_x/2)*(n_x+1) + n_x/2.
    #[test]
    fn center_node_at_center() {
        let center = [3.0, -1.0];
        let c = Circle::new(center, 1.0, 6, None, None);
        let mid = (c.n_x / 2) * (c.n_x + 1) + c.n_x / 2;
        let [x, y, z] = c.get_node(mid);
        assert!(
            (x - center[0]).abs() < 1e-12,
            "center x: got {}, expected {}",
            x,
            center[0]
        );
        assert!(
            (y - center[1]).abs() < 1e-12,
            "center y: got {}, expected {}",
            y,
            center[1]
        );
        assert!(z.abs() < 1e-12);
    }

    // Innermost arm nodes (layer 1) lie outside [-a, a]².
    #[test]
    fn innermost_arm_nodes_outside_inner_square() {
        let radius = 2.0;
        let c = Circle::new([0.0, 0.0], radius, 6, None, None);
        let r_ref = radius / 2f64.sqrt();
        let a = c.alpha * r_ref;
        let n_vert_square = (c.n_x + 1) * (c.n_x + 1);
        for i_theta in 0..c.n_theta {
            let idx = n_vert_square + i_theta;
            let [x, y, _] = c.get_node(idx);
            assert!(
                x.abs() > a - 1e-12 || y.abs() > a - 1e-12,
                "Innermost arm node {} at ({:.6},{:.6}) inside inner square (a={})",
                idx,
                x,
                y,
                a
            );
        }
    }

    // Outer corners of each arm are at (±r_ref, ±r_ref);
    // inner corner of bottom arm is at (-w, -w), one uniform layer out.
    #[test]
    fn arm_bilinear_corners() {
        let radius = 4.0;
        let c = Circle::new([0.0, 0.0], radius, 6, None, None); // n_x=2, n_rho=2, n_theta=8
        let r_ref = radius / 2f64.sqrt();
        let nv = (c.n_x + 1) * (c.n_x + 1);

        // Outer corners (t_rho=1, t_x=0) → (±r_ref, ±r_ref).
        let outer_idxs = [
            nv + (c.n_rho - 1) * c.n_theta,
            nv + (c.n_rho - 1) * c.n_theta + c.n_x,
            nv + (c.n_rho - 1) * c.n_theta + 2 * c.n_x,
            nv + (c.n_rho - 1) * c.n_theta + 3 * c.n_x,
        ];
        let expected_corners = [
            (-r_ref, -r_ref),
            (r_ref, -r_ref),
            (r_ref, r_ref),
            (-r_ref, r_ref),
        ];
        for (&idx, &(ex, ey)) in outer_idxs.iter().zip(expected_corners.iter()) {
            let [x, y, _] = c.get_node(idx);
            assert!(
                (x - ex).abs() < 1e-12 && (y - ey).abs() < 1e-12,
                "Outer corner idx {}: got ({},{}), expected ({},{})",
                idx,
                x,
                y,
                ex,
                ey
            );
        }

        // Inner corner of bottom arm (i_rho=1, i_x=0): the diagonal ray blends
        // (-a,-a) to (-R/√2,-R/√2) at t=1/2, and a = R - R/√2 for n=6,
        // so circle_map sends (-w,-w) to -(a + R/√2)/2·(1,1) = (-R/2, -R/2).
        let ex2 = -radius / 2.0;
        let ey2 = -radius / 2.0;
        let [x2, y2, _] = c.get_node(nv);
        assert!(
            (x2 - ex2).abs() < 1e-10 && (y2 - ey2).abs() < 1e-10,
            "Inner bottom corner: got ({},{}), expected ({},{})",
            x2,
            y2,
            ex2,
            ey2
        );
    }

    // === Dimensions ===

    #[test]
    fn geometric_and_topologic_dimension() {
        let c = Circle::new([0.0, 0.0], 1.0, 3, None, None);
        assert_eq!(c.get_geometric_dimension(), 2);
        assert_eq!(c.get_topologic_dimension(), 2);
    }

    // === get_elements_by_dimension ===

    #[test]
    fn elements_by_dim_partition_covers_all() {
        let c = Circle::new([0.0, 0.0], 1.0, 3, None, None);
        let mut all: Vec<usize> = c.get_elements_by_dimension(2);
        all.extend(c.get_elements_by_dimension(1));
        all.sort();
        let expected: Vec<usize> = (0..c.get_nelements()).collect();
        assert_eq!(all, expected);
    }

    #[test]
    fn elements_by_dim_other_empty() {
        let c = Circle::new([0.0, 0.0], 1.0, 3, None, None);
        assert!(c.get_elements_by_dimension(0).is_empty());
        assert!(c.get_elements_by_dimension(3).is_empty());
    }

    // === Coloring ===

    // No two volume elements sharing a node may have the same color.
    // Covers odd n_x (n=3,5,7,9) and even n_x (n=4,6,10).
    #[test]
    fn coloring_is_valid() {
        for n in [3_usize, 4, 5, 6, 7, 9, 10] {
            let c = Circle::new([0.0, 0.0], 1.0, n, None, None);
            assert_eq!(c.get_ncolors(), 4, "n={}", n);
            let mut node_to_elements: Vec<Vec<usize>> = vec![vec![]; c.get_nnodes()];
            for i in c.get_elements_by_dimension(2) {
                for &v in &c.get_element(i).indexes {
                    node_to_elements[v].push(i);
                }
            }
            for elements in &node_to_elements {
                for (k, &e1) in elements.iter().enumerate() {
                    for &e2 in &elements[k + 1..] {
                        assert_ne!(
                            c.get_element_color(e1),
                            c.get_element_color(e2),
                            "n={}: elements {} and {} share a node but both have color {}",
                            n,
                            e1,
                            e2,
                            c.get_element_color(e1)
                        );
                    }
                }
            }
        }
    }

    // The color classes partition the volume elements, and the
    // (color, index-by-color) -> element map is consistent with the colors.
    #[test]
    fn coloring_partition_consistent() {
        for n in [3_usize, 6, 7] {
            let c = Circle::new([0.0, 0.0], 1.0, n, None, None);
            let n_vol = c.get_elements_by_dimension(2).len();
            let total: usize = (0..c.get_ncolors())
                .map(|ci| c.get_nelements_by_color(ci))
                .sum();
            assert_eq!(total, n_vol, "n={}", n);
            let mut seen = vec![false; n_vol];
            for ci in 0..c.get_ncolors() {
                for k in 0..c.get_nelements_by_color(ci) {
                    let e = c.get_element_index_by_color_to_element_index(ci, k);
                    assert_eq!(c.get_element_color(e), ci, "n={}", n);
                    assert!(!seen[e], "n={}: element {} listed twice", n, e);
                    seen[e] = true;
                }
            }
            assert!(seen.iter().all(|&s| s), "n={}", n);
        }
    }

    // === get_element_grad_def ===

    // Inner-square elements are affine with J = h·I (h = 2a/n_x), and the
    // circle map is the identity there.
    #[test]
    fn grad_def_inner_square_elements() {
        let radius = 2.0;
        let c = Circle::new([1.0, -0.5], radius, 10, None, None);
        let r_ref = radius / 2f64.sqrt();
        let h = 2.0 * c.alpha * r_ref / c.n_x as f64;

        for el in 0..c.n_x * c.n_x {
            let mut grad = [0.0; 4];
            c.get_element_grad_def(el, &[0.3, 0.6, 0.0], &mut grad);
            assert!((grad[0] - h).abs() < 1e-12, "el {}: {:?}", el, grad);
            assert!(grad[1].abs() < 1e-12, "el {}: {:?}", el, grad);
            assert!(grad[2].abs() < 1e-12, "el {}: {:?}", el, grad);
            assert!((grad[3] - h).abs() < 1e-12, "el {}: {:?}", el, grad);
        }
    }

    // det J > 0 at quadrature points of every volume element (the mesh is
    // consistently oriented and nowhere degenerate).
    #[test]
    fn grad_def_positive_determinant() {
        // 3-point Gauss-Legendre on [0,1]
        let sq = (3.0_f64 / 5.0).sqrt();
        let pts = [0.5 - 0.5 * sq, 0.5, 0.5 + 0.5 * sq];

        for n in [3_usize, 6, 7, 10] {
            let c = Circle::new([0.0, 0.0], 1.5, n, None, None);
            for el in c.get_elements_by_dimension(2) {
                for &u in &pts {
                    for &v in &pts {
                        let mut grad = [0.0; 4];
                        c.get_element_grad_def(el, &[u, v, 0.0], &mut grad);
                        let det = grad[0] * grad[3] - grad[1] * grad[2];
                        assert!(
                            det > 0.0,
                            "n={} element {} at ({},{}): det J = {}",
                            n,
                            el,
                            u,
                            v,
                            det
                        );
                    }
                }
            }
        }
    }

    // The element maps tile the disk: summing ∫ det J over all volume
    // elements (Gauss quadrature) must give the disk area π·R².
    #[test]
    fn grad_def_integrates_to_disk_area() {
        // 3-point Gauss-Legendre on [0,1]
        let sq = (3.0_f64 / 5.0).sqrt();
        let pts = [0.5 - 0.5 * sq, 0.5, 0.5 + 0.5 * sq];
        let wts = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0];

        for n in [14, 16, 18] {
            let radius = 1.5;
            let c = Circle::new([0.3, 0.7], radius, n, None, None);
            let mut area = 0.0;
            for el in c.get_elements_by_dimension(2) {
                for (i, &u) in pts.iter().enumerate() {
                    for (j, &v) in pts.iter().enumerate() {
                        let mut grad = [0.0; 4];
                        c.get_element_grad_def(el, &[u, v, 0.0], &mut grad);
                        let det = grad[0] * grad[3] - grad[1] * grad[2];
                        area += wts[i] * wts[j] * det;
                    }
                }
            }
            let exact = std::f64::consts::PI * radius * radius;
            assert!(
                ((area - exact) / exact).abs() < 1e-7,
                "n={}: quadrature area {} vs π·R² = {}",
                n,
                area,
                exact
            );
        }
    }

    // === Visual output ===

    #[test]
    fn write_msh_circle() {
        let c = Circle::new([0.0, 0.0], 1.0, 10, None, None);
        c.write_msh("./tests/results/circle_mesh.msh").unwrap();
    }

    #[test]
    fn write_msh_with_color_labels() {
        // Same pattern as write_msh_with_color_labels() in square.rs:
        // convert to UnstructuredMesh, assign each quad's label = its color, write.
        let c = Circle::new([0.0, 0.0], 1.0, 20, None, None);
        let mut mesh = c.get_unstructured_mesh();
        let vec_elem = mesh.get_elements_by_dimension(2);
        mesh.set_labels(|el, elem| {
            // Volume elements come first when converting a Circle.
            if el < vec_elem.len() {
                c.get_element_color(el)
            } else {
                elem.label
            }
        });
        mesh.write_msh("./tests/results/circle_with_colors.msh")
            .unwrap();
    }

    // === Q2 geometric nodes ===

    #[test]
    fn q2_node_count_and_q1_prefix() {
        for n in [3_usize, 6, 7, 10] {
            let q1 = Circle::new([0.3, -0.2], 1.5, n, None, None);
            let q2 = Circle::new([0.3, -0.2], 1.5, n, None, Some(true));

            let n_q1 = q1.get_nnodes();
            let n_vol = q1.n_x * q1.n_x + q1.n_rho * q1.n_theta;
            assert_eq!(
                q2.get_nnodes(),
                n_q1 + 2 * q1.n_x * (q1.n_x + 1) + 2 * q1.n_rho * q1.n_theta + n_vol,
                "n={}",
                n
            );
            assert_eq!(q1.get_nelements(), q2.get_nelements());

            // The Q1 vertices keep their numbering and positions.
            for i in 0..n_q1 {
                assert_eq!(q1.get_node(i), q2.get_node(i), "n={} node {}", n, i);
            }

            // The Q1 corners of every element are unchanged.
            for e in 0..q1.get_nelements() {
                let a = q1.get_element(e);
                let b = q2.get_element(e);
                let nv = a.indexes.len();
                assert_eq!(a.indexes[..], b.indexes[..nv], "n={} element {}", n, e);
                assert_eq!(a.label, b.label);
            }
        }
    }

    // Every node of the Q2 mesh is referenced by exactly the elements
    // (conforming numbering: no gaps, no unused Q2 node).
    #[test]
    fn q2_element_types_and_node_coverage() {
        for n in [3_usize, 7, 10] {
            let c = Circle::new([0.0, 0.0], 1.0, n, None, Some(true));
            let mut seen = std::collections::HashSet::new();
            for e in c.get_elements_by_dimension(2) {
                let elem = c.get_element(e);
                assert_eq!(elem.geometric_element.msh_id, QUADRANGLE9.msh_id);
                assert_eq!(elem.indexes.len(), 9);
                seen.extend(elem.indexes.iter().copied());
            }
            for e in c.get_elements_by_dimension(1) {
                let elem = c.get_element(e);
                assert_eq!(elem.geometric_element.msh_id, EDGE3.msh_id);
                assert_eq!(elem.indexes.len(), 3);
                seen.extend(elem.indexes.iter().copied());
            }
            assert!(seen.iter().all(|&i| i < c.get_nnodes()), "n={}", n);
            assert_eq!(seen.len(), c.get_nnodes(), "n={}", n);
        }
    }

    #[test]
    fn q2_boundary_and_inner_square_geometry() {
        let radius = 2.0;
        let center = [0.4, -0.1];
        let c = Circle::new(center, radius, 10, None, Some(true));

        // All nodes of the boundary EDGE3 elements (midpoints included) lie
        // on the circle: the Q2 mesh represents the curved boundary.
        for e in c.get_elements_by_dimension(1) {
            for &i in &c.get_element(e).indexes {
                let [x, y, _] = c.get_node(i);
                let r = (x - center[0]).hypot(y - center[1]);
                assert!(
                    (r - radius).abs() < 1e-12,
                    "boundary node {} at distance {}",
                    i,
                    r
                );
            }
        }

        // Inner-square cells are affine (identity circle map): midpoints and
        // center are the averages of the corners.
        for e in 0..c.n_x * c.n_x {
            let elem = c.get_element(e);
            let p: Vec<[f64; 3]> = elem.indexes.iter().map(|&i| c.get_node(i)).collect();
            for k in [0, 1] {
                assert!((p[4][k] - 0.5 * (p[0][k] + p[1][k])).abs() < 1e-12);
                assert!((p[5][k] - 0.5 * (p[1][k] + p[2][k])).abs() < 1e-12);
                assert!((p[6][k] - 0.5 * (p[3][k] + p[2][k])).abs() < 1e-12);
                assert!((p[7][k] - 0.5 * (p[0][k] + p[3][k])).abs() < 1e-12);
                assert!((p[8][k] - 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k])).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn q2_unstructured_conversion_and_write_roundtrip() {
        let c = Circle::new([0.0, 0.0], 1.0, 6, None, Some(true));
        let mesh = c.get_unstructured_mesh();

        // The topology is that of the Q1 mesh: the Q2 geometric nodes are
        // not topological vertices.
        let q1 = Circle::new([0.0, 0.0], 1.0, 6, None, None);
        assert_eq!(mesh.get_nvertexes(), q1.get_nnodes());

        // Write + reread: the Q2 elements survive the roundtrip.
        mesh.write_msh("./tests/results/circle_q2.msh").unwrap();
        let reread = UnstructuredMesh::new_from_msh("./tests/results/circle_q2.msh").unwrap();
        assert_eq!(reread.get_nnodes(), c.get_nnodes());
        for e in 0..reread.get_nelements() {
            let a = c.get_element(e);
            let b = reread.get_element(e);
            assert_eq!(a.geometric_element.msh_id, b.geometric_element.msh_id);
            assert_eq!(a.indexes, b.indexes);
            assert_eq!(a.label, b.label);
        }
    }
}
