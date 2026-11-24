extern crate anamesh;

#[cfg(test)]
mod tests {
    use anamesh::geometric_elements::*;

    // === EDGE2: linear edge shape functions ===

    #[test]
    fn edge2_shape_functions_at_endpoints() {
        // At u=0: shape[0]=1, shape[1]=0
        let uvw0 = [0.0, 0.0, 0.0];
        assert!((EDGE2.shape_functions[0](&uvw0) - 1.0).abs() < 1e-12);
        assert!((EDGE2.shape_functions[1](&uvw0) - 0.0).abs() < 1e-12);

        // At u=1: shape[0]=0, shape[1]=1
        let uvw1 = [1.0, 0.0, 0.0];
        assert!((EDGE2.shape_functions[0](&uvw1) - 0.0).abs() < 1e-12);
        assert!((EDGE2.shape_functions[1](&uvw1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn edge2_partition_of_unity() {
        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let uvw = [u, 0.0, 0.0];
            let sum: f64 = (0..EDGE2.nshape_functions)
                .map(|k| EDGE2.shape_functions[k](&uvw))
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Partition of unity failed at u={}",
                u
            );
        }
    }

    #[test]
    fn edge2_derivatives() {
        // Constant derivatives for linear functions
        let uvw = [0.5, 0.0, 0.0];
        let d0 = EDGE2.d_shape_functions[0](&uvw);
        let d1 = EDGE2.d_shape_functions[1](&uvw);
        assert!((d0[0] - (-1.0)).abs() < 1e-12);
        assert!((d1[0] - 1.0).abs() < 1e-12);
    }

    // === EDGE3: quadratic edge shape functions ===

    #[test]
    fn edge3_shape_functions_at_nodes() {
        let nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]];

        for (i, node) in nodes.iter().enumerate() {
            for j in 0..EDGE3.nshape_functions {
                let val = EDGE3.shape_functions[j](node);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-12,
                    "EDGE3 shape function {} at node {}: expected {}, got {}",
                    j,
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn edge3_partition_of_unity() {
        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let uvw = [u, 0.0, 0.0];
            let sum: f64 = (0..EDGE3.nshape_functions)
                .map(|k| EDGE3.shape_functions[k](&uvw))
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Partition of unity failed at u={}",
                u
            );
        }
    }

    // === TRIANGLE3: linear triangle shape functions ===

    #[test]
    fn triangle3_shape_functions_at_vertices() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        for (i, vtx) in vertices.iter().enumerate() {
            for j in 0..TRIANGLE3.nshape_functions {
                let val = TRIANGLE3.shape_functions[j](vtx);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-12,
                    "TRIANGLE3 shape function {} at vertex {}: expected {}, got {}",
                    j,
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn triangle3_partition_of_unity() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.0],
            [0.5, 0.5, 0.0],
            [0.1, 0.3, 0.0],
        ];

        for pt in &points {
            let sum: f64 = (0..TRIANGLE3.nshape_functions)
                .map(|k| TRIANGLE3.shape_functions[k](pt))
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Partition of unity failed at ({}, {})",
                pt[0],
                pt[1]
            );
        }
    }

    // === QUADRANGLE4: bilinear quad shape functions ===

    #[test]
    fn quadrangle4_shape_functions_at_vertices() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        for (i, vtx) in vertices.iter().enumerate() {
            for j in 0..QUADRANGLE4.nshape_functions {
                let val = QUADRANGLE4.shape_functions[j](vtx);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-12,
                    "QUADRANGLE4 shape function {} at vertex {}: expected {}, got {}",
                    j,
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn quadrangle4_partition_of_unity() {
        for i in 0..=4 {
            for j in 0..=4 {
                let u = i as f64 / 4.0;
                let v = j as f64 / 4.0;
                let uvw = [u, v, 0.0];
                let sum: f64 = (0..QUADRANGLE4.nshape_functions)
                    .map(|k| QUADRANGLE4.shape_functions[k](&uvw))
                    .sum();
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "Partition of unity failed at ({}, {})",
                    u,
                    v
                );
            }
        }
    }

    // === QUADRANGLE9: biquadratic quad shape functions ===

    #[test]
    fn quadrangle9_shape_functions_at_nodes() {
        let nodes = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ];

        for (i, node) in nodes.iter().enumerate() {
            for j in 0..QUADRANGLE9.nshape_functions {
                let val = QUADRANGLE9.shape_functions[j](node);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-10,
                    "QUADRANGLE9 shape function {} at node {}: expected {}, got {}",
                    j,
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn quadrangle9_partition_of_unity() {
        for i in 0..=4 {
            for j in 0..=4 {
                let u = i as f64 / 4.0;
                let v = j as f64 / 4.0;
                let uvw = [u, v, 0.0];
                let sum: f64 = (0..QUADRANGLE9.nshape_functions)
                    .map(|k| QUADRANGLE9.shape_functions[k](&uvw))
                    .sum();
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "Partition of unity failed at ({}, {})",
                    u,
                    v
                );
            }
        }
    }

    // === TRIANGLE6: quadratic (P2) triangle shape functions ===

    // Node order: 3 vertices, then the midpoints of edges (0,1), (1,2), (0,2).
    const TRIANGLE6_NODES: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
    ];

    #[test]
    fn triangle6_shape_functions_at_nodes() {
        for (i, node) in TRIANGLE6_NODES.iter().enumerate() {
            for j in 0..TRIANGLE6.nshape_functions {
                let val = TRIANGLE6.shape_functions[j](node);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-12,
                    "TRIANGLE6 shape function {} at node {}: expected {}, got {}",
                    j,
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn triangle6_partition_of_unity() {
        for i in 0..=4 {
            for j in 0..=(4 - i) {
                let u = i as f64 / 4.0;
                let v = j as f64 / 4.0;
                let uvw = [u, v, 0.0];
                let sum: f64 = (0..TRIANGLE6.nshape_functions)
                    .map(|k| TRIANGLE6.shape_functions[k](&uvw))
                    .sum();
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "Partition of unity failed at ({}, {})",
                    u,
                    v
                );
            }
        }
    }

    // The gradients match central finite differences of the shape functions.
    #[test]
    fn triangle6_derivatives_match_finite_differences() {
        let h = 1e-6;
        for &(u, v) in &[(0.2, 0.3), (0.1, 0.7), (0.5, 0.25), (1.0 / 3.0, 1.0 / 3.0)] {
            for k in 0..TRIANGLE6.nshape_functions {
                let d = TRIANGLE6.d_shape_functions[k](&[u, v, 0.0]);
                let fd_u = (TRIANGLE6.shape_functions[k](&[u + h, v, 0.0])
                    - TRIANGLE6.shape_functions[k](&[u - h, v, 0.0]))
                    / (2.0 * h);
                let fd_v = (TRIANGLE6.shape_functions[k](&[u, v + h, 0.0])
                    - TRIANGLE6.shape_functions[k](&[u, v - h, 0.0]))
                    / (2.0 * h);
                assert!(
                    (d[0] - fd_u).abs() < 1e-8 && (d[1] - fd_v).abs() < 1e-8,
                    "TRIANGLE6 d_shape_function {} at ({},{}): got ({},{}), FD gives ({},{})",
                    k,
                    u,
                    v,
                    d[0],
                    d[1],
                    fd_u,
                    fd_v
                );
            }
        }
    }

    // === TETRAHEDRON10: quadratic (P2) tetrahedron shape functions ===

    // Node order: 4 vertices, then the midpoints of edges
    // (0,1), (1,2), (0,2), (0,3), (1,3), (2,3).
    const TETRAHEDRON10_NODES: [[f64; 3]; 10] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];

    #[test]
    fn tetrahedron10_shape_functions_at_nodes() {
        for (i, node) in TETRAHEDRON10_NODES.iter().enumerate() {
            for j in 0..TETRAHEDRON10.nshape_functions {
                let val = TETRAHEDRON10.shape_functions[j](node);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-12,
                    "TETRAHEDRON10 shape function {} at node {}: expected {}, got {}",
                    j,
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn tetrahedron10_partition_of_unity() {
        for i in 0..=4 {
            for j in 0..=(4 - i) {
                for k in 0..=(4 - i - j) {
                    let uvw = [i as f64 / 4.0, j as f64 / 4.0, k as f64 / 4.0];
                    let sum: f64 = (0..TETRAHEDRON10.nshape_functions)
                        .map(|s| TETRAHEDRON10.shape_functions[s](&uvw))
                        .sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-12,
                        "Partition of unity failed at {:?}",
                        uvw
                    );
                }
            }
        }
    }

    // The gradients match central finite differences of the shape functions.
    #[test]
    fn tetrahedron10_derivatives_match_finite_differences() {
        let h = 1e-6;
        for &(u, v, w) in &[(0.2, 0.3, 0.3), (0.1, 0.6, 0.2), (0.25, 0.25, 0.25)] {
            for k in 0..TETRAHEDRON10.nshape_functions {
                let d = TETRAHEDRON10.d_shape_functions[k](&[u, v, w]);
                let f = TETRAHEDRON10.shape_functions[k];
                let fd = [
                    (f(&[u + h, v, w]) - f(&[u - h, v, w])) / (2.0 * h),
                    (f(&[u, v + h, w]) - f(&[u, v - h, w])) / (2.0 * h),
                    (f(&[u, v, w + h]) - f(&[u, v, w - h])) / (2.0 * h),
                ];
                for c in 0..3 {
                    assert!(
                        (d[c] - fd[c]).abs() < 1e-8,
                        "TETRAHEDRON10 d_shape_function {} at ({},{},{}) component {}: got {}, FD gives {}",
                        k,
                        u,
                        v,
                        w,
                        c,
                        d[c],
                        fd[c]
                    );
                }
            }
        }
    }

    // === HEXAHEDRON27: triquadratic (Q2) hexahedron shape functions ===

    // Node order: 8 vertices, 12 edge midpoints (HEXAHEDRON_EDGES_TOPOLOGY
    // order), 6 face centers (HEXAHEDRON_FACES_TOPOLOGY order), center.
    const HEXAHEDRON27_NODES: [[f64; 3]; 27] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.5, 0.0, 1.0],
        [1.0, 0.5, 1.0],
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5],
    ];

    #[test]
    fn hexahedron27_shape_functions_at_nodes() {
        for (i, node) in HEXAHEDRON27_NODES.iter().enumerate() {
            for j in 0..HEXAHEDRON27.nshape_functions {
                let val = HEXAHEDRON27.shape_functions[j](node);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-12,
                    "HEXAHEDRON27 shape function {} at node {}: expected {}, got {}",
                    j,
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn hexahedron27_partition_of_unity() {
        for i in 0..=4 {
            for j in 0..=4 {
                for k in 0..=4 {
                    let uvw = [i as f64 / 4.0, j as f64 / 4.0, k as f64 / 4.0];
                    let sum: f64 = (0..HEXAHEDRON27.nshape_functions)
                        .map(|s| HEXAHEDRON27.shape_functions[s](&uvw))
                        .sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-12,
                        "Partition of unity failed at {:?}",
                        uvw
                    );
                }
            }
        }
    }

    // The gradients match central finite differences of the shape functions.
    #[test]
    fn hexahedron27_derivatives_match_finite_differences() {
        let h = 1e-6;
        for &(u, v, w) in &[(0.2, 0.3, 0.6), (0.1, 0.7, 0.4), (0.5, 0.25, 0.85)] {
            for k in 0..HEXAHEDRON27.nshape_functions {
                let d = HEXAHEDRON27.d_shape_functions[k](&[u, v, w]);
                let f = HEXAHEDRON27.shape_functions[k];
                let fd = [
                    (f(&[u + h, v, w]) - f(&[u - h, v, w])) / (2.0 * h),
                    (f(&[u, v + h, w]) - f(&[u, v - h, w])) / (2.0 * h),
                    (f(&[u, v, w + h]) - f(&[u, v, w - h])) / (2.0 * h),
                ];
                for c in 0..3 {
                    assert!(
                        (d[c] - fd[c]).abs() < 1e-8,
                        "HEXAHEDRON27 d_shape_function {} at ({},{},{}) component {}: got {}, FD gives {}",
                        k,
                        u,
                        v,
                        w,
                        c,
                        d[c],
                        fd[c]
                    );
                }
            }
        }
    }

    // === Consistency checks ===

    #[test]
    fn geometric_element_msh_ids() {
        assert_eq!(EDGE2.msh_id, 1);
        assert_eq!(EDGE3.msh_id, 8);
        assert_eq!(TRIANGLE3.msh_id, 2);
        assert_eq!(TRIANGLE6.msh_id, 9);
        assert_eq!(QUADRANGLE4.msh_id, 3);
        assert_eq!(QUADRANGLE9.msh_id, 10);
        assert_eq!(TETRAHEDRON4.msh_id, 4);
        assert_eq!(TETRAHEDRON10.msh_id, 11);
        assert_eq!(HEXAHEDRON8.msh_id, 5);
        assert_eq!(HEXAHEDRON27.msh_id, 12);
    }

    #[test]
    fn geometric_element_nshape_functions() {
        assert_eq!(EDGE2.nshape_functions, 2);
        assert_eq!(EDGE3.nshape_functions, 3);
        assert_eq!(TRIANGLE3.nshape_functions, 3);
        assert_eq!(TRIANGLE6.nshape_functions, 6);
        assert_eq!(QUADRANGLE4.nshape_functions, 4);
        assert_eq!(QUADRANGLE9.nshape_functions, 9);
        assert_eq!(TETRAHEDRON4.nshape_functions, 4);
        assert_eq!(TETRAHEDRON10.nshape_functions, 10);
        assert_eq!(HEXAHEDRON8.nshape_functions, 8);
        assert_eq!(HEXAHEDRON27.nshape_functions, 27);
    }

    // === tabulate ===

    fn check_tabulate(ge: &GeometricElement, points: &[[f64; 3]]) {
        let n = ge.nshape_functions;
        let (v, g) = ge.tabulate(points);
        assert_eq!((v.len(), g.len()), (points.len() * n, points.len() * n));
        for (p, uvw) in points.iter().enumerate() {
            for k in 0..n {
                assert_eq!(v[p * n + k], ge.shape_functions[k](uvw));
                assert_eq!(g[p * n + k], ge.d_shape_functions[k](uvw));
            }
            let sum: f64 = v[p * n..(p + 1) * n].iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "partition of unity at {uvw:?}");
            for d in 0..3 {
                let s: f64 = g[p * n..(p + 1) * n].iter().map(|x| x[d]).sum();
                assert!(s.abs() < 1e-12, "gradients sum to zero at {uvw:?}");
            }
        }
    }

    #[test]
    fn tabulate_matches_shape_functions() {
        let pts2 = [[0.0, 0.0, 0.0], [0.3, 0.7, 0.0], [1.0, 0.5, 0.0], [0.1127, 0.8873, 0.0]];
        let pts3 = [[0.0, 0.0, 0.0], [0.3, 0.7, 0.2], [1.0, 0.5, 1.0], [0.1127, 0.5, 0.8873]];
        check_tabulate(&QUADRANGLE4, &pts2);
        check_tabulate(&QUADRANGLE9, &pts2);
        check_tabulate(&HEXAHEDRON8, &pts3);
        check_tabulate(&HEXAHEDRON27, &pts3);
        check_tabulate(&TRIANGLE6, &[[0.2, 0.3, 0.0], [0.0, 1.0, 0.0]]);
        let (v, g) = QUADRANGLE9.tabulate(&[]);
        assert!(v.is_empty() && g.is_empty());
    }

    #[test]
    #[should_panic(expected = "outside the reference element")]
    fn tabulate_outside_panics() {
        let _ = QUADRANGLE4.tabulate(&[[1.5, 0.5, 0.0]]);
    }
}
