extern crate rouage;

#[cfg(test)]
mod tests {
    use rouage::geometric_elements::*;

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
                    j, i, expected, val
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
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        for (i, vtx) in vertices.iter().enumerate() {
            for j in 0..TRIANGLE3.nshape_functions {
                let val = TRIANGLE3.shape_functions[j](vtx);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-12,
                    "TRIANGLE3 shape function {} at vertex {}: expected {}, got {}",
                    j, i, expected, val
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
                pt[0], pt[1]
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
                    j, i, expected, val
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
                    u, v
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
                    j, i, expected, val
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
                    u, v
                );
            }
        }
    }

    // === Consistency checks ===

    #[test]
    fn geometric_element_msh_ids() {
        assert_eq!(EDGE2.msh_id, 1);
        assert_eq!(EDGE3.msh_id, 8);
        assert_eq!(TRIANGLE3.msh_id, 2);
        assert_eq!(QUADRANGLE4.msh_id, 3);
        assert_eq!(QUADRANGLE9.msh_id, 10);
    }

    #[test]
    fn geometric_element_nshape_functions() {
        assert_eq!(EDGE2.nshape_functions, 2);
        assert_eq!(EDGE3.nshape_functions, 3);
        assert_eq!(TRIANGLE3.nshape_functions, 3);
        assert_eq!(QUADRANGLE4.nshape_functions, 4);
        assert_eq!(QUADRANGLE9.nshape_functions, 9);
    }
}
