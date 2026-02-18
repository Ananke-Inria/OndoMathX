extern crate rouage;

#[cfg(test)]
mod tests {
    use rouage::reference_elements::*;

    // === Topology counts ===

    #[test]
    fn vertex_topology_counts() {
        assert_eq!(VERTEX.get_nfaces(), 0);
        assert_eq!(VERTEX.get_nedges(), 0);
        assert_eq!(VERTEX.get_nvertexes(), 1);
        assert_eq!(VERTEX.get_dim(), 0);
    }

    #[test]
    fn edge_topology_counts() {
        assert_eq!(EDGE.get_nfaces(), 0);
        assert_eq!(EDGE.get_nedges(), 1);
        assert_eq!(EDGE.get_nvertexes(), 2);
        assert_eq!(EDGE.get_dim(), 1);
    }

    #[test]
    fn triangle_topology_counts() {
        assert_eq!(TRIANGLE.get_nfaces(), 1);
        assert_eq!(TRIANGLE.get_nedges(), 3);
        assert_eq!(TRIANGLE.get_nvertexes(), 3);
        assert_eq!(TRIANGLE.get_dim(), 2);
    }

    #[test]
    fn quadrangle_topology_counts() {
        assert_eq!(QUADRANGLE.get_nfaces(), 1);
        assert_eq!(QUADRANGLE.get_nedges(), 4);
        assert_eq!(QUADRANGLE.get_nvertexes(), 4);
        assert_eq!(QUADRANGLE.get_dim(), 2);
    }

    #[test]
    fn hexahedron_topology_counts() {
        assert_eq!(HEXAHEDRON.get_nfaces(), 6);
        assert_eq!(HEXAHEDRON.get_nedges(), 12);
        assert_eq!(HEXAHEDRON.get_nvertexes(), 8);
        assert_eq!(HEXAHEDRON.get_dim(), 3);
    }

    #[test]
    fn tetrahedron_topology_counts() {
        assert_eq!(TETRAHEDRON.get_nfaces(), 4);
        assert_eq!(TETRAHEDRON.get_nedges(), 6);
        assert_eq!(TETRAHEDRON.get_nvertexes(), 4);
        assert_eq!(TETRAHEDRON.get_dim(), 3);
    }

    // === Edge: is_inside ===

    #[test]
    fn edge_is_inside() {
        assert!(EDGE.is_inside(&[0.0, 0.0, 0.0]));
        assert!(EDGE.is_inside(&[0.5, 0.0, 0.0]));
        assert!(EDGE.is_inside(&[1.0, 0.0, 0.0]));

        assert!(!EDGE.is_inside(&[-0.001, 0.0, 0.0]));
        assert!(!EDGE.is_inside(&[1.001, 0.0, 0.0]));
        assert!(!EDGE.is_inside(&[0.5, 0.1, 0.0]));
        assert!(!EDGE.is_inside(&[0.5, 0.0, 0.1]));
    }

    #[test]
    fn edge_is_on_vertex() {
        assert!(EDGE.is_on_vertex(&[0.0, 0.0, 0.0], 0));
        assert!(EDGE.is_on_vertex(&[1.0, 0.0, 0.0], 1));
        assert!(!EDGE.is_on_vertex(&[0.5, 0.0, 0.0], 0));
        assert!(!EDGE.is_on_vertex(&[0.5, 0.0, 0.0], 1));
    }

    #[test]
    fn edge_is_on_any_vertex() {
        assert!(EDGE.is_on_any_vertex(&[0.0, 0.0, 0.0]));
        assert!(EDGE.is_on_any_vertex(&[1.0, 0.0, 0.0]));
        assert!(!EDGE.is_on_any_vertex(&[0.5, 0.0, 0.0]));
    }

    #[test]
    fn edge_uvw_edge_transform() {
        // Identity orientation (o=0)
        assert_eq!(
            EDGE.uvw_edge_transform(&[0.3, 0.0, 0.0], 0, 0),
            [0.3, 0.0, 0.0]
        );
        // Reversed orientation (o=1)
        assert_eq!(
            EDGE.uvw_edge_transform(&[0.3, 0.0, 0.0], 0, 1),
            [0.7, 0.0, 0.0]
        );
        // Endpoints stay fixed or swap
        assert_eq!(
            EDGE.uvw_edge_transform(&[0.0, 0.0, 0.0], 0, 1),
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            EDGE.uvw_edge_transform(&[1.0, 0.0, 0.0], 0, 1),
            [0.0, 0.0, 0.0]
        );
    }

    // === Triangle: is_inside ===

    #[test]
    fn triangle_is_inside() {
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
    }

    #[test]
    fn triangle_uvw_edge_transform() {
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
    fn triangle_is_on_edge() {
        // Edge 0: (0,0) -> (1,0)
        assert!(TRIANGLE.is_on_edge(&[0.5, 0.0, 0.0], 0));
        assert!(!TRIANGLE.is_on_edge(&[0.25, 0.25, 0.0], 0));

        // Edge 1: (1,0) -> (0,1)
        assert!(TRIANGLE.is_on_edge(&[0.5, 0.5, 0.0], 1));
        assert!(!TRIANGLE.is_on_edge(&[0.0, 0.5, 0.0], 1));

        // Edge 2: (0,0) -> (0,1)
        assert!(TRIANGLE.is_on_edge(&[0.0, 0.5, 0.0], 2));
        assert!(!TRIANGLE.is_on_edge(&[0.5, 0.0, 0.0], 2));
    }

    #[test]
    fn triangle_is_on_any_edge() {
        assert!(TRIANGLE.is_on_any_edge(&[0.5, 0.0, 0.0]));
        assert!(TRIANGLE.is_on_any_edge(&[0.5, 0.5, 0.0]));
        assert!(TRIANGLE.is_on_any_edge(&[0.0, 0.5, 0.0]));
        assert!(!TRIANGLE.is_on_any_edge(&[0.25, 0.25, 0.0]));
    }

    #[test]
    fn triangle_vertex_detection() {
        assert!(TRIANGLE.is_on_any_vertex(&[0.0, 0.0, 0.0]));
        assert!(TRIANGLE.is_on_any_vertex(&[1.0, 0.0, 0.0]));
        assert!(TRIANGLE.is_on_any_vertex(&[0.0, 1.0, 0.0]));
        assert!(!TRIANGLE.is_on_any_vertex(&[0.5, 0.0, 0.0]));
    }

    // === Quadrangle: is_inside ===

    #[test]
    fn quadrangle_is_inside() {
        assert!(QUADRANGLE.is_inside(&[0.0, 0.0, 0.0]));
        assert!(QUADRANGLE.is_inside(&[1.0, 0.0, 0.0]));
        assert!(QUADRANGLE.is_inside(&[1.0, 1.0, 0.0]));
        assert!(QUADRANGLE.is_inside(&[0.0, 1.0, 0.0]));
        assert!(QUADRANGLE.is_inside(&[0.5, 0.5, 0.0]));

        assert!(!QUADRANGLE.is_inside(&[-0.001, 0.5, 0.0]));
        assert!(!QUADRANGLE.is_inside(&[1.001, 0.5, 0.0]));
        assert!(!QUADRANGLE.is_inside(&[0.5, -0.001, 0.0]));
        assert!(!QUADRANGLE.is_inside(&[0.5, 1.001, 0.0]));
        assert!(!QUADRANGLE.is_inside(&[0.5, 0.5, 0.1]));
    }

    #[test]
    fn quadrangle_is_on_edge() {
        // Edge 0: (0,0) -> (1,0) bottom
        assert!(QUADRANGLE.is_on_edge(&[0.5, 0.0, 0.0], 0));
        // Edge 1: (1,0) -> (1,1) right
        assert!(QUADRANGLE.is_on_edge(&[1.0, 0.5, 0.0], 1));
        // Edge 2: (0,1) -> (1,1) top
        assert!(QUADRANGLE.is_on_edge(&[0.5, 1.0, 0.0], 2));
        // Edge 3: (0,0) -> (0,1) left
        assert!(QUADRANGLE.is_on_edge(&[0.0, 0.5, 0.0], 3));

        // Interior point not on any edge
        assert!(!QUADRANGLE.is_on_edge(&[0.5, 0.5, 0.0], 0));
    }

    #[test]
    fn quadrangle_uvw_edge_transform() {
        // Edge 0: identity
        assert_eq!(
            QUADRANGLE.uvw_edge_transform(&[0.3, 0.0, 0.0], 0, 0),
            [0.3, 0.0, 0.0]
        );
        // Edge 0: reversed
        assert_eq!(
            QUADRANGLE.uvw_edge_transform(&[0.3, 0.0, 0.0], 0, 1),
            [0.7, 0.0, 0.0]
        );
    }

    // === Hexahedron: is_inside ===

    #[test]
    fn hexahedron_is_inside() {
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
    }

    #[test]
    fn hexahedron_uvw_edge_transform() {
        assert_eq!(
            HEXAHEDRON.uvw_edge_transform(&[0.0, 0.0, 0.0], 0, 1),
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            HEXAHEDRON.uvw_edge_transform(&[0.0, 0.9, 0.0], 3, 1),
            [0.0, 1.0 - 0.9, 0.0]
        );
    }

    #[test]
    fn hexahedron_uvw_face_transform() {
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
    fn hexahedron_is_on_face() {
        // Face 0 is the z=0 plane: [0,1,2,3]
        assert!(HEXAHEDRON.is_on_face(&[0.5, 0.5, 0.0], 0));
        assert!(!HEXAHEDRON.is_on_face(&[0.5, 0.5, 0.5], 0));

        // Face 5 is the z=1 plane: [4,5,6,7]
        assert!(HEXAHEDRON.is_on_face(&[0.5, 0.5, 1.0], 5));
        assert!(!HEXAHEDRON.is_on_face(&[0.5, 0.5, 0.5], 5));
    }

    #[test]
    fn hexahedron_is_on_any_face() {
        // Point on z=0 face
        assert!(HEXAHEDRON.is_on_any_face(&[0.5, 0.5, 0.0]));
        // Interior point
        assert!(!HEXAHEDRON.is_on_any_face(&[0.5, 0.5, 0.5]));
    }

    #[test]
    fn hexahedron_vertex_detection() {
        for v in 0..HEXAHEDRON.get_nvertexes() {
            assert!(HEXAHEDRON.is_on_vertex(&HEXAHEDRON.vertexes[v], v));
            assert!(HEXAHEDRON.is_on_any_vertex(&HEXAHEDRON.vertexes[v]));
        }
        assert!(!HEXAHEDRON.is_on_any_vertex(&[0.5, 0.5, 0.5]));
    }

    // === Tetrahedron: is_inside ===

    #[test]
    fn tetrahedron_is_inside() {
        assert!(TETRAHEDRON.is_inside(&[0.0, 0.0, 0.0]));
        assert!(TETRAHEDRON.is_inside(&[1.0, 0.0, 0.0]));
        assert!(TETRAHEDRON.is_inside(&[0.0, 1.0, 0.0]));
        assert!(TETRAHEDRON.is_inside(&[0.0, 0.0, 1.0]));
        assert!(TETRAHEDRON.is_inside(&[0.25, 0.25, 0.25]));

        assert!(!TETRAHEDRON.is_inside(&[-0.001, 0.0, 0.0]));
        assert!(!TETRAHEDRON.is_inside(&[0.0, -0.001, 0.0]));
        assert!(!TETRAHEDRON.is_inside(&[0.0, 0.0, -0.001]));
        assert!(!TETRAHEDRON.is_inside(&[0.5, 0.5, 0.5]));
    }

    #[test]
    fn tetrahedron_uvw_face_transform() {
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
    }

    #[test]
    fn tetrahedron_is_on_face() {
        // Face 0: [0,1,2] in z=0 plane
        assert!(TETRAHEDRON.is_on_face(&[0.25, 0.25, 0.0], 0));
        assert!(!TETRAHEDRON.is_on_face(&[0.1, 0.1, 0.1], 0));

        // Face 1: [0,1,3] in y=0 plane
        assert!(TETRAHEDRON.is_on_face(&[0.25, 0.0, 0.25], 1));
    }

    #[test]
    fn tetrahedron_vertex_detection() {
        for v in 0..TETRAHEDRON.get_nvertexes() {
            assert!(TETRAHEDRON.is_on_vertex(&TETRAHEDRON.vertexes[v], v));
            assert!(TETRAHEDRON.is_on_any_vertex(&TETRAHEDRON.vertexes[v]));
        }
        assert!(!TETRAHEDRON.is_on_any_vertex(&[0.25, 0.25, 0.25]));
    }

    // === Face triangularity ===

    #[test]
    fn face_triangularity() {
        // Triangle has one face which is triangular
        assert!(TRIANGLE.is_face_triangular(0));

        // Quadrangle has one face which is NOT triangular
        assert!(!QUADRANGLE.is_face_triangular(0));

        // Hexahedron: all faces are quads
        for f in 0..HEXAHEDRON.get_nfaces() {
            assert!(!HEXAHEDRON.is_face_triangular(f));
        }

        // Tetrahedron: all faces are triangles
        for f in 0..TETRAHEDRON.get_nfaces() {
            assert!(TETRAHEDRON.is_face_triangular(f));
        }
    }
}
