extern crate anamesh;

#[cfg(test)]
mod tests {
    use anamesh::topology::*;

    // === Edge ===

    #[test]
    fn edge_construction_and_equality() {
        let e1 = Edge::new([1, 2]);
        let e2 = Edge::new([2, 1]);
        assert_eq!(e1, e2);
    }

    #[test]
    fn edge_inequality() {
        let e1 = Edge::new([1, 2]);
        let e2 = Edge::new([1, 3]);
        assert_ne!(e1, e2);
    }

    #[test]
    fn edge_orientation_canonical() {
        // [1, 2] is already sorted -> orientation 0
        let e = Edge::new([1, 2]);
        assert_eq!(e.get_orientation(), 0);
    }

    #[test]
    fn edge_orientation_reversed() {
        // [2, 1] is not sorted -> orientation 1
        let e = Edge::new([2, 1]);
        assert_eq!(e.get_orientation(), 1);
    }

    #[test]
    fn edge_relative_orientation_same() {
        let e1 = Edge::new([1, 2]);
        let e2 = Edge::new([1, 2]);
        assert_eq!(Edge::get_relative_orientation(&e1, &e2), 0);
    }

    #[test]
    fn edge_relative_orientation_reversed() {
        let e1 = Edge::new([1, 2]);
        let e2 = Edge::new([2, 1]);
        assert_eq!(Edge::get_relative_orientation(&e1, &e2), 1);
    }

    #[test]
    fn edge_ordering() {
        let e1 = Edge::new([1, 3]);
        let e2 = Edge::new([2, 4]);
        assert!(e1 < e2);
    }

    // === Face ===

    #[test]
    fn face_triangular_construction() {
        // Convention: repeated last index means triangular
        let f = Face::new([0, 1, 2, 2]);
        assert!(f.is_triangular());
    }

    #[test]
    fn face_quad_construction() {
        let f = Face::new([0, 1, 2, 3]);
        assert!(!f.is_triangular());
    }

    #[test]
    fn face_triangular_equality() {
        // Permutations of a triangular face should be equal
        let f1 = Face::new([0, 1, 2, 2]);
        let f2 = Face::new([1, 2, 0, 0]);
        let f3 = Face::new([2, 0, 1, 1]);
        assert_eq!(f1, f2);
        assert_eq!(f1, f3);
    }

    #[test]
    fn face_quad_equality() {
        // Rotations of a quad face should be equal
        let f1 = Face::new([0, 1, 2, 3]);
        let f2 = Face::new([1, 2, 3, 0]);
        let f3 = Face::new([2, 3, 0, 1]);
        let f4 = Face::new([3, 0, 1, 2]);
        assert_eq!(f1, f2);
        assert_eq!(f1, f3);
        assert_eq!(f1, f4);
    }

    #[test]
    fn face_quad_reversed_equality() {
        // A quad and its reversal should also be equal (same sorted canonical form)
        let f1 = Face::new([0, 1, 2, 3]);
        let f2 = Face::new([0, 3, 2, 1]);
        assert_eq!(f1, f2);
    }

    #[test]
    fn face_inequality() {
        let f1 = Face::new([0, 1, 2, 3]);
        let f2 = Face::new([0, 1, 2, 4]);
        assert_ne!(f1, f2);
    }

    #[test]
    fn face_triangular_orientation() {
        // Orientation 0 = canonical (sorted) order
        let f = Face::new([0, 1, 2, 2]);
        let o = f.get_orientation();
        // The sorted form is [0,1,2,2], same as input, so orientation 0
        assert_eq!(o, 0);
    }

    #[test]
    fn face_triangular_orientation_rotated() {
        // [1, 2, 0, 0] -> sorted is [0, 1, 2, 2]
        // In the original, position of 0 is at index 2, position of 1 is at index 0
        // Orientation should be 1 (cyclic rotation)
        let f = Face::new([1, 2, 0, 0]);
        let o = f.get_orientation();
        assert_eq!(o, 2);
    }

    #[test]
    fn face_relative_orientation_triangular() {
        let f1 = Face::new([0, 1, 2, 2]);
        let f2 = Face::new([1, 2, 0, 0]);
        let o = Face::get_relative_orientation(&f1, &f2);
        // f2's ordering relative to f1
        assert!(o < 6);
    }

    #[test]
    fn face_relative_orientation_quad() {
        let f1 = Face::new([0, 1, 2, 3]);
        let f2 = Face::new([1, 2, 3, 0]);
        let o = Face::get_relative_orientation(&f1, &f2);
        assert!(o < 8);
    }

    #[test]
    fn face_ordering() {
        let f1 = Face::new([0, 1, 2, 2]);
        let f2 = Face::new([1, 2, 3, 3]);
        assert!(f1 < f2);
    }
}
