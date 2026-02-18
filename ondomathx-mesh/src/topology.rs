use crate::reference_elements::*;
use std::cmp::Ordering;

#[derive(Clone, Default, Debug, Hash)]
pub struct Edge {
    indexes: [usize; 2],
    sorted_indexes: [usize; 2],
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sorted_indexes.cmp(&other.sorted_indexes)
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.sorted_indexes == other.sorted_indexes
    }
}

impl Eq for Edge {}

impl Edge {
    pub fn new(indexes: [usize; 2]) -> Edge {
        let mut sorted_indexes = indexes;

        if indexes[1] < indexes[0] {
            sorted_indexes.reverse();
        }

        Edge {
            indexes,
            sorted_indexes,
        }
    }

    pub fn get_orientation(&self) -> usize {
        if self.indexes[0] == self.sorted_indexes[0] {
            return 0;
        } else {
            return 1;
        }
    }

    pub fn get_relative_orientation(edge_1: &Self, edge_2: &Self) -> usize {
        if edge_1.indexes[0] == edge_2.indexes[0] {
            return 0;
        } else {
            return 1;
        }
    }
}

#[derive(Clone, Default, Debug, Hash)]
pub struct Face {
    indexes: [usize; 4],
    sorted_indexes: [usize; 4],
}

impl Ord for Face {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sorted_indexes.cmp(&other.sorted_indexes)
    }
}

impl PartialOrd for Face {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Face {
    fn eq(&self, other: &Self) -> bool {
        self.sorted_indexes == other.sorted_indexes
    }
}

impl Eq for Face {}

impl Face {
    // Remainder: The convention is that repeated last indexes means that the face is a triangle
    fn sort_face_indexes(indexes: &[usize]) -> [usize; 4] {
        let mut sorted_indexes = [indexes[0], indexes[1], indexes[2], indexes[3]];

        //The face is triangular
        if Self::is_triangular_from_indexes(indexes) {
            sorted_indexes[0..3].sort();
            sorted_indexes[3] = sorted_indexes[2];
        } else {
            let mut i_min = 0;

            for k in 1..4 {
                if indexes[k] < indexes[i_min] {
                    i_min = k;
                }
            }

            sorted_indexes.rotate_left(i_min);

            if sorted_indexes[3] < sorted_indexes[1] {
                (sorted_indexes[1], sorted_indexes[3]) = (sorted_indexes[3], sorted_indexes[1]);
            }
        }

        return sorted_indexes;
    }

    pub fn new(indexes: [usize; 4]) -> Face {
        let sorted_indexes = Self::sort_face_indexes(&indexes);

        Face {
            indexes: indexes,
            sorted_indexes: sorted_indexes,
        }
    }

    fn is_triangular_from_indexes(i: &[usize]) -> bool {
        if i[2] == i[3] {
            return true;
        }
        false
    }

    pub fn is_triangular(&self) -> bool {
        return Self::is_triangular_from_indexes(&self.indexes);
    }

    //Recover the number of the orientation of the face f2 relative to face f1
    pub fn get_orientation(&self) -> usize {
        if self.is_triangular() {
            return Self::get_orientation_in_triangle_from_indices(
                &self.sorted_indexes,
                &self.indexes,
            );
        } else {
            return Self::get_orientation_in_quadrangle_from_indices(
                &self.sorted_indexes,
                &self.indexes,
            );
        }
    }

    fn get_orientation_in_triangle_from_indices(
        indexes_1: &[usize; 4],
        indexes_2: &[usize; 4],
    ) -> usize {
        for o in 0..6 {
            if indexes_1[0] == indexes_2[ORIENTATIONS_TRIANGLE[o][0]]
                && indexes_1[1] == indexes_2[ORIENTATIONS_TRIANGLE[o][1]]
            {
                return o;
            }
        }

        panic!("An orientation has not been found, which should be impossible!");
    }

    fn get_orientation_in_quadrangle_from_indices(
        indexes_1: &[usize; 4],
        indexes_2: &[usize; 4],
    ) -> usize {
        for o in 0..8 {
            if indexes_1[0] == indexes_2[ORIENTATIONS_QUADRANGLE[o][0]]
                && indexes_1[1] == indexes_2[ORIENTATIONS_QUADRANGLE[o][1]]
                && indexes_1[2] == indexes_2[ORIENTATIONS_QUADRANGLE[o][2]]
            {
                return o;
            }
        }

        panic!("An orientation has not been found, which should be impossible!");
    }

    pub fn get_relative_orientation(face_1: &Self, face_2: &Self) -> usize {
        assert!(face_1.is_triangular() == face_2.is_triangular());

        if face_1.is_triangular() {
            return Self::get_orientation_in_triangle_from_indices(
                &face_1.indexes,
                &face_2.indexes,
            );
        } else {
            return Self::get_orientation_in_quadrangle_from_indices(
                &face_1.indexes,
                &face_2.indexes,
            );
        }
    }
}
