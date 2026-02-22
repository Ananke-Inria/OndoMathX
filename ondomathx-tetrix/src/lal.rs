/// Dense vector (equivalent to LAL::Vector).
pub type Vector = Vec<f64>;

/// Diagonal matrix stored as its diagonal entries (equivalent to LAL::DiagonalMatrix).
pub type DiagonalMatrix = Vec<f64>;

pub fn new_vector(size: usize) -> Vector {
    vec![0.0; size]
}

pub fn new_diagonal_matrix(size: usize) -> DiagonalMatrix {
    vec![0.0; size]
}

pub fn dimension(v: &Vector) -> usize {
    v.len()
}

pub fn scale(alpha: f64, v: &mut Vector) {
    for x in v.iter_mut() {
        *x *= alpha;
    }
}

/// Accumulate value to the (i, i) diagonal entry (i == j for mass matrices).
pub fn add_interaction(m: &mut DiagonalMatrix, i: usize, _j: usize, value: f64) {
    m[i] += value;
}
