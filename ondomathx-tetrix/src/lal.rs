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

/// Mass-weighted L² norm: sqrt( u^T M u ) for a diagonal mass matrix M.
pub fn mass_norm(u: &Vector, mass: &DiagonalMatrix) -> f64 {
    u.iter()
        .zip(mass.iter())
        .map(|(&ui, &mi)| mi * ui * ui)
        .sum::<f64>()
        .sqrt()
}

pub fn infinite_norm(v: &Vector) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Power iteration for the generalized eigenvalue problem  K u = λ M u.
///
/// Computes the spectral radius of M⁻¹ K by repeatedly applying:
///   v ← M⁻¹ K v,   then normalising by the ∞-norm.
///
/// Arguments:
/// - `solve`   : applies M⁻¹ in-place  (v ← M⁻¹ v)
/// - `mlt_add` : applies K              (w ← K v)
/// - `size`    : length of the eigenvector (DG or CG)
/// - `tol`     : relative convergence tolerance
/// - `n_iter_max` : maximum number of iterations
///
/// Returns `(lambda, n_iter, residue)`.
pub fn power_iteration<S, M>(
    mut solve: S,
    mut mlt_add: M,
    size: usize,
    tol: f64,
    n_iter_max: usize,
) -> (f64, usize, f64)
where
    S: FnMut(&mut Vector),
    M: FnMut(&Vector, &mut Vector),
{
    // Deterministic non-trivial initialisation (mirrors C++ reference)
    let mut v1: Vector = (0..size)
        .map(|i| (i % 7 + 3 * (i % 3) + (i % 2) + i % 20) as f64)
        .collect();
    let mut v2 = vec![0.0_f64; size];

    let mut lambda1 = 0.0_f64;
    let mut residue = f64::MAX;
    let mut n_iter  = 0_usize;

    loop {
        let lambda0 = lambda1;   // local copy; always read below

        mlt_add(&v1, &mut v2);  // v2 ← K v1
        solve(&mut v2);          // v2 ← M⁻¹ K v1
        v1.copy_from_slice(&v2);

        lambda1 = infinite_norm(&v1);
        for x in v1.iter_mut() { *x /= lambda1; }

        if lambda0 != 0.0 {
            residue = (lambda1 - lambda0).abs() / lambda0.abs();
        }
        n_iter += 1;

        if residue <= tol || n_iter >= n_iter_max { break; }
    }

    (lambda1, n_iter, residue)
}
