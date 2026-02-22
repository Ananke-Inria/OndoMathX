use nalgebra::Matrix3;
use tetrix::finite_element::tet_10b::{
    build_local_stiffness, d_shape_function, mlt_stiffness, shape_function, POINTS,
};

const TOL: f64 = 1e-12;
const TOL_K: f64 = 1e-12;

// ---------------------------------------------------------------------------
// 1. P2 sub-basis Kronecker delta: φ_i(P_j) = δ_{ij}  for  i, j < 10
// ---------------------------------------------------------------------------
#[test]
fn test_p2_kronecker_delta() {
    for j in 0..10 {
        let pt = POINTS[j];
        for i in 0..10 {
            let val = shape_function(i, &pt);
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < TOL,
                "φ_{i}(P_{j}) = {val:.2e}, expected {expected}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Bubble functions equal 1 at their own evaluation node
//    φ_i(P_i) = 1  for  i = 10..14
// ---------------------------------------------------------------------------
#[test]
fn test_bubble_node_values() {
    for i in 10..15 {
        let val = shape_function(i, &POINTS[i]);
        assert!(
            (val - 1.0).abs() < TOL,
            "φ_{i}(P_{i}) = {val:.2e}, expected 1.0"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Bubble functions vanish at all P2 nodes (i < 10)
//    φ_i(P_j) = 0  for  i ≥ 10,  j < 10
// ---------------------------------------------------------------------------
#[test]
fn test_bubbles_vanish_at_p2_nodes() {
    for j in 0..10 {
        let pt = POINTS[j];
        for i in 10..15 {
            let val = shape_function(i, &pt);
            assert!(val.abs() < TOL, "φ_{i}(P_{j}) = {val:.2e}, expected 0.0");
        }
    }
}

// ---------------------------------------------------------------------------
// 4. P2 partition of unity: Σ_{i<10} φ_i(P) = 1 at interior points
// ---------------------------------------------------------------------------
#[test]
fn test_p2_partition_of_unity() {
    let pts: [[f64; 3]; 5] = [
        [0.1, 0.2, 0.3],
        [0.25, 0.25, 0.25],
        [0.1, 0.1, 0.1],
        [0.5, 0.25, 0.1],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
    ];
    for pt in &pts {
        let sum: f64 = (0..10).map(|i| shape_function(i, pt)).sum();
        assert!(
            (sum - 1.0).abs() < TOL,
            "P2 partition of unity at {pt:?}: sum = {sum:.2e}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. P2 gradient sum = 0: Σ_{i<10} ∂φ_i/∂d(P) = 0  for d = 0, 1, 2
// ---------------------------------------------------------------------------
#[test]
fn test_p2_gradient_sum_is_zero() {
    let pts: [[f64; 3]; 3] = [[0.1, 0.2, 0.3], [0.25, 0.25, 0.25], [0.15, 0.1, 0.05]];
    for pt in &pts {
        for d in 0..3 {
            let sum: f64 = (0..10).map(|i| d_shape_function(i, d, pt)).sum();
            assert!(
                sum.abs() < TOL,
                "Σ_{{i<10}} ∂φ_i/∂{d} at {pt:?} = {sum:.2e}, expected 0"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Analytical gradients match centred finite differences (all 15 functions)
// ---------------------------------------------------------------------------
#[test]
fn test_gradients_vs_finite_differences() {
    let pt = [0.15, 0.2, 0.25];
    let h = 1e-7;
    let tol_fd = 1e-6;
    for i in 0..15 {
        for d in 0..3 {
            let mut pp = pt;
            let mut pm = pt;
            pp[d] += h;
            pm[d] -= h;
            let fd = (shape_function(i, &pp) - shape_function(i, &pm)) / (2.0 * h);
            let analytic = d_shape_function(i, d, &pt);
            assert!(
                (fd - analytic).abs() < tol_fd,
                "φ_{i} ∂/∂{d} at {pt:?}: FD={fd:.8e}, analytic={analytic:.8e}, Δ={:.2e}",
                (fd - analytic).abs()
            );
        }
    }
}

fn test_matrix() -> Matrix3<f64> {
    Matrix3::new(2.0, 0.5, 0.1, 0.5, 3.0, 0.2, 0.1, 0.2, 1.5)
}

// ---------------------------------------------------------------------------
// 7. mlt_stiffness(e_i) == K · e_i  for each basis vector
// ---------------------------------------------------------------------------
#[test]
fn test_mlt_stiffness_basis_vectors() {
    let sym_grad = test_matrix();
    let k = build_local_stiffness(&sym_grad);

    let mut input = [0.0_f64; 15];
    let mut computed = [0.0_f64; 15];

    for i in 0..15 {
        input.fill(0.0);
        input[i] = 1.0;

        mlt_stiffness(&input, &sym_grad, &mut computed);

        let expected = k.column(i);
        for row in 0..15 {
            let c = computed[row];
            let e = expected[row];
            assert!(
                (c - e).abs() < TOL_K,
                "mlt_stiffness(e_{i}): row {row}: computed={c:.12e}, expected={e:.12e}, Δ={:.2e}",
                (c - e).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Symmetry — K must be symmetric (K[i,j] == K[j,i])
// ---------------------------------------------------------------------------
#[test]
fn test_stiffness_symmetry() {
    let sym_grad = test_matrix();
    let k = build_local_stiffness(&sym_grad);

    for i in 0..15 {
        for j in 0..15 {
            let diff = (k[(i, j)] - k[(j, i)]).abs();
            assert!(
                diff < TOL_K,
                "K not symmetric: K[{i},{j}]={:.12e}, K[{j},{i}]={:.12e}, Δ={:.2e}",
                k[(i, j)],
                k[(j, i)],
                diff
            );
        }
    }
}
