use nalgebra::Matrix3;
use ondomathx_tetrix::finite_element::tet_10::{build_local_stiffness, mlt_stiffness};

const TOL_K: f64 = 1e-14;

fn test_matrix() -> Matrix3<f64> {
    Matrix3::new(2.0, 0.5, 0.1, 0.5, 3.0, 0.2, 0.1, 0.2, 1.5)
}

// ---------------------------------------------------------------------------
// 1. mlt_stiffness(e_i) == K · e_i  for each basis vector
// ---------------------------------------------------------------------------
#[test]
fn test_mlt_stiffness_basis_vectors() {
    let sym_grad = test_matrix();
    let k = build_local_stiffness(&sym_grad);

    let mut input = [0.0_f64; 10];
    let mut computed = [0.0_f64; 10];

    for i in 0..10 {
        input.fill(0.0);
        input[i] = 1.0;

        mlt_stiffness(&input, &sym_grad, &mut computed);

        let expected = k.column(i);
        for row in 0..10 {
            let c = computed[row];
            let e = expected[row];
            assert!(
                (c - e).abs() < TOL_K,
                "mlt_stiffness(e_{i}): row {row}: computed={c:.14e}, expected={e:.14e}, Δ={:.2e}",
                (c - e).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Symmetry — K must be symmetric (K[i,j] == K[j,i])
// ---------------------------------------------------------------------------
#[test]
fn test_stiffness_symmetry() {
    let sym_grad = test_matrix();
    let k = build_local_stiffness(&sym_grad);

    for i in 0..10 {
        for j in 0..10 {
            let diff = (k[(i, j)] - k[(j, i)]).abs();
            assert!(
                diff < TOL_K,
                "K not symmetric: K[{i},{j}]={:.14e}, K[{j},{i}]={:.14e}, Δ={:.2e}",
                k[(i, j)],
                k[(j, i)],
                diff
            );
        }
    }
}
