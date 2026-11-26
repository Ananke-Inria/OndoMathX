use nalgebra::{DMatrix, DVector};
use ondomathx_tetrix::element::tet_14;
use ondomathx_tetrix::element::tet_15::d_shape_function;
use ondomathx_tetrix::element::tet_15_14::{gradient_interpolation, transpose_gradient_interpolation};

// The precomputed coefficients in tet_15_14 have ~10 significant digits,
// so floating-point agreement up to 1e-8 is expected.
const TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Reference implementations
//
// For a function f = Σ_i coeff[i] · φ_i defined in tet_15, its gradient at
// the j-th quadrature point q_j of tet_14 in direction d is:
//
//   grad_ref[j * 3 + d] = Σ_i coeff[i] · ∂φ_i/∂x_d(q_j)
//
// The transpose maps a field z defined at the 42 gradient slots back to the
// 15-dimensional shape-function space:
//
//   transpose_ref[i] = Σ_{j=0..13} Σ_{d=0..2} z[j*3+d] · ∂φ_i/∂x_d(q_j)
// ---------------------------------------------------------------------------

fn gradient_ref(coeff: &[f64; 15], output: &mut [f64; 42]) {
    for j in 0..14 {
        for d in 0..3 {
            output[j * 3 + d] = (0..15)
                .map(|i| coeff[i] * d_shape_function(i, d, &tet_14::POINTS[j]))
                .sum();
        }
    }
}

/// Build the 42×15 gradient matrix M where M[j*3+d, i] = ∂φ_i/∂x_d(q_j).
fn build_nalgebra_matrix() -> DMatrix<f64> {
    DMatrix::from_fn(42, 15, |row, col| {
        let j = row / 3;
        let d = row % 3;
        d_shape_function(col, d, &tet_14::POINTS[j])
    })
}

fn transpose_ref(input: &[f64; 42], output: &mut [f64; 15]) {
    for i in 0..15 {
        output[i] = (0..14)
            .flat_map(|j| (0..3).map(move |d| (j, d)))
            .map(|(j, d)| input[j * 3 + d] * d_shape_function(i, d, &tet_14::POINTS[j]))
            .sum();
    }
}

// ---------------------------------------------------------------------------
// gradient_interpolation: each basis vector e_i
//
// Checks that gradient_interpolation(e_i, ·) equals the direct evaluation
// of ∂φ_i/∂x_d at each tet_14 quadrature point.
// Output layout: output[j * 3 + d] for j ∈ 0..14, d ∈ {0,1,2}.
// ---------------------------------------------------------------------------
#[test]
fn test_gradient_interpolation_basis_vectors() {
    let mut coeff = [0.0_f64; 15];
    let mut computed = [0.0_f64; 42];
    let mut expected = [0.0_f64; 42];

    for i in 0..15 {
        coeff.fill(0.0);
        coeff[i] = 1.0;

        gradient_interpolation(&coeff, &mut computed);
        gradient_ref(&coeff, &mut expected);

        for j in 0..14 {
            for d in 0..3 {
                let c = computed[j * 3 + d];
                let e = expected[j * 3 + d];
                assert!(
                    (c - e).abs() < TOL,
                    "gradient_interpolation(e_{i}): qp {j}, dir {d}: \
                     computed={c:.10e}, expected={e:.10e}, Δ={:.2e}",
                    (c - e).abs()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// transpose_gradient_interpolation: each basis vector e_k of the 42-d space
//
// For each unit vector e_k (k = j*3+d), the transpose maps to:
//   output[i] = ∂φ_i/∂x_d(q_j)
// which is exactly d_shape_function(i, d, POINTS_14[j]).
// ---------------------------------------------------------------------------
#[test]
fn test_transpose_gradient_interpolation() {
    let mut input = [0.0_f64; 42];
    let mut computed = [0.0_f64; 15];
    let mut expected = [0.0_f64; 15];

    for j in 0..14usize {
        for d in 0..3usize {
            input.fill(0.0);
            input[j * 3 + d] = 1.0;

            transpose_gradient_interpolation(&input, &mut computed);
            transpose_ref(&input, &mut expected);

            for i in 0..15 {
                let c = computed[i];
                let e = expected[i];
                assert!(
                    (c - e).abs() < TOL,
                    "transpose_gradient_interpolation(e_({j},{d})): shape fn {i}: \
                     computed={c:.10e}, expected={e:.10e}, Δ={:.2e}",
                    (c - e).abs()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// nalgebra matrix correctness
//
// Verifies that build_nalgebra_matrix() (used in benches/tet_15_14.rs) gives
// results that agree with the reference implementations on all basis vectors.
// ---------------------------------------------------------------------------
#[test]
fn test_nalgebra_matrix_gradient() {
    let mat = build_nalgebra_matrix(); // 42×15

    for i in 0..15 {
        let mut coeff = [0.0_f64; 15];
        coeff[i] = 1.0;
        let mut expected = [0.0_f64; 42];
        gradient_ref(&coeff, &mut expected);

        let vec_in = DVector::from_fn(15, |k, _| coeff[k]);
        let vec_out = &mat * &vec_in;

        for k in 0..42 {
            let c = vec_out[k];
            let e = expected[k];
            assert!(
                (c - e).abs() < TOL,
                "nalgebra M·e_{i}: slot {k}: got={c:.10e}, expected={e:.10e}, Δ={:.2e}",
                (c - e).abs()
            );
        }
    }
}

#[test]
fn test_nalgebra_matrix_transpose() {
    let mat = build_nalgebra_matrix(); // 42×15

    for j in 0..14usize {
        for d in 0..3usize {
            let mut input = [0.0_f64; 42];
            input[j * 3 + d] = 1.0;
            let mut expected = [0.0_f64; 15];
            transpose_ref(&input, &mut expected);

            let vec_in = DVector::from_fn(42, |k, _| input[k]);
            let vec_out = mat.tr_mul(&vec_in);

            for i in 0..15 {
                let c = vec_out[i];
                let e = expected[i];
                assert!(
                    (c - e).abs() < TOL,
                    "nalgebra M^T·e_({j},{d}): shape fn {i}: got={c:.10e}, expected={e:.10e}, Δ={:.2e}",
                    (c - e).abs()
                );
            }
        }
    }
}
