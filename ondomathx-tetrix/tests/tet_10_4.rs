use nalgebra::{DMatrix, DVector};
use ondomathx_tetrix::element::tet_4;
use ondomathx_tetrix::element::tet_10::d_shape_function;
use ondomathx_tetrix::element::tet_10_4::{gradient_interpolation, transpose_gradient_interpolation};

const TOL: f64 = 1e-14;

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

fn gradient_ref(coeff: &[f64; 10], output: &mut [f64; 12]) {
    for j in 0..4 {
        for d in 0..3 {
            output[j * 3 + d] = (0..10)
                .map(|i| coeff[i] * d_shape_function(i, d, &tet_4::POINTS[j]))
                .sum();
        }
    }
}

fn transpose_ref(input: &[f64; 12], output: &mut [f64; 10]) {
    for i in 0..10 {
        output[i] = (0..4)
            .flat_map(|j| (0..3).map(move |d| (j, d)))
            .map(|(j, d)| input[j * 3 + d] * d_shape_function(i, d, &tet_4::POINTS[j]))
            .sum();
    }
}

fn build_nalgebra_matrix() -> DMatrix<f64> {
    DMatrix::from_fn(12, 10, |row, col| {
        let j = row / 3;
        let d = row % 3;
        d_shape_function(col, d, &tet_4::POINTS[j])
    })
}

// ---------------------------------------------------------------------------
// gradient_interpolation: each basis vector e_i
// ---------------------------------------------------------------------------
#[test]
fn test_gradient_interpolation_basis_vectors() {
    let mut coeff = [0.0_f64; 10];
    let mut computed = [0.0_f64; 12];
    let mut expected = [0.0_f64; 12];

    for i in 0..10 {
        coeff.fill(0.0);
        coeff[i] = 1.0;

        gradient_interpolation(&coeff, &mut computed);
        gradient_ref(&coeff, &mut expected);

        for j in 0..4 {
            for d in 0..3 {
                let c = computed[j * 3 + d];
                let e = expected[j * 3 + d];
                assert!(
                    (c - e).abs() < TOL,
                    "gradient_interpolation(e_{i}): qp {j}, dir {d}: \
                     computed={c:.14e}, expected={e:.14e}, Δ={:.2e}",
                    (c - e).abs()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// transpose_gradient_interpolation: each basis vector e_k of the 12-d space
// ---------------------------------------------------------------------------
#[test]
fn test_transpose_gradient_interpolation() {
    let mut input = [0.0_f64; 12];
    let mut computed = [0.0_f64; 10];
    let mut expected = [0.0_f64; 10];

    for j in 0..4_usize {
        for d in 0..3_usize {
            input.fill(0.0);
            input[j * 3 + d] = 1.0;

            transpose_gradient_interpolation(&input, &mut computed);
            transpose_ref(&input, &mut expected);

            for i in 0..10 {
                let c = computed[i];
                let e = expected[i];
                assert!(
                    (c - e).abs() < TOL,
                    "transpose_gradient_interpolation(e_({j},{d})): shape fn {i}: \
                     computed={c:.14e}, expected={e:.14e}, Δ={:.2e}",
                    (c - e).abs()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// nalgebra matrix correctness
// ---------------------------------------------------------------------------
#[test]
fn test_nalgebra_matrix_gradient() {
    let mat = build_nalgebra_matrix(); // 12×10

    for i in 0..10 {
        let mut coeff = [0.0_f64; 10];
        coeff[i] = 1.0;
        let mut expected = [0.0_f64; 12];
        gradient_ref(&coeff, &mut expected);

        let vec_in = DVector::from_fn(10, |k, _| coeff[k]);
        let vec_out = &mat * &vec_in;

        for k in 0..12 {
            let c = vec_out[k];
            let e = expected[k];
            assert!(
                (c - e).abs() < TOL,
                "nalgebra M·e_{i}: slot {k}: got={c:.14e}, expected={e:.14e}, Δ={:.2e}",
                (c - e).abs()
            );
        }
    }
}

#[test]
fn test_nalgebra_matrix_transpose() {
    let mat = build_nalgebra_matrix(); // 12×10

    for j in 0..4_usize {
        for d in 0..3_usize {
            let mut input = [0.0_f64; 12];
            input[j * 3 + d] = 1.0;
            let mut expected = [0.0_f64; 10];
            transpose_ref(&input, &mut expected);

            let vec_in = DVector::from_fn(12, |k, _| input[k]);
            let vec_out = mat.tr_mul(&vec_in);

            for i in 0..10 {
                let c = vec_out[i];
                let e = expected[i];
                assert!(
                    (c - e).abs() < TOL,
                    "nalgebra M^T·e_({j},{d}): shape fn {i}: got={c:.14e}, expected={e:.14e}, Δ={:.2e}",
                    (c - e).abs()
                );
            }
        }
    }
}
