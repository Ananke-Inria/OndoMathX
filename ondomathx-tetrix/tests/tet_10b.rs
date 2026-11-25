use nalgebra::Matrix3;
use ondomathx_tetrix::finite_element::tet_10b::{
    build_local_stiffness, d_shape_function, mlt_stiffness, mlt_stiffness_opt, shape_function,
    BUBBLE_GRAD_NORM_SQ, POINTS,
};
use ondomathx_tetrix::finite_element::{tet_14};

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

/// Generate `n` deterministic SPD 3×3 matrices via a LCG.
/// Each matrix is L·Lᵀ where L is lower-triangular with positive diagonal.
fn random_spd_matrices(n: usize) -> Vec<Matrix3<f64>> {
    let mut state: u64 = 0x123456789abcdef0;
    let mut next = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) * (2.0 / (1u64 << 53) as f64) - 1.0
    };
    (0..n)
        .map(|_| {
            let (l00, l11, l22) = (next().abs() + 0.5, next().abs() + 0.5, next().abs() + 0.5);
            let (l10, l20, l21) = (next(), next(), next());
            let l = Matrix3::new(l00, 0.0, 0.0, l10, l11, 0.0, l20, l21, l22);
            l * l.transpose()
        })
        .collect()
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

// ---------------------------------------------------------------------------
// 9. mlt_stiffness_opt: K_PP block is exact
//    With zero bubble input, the P2 output must match mlt_stiffness exactly.
// ---------------------------------------------------------------------------
#[test]
fn test_mlt_stiffness_opt_pp_exact() {
    let sym_grad = test_matrix();
    let mut input = [0.0_f64; 15];
    input[..10].copy_from_slice(&[1.0, -2.0, 0.5, 3.0, -1.5, 0.7, -0.3, 2.1, -0.9, 1.3]);

    let mut out_exact = [0.0_f64; 15];
    let mut out_opt = [0.0_f64; 15];
    mlt_stiffness(&input, &sym_grad, &mut out_exact);
    mlt_stiffness_opt(&input, &sym_grad, &mut out_opt);

    for i in 0..10 {
        let c = out_opt[i];
        let e = out_exact[i];
        assert!(
            (c - e).abs() < TOL_K,
            "K_PP row {i}: opt={c:.12e}, exact={e:.12e}, Δ={:.2e}",
            (c - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// 10. mlt_stiffness_opt: K_PB and K_BP blocks are exact for all bubbles
// ---------------------------------------------------------------------------
#[test]
fn test_mlt_stiffness_opt_cross_blocks_exact() {
    let sym_grad = test_matrix();
    let k = build_local_stiffness(&sym_grad);
    let mut input = [0.0_f64; 15];
    let mut out_opt = [0.0_f64; 15];

    // K_PB: apply e_j for each bubble j, check P2 rows 0..10
    for j in 10..15 {
        input.fill(0.0);
        input[j] = 1.0;
        mlt_stiffness_opt(&input, &sym_grad, &mut out_opt);
        for i in 0..10 {
            let c = out_opt[i];
            let e = k[(i, j)];
            assert!(
                (c - e).abs() < TOL_K,
                "K_PB: opt[{i}] for e_{j}: got={c:.12e}, K[{i},{j}]={e:.12e}, Δ={:.2e}",
                (c - e).abs()
            );
        }
    }

    // K_BP: apply e_i for each P2 node i, check bubble rows 10..15
    for i in 0..10 {
        input.fill(0.0);
        input[i] = 1.0;
        mlt_stiffness_opt(&input, &sym_grad, &mut out_opt);
        for j in 10..15 {
            let c = out_opt[j];
            let e = k[(j, i)];
            assert!(
                (c - e).abs() < TOL_K,
                "K_BP: opt[{j}] for e_{i}: got={c:.12e}, K[{j},{i}]={e:.12e}, Δ={:.2e}",
                (c - e).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 11. mlt_stiffness_opt: the operator is symmetric  (50 random SPD matrices)
// ---------------------------------------------------------------------------
#[test]
fn test_mlt_stiffness_opt_symmetry() {
    let mut input = [0.0_f64; 15];
    let mut col = [0.0_f64; 15];
    let mut k_opt = [[0.0_f64; 15]; 15];

    for (m, sym_grad) in random_spd_matrices(50).iter().enumerate() {
        for j in 0..15 {
            input.fill(0.0);
            input[j] = 1.0;
            mlt_stiffness_opt(&input, sym_grad, &mut col);
            for i in 0..15 {
                k_opt[i][j] = col[i];
            }
        }

        for i in 0..15 {
            for j in 0..15 {
                let diff = (k_opt[i][j] - k_opt[j][i]).abs();
                assert!(
                    diff < TOL_K,
                    "matrix {m}: K_opt not symmetric: [{i},{j}]={:.12e}, [{j},{i}]={:.12e}, Δ={:.2e}",
                    k_opt[i][j],
                    k_opt[j][i],
                    diff
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 12. BUBBLE_GRAD_NORM_SQ: ∫_T |∇φ_j|² dV for j = 10..14
//
//  Face bubbles (j = 0..3): integrand is degree 4, the 14-point quadrature
//  is exact for degree ≤ 5, so the numerical result is exact.
//
//  Interior bubble (j = 4): integrand is degree 6, beyond the 14-point rule.
//  Verified instead via the barycentric monomial formula
//    ∫_T λ0^a λ1^b λ2^c λ3^d dV = a!b!c!d! / (a+b+c+d+3)!
// ---------------------------------------------------------------------------
#[test]
fn test_bubble_grad_norm_sq() {
    // ── Face bubbles: 14-point quadrature (exact for degree ≤ 5) ────────────
    for j in 0..4 {
        let mut val = 0.0f64;
        for iq in 0..14 {
            let w = tet_14::WEIGHTS[iq];
            let q = &tet_14::POINTS[iq];
            let du = d_shape_function(10 + j, 0, q);
            let dv = d_shape_function(10 + j, 1, q);
            let dw = d_shape_function(10 + j, 2, q);
            val += w * (du * du + dv * dv + dw * dw);
        }
        assert!(
            (val - BUBBLE_GRAD_NORM_SQ[j]).abs() < TOL,
            "face bubble φ{}: quadrature={val:.15e}, const={:.15e}, Δ={:.2e}",
            10 + j,
            BUBBLE_GRAD_NORM_SQ[j],
            (val - BUBBLE_GRAD_NORM_SQ[j]).abs()
        );
    }

    // ── Interior bubble φ14: monomial formula ────────────────────────────────
    // ∫_T λ0^a λ1^b λ2^c λ3^d dV = a!b!c!d! / (a+b+c+d+3)!
    fn fact(n: u32) -> f64 {
        (1..=n).map(|i| i as f64).product()
    }
    fn monomial_tet(a: u32, b: u32, c: u32, d: u32) -> f64 {
        fact(a) * fact(b) * fact(c) * fact(d) / fact(a + b + c + d + 3)
    }

    // |∇φ14|² = 256² × [λ2²λ3²(λ0−λ1)² + λ1²λ3²(λ0−λ2)² + λ1²λ2²(λ0−λ3)²]
    // By tet symmetry all three squared terms are equal, so:
    //   ∫_T |∇φ14|² = 3 · 256² · ∫_T λ2²λ3²(λ0−λ1)²
    //               = 3 · 256² · (∫λ0²λ2²λ3² − 2·∫λ0λ1λ2²λ3² + ∫λ1²λ2²λ3²)
    let int_interior = 3.0 * 256.0f64.powi(2)
        * (monomial_tet(2, 0, 2, 2) - 2.0 * monomial_tet(1, 1, 2, 2) + monomial_tet(0, 2, 2, 2));

    assert!(
        (int_interior - BUBBLE_GRAD_NORM_SQ[4]).abs() < TOL,
        "interior bubble φ14: formula={int_interior:.15e}, const={:.15e}, Δ={:.2e}",
        BUBBLE_GRAD_NORM_SQ[4],
        (int_interior - BUBBLE_GRAD_NORM_SQ[4]).abs()
    );
}

// ---------------------------------------------------------------------------
// 13. mlt_stiffness_opt == mlt_stiffness for the full 15×15 matrix
//     Checks all blocks (K_PP, K_PB, K_BP, K_BB) against the reference
//     quadrature-based operator over 50 random SPD diffusion tensors.
// ---------------------------------------------------------------------------
#[test]
fn test_mlt_stiffness_opt_equals_mlt_stiffness() {
    let mut input = [0.0_f64; 15];
    let mut out_exact = [0.0_f64; 15];
    let mut out_opt = [0.0_f64; 15];

    for (m, sym_grad) in random_spd_matrices(50).iter().enumerate() {
        for j in 0..15 {
            input.fill(0.0);
            input[j] = 1.0;

            mlt_stiffness(&input, sym_grad, &mut out_exact);
            mlt_stiffness_opt(&input, sym_grad, &mut out_opt);

            for i in 0..15 {
                let diff = (out_opt[i] - out_exact[i]).abs();
                assert!(
                    diff < TOL_K,
                    "matrix {m}, col {j}, row {i}: opt={:.12e}, exact={:.12e}, Δ={:.2e}",
                    out_opt[i],
                    out_exact[i],
                    diff
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 14. mlt_stiffness_opt matrix is positive semi-definite
//
//  The matrix assembled column-by-column via mlt_stiffness_opt (as in test 11)
//  should be PSD for any positive definite A:
//    - exactly one eigenvalue ≈ 0 (null space: constant function)
//    - all 14 remaining eigenvalues strictly positive
// ---------------------------------------------------------------------------
#[test]
fn test_stiffness_positive_semidefinite() {
    let mut input = [0.0_f64; 15];
    let mut col = [0.0_f64; 15];
    let mut k_opt = nalgebra::SMatrix::<f64, 15, 15>::zeros();

    let tol = 1e-10;

    for (m, sym_grad) in random_spd_matrices(50).iter().enumerate() {
        for j in 0..15 {
            input.fill(0.0);
            input[j] = 1.0;
            mlt_stiffness_opt(&input, sym_grad, &mut col);
            for i in 0..15 {
                k_opt[(i, j)] = col[i];
            }
        }

        let k_sym = (k_opt + k_opt.transpose()) * 0.5;
        let eigen = k_sym.symmetric_eigen();
        let mut eigenvalues: Vec<f64> = eigen.eigenvalues.as_slice().to_vec();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(
            eigenvalues[0].abs() < tol,
            "matrix {m}: λ_min = {:.6e}: expected ≈ 0 (null space of constants)",
            eigenvalues[0]
        );

        for (i, &ev) in eigenvalues[1..].iter().enumerate() {
            assert!(
                ev > 0.0,
                "matrix {m}: eigenvalue[{}] = {:.6e}: expected > 0",
                i + 1,
                ev
            );
        }
    }
}
