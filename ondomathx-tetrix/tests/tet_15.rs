use nalgebra::Matrix3;
use tetrix::finite_element::tet_15::{
    build_local_stiffness, d_shape_function, mlt_stiffness, shape_function, POINTS, WEIGHTS,
};

const TOL: f64 = 1e-12;

// ---------------------------------------------------------------------------
// 1. Kronecker-delta interpolation: φ_i(P_j) = δ_{ij}
// ---------------------------------------------------------------------------
#[test]
fn test_kronecker_delta() {
    for j in 0..15 {
        let pt = POINTS[j];
        for i in 0..15 {
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
// 2. Partition of unity: Σ_i φ_i(P) = 1 at any interior point P
// ---------------------------------------------------------------------------
#[test]
fn test_partition_of_unity() {
    let pts: [[f64; 3]; 5] = [
        [0.1, 0.2, 0.3],
        [0.25, 0.25, 0.25],
        [0.1, 0.1, 0.1],
        [0.5, 0.25, 0.1],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
    ];
    for pt in &pts {
        let sum: f64 = (0..15).map(|i| shape_function(i, pt)).sum();
        assert!(
            (sum - 1.0).abs() < TOL,
            "Partition of unity at {pt:?}: sum = {sum:.2e}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Sum of partial derivatives = 0 (consequence of partition of unity)
//    Σ_i ∂φ_i/∂d(P) = 0  for d = 0, 1, 2
// ---------------------------------------------------------------------------
#[test]
fn test_gradient_sum_is_zero() {
    let pts: [[f64; 3]; 3] = [
        [0.1, 0.2, 0.3],
        [0.25, 0.25, 0.25],
        [0.15, 0.1, 0.05],
    ];
    for pt in &pts {
        for d in 0..3 {
            let sum: f64 = (0..15).map(|i| d_shape_function(i, d, pt)).sum();
            assert!(
                sum.abs() < TOL,
                "Σ ∂φ_i/∂{d} at {pt:?} = {sum:.2e}, expected 0"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. P1 reproduction: Σ_i φ_i(P) · P_i[d] = P[d]  for d = 0, 1, 2
// ---------------------------------------------------------------------------
#[test]
fn test_p1_reproduction() {
    let pts: [[f64; 3]; 4] = [
        [0.1, 0.2, 0.3],
        [0.25, 0.25, 0.25],
        [0.1, 0.1, 0.1],
        [0.4, 0.2, 0.1],
    ];
    for pt in &pts {
        for d in 0..3 {
            let interp: f64 = (0..15)
                .map(|i| shape_function(i, pt) * POINTS[i][d])
                .sum();
            assert!(
                (interp - pt[d]).abs() < TOL,
                "P1 reproduction coord {d} at {pt:?}: got {interp:.6e}, expected {:.6e}",
                pt[d]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. P2 reproduction: Σ_i φ_i(P) · p(P_i) = p(P) for all degree-2 monomials
// ---------------------------------------------------------------------------
#[test]
fn test_p2_reproduction() {
    let pts: [[f64; 3]; 3] = [
        [0.1, 0.2, 0.3],
        [0.25, 0.25, 0.25],
        [0.15, 0.1, 0.4],
    ];
    let monomials: &[(&str, fn(f64, f64, f64) -> f64)] = &[
        ("u²", |u, _v, _w| u * u),
        ("v²", |_u, v, _w| v * v),
        ("w²", |_u, _v, w| w * w),
        ("uv", |u, v, _w| u * v),
        ("uw", |u, _v, w| u * w),
        ("vw", |_u, v, w| v * w),
    ];
    for pt in &pts {
        let (u, v, w) = (pt[0], pt[1], pt[2]);
        for &(name, mono) in monomials {
            let interp: f64 = (0..15)
                .map(|i| {
                    let [pu, pv, pw] = POINTS[i];
                    shape_function(i, pt) * mono(pu, pv, pw)
                })
                .sum();
            let exact = mono(u, v, w);
            assert!(
                (interp - exact).abs() < TOL,
                "P2 reproduction of {name} at {pt:?}: got {interp:.6e}, expected {exact:.6e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Analytical gradients match centred finite differences
//    |∂φ_i/∂d - [φ_i(P+h·eₐ) - φ_i(P-h·eₐ)]/(2h)| < tol
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

// ---------------------------------------------------------------------------
// 7. Quadrature exactness: Σ_i w_i · u^a v^b w^c = a!b!c!/(3+a+b+c)!
//    verified for all monomials up to total degree 3 and degree-4 bubble monomials
// ---------------------------------------------------------------------------
#[test]
fn test_quadrature_exactness_degree_3() {
    fn factorial(n: u32) -> f64 {
        (1..=n).fold(1.0, |acc, k| acc * k as f64)
    }
    fn exact_tet_integral(a: u32, b: u32, c: u32) -> f64 {
        factorial(a) * factorial(b) * factorial(c) / factorial(3 + a + b + c)
    }
    fn quad(a: u32, b: u32, c: u32) -> f64 {
        (0..15)
            .map(|i| {
                WEIGHTS[i]
                    * POINTS[i][0].powi(a as i32)
                    * POINTS[i][1].powi(b as i32)
                    * POINTS[i][2].powi(c as i32)
            })
            .sum()
    }

    // degree 0
    assert!((quad(0,0,0) - exact_tet_integral(0,0,0)).abs() < TOL, "exact for 1");
    // degree 1
    assert!((quad(1,0,0) - exact_tet_integral(1,0,0)).abs() < TOL, "exact for u");
    assert!((quad(0,1,0) - exact_tet_integral(0,1,0)).abs() < TOL, "exact for v");
    assert!((quad(0,0,1) - exact_tet_integral(0,0,1)).abs() < TOL, "exact for w");
    // degree 2
    assert!((quad(2,0,0) - exact_tet_integral(2,0,0)).abs() < TOL, "exact for u²");
    assert!((quad(1,1,0) - exact_tet_integral(1,1,0)).abs() < TOL, "exact for uv");
    assert!((quad(1,0,1) - exact_tet_integral(1,0,1)).abs() < TOL, "exact for uw");
    assert!((quad(0,2,0) - exact_tet_integral(0,2,0)).abs() < TOL, "exact for v²");
    assert!((quad(0,1,1) - exact_tet_integral(0,1,1)).abs() < TOL, "exact for vw");
    assert!((quad(0,0,2) - exact_tet_integral(0,0,2)).abs() < TOL, "exact for w²");
    // degree 3
    assert!((quad(3,0,0) - exact_tet_integral(3,0,0)).abs() < TOL, "exact for u³");
    assert!((quad(2,1,0) - exact_tet_integral(2,1,0)).abs() < TOL, "exact for u²v");
    assert!((quad(2,0,1) - exact_tet_integral(2,0,1)).abs() < TOL, "exact for u²w");
    assert!((quad(1,2,0) - exact_tet_integral(1,2,0)).abs() < TOL, "exact for uv²");
    assert!((quad(1,1,1) - exact_tet_integral(1,1,1)).abs() < TOL, "exact for uvw");
    assert!((quad(1,0,2) - exact_tet_integral(1,0,2)).abs() < TOL, "exact for uw²");
    assert!((quad(0,3,0) - exact_tet_integral(0,3,0)).abs() < TOL, "exact for v³");
    assert!((quad(0,2,1) - exact_tet_integral(0,2,1)).abs() < TOL, "exact for v²w");
    assert!((quad(0,1,2) - exact_tet_integral(0,1,2)).abs() < TOL, "exact for vw²");
    assert!((quad(0,0,3) - exact_tet_integral(0,0,3)).abs() < TOL, "exact for w³");
    assert!((quad(2,1,1) - exact_tet_integral(2,1,1)).abs() < TOL, "exact for u²vw");
    assert!((quad(1,2,1) - exact_tet_integral(1,2,1)).abs() < TOL, "exact for uv²w");
    assert!((quad(1,1,2) - exact_tet_integral(1,1,2)).abs() < TOL, "exact for uvw²");
}

// ---------------------------------------------------------------------------
// 8. mlt_stiffness(e_i) == K · e_i  for each basis vector
// ---------------------------------------------------------------------------
fn test_matrix() -> Matrix3<f64> {
    Matrix3::new(2.0, 0.5, 0.1, 0.5, 3.0, 0.2, 0.1, 0.2, 1.5)
}

const TOL_K: f64 = 1e-10;

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
// 9. Symmetry — K must be symmetric (K[i,j] == K[j,i])
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
