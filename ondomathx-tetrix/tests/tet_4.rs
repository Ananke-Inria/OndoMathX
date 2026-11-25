use ondomathx_tetrix::finite_element::tet_4::{POINTS, WEIGHTS};

const TOL: f64 = 1e-14;

// Exact integral ∫_T u^a v^b w^c dV = a! b! c! / (3+a+b+c)!
fn exact_tet_integral(a: u32, b: u32, c: u32) -> f64 {
    fn fac(n: u32) -> f64 {
        (1..=n).fold(1.0, |acc, k| acc * k as f64)
    }
    fac(a) * fac(b) * fac(c) / fac(3 + a + b + c)
}

// Quadrature approximation of ∫_T u^a v^b w^c dV
fn quad(a: u32, b: u32, c: u32) -> f64 {
    (0..4)
        .map(|i| {
            WEIGHTS[i]
                * POINTS[i][0].powi(a as i32)
                * POINTS[i][1].powi(b as i32)
                * POINTS[i][2].powi(c as i32)
        })
        .sum()
}


// ---------------------------------------------------------------------------
// Monomial exactness up to degree 2 (10 monomials)
// ---------------------------------------------------------------------------
#[test]
fn test_quadrature_exactness_degree_2() {
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
}
