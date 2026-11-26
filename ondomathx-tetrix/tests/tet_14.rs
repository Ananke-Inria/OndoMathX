use ondomathx_tetrix::element::tet_14::{POINTS, WEIGHTS};

const TOL: f64 = 1e-12;

// Exact integral ∫_T u^a v^b w^c dV = a! b! c! / (3+a+b+c)!
fn exact_tet_integral(a: u32, b: u32, c: u32) -> f64 {
    fn fac(n: u32) -> f64 {
        (1..=n).fold(1.0, |acc, k| acc * k as f64)
    }
    fac(a) * fac(b) * fac(c) / fac(3 + a + b + c)
}

// Quadrature approximation of ∫_T u^a v^b w^c dV
fn quad(a: u32, b: u32, c: u32) -> f64 {
    (0..14)
        .map(|i| {
            WEIGHTS[i]
                * POINTS[i][0].powi(a as i32)
                * POINTS[i][1].powi(b as i32)
                * POINTS[i][2].powi(c as i32)
        })
        .sum()
}


// ---------------------------------------------------------------------------
// 1. Monomial exactness up to degree 5
//
// The rule is exact for every monomial u^a v^b w^c with a+b+c ≤ 5
// (56 monomials total).  This is the algebraic degree of precision of the
// 14-point Dunavant-type rule.
// ---------------------------------------------------------------------------
#[test]
fn test_quadrature_exactness_degree_5() {
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

    // degree 4
    assert!((quad(4,0,0) - exact_tet_integral(4,0,0)).abs() < TOL, "exact for u⁴");
    assert!((quad(3,1,0) - exact_tet_integral(3,1,0)).abs() < TOL, "exact for u³v");
    assert!((quad(3,0,1) - exact_tet_integral(3,0,1)).abs() < TOL, "exact for u³w");
    assert!((quad(2,2,0) - exact_tet_integral(2,2,0)).abs() < TOL, "exact for u²v²");
    assert!((quad(2,1,1) - exact_tet_integral(2,1,1)).abs() < TOL, "exact for u²vw");
    assert!((quad(2,0,2) - exact_tet_integral(2,0,2)).abs() < TOL, "exact for u²w²");
    assert!((quad(1,3,0) - exact_tet_integral(1,3,0)).abs() < TOL, "exact for uv³");
    assert!((quad(1,2,1) - exact_tet_integral(1,2,1)).abs() < TOL, "exact for uv²w");
    assert!((quad(1,1,2) - exact_tet_integral(1,1,2)).abs() < TOL, "exact for uvw²");
    assert!((quad(1,0,3) - exact_tet_integral(1,0,3)).abs() < TOL, "exact for uw³");
    assert!((quad(0,4,0) - exact_tet_integral(0,4,0)).abs() < TOL, "exact for v⁴");
    assert!((quad(0,3,1) - exact_tet_integral(0,3,1)).abs() < TOL, "exact for v³w");
    assert!((quad(0,2,2) - exact_tet_integral(0,2,2)).abs() < TOL, "exact for v²w²");
    assert!((quad(0,1,3) - exact_tet_integral(0,1,3)).abs() < TOL, "exact for vw³");
    assert!((quad(0,0,4) - exact_tet_integral(0,0,4)).abs() < TOL, "exact for w⁴");

    // degree 5
    assert!((quad(5,0,0) - exact_tet_integral(5,0,0)).abs() < TOL, "exact for u⁵");
    assert!((quad(4,1,0) - exact_tet_integral(4,1,0)).abs() < TOL, "exact for u⁴v");
    assert!((quad(4,0,1) - exact_tet_integral(4,0,1)).abs() < TOL, "exact for u⁴w");
    assert!((quad(3,2,0) - exact_tet_integral(3,2,0)).abs() < TOL, "exact for u³v²");
    assert!((quad(3,1,1) - exact_tet_integral(3,1,1)).abs() < TOL, "exact for u³vw");
    assert!((quad(3,0,2) - exact_tet_integral(3,0,2)).abs() < TOL, "exact for u³w²");
    assert!((quad(2,3,0) - exact_tet_integral(2,3,0)).abs() < TOL, "exact for u²v³");
    assert!((quad(2,2,1) - exact_tet_integral(2,2,1)).abs() < TOL, "exact for u²v²w");
    assert!((quad(2,1,2) - exact_tet_integral(2,1,2)).abs() < TOL, "exact for u²vw²");
    assert!((quad(2,0,3) - exact_tet_integral(2,0,3)).abs() < TOL, "exact for u²w³");
    assert!((quad(1,4,0) - exact_tet_integral(1,4,0)).abs() < TOL, "exact for uv⁴");
    assert!((quad(1,3,1) - exact_tet_integral(1,3,1)).abs() < TOL, "exact for uv³w");
    assert!((quad(1,2,2) - exact_tet_integral(1,2,2)).abs() < TOL, "exact for uv²w²");
    assert!((quad(1,1,3) - exact_tet_integral(1,1,3)).abs() < TOL, "exact for uvw³");
    assert!((quad(1,0,4) - exact_tet_integral(1,0,4)).abs() < TOL, "exact for uw⁴");
    assert!((quad(0,5,0) - exact_tet_integral(0,5,0)).abs() < TOL, "exact for v⁵");
    assert!((quad(0,4,1) - exact_tet_integral(0,4,1)).abs() < TOL, "exact for v⁴w");
    assert!((quad(0,3,2) - exact_tet_integral(0,3,2)).abs() < TOL, "exact for v³w²");
    assert!((quad(0,2,3) - exact_tet_integral(0,2,3)).abs() < TOL, "exact for v²w³");
    assert!((quad(0,1,4) - exact_tet_integral(0,1,4)).abs() < TOL, "exact for vw⁴");
    assert!((quad(0,0,5) - exact_tet_integral(0,0,5)).abs() < TOL, "exact for w⁵");
}
