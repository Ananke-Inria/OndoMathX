use nalgebra::{Matrix3, SVector};
use tetrix::finite_element::tet_10b;
use tetrix::finite_element::tet_15;
use tetrix::finite_element::tet_15_10b::{
    tet_10b_to_tet_15, tet_15_to_tet_10b, tet_15_to_tet_10b_transpose,
};

const TOL: f64 = 1e-12;

fn test_coeffs_10b() -> [f64; 15] {
    [1.0, -2.0, 0.5, 3.0, -1.5, 0.7, -0.3, 2.1, -0.9, 1.3, -0.4, 0.8, -1.1, 0.6, -0.2]
}

fn test_coeffs_15() -> [f64; 15] {
    [0.5, 1.2, -0.8, 2.0, -1.0, 0.3, 0.9, -0.4, 1.7, -0.6, 0.1, -1.3, 0.4, -0.7, 0.2]
}

// ---------------------------------------------------------------------------
// 1. Round-trip tet_10b → tet_15 → tet_10b
// ---------------------------------------------------------------------------
#[test]
fn test_roundtrip_10b_to_15_to_10b() {
    let b = test_coeffs_10b();
    let mut a = [0.0f64; 15];
    let mut b2 = [0.0f64; 15];

    tet_10b_to_tet_15(&b, &mut a);
    tet_15_to_tet_10b(&a, &mut b2);

    for i in 0..15 {
        assert!(
            (b2[i] - b[i]).abs() < TOL,
            "round-trip 10b→15→10b: i={i}: got {:.14e}, expected {:.14e}, Δ={:.2e}",
            b2[i],
            b[i],
            (b2[i] - b[i]).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Round-trip tet_15 → tet_10b → tet_15
// ---------------------------------------------------------------------------
#[test]
fn test_roundtrip_15_to_10b_to_15() {
    let a = test_coeffs_15();
    let mut b = [0.0f64; 15];
    let mut a2 = [0.0f64; 15];

    tet_15_to_tet_10b(&a, &mut b);
    tet_10b_to_tet_15(&b, &mut a2);

    for i in 0..15 {
        assert!(
            (a2[i] - a[i]).abs() < TOL,
            "round-trip 15→10b→15: i={i}: got {:.14e}, expected {:.14e}, Δ={:.2e}",
            a2[i],
            a[i],
            (a2[i] - a[i]).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Function consistency: both representations evaluate to the same value
//    at several interior points
// ---------------------------------------------------------------------------
#[test]
fn test_function_consistency_10b_to_15() {
    let b = test_coeffs_10b();
    let mut a = [0.0f64; 15];
    tet_10b_to_tet_15(&b, &mut a);

    let test_pts: [[f64; 3]; 5] = [
        [0.1, 0.2, 0.3],
        [0.25, 0.25, 0.25],
        [0.1, 0.1, 0.1],
        [0.4, 0.2, 0.1],
        [1.0 / 6.0, 1.0 / 3.0, 1.0 / 4.0],
    ];

    for pt in &test_pts {
        let val_10b: f64 = (0..15).map(|j| b[j] * tet_10b::shape_function(j, pt)).sum();
        let val_15: f64 = (0..15).map(|i| a[i] * tet_15::shape_function(i, pt)).sum();
        assert!(
            (val_10b - val_15).abs() < TOL,
            "function mismatch at {pt:?}: tet_10b={val_10b:.14e}, tet_15={val_15:.14e}, Δ={:.2e}",
            (val_10b - val_15).abs()
        );
    }
}

#[test]
fn test_function_consistency_15_to_10b() {
    let a = test_coeffs_15();
    let mut b = [0.0f64; 15];
    tet_15_to_tet_10b(&a, &mut b);

    let test_pts: [[f64; 3]; 5] = [
        [0.1, 0.2, 0.3],
        [0.25, 0.25, 0.25],
        [0.1, 0.1, 0.1],
        [0.4, 0.2, 0.1],
        [1.0 / 6.0, 1.0 / 3.0, 1.0 / 4.0],
    ];

    for pt in &test_pts {
        let val_15: f64 = (0..15).map(|i| a[i] * tet_15::shape_function(i, pt)).sum();
        let val_10b: f64 = (0..15).map(|j| b[j] * tet_10b::shape_function(j, pt)).sum();
        assert!(
            (val_15 - val_10b).abs() < TOL,
            "function mismatch at {pt:?}: tet_15={val_15:.14e}, tet_10b={val_10b:.14e}, Δ={:.2e}",
            (val_15 - val_10b).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Constant function: tet_15 all-ones → tet_10b first 10 = 1, last 5 = 0
//    (the P2 sub-basis already forms a partition of unity)
// ---------------------------------------------------------------------------
#[test]
fn test_constant_function() {
    let a = [1.0f64; 15];
    let mut b = [0.0f64; 15];
    tet_15_to_tet_10b(&a, &mut b);

    for j in 0..10 {
        assert!(
            (b[j] - 1.0).abs() < TOL,
            "constant fn: b[{j}] = {:.14e}, expected 1.0",
            b[j]
        );
    }
    for j in 10..15 {
        assert!(
            b[j].abs() < TOL,
            "constant fn: b[{j}] = {:.14e}, expected 0.0",
            b[j]
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Energy preservation: b^T K_10b b = a^T K_15 a for the same function
//    This verifies that the change of basis preserves the bilinear form.
// ---------------------------------------------------------------------------
#[test]
fn test_energy_preservation() {
    let sym_grad = Matrix3::new(2.0, 0.5, 0.1, 0.5, 3.0, 0.2, 0.1, 0.2, 1.5);
    let k_10b = tet_10b::build_local_stiffness(&sym_grad);
    let k_15 = tet_15::build_local_stiffness(&sym_grad);

    let b = test_coeffs_10b();
    let mut a = [0.0f64; 15];
    tet_10b_to_tet_15(&b, &mut a);

    let b_vec = SVector::<f64, 15>::from_fn(|i, _| b[i]);
    let a_vec = SVector::<f64, 15>::from_fn(|i, _| a[i]);

    let energy_10b = b_vec.dot(&(k_10b * b_vec));
    let energy_15 = a_vec.dot(&(k_15 * a_vec));

    assert!(
        (energy_10b - energy_15).abs() < 1e-10,
        "energy mismatch: tet_10b={energy_10b:.14e}, tet_15={energy_15:.14e}, Δ={:.2e}",
        (energy_10b - energy_15).abs()
    );
}

// ---------------------------------------------------------------------------
// 6. tet_15_to_tet_10b_transpose is the adjoint of tet_15_to_tet_10b:
//    <P u, c> = <u, P^T c>  for all u, c
// ---------------------------------------------------------------------------
#[test]
fn test_tet_15_to_tet_10b_transpose_adjoint() {
    let u = test_coeffs_15();
    let c = test_coeffs_10b();

    let mut pu = [0.0f64; 15];
    tet_15_to_tet_10b(&u, &mut pu);

    let mut pt_c = [0.0f64; 15];
    tet_15_to_tet_10b_transpose(&c, &mut pt_c);

    let lhs: f64 = pu.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
    let rhs: f64 = u.iter().zip(pt_c.iter()).map(|(x, y)| x * y).sum();

    assert!(
        (lhs - rhs).abs() < TOL,
        "<P u, c> = {lhs:.15e}  ≠  <u, P^T c> = {rhs:.15e},  Δ = {:.2e}",
        (lhs - rhs).abs()
    );
}

// ---------------------------------------------------------------------------
// 7. tet_15 basis vector e_i: tet_15_to_tet_10b(e_i) gives the i-th column
//    of the inverse change-of-basis matrix; converting back must recover e_i
// ---------------------------------------------------------------------------
#[test]
fn test_basis_vectors_roundtrip() {
    let mut a = [0.0f64; 15];
    let mut b = [0.0f64; 15];
    let mut a2 = [0.0f64; 15];

    for i in 0..15 {
        a.fill(0.0);
        a[i] = 1.0;

        tet_15_to_tet_10b(&a, &mut b);
        tet_10b_to_tet_15(&b, &mut a2);

        for k in 0..15 {
            let expected = if k == i { 1.0 } else { 0.0 };
            assert!(
                (a2[k] - expected).abs() < TOL,
                "basis vector e_{i}: a2[{k}]={:.14e}, expected {expected}",
                a2[k]
            );
        }
    }
}
