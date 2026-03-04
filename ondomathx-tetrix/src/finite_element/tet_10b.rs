use nalgebra::{Matrix3, SMatrix, Vector3};

/// Enriched tetrahedral element — 15 DOFs, alternative basis.
///
/// Same 15-dimensional function space as `tet_15`, but with a different basis:
///   - Nodes 0–9:  standard P2 Lagrange basis functions
///   - Nodes 10–14: face and interior bubble functions
///
/// Node numbering and positions match `tet_15` exactly:
///   0: (0,0,0)      vertex V0             φ0  = λ0(2λ0−1)
///   1: (1,0,0)      vertex V1             φ1  = λ1(2λ1−1)
///   2: (0,1,0)      vertex V2             φ2  = λ2(2λ2−1)
///   3: (0,0,1)      vertex V3             φ3  = λ3(2λ3−1)
///   4: (½,0,0)      midpoint V0−V1        φ4  = 4λ0λ1
///   5: (½,½,0)      midpoint V1−V2        φ5  = 4λ1λ2
///   6: (0,½,0)      midpoint V0−V2        φ6  = 4λ0λ2
///   7: (0,0,½)      midpoint V0−V3        φ7  = 4λ0λ3
///   8: (½,0,½)      midpoint V1−V3        φ8  = 4λ1λ3
///   9: (0,½,½)      midpoint V2−V3        φ9  = 4λ2λ3
///  10: (⅓,⅓,0)     face V0V1V2 bubble    φ10 = 27λ0λ1λ2
///  11: (⅓,0,⅓)     face V0V1V3 bubble    φ11 = 27λ0λ1λ3
///  12: (0,⅓,⅓)     face V0V2V3 bubble    φ12 = 27λ0λ2λ3
///  13: (⅓,⅓,⅓)    face V1V2V3 bubble    φ13 = 27λ1λ2λ3
///  14: (¼,¼,¼)     interior bubble       φ14 = 256λ0λ1λ2λ3
///
/// with λ0=1−u−v−w,  λ1=u,  λ2=v,  λ3=w.
/// Arguments to function arrays: (u, v, w, z) where z = λ0 = 1−u−v−w.

pub const POINTS: [[f64; 3]; 15] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [1.0 / 3.0, 1.0 / 3.0, 0.0],
    [1.0 / 3.0, 0.0, 1.0 / 3.0],
    [0.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [0.25, 0.25, 0.25],
];

// Shape functions on the reference tetrahedron.
// Arguments: (u, v, w, z) where z = 1 - u - v - w.
pub const SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    // P2 Lagrange basis (nodes 0–9)
    |_u, _v, _w, z| z * (2.0 * z - 1.0), // φ0  = λ0(2λ0−1)
    |u, _v, _w, _z| u * (2.0 * u - 1.0), // φ1  = λ1(2λ1−1)
    |_u, v, _w, _z| v * (2.0 * v - 1.0), // φ2  = λ2(2λ2−1)
    |_u, _v, w, _z| w * (2.0 * w - 1.0), // φ3  = λ3(2λ3−1)
    |u, _v, _w, z| 4.0 * z * u,          // φ4  = 4λ0λ1
    |u, v, _w, _z| 4.0 * u * v,          // φ5  = 4λ1λ2
    |_u, v, _w, z| 4.0 * z * v,          // φ6  = 4λ0λ2
    |_u, _v, w, z| 4.0 * z * w,          // φ7  = 4λ0λ3
    |u, _v, w, _z| 4.0 * u * w,          // φ8  = 4λ1λ3
    |_u, v, w, _z| 4.0 * v * w,          // φ9  = 4λ2λ3
    // Bubble functions (nodes 10–14)
    |u, v, _w, z| 27.0 * z * u * v,     // φ10 = 27λ0λ1λ2
    |u, _v, w, z| 27.0 * z * u * w,     // φ11 = 27λ0λ1λ3
    |_u, v, w, z| 27.0 * z * v * w,     // φ12 = 27λ0λ2λ3
    |u, v, w, _z| 27.0 * u * v * w,     // φ13 = 27λ1λ2λ3
    |u, v, w, z| 256.0 * z * u * v * w, // φ14 = 256λ0λ1λ2λ3
];

pub const DU_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |_u, _v, _w, z| 1.0 - 4.0 * z,        // ∂φ0/∂u  = 1−4z
    |u, _v, _w, _z| 4.0 * u - 1.0,        // ∂φ1/∂u  = 4u−1
    |_u, _v, _w, _z| 0.0,                 // ∂φ2/∂u  = 0
    |_u, _v, _w, _z| 0.0,                 // ∂φ3/∂u  = 0
    |u, _v, _w, z| 4.0 * (z - u),         // ∂φ4/∂u  = 4(z−u)
    |_u, v, _w, _z| 4.0 * v,              // ∂φ5/∂u  = 4v
    |_u, v, _w, _z| -4.0 * v,             // ∂φ6/∂u  = −4v
    |_u, _v, w, _z| -4.0 * w,             // ∂φ7/∂u  = −4w
    |_u, _v, w, _z| 4.0 * w,              // ∂φ8/∂u  = 4w
    |_u, _v, _w, _z| 0.0,                 // ∂φ9/∂u  = 0
    |u, v, _w, z| 27.0 * v * (z - u),     // ∂φ10/∂u = 27v(z−u)
    |u, _v, w, z| 27.0 * w * (z - u),     // ∂φ11/∂u = 27w(z−u)
    |_u, v, w, _z| -27.0 * v * w,         // ∂φ12/∂u = −27vw
    |_u, v, w, _z| 27.0 * v * w,          // ∂φ13/∂u = 27vw
    |u, v, w, z| 256.0 * v * w * (z - u), // ∂φ14/∂u = 256vw(z−u)
];

pub const DV_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |_u, _v, _w, z| 1.0 - 4.0 * z,        // ∂φ0/∂v  = 1−4z
    |_u, _v, _w, _z| 0.0,                 // ∂φ1/∂v  = 0
    |_u, v, _w, _z| 4.0 * v - 1.0,        // ∂φ2/∂v  = 4v−1
    |_u, _v, _w, _z| 0.0,                 // ∂φ3/∂v  = 0
    |u, _v, _w, _z| -4.0 * u,             // ∂φ4/∂v  = −4u
    |u, _v, _w, _z| 4.0 * u,              // ∂φ5/∂v  = 4u
    |_u, v, _w, z| 4.0 * (z - v),         // ∂φ6/∂v  = 4(z−v)
    |_u, _v, w, _z| -4.0 * w,             // ∂φ7/∂v  = −4w
    |_u, _v, _w, _z| 0.0,                 // ∂φ8/∂v  = 0
    |_u, _v, w, _z| 4.0 * w,              // ∂φ9/∂v  = 4w
    |u, v, _w, z| 27.0 * u * (z - v),     // ∂φ10/∂v = 27u(z−v)
    |u, _v, w, _z| -27.0 * u * w,         // ∂φ11/∂v = −27uw
    |_u, v, w, z| 27.0 * w * (z - v),     // ∂φ12/∂v = 27w(z−v)
    |u, _v, w, _z| 27.0 * u * w,          // ∂φ13/∂v = 27uw
    |u, v, w, z| 256.0 * u * w * (z - v), // ∂φ14/∂v = 256uw(z−v)
];

pub const DW_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |_u, _v, _w, z| 1.0 - 4.0 * z,        // ∂φ0/∂w  = 1−4z
    |_u, _v, _w, _z| 0.0,                 // ∂φ1/∂w  = 0
    |_u, _v, _w, _z| 0.0,                 // ∂φ2/∂w  = 0
    |_u, _v, w, _z| 4.0 * w - 1.0,        // ∂φ3/∂w  = 4w−1
    |u, _v, _w, _z| -4.0 * u,             // ∂φ4/∂w  = −4u
    |_u, _v, _w, _z| 0.0,                 // ∂φ5/∂w  = 0
    |_u, v, _w, _z| -4.0 * v,             // ∂φ6/∂w  = −4v
    |_u, _v, w, z| 4.0 * (z - w),         // ∂φ7/∂w  = 4(z−w)
    |u, _v, _w, _z| 4.0 * u,              // ∂φ8/∂w  = 4u
    |_u, v, _w, _z| 4.0 * v,              // ∂φ9/∂w  = 4v
    |u, v, _w, _z| -27.0 * u * v,         // ∂φ10/∂w = −27uv
    |u, _v, w, z| 27.0 * u * (z - w),     // ∂φ11/∂w = 27u(z−w)
    |_u, v, w, z| 27.0 * v * (z - w),     // ∂φ12/∂w = 27v(z−w)
    |u, v, _w, _z| 27.0 * u * v,          // ∂φ13/∂w = 27uv
    |u, v, w, z| 256.0 * u * v * (z - w), // ∂φ14/∂w = 256uv(z−w)
];

/// K_BB basis stiffness matrices: KBB_AB[p][q] = ∫_T A_AB ∇φ_{10+p} · ∇φ_{10+q} dV
/// K_BB = a00·KBB_00 + a01·KBB_01 + a02·KBB_02 + a11·KBB_11 + a12·KBB_12 + a22·KBB_22

/// A = [[1,0,0],[0,0,0],[0,0,0]]
const KBB_00: [[f64; 5]; 5] = [
    [
        5.7857142857142863e-1,
        2.8928571428571426e-1,
        0.0000000000000000e0,
        0.0000000000000000e0,
        6.8571428571428561e-1,
    ], // p=0
    [
        2.8928571428571426e-1,
        5.7857142857142863e-1,
        0.0000000000000000e0,
        0.0000000000000000e0,
        6.8571428571428550e-1,
    ], // p=1
    [
        0.0000000000000000e0,
        0.0000000000000000e0,
        5.7857142857142863e-1,
        -5.7857142857142863e-1,
        0.0000000000000000e0,
    ], // p=2
    [
        0.0000000000000000e0,
        0.0000000000000000e0,
        -5.7857142857142863e-1,
        5.7857142857142863e-1,
        0.0000000000000000e0,
    ], // p=3
    [
        6.8571428571428572e-1,
        6.8571428571428572e-1,
        0.0000000000000000e0,
        0.0000000000000000e0,
        1.5439160309836284e0,
    ], // p=4
];

/// A = [[0,1,0],[1,0,0],[0,0,0]]
const KBB_01: [[f64; 5]; 5] = [
    [
        5.7857142857142874e-1,
        2.8928571428571442e-1,
        2.8928571428571442e-1,
        -2.8928571428571442e-1,
        6.8571428571428616e-1,
    ], // p=0
    [
        2.8928571428571442e-1,
        5.7857142857142885e-1,
        5.7857142857142851e-1,
        -5.7857142857142885e-1,
        6.8571428571428583e-1,
    ], // p=1
    [
        2.8928571428571442e-1,
        5.7857142857142851e-1,
        5.7857142857142896e-1,
        -5.7857142857142874e-1,
        6.8571428571428583e-1,
    ], // p=2
    [
        -2.8928571428571442e-1,
        -5.7857142857142885e-1,
        -5.7857142857142874e-1,
        5.7857142857142863e-1,
        -6.8571428571428605e-1,
    ], // p=3
    [
        6.8571428571428616e-1,
        6.8571428571428583e-1,
        6.8571428571428583e-1,
        -6.8571428571428605e-1,
        1.5439160309836295e0,
    ], // p=4
];

/// A = [[0,0,1],[0,0,0],[1,0,0]]
const KBB_02: [[f64; 5]; 5] = [
    [
        5.7857142857142885e-1,
        2.8928571428571442e-1,
        5.7857142857142874e-1,
        -5.7857142857142885e-1,
        6.8571428571428594e-1,
    ], // p=0
    [
        2.8928571428571442e-1,
        5.7857142857142874e-1,
        2.8928571428571442e-1,
        -2.8928571428571448e-1,
        6.8571428571428616e-1,
    ], // p=1
    [
        5.7857142857142874e-1,
        2.8928571428571442e-1,
        5.7857142857142885e-1,
        -5.7857142857142896e-1,
        6.8571428571428583e-1,
    ], // p=2
    [
        -5.7857142857142885e-1,
        -2.8928571428571448e-1,
        -5.7857142857142896e-1,
        5.7857142857142863e-1,
        -6.8571428571428594e-1,
    ], // p=3
    [
        6.8571428571428594e-1,
        6.8571428571428616e-1,
        6.8571428571428583e-1,
        -6.8571428571428594e-1,
        1.5439160309836297e0,
    ], // p=4
];

/// A = [[0,0,0],[0,1,0],[0,0,0]]
const KBB_11: [[f64; 5]; 5] = [
    [
        5.7857142857142863e-1,
        0.0000000000000000e0,
        2.8928571428571426e-1,
        0.0000000000000000e0,
        6.8571428571428561e-1,
    ], // p=0
    [
        0.0000000000000000e0,
        5.7857142857142863e-1,
        0.0000000000000000e0,
        -5.7857142857142863e-1,
        0.0000000000000000e0,
    ], // p=1
    [
        2.8928571428571426e-1,
        0.0000000000000000e0,
        5.7857142857142863e-1,
        0.0000000000000000e0,
        6.8571428571428550e-1,
    ], // p=2
    [
        0.0000000000000000e0,
        -5.7857142857142863e-1,
        0.0000000000000000e0,
        5.7857142857142863e-1,
        0.0000000000000000e0,
    ], // p=3
    [
        6.8571428571428572e-1,
        0.0000000000000000e0,
        6.8571428571428572e-1,
        0.0000000000000000e0,
        1.5439160309836284e0,
    ], // p=4
];

/// A = [[0,0,0],[0,0,1],[0,1,0]]
const KBB_12: [[f64; 5]; 5] = [
    [
        5.7857142857142885e-1,
        5.7857142857142874e-1,
        2.8928571428571442e-1,
        -5.7857142857142885e-1,
        6.8571428571428594e-1,
    ], // p=0
    [
        5.7857142857142874e-1,
        5.7857142857142896e-1,
        2.8928571428571442e-1,
        -5.7857142857142885e-1,
        6.8571428571428583e-1,
    ], // p=1
    [
        2.8928571428571442e-1,
        2.8928571428571442e-1,
        5.7857142857142874e-1,
        -2.8928571428571437e-1,
        6.8571428571428616e-1,
    ], // p=2
    [
        -5.7857142857142885e-1,
        -5.7857142857142885e-1,
        -2.8928571428571437e-1,
        5.7857142857142863e-1,
        -6.8571428571428594e-1,
    ], // p=3
    [
        6.8571428571428594e-1,
        6.8571428571428583e-1,
        6.8571428571428616e-1,
        -6.8571428571428594e-1,
        1.5439160309836297e0,
    ], // p=4
];

/// A = [[0,0,0],[0,0,0],[0,0,1]]
const KBB_22: [[f64; 5]; 5] = [
    [
        5.7857142857142863e-1,
        0.0000000000000000e0,
        0.0000000000000000e0,
        -5.7857142857142863e-1,
        0.0000000000000000e0,
    ], // p=0
    [
        0.0000000000000000e0,
        5.7857142857142851e-1,
        2.8928571428571426e-1,
        0.0000000000000000e0,
        6.8571428571428572e-1,
    ], // p=1
    [
        0.0000000000000000e0,
        2.8928571428571426e-1,
        5.7857142857142863e-1,
        0.0000000000000000e0,
        6.8571428571428561e-1,
    ], // p=2
    [
        -5.7857142857142863e-1,
        0.0000000000000000e0,
        0.0000000000000000e0,
        5.7857142857142863e-1,
        0.0000000000000000e0,
    ], // p=3
    [
        0.0000000000000000e0,
        6.8571428571428572e-1,
        6.8571428571428572e-1,
        0.0000000000000000e0,
        1.5439160309836284e0,
    ], // p=4
];

/// ∫_T |∇φ_j|² dV on the reference tetrahedron, for j = 10..14 (bubble functions only).
///
/// Derived analytically using ∫_T λ0^a λ1^b λ2^c λ3^d dV = a!b!c!d!/(a+b+c+d+3)!
///
/// **Face bubbles** φ10..φ13 = 27λaλbλc — all four are equal by tet symmetry.
///   ∇φ10 = 27[λ2(λ0−λ1), λ1(λ0−λ2), −λ1λ2]  (and permutations for φ11..φ13)
///   |∇φ10|²/729 = λ2²(λ0−λ1)² + λ1²(λ0−λ2)² + λ1²λ2²
///   Integrating term by term:
///     ∫ λ0²λ1²  = ∫ λ0²λ2²  = ∫ λ1²λ2²  = 1/1260
///     ∫ λ0λ1²λ2 = ∫ λ0λ1λ2² = 1/2520
///   → ∫_T |∇φ|²/729 = 1/420   → ∫_T |∇φ|² = 729/420 = **243/140**
///
/// **Interior bubble** φ14 = 256λ0λ1λ2λ3.
///   ∇φ14 = 256[λ2λ3(λ0−λ1), λ1λ3(λ0−λ2), λ1λ2(λ0−λ3)]
///   By tet symmetry the three squared terms are equal:
///   ∫_T |∇φ14|² = 3·256²·∫_T λ2²λ3²(λ0−λ1)²
///   ∫_T λ0²λ2²λ3² = ∫_T λ1²λ2²λ3² = 1/45360,  ∫_T λ0λ1λ2²λ3² = 1/90720
///   → 3·256²·(1/45360 − 2/90720 + 1/45360) = 3·65536/45360 = **4096/945**
pub const BUBBLE_GRAD_NORM_SQ: [f64; 5] = [
    243.0 / 140.0,  // φ10 = 27λ0λ1λ2  (face bubble)
    243.0 / 140.0,  // φ11 = 27λ0λ1λ3  (face bubble)
    243.0 / 140.0,  // φ12 = 27λ0λ2λ3  (face bubble)
    243.0 / 140.0,  // φ13 = 27λ1λ2λ3  (face bubble)
    4096.0 / 945.0, // φ14 = 256λ0λ1λ2λ3  (interior bubble)
];

pub fn shape_function(i: usize, uvw: &[f64; 3]) -> f64 {
    let (u, v, w) = (uvw[0], uvw[1], uvw[2]);
    let z = 1.0 - u - v - w;
    SHAPE_FUNCTIONS[i](u, v, w, z)
}

pub fn d_shape_function(i: usize, d: usize, uvw: &[f64; 3]) -> f64 {
    let (u, v, w) = (uvw[0], uvw[1], uvw[2]);
    let z = 1.0 - u - v - w;
    match d {
        0 => DU_SHAPE_FUNCTIONS[i](u, v, w, z),
        1 => DV_SHAPE_FUNCTIONS[i](u, v, w, z),
        2 => DW_SHAPE_FUNCTIONS[i](u, v, w, z),
        _ => 0.0,
    }
}

/// Build the 15×15 local stiffness matrix (stack-allocated):
///
///   K[i, j] = Σ_{quad} WEIGHTS[quad] · (∇φ_i(q) · sym_grad · ∇φ_j(q))
///
/// Uses the 14-point quadrature rule (exact for P5).
pub fn build_local_stiffness(sym_grad: &Matrix3<f64>) -> SMatrix<f64, 15, 15> {
    let mut k = SMatrix::<f64, 15, 15>::zeros();
    for i_quad in 0..14 {
        let w = super::tet_14::WEIGHTS[i_quad];
        let q = &super::tet_14::POINTS[i_quad];
        let grads: [Vector3<f64>; 15] = std::array::from_fn(|i| {
            Vector3::new(
                d_shape_function(i, 0, q),
                d_shape_function(i, 1, q),
                d_shape_function(i, 2, q),
            )
        });
        for i in 0..15 {
            let a_grad_i = sym_grad * grads[i];
            for j in 0..15 {
                k[(i, j)] += w * a_grad_i.dot(&grads[j]);
            }
        }
    }
    k
}

/// Apply the local stiffness operator to a 15-DOF element vector.
///
/// Computes:  output = K · input
///
/// Uses on-the-fly gradient evaluation at the 14 quadrature points
/// without assembling the full stiffness matrix.
pub fn mlt_stiffness(input: &[f64; 15], sym_grad: &Matrix3<f64>, output: &mut [f64; 15]) {
    output.fill(0.0);
    for i_quad in 0..14 {
        let w = super::tet_14::WEIGHTS[i_quad];
        let q = &super::tet_14::POINTS[i_quad];
        let grads: [[f64; 3]; 15] = std::array::from_fn(|i| {
            [
                d_shape_function(i, 0, q),
                d_shape_function(i, 1, q),
                d_shape_function(i, 2, q),
            ]
        });
        let mut g = [0.0f64; 3];
        for i in 0..15 {
            g[0] += grads[i][0] * input[i];
            g[1] += grads[i][1] * input[i];
            g[2] += grads[i][2] * input[i];
        }
        let ag = w * (sym_grad * Vector3::new(g[0], g[1], g[2]));
        for i in 0..15 {
            output[i] += grads[i][0] * ag[0] + grads[i][1] * ag[1] + grads[i][2] * ag[2];
        }
    }
}

/// Optimised exact operator for K_PP, K_PB, K_BP on 15 DOFs (K_BB approximated by a diagonal matrix).
///
/// The 15×15 stiffness matrix is split into four blocks:
///
/// - **K_PP (P2 × P2):** exact, delegated to [`super::tet_10::mlt_stiffness`].
///
/// - **K_PB / K_BP (P2 × bubble):** exact via Green's formula with boundary correction.
///   Integration by parts gives:
///   `K[i,j] = -div(A∇φ_i) · ∫_T φ_j + ∫_{∂T} (A∇φ_i · n) · φ_j dσ`
///   Since φ_i is P2, `div(A∇φ_i) = A:H(φ_i)` is constant (from [`super::tet_10::HESSIANS`]).
///   For the interior bubble (φ14 = 256λ0λ1λ2λ3), the boundary term vanishes.
///   For face bubbles (φ10–φ13 = 27λaλbλc), only one face contributes.  Since
///   A∇φ_i · n̂ is linear on that face, the integral is evaluated exactly by:
///   `∫_face f · 27λaλbλc dσ = (9 · area / 20) · f(centroid_face)`
///   Equivalently: weight `9/20` applied to `(A · n_flux) · ∇φ_i(centroid_face)`,
///   where `n_flux = n̂ · area`, where centroid_face is the centtroid of the face.
///
/// - **K_BB (bubble × bubble):** exact, via the precomputed basis matrices `KBB_00..KBB_22`.
pub fn mlt_stiffness_opt(input: &[f64; 15], sym_grad: &Matrix3<f64>, output: &mut [f64; 15]) {
    let input_p2: &[f64; 10] = input[..10].try_into().unwrap();

    // Extract the 6 independent components of sym_grad once.
    // Reused for A·g (K_PP), A·n_flux_k (K_PB/K_BP boundary), and div_p2 (volume term).
    let a00 = sym_grad[(0, 0)];
    let a11 = sym_grad[(1, 1)];
    let a22 = sym_grad[(2, 2)];
    let a01 = sym_grad[(0, 1)];
    let a02 = sym_grad[(0, 2)];
    let a12 = sym_grad[(1, 2)];

    // ── K_BB exact: K_BB = a00·KBB_00 + a01·KBB_01 + ... + a22·KBB_22 ──
    for p in 0..5 {
        let mut s = 0.0f64;
        for q in 0..5 {
            s += (a00 * KBB_00[p][q]
                + a01 * KBB_01[p][q]
                + a02 * KBB_02[p][q]
                + a11 * KBB_11[p][q]
                + a12 * KBB_12[p][q]
                + a22 * KBB_22[p][q])
                * input[10 + q];
        }
        output[10 + p] = s;
    }

    // ── K_PP + K_PB boundary: output[..10] = M^T · (W·A·M·input_p2 + t_bdry) ──
    //
    // A·n_flux_k for the 4 face bubbles (n_flux = outward normal × area, sparse):
    //   k=0: n=(0,0,−½) → −½·(a02, a12, a22)
    //   k=1: n=(0,−½,0) → −½·(a01, a11, a12)
    //   k=2: n=(−½,0,0) → −½·(a00, a01, a02)
    //   k=3: n=(½,½,½)  →  ½·(a00+a01+a02, a01+a11+a12, a02+a12+a22)
    let an = [
        [-0.5 * a02, -0.5 * a12, -0.5 * a22],
        [-0.5 * a01, -0.5 * a11, -0.5 * a12],
        [-0.5 * a00, -0.5 * a01, -0.5 * a02],
        [
            0.5 * (a00 + a01 + a02),
            0.5 * (a01 + a11 + a12),
            0.5 * (a02 + a12 + a22),
        ],
    ];

    // Combined loop over face centroids (k=0..3):
    //   - K_BP boundary: dot(an_k, g_k) computed before overwriting g_k
    //   - buf_grad overwritten with W_FACE·(A·g_k) + WEIGHT·x_{10+k}·an_k
    let mut buf_grad = [0.0f64; 15];
    super::tet_10_5::gradient_interpolation(input_p2, &mut buf_grad);

    const W_FACE: f64 = 3.0 / 40.0;
    const WEIGHT: f64 = 9.0 / 20.0;

    for k in 0..4 {
        let g0 = buf_grad[k * 3];
        let g1 = buf_grad[k * 3 + 1];
        let g2 = buf_grad[k * 3 + 2];
        let (b0, b1, b2) = (an[k][0], an[k][1], an[k][2]);
        output[10 + k] += WEIGHT * (b0 * g0 + b1 * g1 + b2 * g2); // K_BP boundary
        let ag0 = a00 * g0 + a01 * g1 + a02 * g2;
        let ag1 = a01 * g0 + a11 * g1 + a12 * g2;
        let ag2 = a02 * g0 + a12 * g1 + a22 * g2;
        let s = WEIGHT * input[10 + k];
        buf_grad[k * 3] = W_FACE * ag0 + s * b0; // K_PP + K_PB folded
        buf_grad[k * 3 + 1] = W_FACE * ag1 + s * b1;
        buf_grad[k * 3 + 2] = W_FACE * ag2 + s * b2;
    }

    // Interior centroid (k=4): K_PP only, no boundary contribution
    {
        let g0 = buf_grad[12];
        let g1 = buf_grad[13];
        let g2 = buf_grad[14];
        buf_grad[12] = super::tet_5::W_INTERIOR * (a00 * g0 + a01 * g1 + a02 * g2);
        buf_grad[13] = super::tet_5::W_INTERIOR * (a01 * g0 + a11 * g1 + a12 * g2);
        buf_grad[14] = super::tet_5::W_INTERIOR * (a02 * g0 + a12 * g1 + a22 * g2);
    }

    // M^T · buf_grad → K_PP + K_PB boundary in output[..10]
    super::tet_10_5::transpose_gradient_interpolation(&buf_grad, &mut output[..10]);

    // ── K_PB / K_BP volume term: -div(A∇φ_i) · ∫_T φ_j ─────────────────
    //
    // div(A∇φ_i) = A:H(φ_i) is constant for P2; HESSIANS is sparse.
    let d01 = 2.0 * a01;
    let d02 = 2.0 * a02;
    let d12 = 2.0 * a12;
    let div_p2 = [
        4.0 * (a00 + a11 + a22 + d01 + d02 + d12), // φ0
        4.0 * a00,                                 // φ1
        4.0 * a11,                                 // φ2
        4.0 * a22,                                 // φ3
        -4.0 * (2.0 * a00 + d01 + d02),            // φ4
        4.0 * d01,                                 // φ5
        -4.0 * (2.0 * a11 + d01 + d12),            // φ6
        -4.0 * (2.0 * a22 + d02 + d12),            // φ7
        4.0 * d02,                                 // φ8
        4.0 * d12,                                 // φ9
    ];

    // ∫_T φ_j  for j = 10..14
    const BUBBLE_MASS: [f64; 5] = [
        3.0 / 80.0,   // φ10 = 27λ0λ1λ2
        3.0 / 80.0,   // φ11 = 27λ0λ1λ3
        3.0 / 80.0,   // φ12 = 27λ0λ2λ3
        3.0 / 80.0,   // φ13 = 27λ1λ2λ3
        16.0 / 315.0, // φ14 = 256λ0λ1λ2λ3
    ];

    let bubble_dot: f64 = (0..5).map(|j| BUBBLE_MASS[j] * input[10 + j]).sum();
    for i in 0..10 {
        output[i] -= div_p2[i] * bubble_dot;
    }

    let div_dot: f64 = (0..10).map(|i| div_p2[i] * input[i]).sum();
    for j in 0..5 {
        output[10 + j] -= BUBBLE_MASS[j] * div_dot;
    }
}
