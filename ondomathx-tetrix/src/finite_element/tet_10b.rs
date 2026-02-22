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
    |_u, _v, _w,  z| z * (2.0 * z - 1.0),          // φ0  = λ0(2λ0−1)
    | u, _v, _w, _z| u * (2.0 * u - 1.0),           // φ1  = λ1(2λ1−1)
    |_u,  v, _w, _z| v * (2.0 * v - 1.0),           // φ2  = λ2(2λ2−1)
    |_u, _v,  w, _z| w * (2.0 * w - 1.0),           // φ3  = λ3(2λ3−1)
    | u, _v, _w,  z| 4.0 * z * u,                   // φ4  = 4λ0λ1
    | u,  v, _w, _z| 4.0 * u * v,                   // φ5  = 4λ1λ2
    |_u,  v, _w,  z| 4.0 * z * v,                   // φ6  = 4λ0λ2
    |_u, _v,  w,  z| 4.0 * z * w,                   // φ7  = 4λ0λ3
    | u, _v,  w, _z| 4.0 * u * w,                   // φ8  = 4λ1λ3
    |_u,  v,  w, _z| 4.0 * v * w,                   // φ9  = 4λ2λ3
    // Bubble functions (nodes 10–14)
    | u,  v, _w,  z| 27.0 * z * u * v,              // φ10 = 27λ0λ1λ2
    | u, _v,  w,  z| 27.0 * z * u * w,              // φ11 = 27λ0λ1λ3
    |_u,  v,  w,  z| 27.0 * z * v * w,              // φ12 = 27λ0λ2λ3
    | u,  v,  w, _z| 27.0 * u * v * w,              // φ13 = 27λ1λ2λ3
    | u,  v,  w,  z| 256.0 * z * u * v * w,         // φ14 = 256λ0λ1λ2λ3
];

pub const DU_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |_u, _v, _w,  z| 1.0 - 4.0 * z,                // ∂φ0/∂u  = 1−4z
    | u, _v, _w, _z| 4.0 * u - 1.0,                // ∂φ1/∂u  = 4u−1
    |_u, _v, _w, _z| 0.0,                           // ∂φ2/∂u  = 0
    |_u, _v, _w, _z| 0.0,                           // ∂φ3/∂u  = 0
    | u, _v, _w,  z| 4.0 * (z - u),                // ∂φ4/∂u  = 4(z−u)
    |_u,  v, _w, _z| 4.0 * v,                       // ∂φ5/∂u  = 4v
    |_u,  v, _w, _z| -4.0 * v,                      // ∂φ6/∂u  = −4v
    |_u, _v,  w, _z| -4.0 * w,                      // ∂φ7/∂u  = −4w
    |_u, _v,  w, _z| 4.0 * w,                       // ∂φ8/∂u  = 4w
    |_u, _v, _w, _z| 0.0,                           // ∂φ9/∂u  = 0
    | u,  v, _w,  z| 27.0 * v * (z - u),            // ∂φ10/∂u = 27v(z−u)
    | u, _v,  w,  z| 27.0 * w * (z - u),            // ∂φ11/∂u = 27w(z−u)
    |_u,  v,  w, _z| -27.0 * v * w,                 // ∂φ12/∂u = −27vw
    |_u,  v,  w, _z| 27.0 * v * w,                  // ∂φ13/∂u = 27vw
    | u,  v,  w,  z| 256.0 * v * w * (z - u),       // ∂φ14/∂u = 256vw(z−u)
];

pub const DV_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |_u, _v, _w,  z| 1.0 - 4.0 * z,                // ∂φ0/∂v  = 1−4z
    |_u, _v, _w, _z| 0.0,                           // ∂φ1/∂v  = 0
    |_u,  v, _w, _z| 4.0 * v - 1.0,                // ∂φ2/∂v  = 4v−1
    |_u, _v, _w, _z| 0.0,                           // ∂φ3/∂v  = 0
    | u, _v, _w, _z| -4.0 * u,                      // ∂φ4/∂v  = −4u
    | u, _v, _w, _z| 4.0 * u,                       // ∂φ5/∂v  = 4u
    |_u,  v, _w,  z| 4.0 * (z - v),                // ∂φ6/∂v  = 4(z−v)
    |_u, _v,  w, _z| -4.0 * w,                      // ∂φ7/∂v  = −4w
    |_u, _v, _w, _z| 0.0,                           // ∂φ8/∂v  = 0
    |_u, _v,  w, _z| 4.0 * w,                       // ∂φ9/∂v  = 4w
    | u,  v, _w,  z| 27.0 * u * (z - v),            // ∂φ10/∂v = 27u(z−v)
    | u, _v,  w, _z| -27.0 * u * w,                 // ∂φ11/∂v = −27uw
    |_u,  v,  w,  z| 27.0 * w * (z - v),            // ∂φ12/∂v = 27w(z−v)
    | u, _v,  w, _z| 27.0 * u * w,                  // ∂φ13/∂v = 27uw
    | u,  v,  w,  z| 256.0 * u * w * (z - v),       // ∂φ14/∂v = 256uw(z−v)
];

pub const DW_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |_u, _v, _w,  z| 1.0 - 4.0 * z,                // ∂φ0/∂w  = 1−4z
    |_u, _v, _w, _z| 0.0,                           // ∂φ1/∂w  = 0
    |_u, _v, _w, _z| 0.0,                           // ∂φ2/∂w  = 0
    |_u, _v,  w, _z| 4.0 * w - 1.0,                // ∂φ3/∂w  = 4w−1
    | u, _v, _w, _z| -4.0 * u,                      // ∂φ4/∂w  = −4u
    |_u, _v, _w, _z| 0.0,                           // ∂φ5/∂w  = 0
    |_u,  v, _w, _z| -4.0 * v,                      // ∂φ6/∂w  = −4v
    |_u, _v,  w,  z| 4.0 * (z - w),                // ∂φ7/∂w  = 4(z−w)
    | u, _v, _w, _z| 4.0 * u,                       // ∂φ8/∂w  = 4u
    |_u,  v, _w, _z| 4.0 * v,                       // ∂φ9/∂w  = 4v
    | u,  v, _w, _z| -27.0 * u * v,                 // ∂φ10/∂w = −27uv
    | u, _v,  w,  z| 27.0 * u * (z - w),            // ∂φ11/∂w = 27u(z−w)
    |_u,  v,  w,  z| 27.0 * v * (z - w),            // ∂φ12/∂w = 27v(z−w)
    | u,  v, _w, _z| 27.0 * u * v,                  // ∂φ13/∂w = 27uv
    | u,  v,  w,  z| 256.0 * u * v * (z - w),       // ∂φ14/∂w = 256uv(z−w)
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
