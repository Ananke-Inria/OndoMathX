use nalgebra::{Matrix3, SMatrix, Vector3};

/// P2 Lagrange finite element on the reference tetrahedron — 10 nodes.
///
/// Node numbering and positions:
///   0: (0,0,0)   vertex V0            φ0 = λ0(2λ0−1)
///   1: (1,0,0)   vertex V1            φ1 = λ1(2λ1−1)
///   2: (0,1,0)   vertex V2            φ2 = λ2(2λ2−1)
///   3: (0,0,1)   vertex V3            φ3 = λ3(2λ3−1)
///   4: (½,0,0)   midpoint edge V0−V1  φ4 = 4λ0λ1
///   5: (½,½,0)   midpoint edge V1−V2  φ5 = 4λ1λ2
///   6: (0,½,0)   midpoint edge V0−V2  φ6 = 4λ0λ2
///   7: (0,0,½)   midpoint edge V0−V3  φ7 = 4λ0λ3
///   8: (½,0,½)   midpoint edge V1−V3  φ8 = 4λ1λ3
///   9: (0,½,½)   midpoint edge V2−V3  φ9 = 4λ2λ3
///
/// with λ0=1−u−v−w,  λ1=u,  λ2=v,  λ3=w.
/// Arguments to function arrays: (u, v, w, z) where z = λ0 = 1−u−v−w.

pub const POINTS: [[f64; 3]; 10] = [
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
];

// Shape functions on the reference tetrahedron.
// Arguments: (u, v, w, z) where z = 1 - u - v - w.
pub const SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 10] = [
    |_u, _v, _w, z| z * (2.0 * z - 1.0), // φ0 = λ0(2λ0−1)
    |u, _v, _w, _z| u * (2.0 * u - 1.0), // φ1 = λ1(2λ1−1)
    |_u, v, _w, _z| v * (2.0 * v - 1.0), // φ2 = λ2(2λ2−1)
    |_u, _v, w, _z| w * (2.0 * w - 1.0), // φ3 = λ3(2λ3−1)
    |u, _v, _w, z| 4.0 * z * u,          // φ4 = 4λ0λ1
    |u, v, _w, _z| 4.0 * u * v,          // φ5 = 4λ1λ2
    |_u, v, _w, z| 4.0 * z * v,          // φ6 = 4λ0λ2
    |_u, _v, w, z| 4.0 * z * w,          // φ7 = 4λ0λ3
    |u, _v, w, _z| 4.0 * u * w,          // φ8 = 4λ1λ3
    |_u, v, w, _z| 4.0 * v * w,          // φ9 = 4λ2λ3
];

pub const DU_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 10] = [
    |_u, _v, _w, z| 1.0 - 4.0 * z, // ∂φ0/∂u = 1−4z
    |u, _v, _w, _z| 4.0 * u - 1.0, // ∂φ1/∂u = 4u−1
    |_u, _v, _w, _z| 0.0,          // ∂φ2/∂u = 0
    |_u, _v, _w, _z| 0.0,          // ∂φ3/∂u = 0
    |u, _v, _w, z| 4.0 * (z - u),  // ∂φ4/∂u = 4(z−u)
    |_u, v, _w, _z| 4.0 * v,       // ∂φ5/∂u = 4v
    |_u, v, _w, _z| -4.0 * v,      // ∂φ6/∂u = −4v
    |_u, _v, w, _z| -4.0 * w,      // ∂φ7/∂u = −4w
    |_u, _v, w, _z| 4.0 * w,       // ∂φ8/∂u = 4w
    |_u, _v, _w, _z| 0.0,          // ∂φ9/∂u = 0
];

pub const DV_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 10] = [
    |_u, _v, _w, z| 1.0 - 4.0 * z, // ∂φ0/∂v = 1−4z
    |_u, _v, _w, _z| 0.0,          // ∂φ1/∂v = 0
    |_u, v, _w, _z| 4.0 * v - 1.0, // ∂φ2/∂v = 4v−1
    |_u, _v, _w, _z| 0.0,          // ∂φ3/∂v = 0
    |u, _v, _w, _z| -4.0 * u,      // ∂φ4/∂v = −4u
    |u, _v, _w, _z| 4.0 * u,       // ∂φ5/∂v = 4u
    |_u, v, _w, z| 4.0 * (z - v),  // ∂φ6/∂v = 4(z−v)
    |_u, _v, w, _z| -4.0 * w,      // ∂φ7/∂v = −4w
    |_u, _v, _w, _z| 0.0,          // ∂φ8/∂v = 0
    |_u, _v, w, _z| 4.0 * w,       // ∂φ9/∂v = 4w
];

pub const DW_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 10] = [
    |_u, _v, _w, z| 1.0 - 4.0 * z, // ∂φ0/∂w = 1−4z
    |_u, _v, _w, _z| 0.0,          // ∂φ1/∂w = 0
    |_u, _v, _w, _z| 0.0,          // ∂φ2/∂w = 0
    |_u, _v, w, _z| 4.0 * w - 1.0, // ∂φ3/∂w = 4w−1
    |u, _v, _w, _z| -4.0 * u,      // ∂φ4/∂w = −4u
    |_u, _v, _w, _z| 0.0,          // ∂φ5/∂w = 0
    |_u, v, _w, _z| -4.0 * v,      // ∂φ6/∂w = −4v
    |_u, _v, w, z| 4.0 * (z - w),  // ∂φ7/∂w = 4(z−w)
    |u, _v, _w, _z| 4.0 * u,       // ∂φ8/∂w = 4u
    |_u, v, _w, _z| 4.0 * v,       // ∂φ9/∂w = 4v
];

/// Symmetric Hessian of each basis function, stored in Voigt order:
///   [∂²φ/∂u², ∂²φ/∂v², ∂²φ/∂w², ∂²φ/∂u∂v, ∂²φ/∂u∂w, ∂²φ/∂v∂w]
///
/// Because all φ_i are quadratic polynomials, their second derivatives are
/// constant over the element — no point evaluation is needed.
pub const HESSIANS: [[f64; 6]; 10] = [
    //  uu    vv    ww    uv    uw    vw
    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],    // φ0 = λ0(2λ0−1)
    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],    // φ1 = λ1(2λ1−1)
    [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],    // φ2 = λ2(2λ2−1)
    [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],    // φ3 = λ3(2λ3−1)
    [-8.0, 0.0, 0.0, -4.0, -4.0, 0.0], // φ4 = 4λ0λ1
    [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],    // φ5 = 4λ1λ2
    [0.0, -8.0, 0.0, -4.0, 0.0, -4.0], // φ6 = 4λ0λ2
    [0.0, 0.0, -8.0, 0.0, -4.0, -4.0], // φ7 = 4λ0λ3
    [0.0, 0.0, 0.0, 0.0, 4.0, 0.0],    // φ8 = 4λ1λ3
    [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],    // φ9 = 4λ2λ3
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

/// Build the 10×10 local stiffness matrix (stack-allocated):
///
///   K[i, j] = Σ_{quad} WEIGHTS[quad] · (∇φ_i(q) · sym_grad · ∇φ_j(q))
///
/// Useful for testing or when the same matrix is applied to many vectors.
pub fn build_local_stiffness(sym_grad: &Matrix3<f64>) -> SMatrix<f64, 10, 10> {
    let mut k = SMatrix::<f64, 10, 10>::zeros();
    for i_quad in 0..4 {
        let w = super::tet_4::WEIGHTS[i_quad];
        let q = &super::tet_4::POINTS[i_quad];
        let grads: [Vector3<f64>; 10] = std::array::from_fn(|i| {
            Vector3::new(
                d_shape_function(i, 0, q),
                d_shape_function(i, 1, q),
                d_shape_function(i, 2, q),
            )
        });
        for i in 0..10 {
            let a_grad_i = sym_grad * grads[i];
            for j in 0..10 {
                k[(i, j)] += w * a_grad_i.dot(&grads[j]);
            }
        }
    }
    k
}

/// Apply the local stiffness operator to a 10-DOF element vector.
///
/// Computes:  output = M^T · (W · A · M · input)
///
/// where:
///   M      — gradient interpolation matrix (12×10): maps the 10 nodal DOFs
///             to gradients at the 4 quadrature points (3 directions each)
///   A      — the 3×3 symmetric matrix passed as `sym_grad`
///   W      — diagonal matrix of quadrature weights from `tet_4::WEIGHTS`
///   M^T    — transpose interpolation (12→10)
pub fn mlt_stiffness(input: &[f64; 10], sym_grad: &Matrix3<f64>, output: &mut [f64; 10]) {
    // Step 1: M · input  →  gradients at the 4 quadrature points (12 values)
    let mut buf_grad = [0.0f64; 12];
    super::tet_10_4::gradient_interpolation(input, &mut buf_grad);

    // Step 2: for each quadrature point, apply  W[i_quad] · sym_grad  to the
    //         3-component gradient vector
    for i_quad in 0..4 {
        let g0 = buf_grad[i_quad * 3];
        let g1 = buf_grad[i_quad * 3 + 1];
        let g2 = buf_grad[i_quad * 3 + 2];
        let result = sym_grad * Vector3::new(g0, g1, g2);
        let w = super::tet_4::WEIGHTS[i_quad];
        buf_grad[i_quad * 3] = w * result[0];
        buf_grad[i_quad * 3 + 1] = w * result[1];
        buf_grad[i_quad * 3 + 2] = w * result[2];
    }

    // Step 3: M^T · buf_grad  →  10 output DOFs
    super::tet_10_4::transpose_gradient_interpolation(&buf_grad, output);
}
