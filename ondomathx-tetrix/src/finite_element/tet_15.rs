/// 15-point quadrature rule for the reference tetrahedron,  
///
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

pub const WV: f64 = 17.0 / 5040.0;
pub const WE: f64 = 2.0 / 315.0;
pub const WF: f64 = 9.0 / 560.0;
pub const WB: f64 = 16.0 / 315.0;

pub const WEIGHTS: [f64; 15] = [WV, WV, WV, WV, WE, WE, WE, WE, WE, WE, WF, WF, WF, WF, WB];

// Shape functions on the reference tetrahedron.
// Arguments: (u, v, w, z) where z = 1 - u - v - w.
pub const SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |u, v, w, z| {
        z * (1.0 - 2.0 * u - 2.0 * v - 2.0 * w + 3.0 * u * v + 3.0 * u * w + 3.0 * v * w
            - 4.0 * u * v * w)
    },
    |u, v, w, z| {
        u * (1.0 - 2.0 * v - 2.0 * w - 2.0 * z + 3.0 * v * w + 3.0 * v * z + 3.0 * w * z
            - 4.0 * v * w * z)
    },
    |u, v, w, z| {
        v * (1.0 - 2.0 * u - 2.0 * w - 2.0 * z + 3.0 * u * w + 3.0 * u * z + 3.0 * w * z
            - 4.0 * u * w * z)
    },
    |u, v, w, z| {
        w * (1.0 - 2.0 * u - 2.0 * v - 2.0 * z + 3.0 * u * v + 3.0 * u * z + 3.0 * v * z
            - 4.0 * u * v * z)
    },
    |u, v, w, z| 4.0 * u * z * (1.0 - 3.0 * v - 3.0 * w + 8.0 * v * w),
    |u, v, w, z| 4.0 * u * v * (1.0 - 3.0 * w - 3.0 * z + 8.0 * w * z),
    |u, v, w, z| 4.0 * v * z * (1.0 - 3.0 * u - 3.0 * w + 8.0 * u * w),
    |u, v, w, z| 4.0 * w * z * (1.0 - 3.0 * u - 3.0 * v + 8.0 * u * v),
    |u, v, w, z| 4.0 * u * w * (1.0 - 3.0 * v - 3.0 * z + 8.0 * v * z),
    |u, v, w, z| 4.0 * v * w * (1.0 - 3.0 * u - 3.0 * z + 8.0 * u * z),
    |u, v, w, z| 27.0 * z * u * v * (1.0 - 4.0 * w),
    |u, v, w, z| 27.0 * z * u * w * (1.0 - 4.0 * v),
    |u, v, w, z| 27.0 * z * v * w * (1.0 - 4.0 * u),
    |u, v, w, z| 27.0 * u * v * w * (1.0 - 4.0 * z),
    |u, v, w, _z| 256.0 * u * v * w * (1.0 - u - v - w),
];

pub const DU_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |u, v, w, z| {
        2.0 * u + 2.0 * v + 2.0 * w - 2.0 * z - 3.0 * u * v - 3.0 * u * w - 3.0 * v * w
            + 3.0 * v * z
            + 3.0 * w * z
            + 4.0 * u * v * w
            - 4.0 * v * w * z
            - 1.0
    },
    |u, v, w, z| {
        2.0 * u - 2.0 * v - 2.0 * w - 2.0 * z - 3.0 * u * v - 3.0 * u * w
            + 3.0 * v * w
            + 3.0 * v * z
            + 3.0 * w * z
            + 4.0 * u * v * w
            - 4.0 * v * w * z
            + 1.0
    },
    |u, v, w, z| v * (4.0 * w - 3.0) * (u - z),
    |u, v, w, z| w * (4.0 * v - 3.0) * (u - z),
    |u, v, w, z| 4.0 * (u - z) * (3.0 * v + 3.0 * w - 8.0 * v * w - 1.0),
    |u, v, w, z| 4.0 * v * (3.0 * u - 3.0 * w - 3.0 * z - 8.0 * u * w + 8.0 * w * z + 1.0),
    |u, v, w, z| 4.0 * v * (3.0 * u + 3.0 * w - 3.0 * z - 8.0 * u * w + 8.0 * w * z - 1.0),
    |u, v, w, z| 4.0 * w * (3.0 * u + 3.0 * v - 3.0 * z - 8.0 * u * v + 8.0 * v * z - 1.0),
    |u, v, w, z| 4.0 * w * (3.0 * u - 3.0 * v - 3.0 * z - 8.0 * u * v + 8.0 * v * z + 1.0),
    |u, v, w, z| 32.0 * v * w * (z - u),
    |u, v, w, z| 27.0 * v * (4.0 * w - 1.0) * (u - z),
    |u, v, w, z| 27.0 * w * (4.0 * v - 1.0) * (u - z),
    |u, v, w, z| 27.0 * v * w * (4.0 * u - 4.0 * z - 1.0),
    |u, v, w, z| 27.0 * v * w * (4.0 * u - 4.0 * z + 1.0),
    |u, v, w, z| 256.0 * v * w * (z - u),
];

pub const DV_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |u, v, w, z| {
        2.0 * u + 2.0 * v + 2.0 * w - 2.0 * z - 3.0 * u * v - 3.0 * u * w - 3.0 * v * w
            + 3.0 * u * z
            + 3.0 * w * z
            + 4.0 * u * v * w
            - 4.0 * u * w * z
            - 1.0
    },
    |u, v, w, z| u * (4.0 * w - 3.0) * (v - z),
    |u, v, w, z| {
        2.0 * v - 2.0 * u - 2.0 * w - 2.0 * z - 3.0 * u * v + 3.0 * u * w - 3.0 * v * w
            + 3.0 * u * z
            + 3.0 * w * z
            + 4.0 * u * v * w
            - 4.0 * u * w * z
            + 1.0
    },
    |u, v, w, z| w * (4.0 * u - 3.0) * (v - z),
    |u, v, w, z| 4.0 * u * (3.0 * v + 3.0 * w - 3.0 * z - 8.0 * v * w + 8.0 * w * z - 1.0),
    |u, v, w, z| 4.0 * u * (3.0 * v - 3.0 * w - 3.0 * z - 8.0 * v * w + 8.0 * w * z + 1.0),
    |u, v, w, z| 4.0 * (v - z) * (3.0 * u + 3.0 * w - 8.0 * u * w - 1.0),
    |u, v, w, z| 4.0 * w * (3.0 * u + 3.0 * v - 3.0 * z - 8.0 * u * v + 8.0 * u * z - 1.0),
    |u, v, w, z| 32.0 * u * w * (z - v),
    |u, v, w, z| 4.0 * w * (1.0 - 3.0 * u + 3.0 * v - 3.0 * z - 8.0 * u * v + 8.0 * u * z),
    |u, v, w, z| 27.0 * u * (4.0 * w - 1.0) * (v - z),
    |u, v, w, z| 27.0 * u * w * (4.0 * v - 4.0 * z - 1.0),
    |u, v, w, z| 27.0 * w * (4.0 * u - 1.0) * (v - z),
    |u, v, w, z| 27.0 * u * w * (4.0 * v - 4.0 * z + 1.0),
    |u, v, w, z| 256.0 * u * w * (z - v),
];

pub const DW_SHAPE_FUNCTIONS: [fn(f64, f64, f64, f64) -> f64; 15] = [
    |u, v, w, z| {
        2.0 * u + 2.0 * v + 2.0 * w - 2.0 * z - 3.0 * u * v - 3.0 * u * w - 3.0 * v * w
            + 3.0 * u * z
            + 3.0 * v * z
            + 4.0 * u * v * w
            - 4.0 * u * v * z
            - 1.0
    },
    |u, v, w, z| u * (4.0 * v - 3.0) * (w - z),
    |u, v, w, z| v * (4.0 * u - 3.0) * (w - z),
    |u, v, w, z| {
        2.0 * w - 2.0 * v - 2.0 * u - 2.0 * z + 3.0 * u * v - 3.0 * u * w - 3.0 * v * w
            + 3.0 * u * z
            + 3.0 * v * z
            + 4.0 * u * v * w
            - 4.0 * u * v * z
            + 1.0
    },
    |u, v, w, z| 4.0 * u * (3.0 * v + 3.0 * w - 3.0 * z - 8.0 * v * w + 8.0 * v * z - 1.0),
    |u, v, w, z| 32.0 * u * v * (z - w),
    |u, v, w, z| 4.0 * v * (3.0 * u + 3.0 * w - 3.0 * z - 8.0 * u * w + 8.0 * u * z - 1.0),
    |u, v, w, z| 4.0 * (w - z) * (3.0 * u + 3.0 * v - 8.0 * u * v - 1.0),
    |u, v, w, z| 4.0 * u * (1.0 - 3.0 * v + 3.0 * w - 3.0 * z - 8.0 * v * w + 8.0 * v * z),
    |u, v, w, z| 4.0 * v * (1.0 - 3.0 * u + 3.0 * w - 3.0 * z - 8.0 * u * w + 8.0 * u * z),
    |u, v, w, z| 27.0 * u * v * (4.0 * w - 4.0 * z - 1.0),
    |u, v, w, z| 27.0 * u * (4.0 * v - 1.0) * (w - z),
    |u, v, w, z| 27.0 * v * (4.0 * u - 1.0) * (w - z),
    |u, v, w, z| 27.0 * u * v * (4.0 * w - 4.0 * z + 1.0),
    |u, v, w, z| 256.0 * u * v * (z - w),
];

use nalgebra::{Matrix3, SMatrix, Vector3};

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
/// Useful for testing or when the same matrix is applied to many vectors.
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
/// Computes:  output = M^T · (W · A · M · input)
///
/// where:
///   M      — gradient interpolation matrix (42×15): maps the 15 nodal DOFs
///             to gradients at the 14 quadrature points (3 directions each)
///   A      — the 3×3 symmetric matrix passed as `sym_grad`
///   W      — diagonal matrix of quadrature weights from `tet_14::WEIGHTS`
///   M^T    — transpose interpolation (42→15)
pub fn mlt_stiffness(input: &[f64; 15], sym_grad: &Matrix3<f64>, output: &mut [f64; 15]) {
    // Step 1: M · input  →  gradients at the 14 quadrature points (42 values)
    let mut buf_grad = [0.0f64; 42];
    super::tet_15_14::gradient_interpolation(input, &mut buf_grad);

    // Step 2: for each quadrature point, apply  W[i_quad] · sym_grad  to the
    //         3-component gradient vector
    for i_quad in 0..14 {
        let g0 = buf_grad[i_quad * 3];
        let g1 = buf_grad[i_quad * 3 + 1];
        let g2 = buf_grad[i_quad * 3 + 2];
        let result = sym_grad * Vector3::new(g0, g1, g2);
        let w = super::tet_14::WEIGHTS[i_quad];
        buf_grad[i_quad * 3] = w * result[0];
        buf_grad[i_quad * 3 + 1] = w * result[1];
        buf_grad[i_quad * 3 + 2] = w * result[2];
    }

    // Step 3: M^T · buf_grad  →  15 output DOFs
    super::tet_15_14::transpose_gradient_interpolation(&buf_grad, output);
}

/// Apply the local stiffness operator via the `tet_10b` basis.
///
/// Converts `input` to the `tet_10b` representation, delegates to
/// [`super::tet_10b::mlt_stiffness_opt`], then converts the result back.
pub fn mlt_stiffness_opt(input: &[f64; 15], sym_grad: &Matrix3<f64>, output: &mut [f64; 15]) {
    let mut b = [0.0f64; 15];
    super::tet_15_10b::tet_15_to_tet_10b(input, &mut b);
    let mut c = [0.0f64; 15];
    super::tet_10b::mlt_stiffness_opt(&b, sym_grad, &mut c);
    super::tet_15_10b::tet_15_to_tet_10b_transpose(&c, output);
}
