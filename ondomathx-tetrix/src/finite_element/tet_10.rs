use nalgebra::{Matrix3, SMatrix, Vector3};

/// P2 Lagrange finite element on the reference tetrahedron — 10 nodes.
///
/// Node numbering and positions (x, y, z):
///   0: (0,0,0)   vertex V0            φ0 = λ0(2λ0−1)
///   1: (1,0,0)   vertex V1            φ1 = λ1(2λ1−1)
///   2: (0,1,0)   vertex V2            φ2 = λ2(2λ2−1)
///   3: (0,0,1)   vertex V3            φ3 = λ3(2λ3−1)
///   4: (½,0,0)   midpoint edge V0−V1  φ4 = 4λ0λ1
///   5: (0,½,0)   midpoint edge V0−V2  φ5 = 4λ0λ2
///   6: (0,0,½)   midpoint edge V0−V3  φ6 = 4λ0λ3
///   7: (½,½,0)   midpoint edge V1−V2  φ7 = 4λ1λ2
///   8: (½,0,½)   midpoint edge V1−V3  φ8 = 4λ1λ3
///   9: (0,½,½)   midpoint edge V2−V3  φ9 = 4λ2λ3
///
/// with barycentric coordinates  λ0=1−x−y−z,  λ1=x,  λ2=y,  λ3=z.

pub const POINTS: [[f64; 3]; 10] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
];

/// Partial derivative ∂φ_i/∂x_d evaluated at point `uvw`.
///
/// d=0 → ∂/∂x,  d=1 → ∂/∂y,  d=2 → ∂/∂z.
pub fn d_shape_function(i: usize, d: usize, uvw: &[f64; 3]) -> f64 {
    let (x, y, z) = (uvw[0], uvw[1], uvw[2]);
    // λ0 = 1−x−y−z
    match (i, d) {
        // φ0 = λ0(2λ0−1):  ∂φ0/∂x_d = −(4λ0−1) = 4x+4y+4z−3 for all d
        (0, 0) | (0, 1) | (0, 2) => 4.0 * (x + y + z) - 3.0,
        // φ1 = x(2x−1)
        (1, 0) => 4.0 * x - 1.0,
        (1, 1) | (1, 2) => 0.0,
        // φ2 = y(2y−1)
        (2, 1) => 4.0 * y - 1.0,
        (2, 0) | (2, 2) => 0.0,
        // φ3 = z(2z−1)
        (3, 2) => 4.0 * z - 1.0,
        (3, 0) | (3, 1) => 0.0,
        // φ4 = 4(1−x−y−z)x
        (4, 0) => 4.0 - 8.0 * x - 4.0 * y - 4.0 * z,
        (4, 1) => -4.0 * x,
        (4, 2) => -4.0 * x,
        // φ5 = 4(1−x−y−z)y
        (5, 0) => -4.0 * y,
        (5, 1) => 4.0 - 4.0 * x - 8.0 * y - 4.0 * z,
        (5, 2) => -4.0 * y,
        // φ6 = 4(1−x−y−z)z
        (6, 0) => -4.0 * z,
        (6, 1) => -4.0 * z,
        (6, 2) => 4.0 - 4.0 * x - 4.0 * y - 8.0 * z,
        // φ7 = 4xy
        (7, 0) => 4.0 * y,
        (7, 1) => 4.0 * x,
        (7, 2) => 0.0,
        // φ8 = 4xz
        (8, 0) => 4.0 * z,
        (8, 1) => 0.0,
        (8, 2) => 4.0 * x,
        // φ9 = 4yz
        (9, 0) => 0.0,
        (9, 1) => 4.0 * z,
        (9, 2) => 4.0 * y,
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
        let grads: [Vector3<f64>; 10] = std::array::from_fn(|i| Vector3::new(
            d_shape_function(i, 0, q),
            d_shape_function(i, 1, q),
            d_shape_function(i, 2, q),
        ));
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
        buf_grad[i_quad * 3]     = w * result[0];
        buf_grad[i_quad * 3 + 1] = w * result[1];
        buf_grad[i_quad * 3 + 2] = w * result[2];
    }

    // Step 3: M^T · buf_grad  →  10 output DOFs
    super::tet_10_4::transpose_gradient_interpolation(&buf_grad, output);
}
