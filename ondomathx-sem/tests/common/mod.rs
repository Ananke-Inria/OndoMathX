//! Meshes and deformations shared by the integration tests.
#![allow(dead_code)]

use ondomathx_mesh::diffeomorphism::Diffeomorphism;
use ondomathx_mesh::mesh::cube::Cube;
use ondomathx_mesh::mesh::square::Square;
use ondomathx_mesh::mesh::{Mesh, UnstructuredMesh};

/// f(x, y) = (x + 0.1·sin(y) + 0.05·xy, y + 0.2·x²): curved, non-affine.
/// The coupled `xy` term matters: a separable map (x-shift depending on y
/// only, y-shift on x only) sends rectangles to parallelograms.
pub fn bending_2d() -> Diffeomorphism {
    Diffeomorphism {
        inner_dim: 2,
        outer_dim: 2,
        params: vec![],
        diffeo: |_p, x, f| {
            f[0] = x[0] + 0.1 * x[1].sin() + 0.05 * x[0] * x[1];
            f[1] = x[1] + 0.2 * x[0] * x[0];
        },
        grad_diffeo: |_p, x, g| {
            g[0] = 1.0 + 0.05 * x[1];
            g[1] = 0.1 * x[1].cos() + 0.05 * x[0];
            g[2] = 0.4 * x[0];
            g[3] = 1.0;
        },
        inner_deformation: None,
    }
}

/// Rotation by 0.4 rad about the origin, plus a shift: affine.
pub fn rotation_2d() -> Diffeomorphism {
    Diffeomorphism {
        inner_dim: 2,
        outer_dim: 2,
        params: vec![0.4],
        diffeo: |p, x, f| {
            let (c, s) = (p[0].cos(), p[0].sin());
            f[0] = c * x[0] - s * x[1] + 0.3;
            f[1] = s * x[0] + c * x[1] - 0.1;
        },
        grad_diffeo: |p, _x, g| {
            let (c, s) = (p[0].cos(), p[0].sin());
            g.copy_from_slice(&[c, -s, s, c]);
        },
        inner_deformation: None,
    }
}

/// f(x, y) = (x, y·(1 + 0.3·x)): bilinear, so straight Q2 elements stay Q1.
pub fn bilinear_2d() -> Diffeomorphism {
    Diffeomorphism {
        inner_dim: 2,
        outer_dim: 2,
        params: vec![],
        diffeo: |_p, x, f| {
            f[0] = x[0];
            f[1] = x[1] * (1.0 + 0.3 * x[0]);
        },
        grad_diffeo: |_p, x, g| {
            g[0] = 1.0;
            g[1] = 0.0;
            g[2] = 0.3 * x[1];
            g[3] = 1.0 + 0.3 * x[0];
        },
        inner_deformation: None,
    }
}

/// f(x, y, z) = (x + 0.1·sin(z), y + 0.1·x², z + 0.05·xy).
pub fn bending_3d() -> Diffeomorphism {
    Diffeomorphism {
        inner_dim: 3,
        outer_dim: 3,
        params: vec![],
        diffeo: |_p, x, f| {
            f[0] = x[0] + 0.1 * x[2].sin();
            f[1] = x[1] + 0.1 * x[0] * x[0];
            f[2] = x[2] + 0.05 * x[0] * x[1];
        },
        grad_diffeo: |_p, x, g| {
            g.copy_from_slice(&[
                1.0,
                0.0,
                0.1 * x[2].cos(),
                0.2 * x[0],
                1.0,
                0.0,
                0.05 * x[1],
                0.05 * x[0],
                1.0,
            ]);
        },
        inner_deformation: None,
    }
}

pub fn square(
    nx: usize,
    ny: usize,
    q2: bool,
    deformation: Option<Diffeomorphism>,
) -> UnstructuredMesh {
    let mut mesh =
        Square::new([0.0, 0.0], 2.0, 1.0, nx, ny, deformation, Some(q2)).get_unstructured_mesh();
    mesh.apply_deformation();
    mesh
}

pub fn cube(n: [usize; 3], q2: bool, deformation: Option<Diffeomorphism>) -> UnstructuredMesh {
    let mut mesh = Cube::new(
        [0.0, 0.0, 0.0],
        1.0,
        2.0,
        1.5,
        n[0],
        n[1],
        n[2],
        deformation,
        Some(q2),
    )
    .get_unstructured_mesh();
    mesh.apply_deformation();
    mesh
}

/// Identity for `x ≤ 1`, non-affine for `x > 1`: with `s = max(x − 1, 0)²`,
/// f(x, y) = (x + 0.1·s·y, y + 0.05·s). On Square([0,2]×[0,1]) the elements
/// left of `x = 1` stay affine and the others become Q1 (or Q2), so one mesh
/// has two geometric groups.
pub fn partial_2d() -> Diffeomorphism {
    Diffeomorphism {
        inner_dim: 2,
        outer_dim: 2,
        params: vec![],
        diffeo: |_p, x, f| {
            let s = (x[0] - 1.0).max(0.0).powi(2);
            f[0] = x[0] + 0.1 * s * x[1];
            f[1] = x[1] + 0.05 * s;
        },
        grad_diffeo: |_p, x, g| {
            let (s, ds) = ((x[0] - 1.0).max(0.0).powi(2), 2.0 * (x[0] - 1.0).max(0.0));
            g[0] = 1.0 + 0.1 * ds * x[1];
            g[1] = 0.1 * s;
            g[2] = 0.05 * ds;
            g[3] = 1.0;
        },
        inner_deformation: None,
    }
}

/// 3D analogue of [`partial_2d`] on Cube([0,1]×[0,2]×[0,1.5]), switching at
/// `y = 1`: f = (x + 0.05·s, y + 0.1·s·z, z), `s = max(y − 1, 0)²`.
pub fn partial_3d() -> Diffeomorphism {
    Diffeomorphism {
        inner_dim: 3,
        outer_dim: 3,
        params: vec![],
        diffeo: |_p, x, f| {
            let s = (x[1] - 1.0).max(0.0).powi(2);
            f[0] = x[0] + 0.05 * s;
            f[1] = x[1] + 0.1 * s * x[2];
            f[2] = x[2];
        },
        grad_diffeo: |_p, x, g| {
            let (s, ds) = ((x[1] - 1.0).max(0.0).powi(2), 2.0 * (x[1] - 1.0).max(0.0));
            g.copy_from_slice(&[
                1.0,
                0.05 * ds,
                0.0,
                0.0,
                1.0 + 0.1 * ds * x[2],
                0.1 * s,
                0.0,
                0.0,
                1.0,
            ]);
        },
        inner_deformation: None,
    }
}
