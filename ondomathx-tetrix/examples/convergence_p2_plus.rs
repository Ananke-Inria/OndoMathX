//! Convergence of the P2⁺ space (`P2PlusSpace`) on a static problem.
//!
//! On the unit cube (`ondomathx-mesh/assets/Cube.mesh`, 6 tetrahedra,
//! refined uniformly: Freudenthal-Kuhn, shape-regular at every level), solves
//!
//! ```text
//! −Δu + u = f  in Ω = (0, 1)³,    ∂u/∂n = 0  on ∂Ω,
//! u = cos πx cos πy cos πz,       f = (3π² + 1) u,
//! ```
//!
//! (the Neumann condition is the natural one of the space, satisfied by `u`)
//! with the operators of the time loops: the stiffness `K` applied in DG
//! storage and the lumped mass `M`, the right-hand side `M f(x_i)`. The
//! system `(K + M) U = M F` is solved by conjugate gradients preconditioned
//! by the diagonal of `K + M`. The errors are computed with the 14-point
//! rule (`tet_14`) and the P2⁺ shape functions (`tet_15`); the expected
//! rates are 3 in L² and 2 in the H¹ seminorm.
//!
//! `cargo run --release -p ondomathx-tetrix --example convergence_p2_plus`

use std::f64::consts::PI;
use std::time::Instant;

use nalgebra::Vector3;
use ondomathx_mesh::mesh::UnstructuredMesh;
use ondomathx_tetrix::element::{tet_14, tet_15};
use ondomathx_tetrix::space::p2_plus::P2PlusSpace;
use ondomathx_toolbox::field::Constant;
use ondomathx_toolbox::lal;

/// Refinements of `Cube.mesh`: 6·8^level tetrahedra, h = 2^−level (the edge
/// of the cube cells, `Cube.mesh` being one cell split in 6).
const LEVELS: std::ops::RangeInclusive<usize> = 1..=5;
/// Relative residual of the conjugate gradients.
const TOL: f64 = 1e-12;

fn exact(x: &[f64; 3]) -> f64 {
    (PI * x[0]).cos() * (PI * x[1]).cos() * (PI * x[2]).cos()
}

fn exact_gradient(x: &[f64; 3]) -> Vector3<f64> {
    let (c, s) = (x.map(|t| (PI * t).cos()), x.map(|t| (PI * t).sin()));
    -PI * Vector3::new(s[0] * c[1] * c[2], c[0] * s[1] * c[2], c[0] * c[1] * s[2])
}

fn rhs(x: &[f64; 3]) -> f64 {
    (3.0 * PI * PI + 1.0) * exact(x)
}

/// The result on one mesh.
struct Level {
    h: f64,
    n_elem: usize,
    n_dof: usize,
    cg_iters: usize,
    err_l2: f64,
    err_h1: f64,
}

fn main() {
    println!("TetriX – convergence of P2⁺ on −Δu + u = f, unit cube, Neumann");
    let mut mesh = UnstructuredMesh::new_from_mesh(concat!(env!("CARGO_MANIFEST_DIR"), "/../ondomathx-mesh/assets/Cube.mesh"))
        .expect("Failed to load Cube.mesh");

    let mut results: Vec<Level> = Vec::new();
    for level in 1..=*LEVELS.end() {
        mesh = mesh.uniform_refine().0;
        if !LEVELS.contains(&level) {
            continue;
        }
        let t0 = Instant::now();
        let r = solve(&mesh, 1.0 / (1 << level) as f64);
        println!(
            "  level {level}: {:7} elements, {:8} DoFs, {:4} CG iterations, {:.2?}",
            r.n_elem,
            r.n_dof,
            r.cg_iters,
            t0.elapsed()
        );
        results.push(r);
    }

    println!();
    println!("      h       ‖u − u_h‖_L²   rate    |u − u_h|_H¹   rate");
    for (k, r) in results.iter().enumerate() {
        let rate = |e: fn(&Level) -> f64| match k {
            0 => "     ".to_string(),
            _ => {
                let p = &results[k - 1];
                format!("{:5.2}", (e(p) / e(r)).ln() / (p.h / r.h).ln())
            }
        };
        println!(
            "  {:8.5}   {:.4e}    {}   {:.4e}     {}",
            r.h,
            r.err_l2,
            rate(|r| r.err_l2),
            r.err_h1,
            rate(|r| r.err_h1)
        );
    }
}

/// Solves the problem on `mesh` and computes the errors.
fn solve(mesh: &UnstructuredMesh, h: f64) -> Level {
    let space = P2PlusSpace::new(mesh);
    let (n_dof, n_elem, n_loc) = (space.num_dofs(), space.num_elements(), space.num_loc_dofs());

    let mut mass = vec![0.0; n_dof];
    space.assemble_mass(&Constant(1.0), &mut mass, 1.0);
    let b: Vec<f64> = (0..n_dof).map(|i| mass[i] * rhs(space.dof_coordinate(i))).collect();

    // Diagonal of K + M (the element stiffness diagonals, summed).
    let mut diag = mass.clone();
    for el in 0..n_elem {
        let g = space.grad_def(el);
        let g_inv = g.try_inverse().expect("singular element Jacobian");
        let k_loc = tet_15::build_local_stiffness(&(g.determinant() * g_inv * g_inv.transpose()));
        for i in 0..n_loc {
            diag[space.loc2glob(el, i)] += k_loc[(i, i)];
        }
    }

    // (K + M) u: K in DG storage (scatter, element-local K, sum of the copies).
    let (mut u_dg, mut ku_dg) = (vec![0.0; space.num_dg_dofs()], vec![0.0; space.num_dg_dofs()]);
    let mut apply = |u: &[f64], v: &mut [f64]| {
        space.scatter_cg(u, &mut u_dg);
        space.mlt_stiffness_dg(&Constant(1.0), &u_dg, &mut ku_dg, 1.0);
        space.collect_dg(&ku_dg, v);
        for i in 0..n_dof {
            v[i] += mass[i] * u[i];
        }
    };
    let mut u = vec![0.0; n_dof];
    let cg_iters = pcg(&mut apply, &diag, &b, &mut u);

    let (err_l2, err_h1) = errors(&space, &u);
    Level { h, n_elem, n_dof, cg_iters, err_l2, err_h1 }
}

/// Conjugate gradients preconditioned by `diag`, from `u`; returns the
/// number of iterations (panics if `TOL` is not reached).
fn pcg(apply: &mut impl FnMut(&[f64], &mut [f64]), diag: &[f64], b: &[f64], u: &mut [f64]) -> usize {
    let n = b.len();
    let mut r = vec![0.0; n];
    apply(u, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }
    let mut z: Vec<f64> = r.iter().zip(diag).map(|(r, d)| r / d).collect();
    let mut p = z.clone();
    let mut ap = vec![0.0; n];
    let mut rz = lal::dot(&r, &z);
    let b_norm = lal::dot(b, b).sqrt();
    for iter in 1..=10 * n {
        apply(&p, &mut ap);
        let alpha = rz / lal::dot(&p, &ap);
        for i in 0..n {
            u[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if lal::dot(&r, &r).sqrt() <= TOL * b_norm {
            return iter;
        }
        for i in 0..n {
            z[i] = r[i] / diag[i];
        }
        let rz_new = lal::dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    panic!("CG did not converge to {TOL:e}");
}

/// ‖u − u_h‖_L² and |u − u_h|_H¹, by the 14-point rule on every element
/// (affine elements: x = x₀ + G ξ, ∇ = G⁻ᵀ ∇_ξ).
fn errors(space: &P2PlusSpace, u: &[f64]) -> (f64, f64) {
    let n_loc = space.num_loc_dofs();
    let (mut l2, mut h1) = (0.0, 0.0);
    for el in 0..space.num_elements() {
        let g = space.grad_def(el);
        let g_inv_t = g.try_inverse().expect("singular element Jacobian").transpose();
        let jac = g.determinant().abs();
        let x0 = Vector3::from(*space.dof_coordinate(space.loc2glob(el, 0)));
        let u_loc: Vec<f64> = (0..n_loc).map(|i| u[space.loc2glob(el, i)]).collect();
        for (q, xi) in tet_14::POINTS.iter().enumerate() {
            let x: [f64; 3] = (x0 + g * Vector3::from(*xi)).into();
            let mut uh = 0.0;
            let mut grad_ref = Vector3::zeros();
            for (i, ui) in u_loc.iter().enumerate() {
                uh += ui * tet_15::shape_function(i, xi);
                for d in 0..3 {
                    grad_ref[d] += ui * tet_15::d_shape_function(i, d, xi);
                }
            }
            let w = jac * tet_14::WEIGHTS[q];
            l2 += w * (uh - exact(&x)).powi(2);
            h1 += w * (g_inv_t * grad_ref - exact_gradient(&x)).norm_squared();
        }
    }
    (l2.sqrt(), h1.sqrt())
}
