//! Shared by the examples: meshes, the leapfrog loop, and the harness for the
//! convergence studies and the benchmark.
//!
//! The library provides the operators ([`Space::apply`] / `mlt_add`,
//! `c_inverse_mass`, `c_max_eigenvalue`, …); the time loop lives here.
#![allow(dead_code)]

use std::path::PathBuf;
use std::time::Instant;

use anamesh::diffeomorphism::Diffeomorphism;
use anamesh::mesh::cube::Cube;
use anamesh::mesh::square::Square;
use anamesh::mesh::{Mesh, UnstructuredMesh};
use lobattox::fields::Field;
use lobattox::{
    dot, stable_dt, GeoStorage, Gl, Gll2, Gll3, Numbering, PointKernel, Quad2, Quad3, Shape, Space, Sweep, VtkField,
    VtkWriter,
};
use rayon::prelude::*;

// ── Meshes ─────────────────────────────────────────────────────────────

/// Interior bump `x + ε·L·s(x)·(…)`, `s = Π sin(π x_d / L_d)`: zero on the
/// boundary, so the domain stays the box and the eigenmodes stay exact, while
/// the elements become curved (Q2) or non-affine (Q1).
/// `params = [ε, L_0, …, L_{D−1}]`.
fn bump(p: &[f64], x: &[f64], f: &mut [f64]) {
    let d = p.len() - 1;
    f[..x.len()].copy_from_slice(x);
    let s: f64 = (0..d)
        .map(|a| (std::f64::consts::PI * x[a] / p[1 + a]).sin())
        .product();
    for a in 0..d {
        // Different shapes per direction, so the map is not a scaling.
        let twist = (std::f64::consts::PI * x[(a + 1) % d] / p[1 + (a + 1) % d]).cos();
        f[a] = x[a] + p[0] * p[1 + a] * s * (0.5 + 0.5 * twist);
    }
}

/// Only `diffeo` is used by `apply_deformation`; the gradient is not needed
/// for meshes whose deformation is applied.
fn no_grad(_p: &[f64], _x: &[f64], g: &mut [f64]) {
    g.fill(f64::NAN);
}

fn bump_diffeo(dim: usize, eps: f64, lengths: &[f64]) -> Diffeomorphism {
    let mut params = vec![eps];
    params.extend_from_slice(lengths);
    Diffeomorphism {
        inner_dim: dim,
        outer_dim: dim,
        params,
        diffeo: bump,
        grad_diffeo: no_grad,
        inner_deformation: None,
    }
}

/// How the elements of a box mesh are shaped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geometry {
    /// Straight Q1 elements (all affine).
    Straight,
    /// Bump-deformed Q1 elements (bilinear/trilinear, not affine).
    DeformedQ1,
    /// Bump-deformed Q2 elements (curved).
    CurvedQ2,
}

impl Geometry {
    pub const ALL: [Geometry; 3] = [Geometry::Straight, Geometry::DeformedQ1, Geometry::CurvedQ2];

    pub fn name(self) -> &'static str {
        match self {
            Geometry::Straight => "straight Q1",
            Geometry::DeformedQ1 => "deformed Q1",
            Geometry::CurvedQ2 => "curved Q2",
        }
    }

    fn parts(self, dim: usize, lengths: &[f64]) -> (Option<Diffeomorphism>, bool) {
        match self {
            Geometry::Straight => (None, false),
            Geometry::DeformedQ1 => (Some(bump_diffeo(dim, 0.04, lengths)), false),
            Geometry::CurvedQ2 => (Some(bump_diffeo(dim, 0.04, lengths)), true),
        }
    }
}

/// `[0, lx] × [0, ly]`, `nx × ny` elements.
pub fn rectangle(l: [f64; 2], n: [usize; 2], geo: Geometry) -> UnstructuredMesh {
    let (def, q2) = geo.parts(2, &l);
    let mut mesh = Square::new([0.0; 2], l[0], l[1], n[0], n[1], def, Some(q2)).get_unstructured_mesh();
    mesh.apply_deformation();
    mesh
}

/// `[0, lx] × [0, ly] × [0, lz]`, `nx × ny × nz` elements.
pub fn cuboid(l: [f64; 3], n: [usize; 3], geo: Geometry) -> UnstructuredMesh {
    let (def, q2) = geo.parts(3, &l);
    let mut mesh = Cube::new([0.0; 3], l[0], l[1], l[2], n[0], n[1], n[2], def, Some(q2))
        .get_unstructured_mesh();
    mesh.apply_deformation();
    mesh
}

// ── Spaces and errors ──────────────────────────────────────────────────

/// The over-integration rule of the final L² error: 10 (2D) or 8 (3D)
/// Gauss–Legendre points per direction, exact for `u_h²` up to degree 19 (2D,
/// P ≤ 9) or 15 (3D, P ≤ 7), and accurate for the smooth reference.
pub trait ErrorQuadrature<const D: usize>: Shape<D> + Sized {
    type Over: Sweep<D, Self>;
}
impl<const N0: usize, const N1: usize> ErrorQuadrature<2> for Gll2<N0, N1> {
    type Over = Quad2<Gl<10>, Gl<10>>;
}
impl<const N0: usize, const N1: usize, const N2: usize> ErrorQuadrature<3> for Gll3<N0, N1, N2> {
    type Over = Quad3<Gl<8>, Gl<8>, Gl<8>>;
}

/// Continuous space (stored node geometry) with the error quadrature declared
/// (recomputed geometry: it is used once per run).
pub fn continuous_space<const D: usize, S: ErrorQuadrature<D>>(mesh: &UnstructuredMesh) -> Space<D, S> {
    Space::builder(mesh)
        .numbering(Numbering::Continuous)
        .quadrature::<S::Over>(GeoStorage::Recompute)
        .build()
}

/// Relative L² error `‖u_h − a·shape‖ / ‖a·shape‖`, integrated with the error
/// quadrature (true L² norm, not the GLL-node norm, which superconverges).
pub fn rel_l2_error<const D: usize, S: ErrorQuadrature<D>, const C: usize>(
    space: &Space<D, S>,
    u: &[f64],
    shape: fn([f64; D]) -> [f64; C],
    a: f64,
) -> f64 {
    let nq = <S::Over as lobattox::Quadrature<D>>::NQ;
    let uq = space.quadrature_values::<S::Over, C>(u);
    let xq = space.quadrature_xyz::<S::Over>();
    let wq = space.quadrature_weights::<S::Over>();
    let (num, den) = (0..space.n_elem())
        .into_par_iter()
        .map(|e| {
            let (mut num, mut den) = (0.0, 0.0);
            for q in 0..nq {
                let (x, w) = (xq[e * nq + q], wq[e * nq + q]);
                let ex = shape(x);
                for c in 0..C {
                    let d = uq[(e * C + c) * nq + q] - a * ex[c];
                    num += w * d * d;
                    den += w * a * a * ex[c] * ex[c];
                }
            }
            (num, den)
        })
        .reduce(|| (0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1));
    (num / den).sqrt()
}

// ── Leapfrog ───────────────────────────────────────────────────────────

/// A standing wave `u(x, t) = shape(x)·cos(ω t)` of `M ü + K u = 0`.
pub struct Mode<const D: usize, const C: usize> {
    pub shape: fn([f64; D]) -> [f64; C],
    pub omega: f64,
}

/// Which reference the final error is measured against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reference {
    /// `cos(ω t)`: space and time errors.
    Exact,
    /// `cos(ω̃ t)` with the leapfrog frequency `sin(ω̃ dt/2) = ω dt/2`. For an
    /// eigenmode the scheme (with the Taylor start below) is exact in time up
    /// to this frequency shift, so only the space error remains.
    DispersionCorrected,
}

/// VTK snapshots every `every` steps (and at steps 0 and `steps`).
pub struct Output {
    pub dir: PathBuf,
    pub prefix: String,
    pub every: usize,
}

pub struct RunOptions {
    pub t_end: f64,
    /// Number of steps; `dt = t_end / steps`.
    pub steps: usize,
    pub reference: Reference,
    /// Track the staggered energy (two extra passes per step).
    pub energy: bool,
    pub output: Option<Output>,
}

#[derive(Debug, Default)]
pub struct Report {
    pub dt: f64,
    /// `‖u − u_ref‖ / ‖u_ref‖` in L² at `t_end` (over-integrated, [`rel_l2_error`]).
    pub rel_error: f64,
    /// `max |Eⁿ⁺½ − E½| / E½` of the staggered energy.
    pub energy_drift: f64,
    /// Time stepping only (operator, update, energy).
    pub loop_seconds: f64,
    /// VTK writing (geometry once + snapshots).
    pub output_seconds: f64,
    pub snapshots: usize,
}

/// Leapfrog for `M ü + K u = 0`, `u(0) = shape`, `u̇(0) = 0`:
/// ```text
/// u¹   = u⁰ − dt²/2 · M⁻¹K u⁰                   (Taylor start)
/// uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² · M⁻¹K uⁿ
/// Eⁿ⁺½ = ½‖(uⁿ⁺¹ − uⁿ)/dt‖²_M + ½⟨K uⁿ⁺¹, uⁿ⟩   (conserved below CFL)
/// ```
pub fn leapfrog<const D: usize, S: ErrorQuadrature<D>, const C: usize, K, F>(
    space: &Space<D, S>,
    kernel: &K,
    mass: &F,
    mode: &Mode<D, C>,
    opts: &RunOptions,
) -> Report
where
    K: PointKernel<D, C>,
    F: Field<D>,
{
    let nloc = S::NLOC;
    let dt = opts.t_end / opts.steps as f64;
    let dt2 = dt * dt;
    let m = space.local_mass(mass);
    let inv_m = space.c_inverse_mass(mass);
    let inv_d = inv_m.diagonal().expect("GLL lumped mass is diagonal");
    let mut report = Report {
        dt,
        ..Default::default()
    };

    let writer = opts.output.as_ref().map(|o| {
        std::fs::create_dir_all(&o.dir).expect("output directory");
        let t = Instant::now();
        let w = VtkWriter::new(space);
        report.output_seconds += t.elapsed().as_secs_f64();
        w
    });
    let snapshot = |report: &mut Report, step: usize, u: &[f64]| {
        let (Some(o), Some(w)) = (&opts.output, &writer) else {
            return;
        };
        if !step.is_multiple_of(o.every) && step != opts.steps {
            return;
        }
        let t = Instant::now();
        let path = o.dir.join(format!("{}_{step:06}.vtk", o.prefix));
        w.write(
            path,
            &[VtkField {
                name: "u",
                components: C,
                data: u,
            }],
        )
        .expect("write VTK");
        report.output_seconds += t.elapsed().as_secs_f64();
        report.snapshots += 1;
    };

    let start = Instant::now();
    // `prev` = uⁿ⁻¹, `cur` = uⁿ, `y` = K_loc uⁿ then M⁻¹K uⁿ.
    let mut prev = space.sample::<C>(mode.shape);
    let mut y = vec![0.0; prev.len()];
    space.c_apply::<C, K>(kernel, &prev, &mut y);
    inv_m.apply::<C>(&mut y);
    let mut cur: Vec<f64> = prev.par_iter().zip(&y).map(|(u, a)| u - 0.5 * dt2 * a).collect();
    report.loop_seconds += start.elapsed().as_secs_f64();
    snapshot(&mut report, 0, &prev);

    let mut diff = if opts.energy { vec![0.0; cur.len()] } else { vec![] };
    let mut energy_0 = None;
    let block = C * nloc;
    for step in 1..opts.steps {
        let t = Instant::now();
        space.mlt_add::<C, K>(kernel, 1.0, &cur, 0.0, &mut y);
        if opts.energy {
            // E^{step−½}: uses uⁿ = cur, uⁿ⁻¹ = prev and K_loc uⁿ before assembly.
            diff.par_iter_mut()
                .zip(cur.par_iter().zip(&prev))
                .for_each(|(d, (a, b))| *d = (a - b) / dt);
            let e = 0.5 * space.mass_inner::<C>(&m, &diff, &diff) + 0.5 * dot(&y, &prev);
            match energy_0 {
                None => energy_0 = Some(e),
                Some(e0) => {
                    report.energy_drift = report.energy_drift.max(((e - e0) / e0).abs());
                }
            }
        }
        space.c_sum_shared::<C>(&mut y);
        // uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹ (K uⁿ), written into `prev` (M⁻¹ fused).
        prev.par_chunks_mut(block)
            .zip(cur.par_chunks(block))
            .zip(y.par_chunks(block))
            .zip(inv_d.par_chunks(nloc))
            .for_each(|(((p, c), a), im)| {
                for k in 0..block {
                    p[k] = 2.0 * c[k] - p[k] - dt2 * im[k % nloc] * a[k];
                }
            });
        std::mem::swap(&mut prev, &mut cur);
        report.loop_seconds += t.elapsed().as_secs_f64();
        snapshot(&mut report, step, &prev);
    }
    snapshot(&mut report, opts.steps, &cur);

    // Error at t_end.
    let omega = match opts.reference {
        Reference::Exact => mode.omega,
        Reference::DispersionCorrected => 2.0 / dt * (mode.omega * dt / 2.0).asin(),
    };
    report.rel_error = rel_l2_error(space, &cur, mode.shape, (omega * opts.t_end).cos());
    report
}

/// Power iteration for the CFL step: relative change of the Rayleigh quotient
/// below `EIG_RTOL` (the C++ `PowerIteration` uses 1e-6), at most
/// `EIG_MAX_ITER` iterations.
pub const EIG_RTOL: f64 = 1e-6;
pub const EIG_MAX_ITER: usize = 5000;

/// Smallest step count with `t_end / steps ≤ stable_dt(λ_max(M⁻¹K), 0.95)`.
pub fn cfl_steps<const D: usize, S: Shape<D>, const C: usize, K, F>(
    space: &Space<D, S>,
    kernel: &K,
    mass: &F,
    t_end: f64,
) -> usize
where
    K: PointKernel<D, C>,
    F: Field<D>,
{
    let est = space.c_max_eigenvalue::<C, K, F>(kernel, mass, EIG_MAX_ITER, EIG_RTOL);
    (t_end / stable_dt(est.lambda, 0.95)).ceil() as usize
}

// ── Tables and arguments ───────────────────────────────────────────────

/// Prints `label  error  rate` rows, the rate against the previous row being
/// `log(e_{k−1}/e_k) / log(s_{k−1}/s_k)` for step sizes `s`.
pub fn print_rates(header: &str, rows: &[(String, f64, f64, f64)]) {
    println!("  {header:<14} {:>12} {:>7} {:>12}", "rel. L² err", "rate", "energy drift");
    for (k, (label, s, e, drift)) in rows.iter().enumerate() {
        let rate = if k == 0 {
            "".to_string()
        } else {
            let (s0, e0) = (rows[k - 1].1, rows[k - 1].2);
            format!("{:.2}", (e0 / e).ln() / (s0 / s).ln())
        };
        println!("  {label:<14} {e:>12.3e} {rate:>7} {drift:>12.1e}");
    }
}

/// Command line: `[study|bench] [--vtk DIR] [--every N] [--steps N]`.
pub struct Args {
    pub mode: String,
    pub vtk: Option<PathBuf>,
    pub every: usize,
    pub steps: usize,
}

pub fn args() -> Args {
    let mut a = Args {
        mode: "study".into(),
        vtk: None,
        every: 50,
        steps: 500,
    };
    let mut it = std::env::args().skip(1);
    while let Some(s) = it.next() {
        match s.as_str() {
            "--vtk" => a.vtk = Some(it.next().expect("--vtk DIR").into()),
            "--every" => a.every = it.next().expect("--every N").parse().expect("integer"),
            "--steps" => a.steps = it.next().expect("--steps N").parse().expect("integer"),
            "study" | "bench" => a.mode = s,
            _ => panic!("usage: [study|bench] [--vtk DIR] [--every N] [--steps N]"),
        }
    }
    a
}

/// Runs a benchmark case without and, if `--vtk` is given, with output.
pub fn bench<const D: usize, S: ErrorQuadrature<D>, const C: usize, K, F>(
    name: &str,
    space: &Space<D, S>,
    kernel: &K,
    mass: &F,
    mode: &Mode<D, C>,
    args: &Args,
) where
    K: PointKernel<D, C>,
    F: Field<D>,
{
    let n_dof = space.n_dup() * C;
    let t = Instant::now();
    let est = space.c_max_eigenvalue::<C, K, F>(kernel, mass, EIG_MAX_ITER, EIG_RTOL);
    let t_eig = t.elapsed().as_secs_f64();
    let dt = stable_dt(est.lambda, 0.95);
    let steps = args.steps;
    println!(
        "{name}: {} elements, {n_dof} duplicated dofs, P = {:?}, dt = {dt:.3e} ({} power iterations, {t_eig:.2} s)",
        space.n_elem(),
        S::N[..D].iter().map(|n| n - 1).collect::<Vec<_>>(),
        est.iterations
    );
    let mut runs = vec![("no output", None)];
    if let Some(dir) = &args.vtk {
        runs.push((
            "VTK output",
            Some(Output {
                dir: dir.join(name),
                prefix: name.into(),
                every: args.every,
            }),
        ));
    }
    for (label, output) in runs {
        let r = leapfrog(
            space,
            kernel,
            mass,
            mode,
            &RunOptions {
                t_end: dt * steps as f64,
                steps,
                reference: Reference::Exact,
                energy: false,
                output,
            },
        );
        let total = r.loop_seconds + r.output_seconds;
        print!(
            "  {label:<10}: {steps} steps, loop {:.3} s ({:.2} ns/dof/step)",
            r.loop_seconds,
            r.loop_seconds * 1e9 / (n_dof as f64 * steps as f64)
        );
        if r.snapshots > 0 {
            print!(
                ", output {:.3} s ({} snapshots, {:.1} ms each), total {total:.3} s, output share {:.0} %",
                r.output_seconds,
                r.snapshots,
                r.output_seconds * 1e3 / r.snapshots as f64,
                100.0 * r.output_seconds / total
            );
        }
        println!();
    }
}
