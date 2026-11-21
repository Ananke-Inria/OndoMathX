//! Acoustic waves `∂ₜ²u − div(A ∇u) = f(x)·g(t)`, free (Neumann) boundary,
//! leapfrog in time: ports of the C++ examples of
//! `OndoMathX_gitlab/core/Testing/FESpace`, same parameters, for efficiency
//! comparisons and visualization (ParaView `.vtk`, Vizir `.mesh` + `.sol`).
//!
//! | case     | C++ example            | mesh                         | P | A               | excitation |
//! |----------|------------------------|------------------------------|---|-----------------|------------|
//! | `square` | `AcousticWave_Square`  | `[0,1]²`, 100 × 100          | 4 | `I`             | initial velocity `G(x)G(y)cos(2π·20(x+y))` |
//! | `cube`   | `AcousticWave_Cube`    | `[0,1]³`, 3 × 3 × 3          | 4 | `diag(4,2,0.5)` | `CInfty0(·, ½, 0.05, 1)·Ricker(t, 0.2, 2000)` |
//! | `circle` | `AcousticWave_Circle`  | unit disc, exact geometry    | 6 | `diag(2,1)`     | `CInfty0(·, 0, 0.2, 4)·Ricker(t, 0.2, 500)` |
//! | `mesh`   | `AcousticWave_2D_Mesh` | `Potatoe.msh` (Anamesh asset) | 4 | `diag(2,1)`     | `CInfty0(·, (−1,1.5), 0.2, 4)·Ricker(t, 0.2, 500)` |
//!
//! ```text
//! cargo run --release --example acoustic_wave -- [square|cube|circle|mesh|all] [--out DIR] [--mesh FILE] [--t-end T]
//! ```
//! Without `--out` nothing is written (pure computation). With `--out DIR`,
//! as the C++ examples: `DIR/<case>.mesh` (Anamesh `write_mesh`), and every
//! `Δt_out` (0.01 for `square`, 0.1 otherwise) `DIR/<case>.<k>.vtk` and
//! `DIR/<case>.<k:05>.sol`, written by a background thread (as the C++
//! `ParallelWriteVTK` / `ParallelWriteSolutionVIZIR`).
//!
//! Differences with the C++: the disc is Anamesh's `Circle` with n = 54
//! (1620 elements, exact geometry) instead of the C++ `Circle(6, 6)` (1728
//! elements); compare per DoF and per step.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anamesh::mesh::circle::Circle;
use anamesh::mesh::cube::Cube;
use anamesh::mesh::square::Square;
use anamesh::mesh::{Mesh, UnstructuredMesh};
use lobattox::fields::Constant;
use lobattox::kernels::AnisotropicAcoustic;
use lobattox::{stable_dt, Iso2, Iso3, Numbering, Shape, Space, VizirWriter, VtkField, VtkWriter};
use rayon::prelude::*;

/// C++ `Ricker(t, t0, s0)`.
fn ricker(t: f64, t0: f64, s0: f64) -> f64 {
    let t = t - t0;
    (4.0 * s0 * s0 * t * t - 2.0 * s0) * (-t * t * s0).exp()
}

/// C++ `Gaussian(t, t0, s0)`.
fn gaussian(t: f64, t0: f64, s0: f64) -> f64 {
    let t = t - t0;
    (-t * t * s0).exp()
}

/// C++ `CInfty0(x, c, r0, s0)`: `exp(s0 r0² / (r² − r0²) + s0)` for `r < r0`
/// (1 at the centre), 0 outside.
fn c_infty0(x: &[f64], c: &[f64], r0: f64, s0: f64) -> f64 {
    let r2: f64 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
    let r02 = r0 * r0;
    if r2 >= r02 {
        0.0
    } else {
        (s0 * r02 / (r2 - r02) + s0).exp()
    }
}

/// How the wave is excited.
enum Excitation<const D: usize> {
    /// `u(0) = 0`, `u'(0) = v0(x)`.
    InitialVelocity(fn([f64; D]) -> f64),
    /// `f(x)·g(t)`: `CInfty0(x, center, r0, s0)·Ricker(t, t0, sigma)`.
    Source {
        center: [f64; D],
        r0: f64,
        s0: f64,
        t0: f64,
        sigma: f64,
    },
}

struct Case<'a, const D: usize, M: Mesh> {
    name: &'a str,
    mesh: &'a M,
    a: [[f64; D]; D],
    t_end: f64,
    output_dt: f64,
    excitation: Excitation<D>,
}

fn run<const D: usize, S: Shape<D>, M: Mesh + Sync>(case: Case<'_, D, M>, out: Option<&Path>) {
    println!("== {} ({} threads)", case.name, rayon::current_num_threads());
    let t = Instant::now();
    let space = Space::<D, S>::new(case.mesh, Numbering::Continuous);
    let t_setup = t.elapsed().as_secs_f64();
    let kernel = AnisotropicAcoustic { a: case.a };
    let rho = Constant(1.0);
    let im = space.c_inverse_mass(&rho);
    let inv_d = im.diagonal().expect("lumped mass").to_vec();

    // CFL step (C++: PowerIteration, tolerance 1e-6), dt = 0.95 · 2/√λ.
    let t = Instant::now();
    let est = space.c_max_eigenvalue::<1, _, _>(&kernel, &rho, 5000, 1e-6);
    let t_eig = t.elapsed().as_secs_f64();
    let dt = stable_dt(est.lambda, 0.95);
    let mut steps = 0;
    while (steps as f64) * dt < case.t_end {
        steps += 1;
    }
    let every = ((case.output_dt / dt).floor() as usize).max(1);
    println!(
        "  {} elements, P = {}, {} DoFs ({} duplicated), dt = {dt:.4e} ({} power iterations, {t_eig:.2} s), {steps} steps, setup {t_setup:.2} s",
        space.n_elem(),
        S::N[0] - 1,
        space.c_n_global(),
        space.n_dup(),
        est.iterations
    );

    // Excitation: source s = M⁻¹ Σ_shared F (the nodal f), or the first iterate.
    let n = space.n_dup();
    let mut prev = vec![0.0; n];
    let mut cur = vec![0.0; n];
    let mut y = vec![0.0; n];
    let (source, ricker_params) = match &case.excitation {
        Excitation::Source {
            center,
            r0,
            s0,
            t0,
            sigma,
        } => {
            let (c, r0, s0) = (*center, *r0, *s0);
            let mut s = space.local_load::<1>(|x| [c_infty0(&x, &c, r0, s0)]);
            space.c_sum_shared::<1>(&mut s);
            im.apply::<1>(&mut s);
            (s, Some((*t0, *sigma)))
        }
        Excitation::InitialVelocity(v0) => {
            // u¹ = dt·v0 − dt³/6·M⁻¹K v0 (C++), u⁰ = 0.
            let v = space.sample::<1>(|x| [v0(x)]);
            space.c_apply::<1, _>(&kernel, &v, &mut y);
            im.apply::<1>(&mut y);
            for i in 0..n {
                cur[i] = dt * v[i] - dt * dt * dt / 6.0 * y[i];
            }
            (vec![0.0; n], None)
        }
    };

    // Output (as the C++: mesh once, VTK + Vizir every `every` steps).
    let writers = out.map(|dir| {
        std::fs::create_dir_all(dir).expect("output directory");
        let t = Instant::now();
        case.mesh
            .write_mesh(dir.join(format!("{}.mesh", case.name)).to_str().unwrap())
            .expect("write .mesh");
        let w = (VtkWriter::new(&space), VizirWriter::new(&space, case.mesh));
        println!("  output to {} (writers {:.2} s)", dir.display(), t.elapsed().as_secs_f64());
        w
    });

    let dt2 = dt * dt;
    let mut n_out = 0;
    let (mut t_loop, mut t_wait) = (0.0, 0.0);
    std::thread::scope(|scope| {
        let mut pending: Option<std::thread::ScopedJoinHandle<'_, ()>> = None;
        for step in 0..steps {
            let t = Instant::now();
            // uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹K uⁿ + dt² g(tⁿ) s, written into `prev`.
            space.c_apply::<1, _>(&kernel, &cur, &mut y);
            let g = ricker_params.map_or(0.0, |(t0, sigma)| ricker(step as f64 * dt, t0, sigma));
            prev.par_iter_mut()
                .zip(cur.par_iter())
                .zip(y.par_iter().zip(inv_d.par_iter()))
                .zip(source.par_iter())
                .for_each(|(((p, c), (a, im)), s)| *p = 2.0 * c - *p - dt2 * im * a + dt2 * g * s);
            std::mem::swap(&mut prev, &mut cur);
            t_loop += t.elapsed().as_secs_f64();

            if let (Some(dir), Some((vtk, vizir))) = (out, &writers) {
                let snapshot = if ricker_params.is_some() { step % every == 0 && step > 0 } else { step % every == 0 };
                if snapshot {
                    let t = Instant::now();
                    if let Some(h) = pending.take() {
                        h.join().unwrap();
                    }
                    let data = cur.clone();
                    t_wait += t.elapsed().as_secs_f64();
                    let (vtk_path, sol_path) = (
                        dir.join(format!("{}.{n_out}.vtk", case.name)),
                        dir.join(format!("{}.{n_out:05}.sol", case.name)),
                    );
                    pending = Some(scope.spawn(move || {
                        vtk.write(&vtk_path, &[VtkField { name: "u", components: 1, data: &data }]).unwrap();
                        vizir.write(&sol_path, &data, 1).unwrap();
                    }));
                    n_out += 1;
                }
            }
        }
        let t = Instant::now();
        if let Some(h) = pending.take() {
            h.join().unwrap();
        }
        t_wait += t.elapsed().as_secs_f64();
    });
    let max_u = cur.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    println!(
        "  loop {t_loop:.3} s: {:.3} ms/step, {:.2} ns/DoF/step (global DoFs); {n_out} snapshots, waiting for output {t_wait:.3} s; max |u(T)| = {max_u:.4e}",
        1e3 * t_loop / steps as f64,
        1e9 * t_loop / (steps as f64 * space.c_n_global() as f64)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let out = flag("--out").map(PathBuf::from);
    let mesh_file = flag("--mesh").unwrap_or_else(|| "../Anamesh/assets/Potatoe.msh".into());
    let t_end_override = flag("--t-end").map(|t| t.parse::<f64>().expect("--t-end"));
    let which = args.first().filter(|a| !a.starts_with("--")).cloned().unwrap_or_else(|| "all".into());
    let t_end = |t: f64| t_end_override.unwrap_or(t);
    let out = out.as_deref();

    if which == "square" || which == "all" {
        let mesh = Square::new([0.0, 0.0], 1.0, 1.0, 100, 100, None, None);
        run::<2, Iso2<5>, _>(
            Case {
                name: "acoustic_wave_square",
                mesh: &mesh,
                a: [[1.0, 0.0], [0.0, 1.0]],
                t_end: t_end(1.0),
                output_dt: 0.01,
                excitation: Excitation::InitialVelocity(|x| {
                    gaussian(x[0], 0.9, 500.0)
                        * gaussian(x[1], 0.5, 500.0)
                        * (2.0 * std::f64::consts::PI * 20.0 * (x[0] + x[1])).cos()
                }),
            },
            out,
        );
    }
    if which == "cube" || which == "all" {
        let mesh = Cube::new([0.0; 3], 1.0, 1.0, 1.0, 3, 3, 3, None, None);
        run::<3, Iso3<5>, _>(
            Case {
                name: "acoustic_wave_cube",
                mesh: &mesh,
                a: [[4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]],
                t_end: t_end(1.0),
                output_dt: 0.1,
                excitation: Excitation::Source {
                    center: [0.5; 3],
                    r0: 0.05,
                    s0: 1.0,
                    t0: 0.2,
                    sigma: 2000.0,
                },
            },
            out,
        );
    }
    if which == "circle" || which == "all" {
        let mesh = Circle::new([0.0, 0.0], 1.0, 54, None, Some(true));
        run::<2, Iso2<7>, _>(
            Case {
                name: "acoustic_wave_circle",
                mesh: &mesh,
                a: [[2.0, 0.0], [0.0, 1.0]],
                t_end: t_end(2.0),
                output_dt: 0.1,
                excitation: Excitation::Source {
                    center: [0.0, 0.0],
                    r0: 0.2,
                    s0: 4.0,
                    t0: 0.2,
                    sigma: 500.0,
                },
            },
            out,
        );
    }
    if which == "mesh" || which == "all" {
        let mesh = UnstructuredMesh::new_from_msh(&mesh_file).unwrap_or_else(|e| panic!("{mesh_file}: {e}"));
        run::<2, Iso2<5>, _>(
            Case {
                name: "acoustic_wave_2d_mesh",
                mesh: &mesh,
                a: [[2.0, 0.0], [0.0, 1.0]],
                t_end: t_end(10.0),
                output_dt: 0.1,
                excitation: Excitation::Source {
                    center: [-1.0, 1.5],
                    r0: 0.2,
                    s0: 4.0,
                    t0: 0.2,
                    sigma: 500.0,
                },
            },
            out,
        );
    }
}
