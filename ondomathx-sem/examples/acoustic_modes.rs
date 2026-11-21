//! Acoustic standing waves, `(1/κ) p̈ − div((1/ρ) ∇p) = 0`, free (Neumann)
//! boundary, leapfrog in time.
//!
//! The Neumann eigenmodes of a box are products of cosines,
//! `p = Π cos(k_d π x_d / L_d) · cos(ω t)` with `ω = c π |k/L|`, `c² = κ/ρ`.
//! The meshes are straight, bump-deformed Q1, or curved Q2; the bump vanishes
//! on the boundary, so the domain (and the exact solution) stays the box.
//!
//! ```text
//! cargo run --release --example acoustic_modes            # convergence studies
//! cargo run --release --example acoustic_modes -- bench   # timing, no output
//! cargo run --release --example acoustic_modes -- bench --vtk out --every 100
//! ```
//! - Space study: 1.3 periods at the CFL step, error against the
//!   dispersion-corrected reference (the leapfrog error on an eigenmode is a
//!   pure frequency shift), so the rate P+1 of the space error shows.
//! - Time study: P = 6 on a fine curved mesh, error against `cos(ω t)`, step
//!   halved: rate 2.

mod common;

use std::f64::consts::PI;

use common::*;
use lobattox::fields::{Constant, Inverse};
use lobattox::kernels::AcousticStiffness;
use lobattox::{Iso2, Iso3, Space};

/// Final time, in periods. Not an integer: at whole periods `cos(ωt)` is at an
/// extremum, the phase error enters squared and the time rate would read 4.
const T_PERIODS: f64 = 1.3;

const RHO: f64 = 2.0;
const KAPPA: f64 = 3.0;

fn c() -> f64 {
    (KAPPA / RHO).sqrt()
}

/// `cos(πx) cos(πy)` on `[0, 2] × [0, 1]` (k = (2, 1)).
fn mode_2d() -> Mode<2, 1> {
    Mode {
        shape: |x| [(PI * x[0]).cos() * (PI * x[1]).cos()],
        omega: c() * PI * 2f64.sqrt(),
    }
}

/// `cos(πx) cos(πy/2) cos(πz/1.5)` on `[0, 1] × [0, 2] × [0, 1.5]`.
fn mode_3d() -> Mode<3, 1> {
    Mode {
        shape: |x| [(PI * x[0]).cos() * (PI * x[1] / 2.0).cos() * (PI * x[2] / 1.5).cos()],
        omega: c() * PI * (1.0 + 0.25 + 1.0 / 2.25f64).sqrt(),
    }
}

const L2: [f64; 2] = [2.0, 1.0];
const L3: [f64; 3] = [1.0, 2.0, 1.5];

fn kernel() -> AcousticStiffness<Constant> {
    AcousticStiffness { rho: Constant(RHO) }
}

fn mass() -> Inverse<Constant> {
    Inverse(Constant(KAPPA))
}

fn run<const D: usize, S: ErrorQuadrature<D>>(
    space: &Space<D, S>,
    mode: &Mode<D, 1>,
    steps: Option<usize>,
    reference: Reference,
) -> Report {
    let t_end = T_PERIODS * 2.0 * PI / mode.omega;
    let steps = steps.unwrap_or_else(|| cfl_steps::<D, S, 1, _, _>(space, &kernel(), &mass(), t_end));
    leapfrog(
        space,
        &kernel(),
        &mass(),
        mode,
        &RunOptions {
            t_end,
            steps,
            reference,
            energy: true,
            output: None,
        },
    )
}

fn space_study_2d<const NP: usize>(geo: Geometry, ns: &[usize]) {
    println!("2D, {}, P = {}", geo.name(), NP - 1);
    let rows: Vec<_> = ns
        .iter()
        .map(|&n| {
            let space = continuous_space::<2, Iso2<NP>>(&rectangle(L2, [2 * n, n], geo));
            let r = run(&space, &mode_2d(), None, Reference::DispersionCorrected);
            (format!("h = 1/{n}"), 1.0 / n as f64, r.rel_error, r.energy_drift)
        })
        .collect();
    print_rates("mesh", &rows);
}

fn space_study_3d<const NP: usize>(geo: Geometry, ns: &[usize]) {
    println!("3D, {}, P = {}", geo.name(), NP - 1);
    let rows: Vec<_> = ns
        .iter()
        .map(|&n| {
            let space = continuous_space::<3, Iso3<NP>>(&cuboid(L3, [n, 2 * n, 2 * n], geo));
            let r = run(&space, &mode_3d(), None, Reference::DispersionCorrected);
            (format!("h = 1/{n}"), 1.0 / n as f64, r.rel_error, r.energy_drift)
        })
        .collect();
    print_rates("mesh", &rows);
}

fn time_study() {
    println!("2D, curved Q2, P = 6, h = 1/8: time convergence against cos(ωt)");
    let space = continuous_space::<2, Iso2<7>>(&rectangle(L2, [16, 8], Geometry::CurvedQ2));
    let mode = mode_2d();
    let t_end = T_PERIODS * 2.0 * PI / mode.omega;
    let n0 = cfl_steps::<2, Iso2<7>, 1, _, _>(&space, &kernel(), &mass(), t_end);
    let rows: Vec<_> = [1, 2, 4, 8]
        .iter()
        .map(|&f| {
            let r = run(&space, &mode, Some(n0 * f), Reference::Exact);
            (format!("dt = {:.2e}", r.dt), r.dt, r.rel_error, r.energy_drift)
        })
        .collect();
    print_rates("step", &rows);
}

fn main() {
    let args = args();
    match args.mode.as_str() {
        "bench" => {
            let space = continuous_space::<2, Iso2<5>>(&rectangle(L2, [256, 128], Geometry::CurvedQ2));
            bench("acoustic_2d", &space, &kernel(), &mass(), &mode_2d(), &args);
            let space = continuous_space::<3, Iso3<5>>(&cuboid(L3, [16, 32, 24], Geometry::CurvedQ2));
            bench("acoustic_3d", &space, &kernel(), &mass(), &mode_3d(), &args);
        }
        _ => {
            for geo in Geometry::ALL {
                space_study_2d::<3>(geo, &[2, 4, 8, 16]);
                space_study_2d::<5>(geo, &[1, 2, 4, 8]);
            }
            space_study_3d::<4>(Geometry::Straight, &[1, 2, 4]);
            space_study_3d::<4>(Geometry::CurvedQ2, &[1, 2, 4]);
            time_study();
        }
    }
}
