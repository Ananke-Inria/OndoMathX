//! Elastic standing waves, `ρ ü − div σ(u) = 0`, `σ = λ tr(ε) I + 2μ ε`,
//! traction-free boundary, leapfrog in time.
//!
//! Traction-free eigenmodes of a box are not separable in general, but the
//! divergence-free shear modes with equal wavenumbers in two directions are:
//! ```text
//! u = (cos(ax) sin(ay), −sin(ax) cos(ay)) · cos(ω t),   ω = a √(2μ/ρ)
//! ```
//! (`div u = 0`, so `λ` drops out of the stress, and `σ·n = 0` on every face
//! of a box whose sides are multiples of π/a). In 3D the example superposes
//! the `(x, y)` and `(y, z)` modes, both with `a = π`.
//!
//! ```text
//! cargo run --release --example elastic_modes            # convergence studies
//! cargo run --release --example elastic_modes -- bench   # timing, no output
//! cargo run --release --example elastic_modes -- bench --vtk out --every 100
//! ```
//! Same studies as `acoustic_modes`: space convergence against the
//! dispersion-corrected reference, time convergence against `cos(ω t)`.

mod common;

use std::f64::consts::PI;

use common::*;
use lobattox::fields::Constant;
use lobattox::kernels::IsotropicElastic;
use lobattox::{Iso2, Iso3, Space};

/// Final time, in periods (not an integer, see `acoustic_modes`).
const T_PERIODS: f64 = 1.3;

const RHO: f64 = 2.0;
const LAMBDA: f64 = 3.0;
const MU: f64 = 1.0;

fn omega() -> f64 {
    PI * (2.0 * MU / RHO).sqrt()
}

/// Shear mode `a = π` on `[0, 2] × [0, 1]`.
fn mode_2d() -> Mode<2, 2> {
    Mode {
        shape: |x| {
            let (sx, cx, sy, cy) = ((PI * x[0]).sin(), (PI * x[0]).cos(), (PI * x[1]).sin(), (PI * x[1]).cos());
            [cx * sy, -sx * cy]
        },
        omega: omega(),
    }
}

/// `(x, y)` plus `(y, z)` shear modes, `a = π`, on `[0, 1] × [0, 2] × [0, 1]`.
fn mode_3d() -> Mode<3, 3> {
    Mode {
        shape: |x| {
            let (s, c) = (x.map(|v| (PI * v).sin()), x.map(|v| (PI * v).cos()));
            [c[0] * s[1], -s[0] * c[1] + c[1] * s[2], -s[1] * c[2]]
        },
        omega: omega(),
    }
}

const L2: [f64; 2] = [2.0, 1.0];
const L3: [f64; 3] = [1.0, 2.0, 1.0];

fn kernel() -> IsotropicElastic<Constant, Constant> {
    IsotropicElastic {
        lambda: Constant(LAMBDA),
        mu: Constant(MU),
    }
}

fn mass() -> Constant {
    Constant(RHO)
}

fn run<const D: usize, S: ErrorQuadrature<D>>(
    space: &Space<D, S>,
    mode: &Mode<D, D>,
    steps: Option<usize>,
    reference: Reference,
) -> Report
where
    IsotropicElastic<Constant, Constant>: lobattox::PointKernel<D, D>,
{
    let t_end = T_PERIODS * 2.0 * PI / mode.omega;
    let steps = steps.unwrap_or_else(|| cfl_steps::<D, S, D, _, _>(space, &kernel(), &mass(), t_end));
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
            let space = continuous_space::<3, Iso3<NP>>(&cuboid(L3, [n, 2 * n, n], geo));
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
    let n0 = cfl_steps::<2, Iso2<7>, 2, _, _>(&space, &kernel(), &mass(), t_end);
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
            bench("elastic_2d", &space, &kernel(), &mass(), &mode_2d(), &args);
            let space = continuous_space::<3, Iso3<5>>(&cuboid(L3, [16, 32, 16], Geometry::CurvedQ2));
            bench("elastic_3d", &space, &kernel(), &mass(), &mode_3d(), &args);
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
