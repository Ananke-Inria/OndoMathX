//! Volume sources: `Space::local_load` (GLL rule) and `local_load_with`
//! (declared quadrature).

mod common;

use std::f64::consts::PI;

use common::*;
use lobattox::fields::Constant;
use lobattox::kernels::AcousticStiffness;
use lobattox::{GeoStorage, Gl, Iso2, Numbering, Quad2, Space};

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(1e-300_f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / scale
}

/// With the lumped mass, `M⁻¹ Σ_shared F` is the nodal `f / ρ`.
#[test]
fn lumped_load_is_nodal() {
    let mesh = square(4, 3, true, Some(bending_2d()));
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let f = |x: [f64; 2]| [x[0].sin() * x[1], 1.0 + x[0] * x[0]];
    let mut load = space.local_load::<2>(f);
    // f = 1: the local mass.
    let one = space.local_load::<1>(|_| [1.0]);
    assert!(max_rel(&one, &space.local_mass(&Constant(1.0))) < 1e-15);
    space.c_sum_shared::<2>(&mut load);
    space.c_inverse_mass(&Constant(2.0)).apply::<2>(&mut load);
    let nodal: Vec<f64> = space.sample::<2>(f).iter().map(|v| v / 2.0).collect();
    assert!(max_rel(&load, &nodal) < 1e-14);
}

/// `Σ_i F_i = ∫ f` and `⟨F, v⟩ = ∫ f v` (partition of unity, `v` in the
/// space) with an exact-enough Gauss–Legendre rule; the GLL node rule gives
/// `local_load`.
#[test]
fn quadrature_load_integrates() {
    type G = Quad2<Gl<5>, Gl<5>>;
    type Nodes = Quad2<lobattox::Gll<3>, lobattox::Gll<3>>;
    let mesh = square(4, 2, false, None); // [0, 2] × [0, 1]
    let space = Space::<2, Iso2<3>>::builder(&mesh)
        .quadrature::<G>(GeoStorage::Recompute)
        .quadrature::<Nodes>(GeoStorage::Store)
        .build();
    let f = |x: [f64; 2]| [x[0] * x[0] * x[1]];
    let load = space.local_load_with::<G, 1>(f);
    assert!((load.iter().sum::<f64>() - 4.0 / 3.0).abs() < 1e-13);
    // v = x y (in the space): ∫ x³ y² = 4 · 1/3.
    let v = space.sample::<1>(|x| [x[0] * x[1]]);
    let fv: f64 = load.iter().zip(&v).map(|(a, b)| a * b).sum();
    assert!((fv - 4.0 / 3.0).abs() < 1e-13, "{fv}");
    assert!(
        max_rel(
            &space.local_load_with::<Nodes, 1>(f),
            &space.local_load::<1>(f)
        ) < 1e-15
    );
}

/// Forced leapfrog `u'' − Δu = f` on the free (Neumann) square, with
/// `u = cos(πx) cos(πy)·t²` (so `f = (2 + 2π² t²) cos(πx) cos(πy)`, a
/// separable source `F·g(t)` per term): the scheme reproduces `u` to the
/// space-time error.
#[test]
fn forced_leapfrog() {
    let mesh = square(8, 4, false, None); // [0, 2] × [0, 1]
    let space = Space::<2, Iso2<7>>::new(&mesh, Numbering::Continuous);
    let k = AcousticStiffness { rho: Constant(1.0) };
    let im = space.c_inverse_mass(&Constant(1.0));
    let shape = |x: [f64; 2]| [(PI * x[0]).cos() * (PI * x[1]).cos()];
    let mut mf = space.local_load::<1>(shape); // M⁻¹F of the spatial profile
    space.c_sum_shared::<1>(&mut mf);
    im.apply::<1>(&mut mf);
    let s = space.sample::<1>(shape);
    let (t_end, steps) = (0.5, 2000);
    let dt = t_end / steps as f64;
    // u⁰ = 0, u¹ = dt² s (exact: u(dt) = dt² s).
    let mut prev = vec![0.0; s.len()];
    let mut cur: Vec<f64> = s.iter().map(|v| dt * dt * v).collect();
    let mut y = vec![0.0; s.len()];
    for n in 1..steps {
        let t = n as f64 * dt;
        let g = 2.0 + 2.0 * PI * PI * t * t;
        space.c_apply::<1>(&k, &cur, &mut y);
        im.apply::<1>(&mut y);
        for i in 0..s.len() {
            let next = 2.0 * cur[i] - prev[i] - dt * dt * (y[i] - g * mf[i]);
            prev[i] = cur[i];
            cur[i] = next;
        }
    }
    let exact: Vec<f64> = s.iter().map(|v| v * t_end * t_end).collect();
    let err = max_rel(&cur, &exact);
    assert!(err < 1e-5, "forced leapfrog error {err:e}");
}
