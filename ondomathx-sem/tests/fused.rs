//! The fused pressure stiffness `operators::FusedKp` against the three global
//! operators `B`, `(M^P)⁻¹`, `Bᵀ`.

mod common;

use anamesh::mesh::square::Square;
use anamesh::mesh::Mesh;
use common::*;
use lobattox::fields::Constant;
use lobattox::kernels::{Divergence, DivergenceTranspose};
use lobattox::operators::FusedKp;
use lobattox::{
    mlt_add_element, mlt_add_mixed, GeoStorage, Gl, Gll, Iso2, Iso3, Numbering, Quad2, Quad3,
    Shape, Space, Sweep,
};

fn seeded(n: usize, seed: u64) -> Vec<f64> {
    let mut s = 0x9E37_79B9_7F4A_7C15_u64 ^ seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(1e-300_f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / scale
}

/// `y0 + λ K^P u` both ways; returns the max relative difference and
/// `⟨K^P u, u⟩`, `⟨B u, (M^P)⁻¹ B u⟩`.
fn compare<const D: usize, SU: Shape<D>, SP: Shape<D>, QB, QM>(
    mesh: &impl Mesh,
    c: f64,
) -> (f64, f64, f64)
where
    QB: Sweep<D, SU> + Sweep<D, SP>,
    QM: Sweep<D, SP>,
{
    let disp = Space::<D, SU>::builder(mesh)
        .numbering(Numbering::Continuous)
        .quadrature::<QB>(GeoStorage::Store)
        .build();
    let pres = Space::<D, SP>::builder(mesh)
        .quadrature::<QM>(GeoStorage::Store)
        .build();
    let lambda = 1e3;
    let u = seeded(disp.n_dup() * D, 1);
    let y0 = seeded(disp.n_dup() * D, 2);

    // Three global operators.
    let mut p = vec![0.0; pres.n_dup()];
    mlt_add_mixed::<QB, D, 1, D, _, _, _>(
        &disp,
        &pres,
        &Divergence {
            coef: Constant(1.0),
        },
        1.0,
        &u,
        0.0,
        &mut p,
    );
    let bu = p.clone();
    pres.inverse_mass_with::<QM, _>(&Constant(c))
        .apply::<1>(&mut p);
    let mut a = y0.clone();
    mlt_add_mixed::<QB, 1, D, D, _, _, _>(
        &pres,
        &disp,
        &DivergenceTranspose {
            coef: Constant(1.0),
        },
        lambda,
        &p,
        1.0,
        &mut a,
    );

    // Fused.
    let kp = FusedKp::<D, SU, SP, QB, QM>::new(&disp, &pres, c);
    let mut b = y0.clone();
    mlt_add_element::<D, _, _, D, D, _>(&disp, &disp, &kp, lambda, &u, 1.0, &mut b);

    let mut k = vec![0.0; u.len()];
    mlt_add_element::<D, _, _, D, D, _>(&disp, &disp, &kp, 1.0, &u, 0.0, &mut k);
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
    let diff: Vec<f64> = a.iter().zip(&y0).map(|(x, y)| x - y).collect();
    let fused: Vec<f64> = b.iter().zip(&y0).map(|(x, y)| x - y).collect();
    (max_rel(&fused, &diff), dot(&k, &u), dot(&bu, &p))
}

fn check(r: (f64, f64, f64), what: &str) {
    let (err, kuu, bmb) = r;
    assert!(err < 1e-12, "{what}: fused vs three operators {err:e}");
    assert!(
        kuu >= 0.0 && (kuu - bmb).abs() < 1e-11 * kuu.abs().max(1.0),
        "{what}: {kuu} vs {bmb}"
    );
}

/// k = 4: displacement Q4 (GLL 5 for b_h, option 2), pressure Q2 with the
/// 3-point Gauss–Legendre mass.
#[test]
fn fused_kp_2d() {
    type QB = Quad2<Gll<5>, Gll<5>>;
    type QM = Quad2<Gl<3>, Gl<3>>;
    for q2 in [false, true] {
        let mesh = square(5, 3, q2, Some(partial_2d()));
        check(
            compare::<2, Iso2<5>, Iso2<3>, QB, QM>(&mesh, 1.0),
            &format!("partial q2={q2}"),
        );
    }
    let mesh = square(3, 2, true, Some(bending_2d()));
    check(
        compare::<2, Iso2<5>, Iso2<3>, QB, QM>(&mesh, 2.5),
        "curved Q2, c = 2.5",
    );
    let sq = Square::new([0.0, 0.0], 2.0, 1.0, 3, 2, Some(bending_2d()), None);
    check(compare::<2, Iso2<5>, Iso2<3>, QB, QM>(&sq, 1.0), "mapped");
    // A Gauss–Legendre b_h rule (k + 1 points) instead of the GLL nodes.
    type QG = Quad2<Gl<5>, Gl<5>>;
    check(
        compare::<2, Iso2<5>, Iso2<3>, QG, QM>(&mesh, 1.0),
        "b_h with Gauss–Legendre",
    );
}

/// k = 3 in 3D: displacement Q3, pressure Q1 with the 2-point Gauss–Legendre mass.
#[test]
fn fused_kp_3d() {
    type QB = Quad3<Gll<4>, Gll<4>, Gll<4>>;
    type QM = Quad3<Gl<2>, Gl<2>, Gl<2>>;
    for q2 in [false, true] {
        let mesh = cube([2, 1, 2], q2, Some(bending_3d()));
        check(
            compare::<3, Iso3<4>, Iso3<2>, QB, QM>(&mesh, 1.0),
            &format!("3D q2={q2}"),
        );
    }
}

#[test]
#[should_panic(expected = "pressure counts")]
fn fused_kp_wrong_mass_rule_panics() {
    type QB = Quad2<Gll<5>, Gll<5>>;
    type QM = Quad2<Gl<4>, Gl<4>>;
    let mesh = square(2, 2, false, None);
    let disp = Space::<2, Iso2<5>>::builder(&mesh)
        .quadrature::<QB>(GeoStorage::Store)
        .build();
    let pres = Space::<2, Iso2<3>>::builder(&mesh)
        .quadrature::<QM>(GeoStorage::Store)
        .build();
    let _ = FusedKp::<2, _, _, QB, QM>::new(&disp, &pres, 1.0);
}
