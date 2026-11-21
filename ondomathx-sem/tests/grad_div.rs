//! `operators::GradDivQ4Q2` (cornea `GradDiv.h`: `b_h` by Green's formula,
//! lumped GLL 3 pressure mass) against `operators::FusedKp` with `b_h` on the
//! displacement GLL rule (option 2) and the same lumped pressure mass: the
//! article's equivalence of the options on Q1 meshes.

mod common;

use anamesh::mesh::Mesh;
use common::*;
use lobattox::operators::{FusedKp, GradDivQ4Q2};
use lobattox::{mlt_add_element, ElementOperator, GeoStorage, Gll, Iso2, Iso3, Numbering, Quad2, Quad3, Shape, Space};

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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(1e-300_f64, |m, v| m.max(v.abs()));
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / scale
}

fn apply<const D: usize, S: Shape<D>, Op: ElementOperator<D, S, S, D, D>>(s: &Space<D, S>, op: &Op, u: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0; u.len()];
    mlt_add_element::<D, _, _, D, D, _>(s, s, op, 1.0, u, 0.0, &mut v);
    v
}

/// Returns `max |K₃ u − K₂ u| / max |K₂ u|` (option 3 vs option 2), after
/// checking that `K₃` is symmetric and positive semi-definite.
fn options_2d(mesh: &impl Mesh, c: f64) -> f64 {
    type QB = Quad2<Gll<5>, Gll<5>>;
    type QM = Quad2<Gll<3>, Gll<3>>;
    let disp = Space::<2, Iso2<5>>::builder(mesh)
        .numbering(Numbering::Continuous)
        .quadrature::<QB>(GeoStorage::Store)
        .build();
    let pres = Space::<2, Iso2<3>>::builder(mesh).quadrature::<QM>(GeoStorage::Store).build();
    let k3 = GradDivQ4Q2::<2>::new(&disp, c);
    let k2 = FusedKp::<2, Iso2<5>, Iso2<3>, QB, QM>::new(&disp, &pres, c);
    let (u, w) = (seeded(disp.n_dup() * 2, 1), seeded(disp.n_dup() * 2, 2));
    let (k3u, k3w) = (apply(&disp, &k3, &u), apply(&disp, &k3, &w));
    let (a, b) = (dot(&k3u, &w), dot(&u, &k3w));
    assert!((a - b).abs() < 1e-11 * a.abs().max(1.0), "symmetry {a} vs {b}");
    assert!(dot(&k3u, &u) >= 0.0);
    max_rel(&k3u, &apply(&disp, &k2, &u))
}

fn options_3d(mesh: &impl Mesh) -> f64 {
    type QB = Quad3<Gll<5>, Gll<5>, Gll<5>>;
    type QM = Quad3<Gll<3>, Gll<3>, Gll<3>>;
    let disp = Space::<3, Iso3<5>>::builder(mesh)
        .numbering(Numbering::Continuous)
        .quadrature::<QB>(GeoStorage::Store)
        .build();
    let pres = Space::<3, Iso3<3>>::builder(mesh).quadrature::<QM>(GeoStorage::Store).build();
    let k3 = GradDivQ4Q2::<3>::new(&disp, 1.0);
    let k2 = FusedKp::<3, Iso3<5>, Iso3<3>, QB, QM>::new(&disp, &pres, 1.0);
    let u = seeded(disp.n_dup() * 3, 3);
    max_rel(&apply(&disp, &k3, &u), &apply(&disp, &k2, &u))
}

#[test]
fn green_and_quadrature_options_agree_on_q1_meshes_2d() {
    for (mesh, what) in [
        (square(3, 2, false, None), "affine"),
        (square(4, 3, false, Some(bilinear_2d())), "bilinear Q1"),
        (square(5, 3, false, Some(partial_2d())), "affine + Q1"),
        (square(3, 2, false, Some(bending_2d())), "bending interpolated by Q1"),
    ] {
        let err = options_2d(&mesh, 1.0);
        assert!(err < 1e-12, "{what}: option 3 vs option 2 {err:e}");
    }
    assert!(options_2d(&square(3, 2, false, Some(bending_2d())), 2.5) < 1e-12, "c = 2.5");
}

#[test]
fn green_and_quadrature_options_agree_on_q1_meshes_3d() {
    for q1 in [cube([2, 1, 2], false, None), cube([2, 1, 2], false, Some(bending_3d()))] {
        let err = options_3d(&q1);
        assert!(err < 1e-12, "3D Q1: option 3 vs option 2 {err:e}");
    }
}

/// Non-separable 3D map (non-polynomial in every reference direction).
fn twist_3d() -> anamesh::diffeomorphism::Diffeomorphism {
    anamesh::diffeomorphism::Diffeomorphism {
        inner_dim: 3,
        outer_dim: 3,
        params: vec![],
        diffeo: |_p, x, f| {
            f[0] = x[0] + 0.08 * (x[1] + x[2]).sin() * x[0];
            f[1] = x[1] + 0.08 * (x[0] * x[2]).cos();
            f[2] = x[2] + 0.08 * (x[0] + x[1]).sin() * x[2];
        },
        grad_diffeo: |_p, x, g| {
            let (s12, c12) = ((x[1] + x[2]).sin(), (x[1] + x[2]).cos());
            let s02 = (x[0] * x[2]).sin();
            let (s01, c01) = ((x[0] + x[1]).sin(), (x[0] + x[1]).cos());
            g.copy_from_slice(&[
                1.0 + 0.08 * s12,
                0.08 * c12 * x[0],
                0.08 * c12 * x[0],
                -0.08 * s02 * x[2],
                1.0,
                -0.08 * s02 * x[0],
                0.08 * c01 * x[2],
                0.08 * c01 * x[2],
                1.0 + 0.08 * s01,
            ]);
        },
        inner_deformation: None,
    }
}

/// Beyond Q1: on Q2 elements the options still agree with k = 4 (GLL 5
/// summation by parts is exact, and so is the discrete Piola identity
/// `Σ_b ∂_b cof(J)_cb = 0`: each `cof_cb` is a polynomial of degree ≤ 4 in
/// `ξ_b` for a Q2 map). On mapped elements with a non-separable map (the
/// `Circle` arms, a twist) the identity is only approximate and the options
/// differ.
#[test]
fn options_on_curved_and_mapped_elements() {
    let q2 = options_2d(&square(3, 2, true, Some(bending_2d())), 1.0);
    let q2_3d = options_3d(&cube([2, 1, 2], true, Some(bending_3d())));
    let circle = anamesh::mesh::circle::Circle::new([0.3, -0.2], 1.5, 7, None, None);
    let mapped = options_2d(&circle, 1.0);
    let cb = anamesh::mesh::cube::Cube::new([0.0; 3], 1.0, 2.0, 1.5, 2, 1, 2, Some(twist_3d()), None);
    let mapped_3d = options_3d(&cb);
    println!("option 3 vs option 2: Q2 2D {q2:e}, Q2 3D {q2_3d:e}, Circle {mapped:e}, twisted cube {mapped_3d:e}");
    assert!(q2 < 1e-12 && q2_3d < 1e-12, "Q2: {q2:e}, {q2_3d:e}");
    assert!(mapped > 1e-10 && mapped_3d > 1e-10, "mapped: {mapped:e}, {mapped_3d:e}");
}

/// `FusedKpQ4Q2` (Green's `B` of `GradDivQ4Q2`, Gauss–Legendre pressure mass)
/// against `FusedKp` (option 2, same Gauss–Legendre mass): equal on Q1/Q2
/// meshes, different on non-separable mapped elements.
fn gl_options_2d(mesh: &impl Mesh) -> f64 {
    use lobattox::operators::{FusedKpQ4Q2, PressureMass2};
    use lobattox::Gl;
    type QB = Quad2<Gll<5>, Gll<5>>;
    let disp = Space::<2, Iso2<5>>::builder(mesh)
        .numbering(Numbering::Continuous)
        .quadrature::<QB>(GeoStorage::Store)
        .quadrature::<PressureMass2>(GeoStorage::Store)
        .build();
    let pres = Space::<2, Iso2<3>>::builder(mesh)
        .quadrature::<Quad2<Gl<3>, Gl<3>>>(GeoStorage::Store)
        .build();
    let k3 = FusedKpQ4Q2::<2>::new(&disp, 1.3);
    let k2 = FusedKp::<2, Iso2<5>, Iso2<3>, QB, Quad2<Gl<3>, Gl<3>>>::new(&disp, &pres, 1.3);
    let (u, w) = (seeded(disp.n_dup() * 2, 5), seeded(disp.n_dup() * 2, 6));
    let (k3u, k3w) = (apply(&disp, &k3, &u), apply(&disp, &k3, &w));
    let (a, b) = (dot(&k3u, &w), dot(&u, &k3w));
    assert!((a - b).abs() < 1e-11 * a.abs().max(1.0), "symmetry {a} vs {b}");
    assert!(dot(&k3u, &u) >= 0.0);
    max_rel(&k3u, &apply(&disp, &k2, &u))
}

fn gl_options_3d(mesh: &impl Mesh) -> f64 {
    use lobattox::operators::{FusedKpQ4Q2, PressureMass3};
    use lobattox::Gl;
    type QB = Quad3<Gll<5>, Gll<5>, Gll<5>>;
    type QM = Quad3<Gl<3>, Gl<3>, Gl<3>>;
    let disp = Space::<3, Iso3<5>>::builder(mesh)
        .numbering(Numbering::Continuous)
        .quadrature::<QB>(GeoStorage::Store)
        .quadrature::<PressureMass3>(GeoStorage::Store)
        .build();
    let pres = Space::<3, Iso3<3>>::builder(mesh).quadrature::<QM>(GeoStorage::Store).build();
    let k3 = FusedKpQ4Q2::<3>::new(&disp, 1.0);
    let k2 = FusedKp::<3, Iso3<5>, Iso3<3>, QB, QM>::new(&disp, &pres, 1.0);
    let u = seeded(disp.n_dup() * 3, 7);
    max_rel(&apply(&disp, &k3, &u), &apply(&disp, &k2, &u))
}

#[test]
fn fused_kp_q4q2_with_gauss_legendre_mass() {
    for (mesh, what) in [
        (square(3, 2, false, None), "affine"),
        (square(5, 3, false, Some(partial_2d())), "affine + Q1"),
        (square(3, 2, true, Some(bending_2d())), "curved Q2"),
    ] {
        let err = gl_options_2d(&mesh);
        assert!(err < 1e-12, "{what}: {err:e}");
    }
    for mesh in [cube([2, 1, 2], false, Some(bending_3d())), cube([2, 1, 2], true, Some(bending_3d()))] {
        let err = gl_options_3d(&mesh);
        assert!(err < 1e-12, "3D: {err:e}");
    }
    let circle = anamesh::mesh::circle::Circle::new([0.3, -0.2], 1.5, 7, None, None);
    let mapped = gl_options_2d(&circle);
    println!("FusedKpQ4Q2 vs FusedKp (GL mass) on the Circle: {mapped:e}");
    assert!(mapped > 1e-10);
}
