//! Spaces with and without the continuous numbering ([`Numbering`]), and the
//! inverse mass operators [`Space::inverse_mass`] / [`Space::c_inverse_mass`].

mod common;

use common::*;
use lobattox::fields::{Constant, PerLabel};
use lobattox::kernels::{IsotropicElastic, Mass};
use lobattox::{Iso2, Iso3, Numbering, Space};

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

fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / scale
}

fn discontinuous_2d() -> Space<2, Iso2<4>> {
    Space::new(&square(3, 2, true, Some(bending_2d())), Numbering::None)
}

#[test]
fn numbering_does_not_change_storage_or_local_operators() {
    let mesh = cube([2, 2, 1], false, Some(bending_3d()));
    let (sc, sd) = (
        Space::<3, Iso3<3>>::new(&mesh, Numbering::Continuous),
        Space::<3, Iso3<3>>::new(&mesh, Numbering::None),
    );
    assert!(sc.has_numbering() && !sd.has_numbering());
    assert_eq!((sc.n_elem(), sc.n_dup()), (sd.n_elem(), sd.n_dup()));
    for e in 0..sc.n_elem() {
        assert_eq!(sc.mesh_element(e), sd.mesh_element(e));
        assert_eq!(sc.kind(e), sd.kind(e));
    }
    let k = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(1.0),
    };
    let u = seeded(sc.n_dup() * 3, 1);
    let (mut vc, mut vd) = (vec![0.0; u.len()], vec![0.0; u.len()]);
    sc.mlt_add::<3>(&k, 1.0, &u, 0.0, &mut vc);
    sd.mlt_add::<3>(&k, 1.0, &u, 0.0, &mut vd);
    assert_eq!(vc, vd);
    assert_eq!(sc.local_mass(&Constant(2.0)), sd.local_mass(&Constant(2.0)));
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn c_sum_shared_panics_without_numbering() {
    let s = discontinuous_2d();
    let mut v = vec![0.0; s.n_dup()];
    s.c_sum_shared::<1>(&mut v);
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn c_n_global_panics_without_numbering() {
    let _ = discontinuous_2d().c_n_global();
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn c_multiplicity_panics_without_numbering() {
    let _ = discontinuous_2d().c_multiplicity();
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn c_inverse_mass_panics_without_numbering() {
    let _ = discontinuous_2d().c_inverse_mass(&Constant(1.0));
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn c_max_eigenvalue_panics_without_numbering() {
    let s = discontinuous_2d();
    let k = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(1.0),
    };
    let _ = s.c_max_eigenvalue::<2>(&k, &Constant(1.0), 10, 1e-6);
}

/// `inverse_mass` is the element-local mass inverse on any space, bit for bit
/// `1 / local_mass`; `c_inverse_mass` is `1 / Σ_shared local_mass`.
#[test]
fn inverse_mass_is_local_and_c_inverse_mass_is_assembled() {
    let mesh = square(3, 2, true, Some(bending_2d()));
    let rho = PerLabel(vec![1.5; 8]);
    for numbering in [Numbering::None, Numbering::Continuous] {
        let s = Space::<2, Iso2<4>>::new(&mesh, numbering);
        let m = s.local_mass(&rho);
        let inv = s.inverse_mass(&rho);
        let expected: Vec<f64> = m.iter().map(|x| 1.0 / x).collect();
        assert_eq!(inv.diagonal().unwrap(), expected.as_slice());
    }
    let s = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let mut m = s.local_mass(&rho);
    s.c_sum_shared::<1>(&mut m);
    let expected: Vec<f64> = m.iter().map(|x| 1.0 / x).collect();
    assert_eq!(
        s.c_inverse_mass(&rho).diagonal().unwrap(),
        expected.as_slice()
    );
}

/// `M⁻¹ (M u) = u`: element-local mass on the discontinuous space (any `u`),
/// assembled mass on the continuous space (continuous `u`).
#[test]
fn inverse_mass_inverts_the_mass_operator() {
    let mesh = cube([2, 1, 2], true, Some(bending_3d()));
    let coef = Constant(3.0);
    let mass = Mass {
        coef: Constant(3.0),
    };

    let sd = Space::<3, Iso3<4>>::new(&mesh, Numbering::None);
    let u = seeded(sd.n_dup() * 2, 2);
    let mut v = vec![0.0; u.len()];
    sd.mlt_add::<2>(&mass, 1.0, &u, 0.0, &mut v);
    sd.inverse_mass(&coef).apply::<2>(&mut v);
    assert!(max_rel_diff(&v, &u) < 1e-14, "discontinuous");

    let sc = Space::<3, Iso3<4>>::new(&mesh, Numbering::Continuous);
    let mut u = seeded(sc.n_dup() * 2, 3);
    sc.c_sum_shared::<2>(&mut u); // continuous
    let mut v = vec![0.0; u.len()];
    sc.c_apply::<2>(&mass, &u, &mut v);
    sc.c_inverse_mass(&coef).apply::<2>(&mut v);
    assert!(max_rel_diff(&v, &u) < 1e-14, "continuous");
}

/// `InverseMass::apply` is `scale_by` with its diagonal, bit for bit.
#[test]
fn inverse_mass_apply_matches_scale_by() {
    let s = Space::<2, Iso2<5>>::new(
        &square(2, 3, false, Some(bilinear_2d())),
        Numbering::Continuous,
    );
    let im = s.c_inverse_mass(&Constant(0.7));
    let u = seeded(s.n_dup() * 3, 4);
    let (mut a, mut b) = (u.clone(), u);
    im.apply::<3>(&mut a);
    s.scale_by::<3>(im.diagonal().unwrap(), &mut b);
    assert_eq!(a, b);
}
