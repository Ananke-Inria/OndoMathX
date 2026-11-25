//! Spaces with and without the continuous numbering ([`Numbering`]), and the
//! inverse mass operators [`Space::inverse_mass_dg`] / [`Space::inverse_mass`].

mod common;

use common::*;
use ondomathx_sem::fields::{Constant, PerLabel};
use ondomathx_sem::kernels::{IsotropicElastic, Mass};
use ondomathx_sem::{Iso2, Iso3, Numbering, Space};

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
    assert_eq!((sc.num_elements(), sc.num_dg_dofs()), (sd.num_elements(), sd.num_dg_dofs()));
    for e in 0..sc.num_elements() {
        assert_eq!(sc.mesh_element(e), sd.mesh_element(e));
        assert_eq!(sc.kind(e), sd.kind(e));
    }
    let k = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(1.0),
    };
    let u = seeded(sc.num_dg_dofs() * 3, 1);
    let (mut vc, mut vd) = (vec![0.0; u.len()], vec![0.0; u.len()]);
    sc.mlt_dg::<3>(&k, &u, &mut vc, 1.0);
    sd.mlt_dg::<3>(&k, &u, &mut vd, 1.0);
    assert_eq!(vc, vd);
    assert_eq!(sc.local_mass(&Constant(2.0)), sd.local_mass(&Constant(2.0)));
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn sum_shared_panics_without_numbering() {
    let s = discontinuous_2d();
    let mut v = vec![0.0; s.num_dg_dofs()];
    s.sum_shared::<1>(&mut v);
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn num_dofs_panics_without_numbering() {
    let _ = discontinuous_2d().num_dofs();
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn multiplicity_dg_panics_without_numbering() {
    let _ = discontinuous_2d().multiplicity_dg();
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn inverse_mass_panics_without_numbering() {
    let _ = discontinuous_2d().inverse_mass(&Constant(1.0));
}

#[test]
#[should_panic(expected = "Numbering::Continuous")]
fn max_eigenvalue_panics_without_numbering() {
    let s = discontinuous_2d();
    let k = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(1.0),
    };
    let _ = s.max_eigenvalue::<2>(&k, &Constant(1.0), 10, 1e-6);
}

/// `inverse_mass_dg` is the element-local mass inverse on any space, bit for bit
/// `1 / local_mass`; `inverse_mass` is `1 / Σ_shared local_mass`.
#[test]
fn inverse_mass_is_local_and_c_inverse_mass_is_assembled() {
    let mesh = square(3, 2, true, Some(bending_2d()));
    let rho = PerLabel(vec![1.5; 8]);
    for numbering in [Numbering::None, Numbering::Continuous] {
        let s = Space::<2, Iso2<4>>::new(&mesh, numbering);
        let m = s.local_mass(&rho);
        let inv = s.inverse_mass_dg(&rho);
        let expected: Vec<f64> = m.iter().map(|x| 1.0 / x).collect();
        assert_eq!(inv.diagonal().unwrap(), expected.as_slice());
    }
    let s = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let mut m = s.local_mass(&rho);
    s.sum_shared::<1>(&mut m);
    let expected: Vec<f64> = m.iter().map(|x| 1.0 / x).collect();
    assert_eq!(
        s.inverse_mass(&rho).diagonal().unwrap(),
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
    let u = seeded(sd.num_dg_dofs() * 2, 2);
    let mut v = vec![0.0; u.len()];
    sd.mlt_dg::<2>(&mass, &u, &mut v, 1.0);
    sd.inverse_mass_dg(&coef).apply::<2>(&mut v);
    assert!(max_rel_diff(&v, &u) < 1e-14, "discontinuous");

    let sc = Space::<3, Iso3<4>>::new(&mesh, Numbering::Continuous);
    let mut u = seeded(sc.num_dg_dofs() * 2, 3);
    sc.sum_shared::<2>(&mut u); // continuous
    let mut v = vec![0.0; u.len()];
    sc.mlt_assembled::<2>(&mass, &u, &mut v);
    sc.inverse_mass(&coef).apply::<2>(&mut v);
    assert!(max_rel_diff(&v, &u) < 1e-14, "continuous");
}

/// `InverseMass::apply` is `scale_by` with its diagonal, bit for bit.
#[test]
fn inverse_mass_apply_matches_scale_by() {
    let s = Space::<2, Iso2<5>>::new(
        &square(2, 3, false, Some(bilinear_2d())),
        Numbering::Continuous,
    );
    let im = s.inverse_mass(&Constant(0.7));
    let u = seeded(s.num_dg_dofs() * 3, 4);
    let (mut a, mut b) = (u.clone(), u);
    im.apply::<3>(&mut a);
    s.scale_by::<3>(im.diagonal().unwrap(), &mut b);
    assert_eq!(a, b);
}
