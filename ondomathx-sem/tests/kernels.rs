mod common;

use anamesh::mesh::UnstructuredMesh;
use common::*;
use lobattox::fields::{Constant, Field, Function, Inverse, PerLabel, PointData};
use lobattox::kernels::{AcousticStiffness, IsotropicElastic, Mass};
use lobattox::{Iso2, Iso3, Numbering, PointKernel, Shape, Space};

// ── Helpers ────────────────────────────────────────────────────────────

/// Labels volume elements by the x-coordinate of their first node's element
/// centre: 1 left of `x = split`, 2 right of it.
fn two_materials(mut mesh: UnstructuredMesh, split: f64) -> UnstructuredMesh {
    let nodes = mesh.get_nodes().to_vec();
    mesh.set_labels(|_, el| {
        let cx = el.indexes.iter().map(|&i| nodes[i][0]).sum::<f64>() / el.indexes.len() as f64;
        if cx < split {
            1
        } else {
            2
        }
    });
    mesh
}

fn apply<const D: usize, S: Shape<D>, const C: usize, K: PointKernel<D, C>>(
    space: &Space<D, S>,
    k: &K,
    u: &[f64],
) -> Vec<f64> {
    let mut v = vec![f64::NAN; u.len()];
    space.mlt_add::<C, K>(k, 1.0, u, 0.0, &mut v);
    v
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * scale,
            "{what}: entry {i}: {x:e} vs {y:e}"
        );
    }
}

fn rho_fn(p: &[f64], x: &[f64]) -> f64 {
    p[0] + p[1] * x[0] * x[1]
}

// ── Fields ─────────────────────────────────────────────────────────────

#[test]
fn field_flags_follow_the_regularity() {
    type K<R> = AcousticStiffness<R>;
    const { assert!(!<K<Constant> as PointKernel<2, 1>>::USES_XYZ) };
    const { assert!(!<K<Constant> as PointKernel<2, 1>>::USES_LABEL) };
    const { assert!(!<K<Constant> as PointKernel<2, 1>>::USES_DOF) };
    const { assert!(<K<Function> as PointKernel<2, 1>>::USES_XYZ) };
    const { assert!(<K<PerLabel> as PointKernel<2, 1>>::USES_LABEL) };
    const { assert!(<K<PointData> as PointKernel<2, 1>>::USES_DOF) };
    const { assert!(<K<Inverse<Function>> as PointKernel<2, 1>>::USES_XYZ) };
    type E = IsotropicElastic<Constant, PerLabel>;
    const { assert!(<E as PointKernel<3, 3>>::USES_LABEL) };
    const { assert!(!<E as PointKernel<3, 3>>::USES_XYZ) };
    const { assert!(!<Mass<Constant> as PointKernel<2, 2>>::USES_GRAD) };
    const { assert!(!<Mass<Constant> as PointKernel<2, 2>>::OUTPUTS_FLUX) };
    const { assert!(<Function as Field<2>>::NEEDS_XYZ && !<Function as Field<2>>::NEEDS_DOF) };
}

#[test]
fn per_label_from_pairs() {
    let t = PerLabel::from_pairs(&[(1, 2.0), (3, 5.0)]);
    assert_eq!(t.0.len(), 4);
    assert!(t.0[0].is_nan() && t.0[2].is_nan());
    assert_eq!((t.0[1], t.0[3]), (2.0, 5.0));
}

/// `Function` (evaluated on the fly at the engine's points) and `PointData`
/// (sampled once at the nodes) describe the same smooth field.
#[test]
fn function_and_point_data_agree() {
    let mesh = square(4, 3, true, Some(bending_2d()));
    let space = Space::<2, Iso2<5>>::new(&mesh, Numbering::Continuous);
    let f = Function {
        f: rho_fn,
        params: vec![1.0, 0.5],
    };
    let pd = PointData::sample(&space, |x| rho_fn(&[1.0, 0.5], &x));
    let u = space.sample::<1>(|x| [x[0].sin() * x[1]]);
    assert_close(
        &apply(&space, &AcousticStiffness { rho: f.clone() }, &u),
        &apply(&space, &AcousticStiffness { rho: pd.clone() }, &u),
        1e-13,
        "acoustic stiffness",
    );
    assert_close(
        space.c_inverse_mass(&f).diagonal().unwrap(),
        space.c_inverse_mass(&pd).diagonal().unwrap(),
        1e-13,
        "inverse mass",
    );
    let u2 = space.sample::<2>(|x| [x[0], x[1] * x[1]]);
    assert_close(
        &apply::<2, Iso2<5>, 2, _>(&space, &Mass { coef: f }, &u2),
        &apply::<2, Iso2<5>, 2, _>(&space, &Mass { coef: pd }, &u2),
        1e-13,
        "mass, C = 2",
    );
}

#[test]
fn inverse_mass_with_fields() {
    let mesh = two_materials(square(4, 2, false, Some(partial_2d())), 1.0);
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let one = space
        .c_inverse_mass(&Constant(1.0))
        .diagonal()
        .unwrap()
        .to_vec();
    // Acoustic mass coefficient 1/κ: inverse mass scales by κ.
    let inv_kappa = space
        .c_inverse_mass(&Inverse(Constant(4.0)))
        .diagonal()
        .unwrap()
        .to_vec();
    let expected: Vec<f64> = one.iter().map(|x| 4.0 * x).collect();
    assert_close(&inv_kappa, &expected, 1e-14, "Inverse(κ)");
    // Mass kernel assembled = 1 / inverse_mass, for a per-label coefficient.
    let rho = PerLabel::from_pairs(&[(1, 1.0), (2, 3.0)]);
    let mut m = apply::<2, Iso2<4>, 1, _>(
        &space,
        &Mass { coef: rho.clone() },
        &vec![1.0; space.n_dup()],
    );
    space.c_sum_shared::<1>(&mut m);
    let inv = space.c_inverse_mass(&rho).diagonal().unwrap().to_vec();
    let expected: Vec<f64> = inv.iter().map(|x| 1.0 / x).collect();
    assert_close(&m, &expected, 1e-14, "Σ_shared(M_ρ·1) vs 1/inverse_mass(ρ)");
    // Total mass on a straight mesh = Σ_labels ρ·area = 1·1 + 3·1.
    let space = Space::<2, Iso2<3>>::new(
        &two_materials(square(4, 2, false, None), 1.0),
        Numbering::Continuous,
    );
    let inv = space.c_inverse_mass(&rho).diagonal().unwrap().to_vec();
    let mut mult = vec![1.0; space.n_dup()];
    space.c_sum_shared::<1>(&mut mult);
    let total: f64 = inv.iter().zip(&mult).map(|(i, k)| 1.0 / (i * k)).sum();
    assert!((total - 4.0).abs() < 1e-12, "total mass {total}");
}

// ── Acoustics ──────────────────────────────────────────────────────────

#[test]
fn acoustic_two_material_energy() {
    // p = x on [0,2]×[0,1], ρ = 1 left of x = 1 and 4 right of it:
    // ∫ |∇p|²/ρ = 1/1 + 1/4.
    for np_case in 0..2 {
        let mesh = two_materials(square(4, 3, np_case == 1, None), 1.0);
        let rho = PerLabel::from_pairs(&[(1, 1.0), (2, 4.0)]);
        let e = if np_case == 0 {
            let space = Space::<2, Iso2<3>>::new(&mesh, Numbering::Continuous);
            let p = space.sample::<1>(|x| [x[0]]);
            dot(&p, &apply(&space, &AcousticStiffness { rho }, &p))
        } else {
            let space = Space::<2, Iso2<6>>::new(&mesh, Numbering::Continuous);
            let p = space.sample::<1>(|x| [x[0]]);
            dot(&p, &apply(&space, &AcousticStiffness { rho }, &p))
        };
        assert!((e - 1.25).abs() < 1e-12, "energy {e}");
    }
}

#[test]
fn acoustic_stiffness_kills_constants_and_is_symmetric() {
    let mesh = two_materials(cube([2, 2, 2], true, Some(bending_3d())), 0.5);
    let space = Space::<3, Iso3<4>>::new(&mesh, Numbering::Continuous);
    let k = AcousticStiffness {
        rho: Function {
            f: rho_fn,
            params: vec![2.0, 1.0],
        },
    };
    let a1 = apply(&space, &k, &vec![1.0; space.n_dup()]);
    assert!(a1.iter().all(|x| x.abs() < 1e-11));
    let u = space.sample::<1>(|x| [(x[0] + 2.0 * x[1]).sin() + x[2]]);
    let w = space.sample::<1>(|x| [x[0] * x[1] - x[2] * x[2]]);
    let (a, b) = (
        dot(&w, &apply(&space, &k, &u)),
        dot(&u, &apply(&space, &k, &w)),
    );
    assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0), "{a} vs {b}");
}

// ── Elasticity ─────────────────────────────────────────────────────────

#[test]
fn elastic_two_material_energy_and_rigid_modes() {
    let mesh = two_materials(square(4, 3, false, None), 1.0);
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let k = IsotropicElastic {
        lambda: PerLabel::from_pairs(&[(1, 2.0), (2, 5.0)]),
        mu: PerLabel::from_pairs(&[(1, 0.7), (2, 1.3)]),
    };
    // u = E x, E symmetric: energy = Σ_materials area·(λ tr(E)² + 2μ E:E).
    let u = space.sample::<2>(|x| [0.3 * x[0] + 0.2 * x[1], 0.2 * x[0] - 0.1 * x[1]]);
    let (tr, ee) = (0.2f64, 0.09 + 0.01 + 2.0 * 0.04);
    let exact = (2.0 * tr * tr + 2.0 * 0.7 * ee) + (5.0 * tr * tr + 2.0 * 1.3 * ee);
    let e = dot(&u, &apply(&space, &k, &u));
    assert!((e - exact).abs() < 1e-12, "energy {e} vs {exact}");

    let mesh = two_materials(cube([2, 2, 2], false, Some(bending_3d())), 0.5);
    let space = Space::<3, Iso3<3>>::new(&mesh, Numbering::Continuous);
    let k = IsotropicElastic {
        lambda: Constant(2.0),
        mu: PerLabel::from_pairs(&[(1, 0.7), (2, 1.3)]),
    };
    for f in [
        |x: [f64; 3]| [-x[1], x[0], 0.0],
        |x: [f64; 3]| [0.0, x[2], -x[1]],
        |_x: [f64; 3]| [1.0, 2.0, 3.0],
    ] {
        let u = space.sample::<3>(f);
        assert!(
            apply(&space, &k, &u).iter().all(|x| x.abs() < 1e-11),
            "rigid mode"
        );
    }
}
