//! Declared quadratures (`SpaceBuilder::quadrature`): weights, points and
//! values at the quadrature points, stored or recomputed geometry, exact
//! (mapped) geometry from parametric meshes.

mod common;

use anamesh::mesh::circle::Circle;
use anamesh::mesh::cube::Cube;
use anamesh::mesh::square::Square;
use anamesh::mesh::Mesh;
use common::*;
use lobattox::fields::Constant;
use lobattox::kernels::IsotropicElastic;
use lobattox::{GeoKind, GeoStorage, Gl, Gll, Gll3, Iso2, Iso3, Numbering, Quad2, Quad3, Space};

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

fn integrate<const D: usize>(w: &[f64], x: &[[f64; D]], f: impl Fn([f64; D]) -> f64) -> f64 {
    w.iter().zip(x).map(|(w, x)| w * f(*x)).sum()
}

fn kinds<const D: usize, S: lobattox::Shape<D>>(space: &Space<D, S>) -> [usize; 4] {
    GeoKind::ALL.map(|k| space.elements(k).len())
}

type G34 = Quad2<Gl<3>, Gl<4>>;
type L45 = Quad2<Gll<4>, Gll<5>>;

/// `∫ x^a y^b` over `[0, 2] × [0, 1]`, exact up to the degree of each rule:
/// Gauss–Legendre `M` points to `2M − 1`, Gauss–Lobatto to `2M − 3`.
#[test]
fn polynomials_are_integrated_exactly() {
    let mesh = square(4, 2, false, None);
    let space = Space::<2, Iso2<3>>::builder(&mesh)
        .quadrature::<G34>(GeoStorage::Recompute)
        .quadrature::<L45>(GeoStorage::Store)
        .build();
    let exact = |a: i32, b: i32| 2f64.powi(a + 1) / (a + 1) as f64 / (b + 1) as f64;
    let (w, x) = (
        space.quadrature_weights::<G34>(),
        space.quadrature_xyz::<G34>(),
    );
    for (a, b) in [(0, 0), (5, 7), (3, 6), (5, 0)] {
        let i = integrate(&w, &x, |x| x[0].powi(a) * x[1].powi(b));
        assert!(
            (i - exact(a, b)).abs() < 1e-13 * exact(a, b),
            "GL x^{a} y^{b}: {i}"
        );
    }
    let (w, x) = (
        space.quadrature_weights::<L45>(),
        space.quadrature_xyz::<L45>(),
    );
    for (a, b) in [(0, 0), (3, 7), (2, 5)] {
        let i = integrate(&w, &x, |x| x[0].powi(a) * x[1].powi(b));
        assert!(
            (i - exact(a, b)).abs() < 1e-13 * exact(a, b),
            "GLL x^{a} y^{b}: {i}"
        );
    }
}

/// Stored and recomputed geometry give the same bits (same tables, same
/// arithmetic), for the quadratures and for the operator at the nodes.
#[test]
fn stored_geometry_equals_recomputed() {
    type Q = Quad3<Gl<3>, Gl<4>, Gll<3>>;
    for q2 in [false, true] {
        let mesh = cube([2, 1, 2], q2, Some(bending_3d()));
        let build = |storage| {
            Space::<3, Iso3<4>>::builder(&mesh)
                .node_geometry(storage)
                .quadrature::<Q>(storage)
                .build()
        };
        let (r, s) = (build(GeoStorage::Recompute), build(GeoStorage::Store));
        assert_eq!(r.quadrature_weights::<Q>(), s.quadrature_weights::<Q>());
        assert_eq!(r.quadrature_xyz::<Q>(), s.quadrature_xyz::<Q>());
        let u = seeded(r.n_dup() * 3, 1);
        assert_eq!(
            r.quadrature_values::<Q, 3>(&u),
            s.quadrature_values::<Q, 3>(&u)
        );
        let k = IsotropicElastic {
            lambda: Constant(2.0),
            mu: Constant(1.0),
        };
        let (mut a, mut b) = (vec![0.0; u.len()], vec![0.0; u.len()]);
        r.mlt_add::<3>(&k, 1.0, &u, 0.0, &mut a);
        s.mlt_add::<3>(&k, 1.0, &u, 0.0, &mut b);
        assert_eq!(a, b, "q2={q2}");
        assert_eq!(r.local_mass(&Constant(1.0)), s.local_mass(&Constant(1.0)));
    }
}

/// The quadrature on the space's own GLL nodes reproduces the nodes: values
/// are `u`, weights the unit local mass, points the node images (bitwise).
#[test]
fn node_quadrature_is_the_identity() {
    type Nodes = Quad2<Gll<5>, Gll<5>>;
    let mesh = square(3, 2, true, Some(bending_2d()));
    let space = Space::<2, Iso2<5>>::builder(&mesh)
        .quadrature::<Nodes>(GeoStorage::Recompute)
        .build();
    let u = seeded(space.n_dup() * 2, 2);
    assert_eq!(space.quadrature_values::<Nodes, 2>(&u), u);
    assert_eq!(
        space.quadrature_weights::<Nodes>(),
        space.local_mass(&Constant(1.0))
    );
    let x = space.quadrature_xyz::<Nodes>();
    let s = space.sample::<2>(|x| x);
    for (i, xi) in x.iter().enumerate() {
        let (e, n) = (i / Space::<2, Iso2<5>>::NLOC, i % Space::<2, Iso2<5>>::NLOC);
        assert_eq!(*xi, [s[e * 50 + n], s[e * 50 + 25 + n]]);
    }
}

/// The node rule `S::Nodes` is available without being declared, with the
/// node geometry: the same results as when declared, `mlt_add_mixed` on it is
/// `mlt_add`, and `local_load` is `local_load_with` on it (all bitwise).
#[test]
fn node_rule_is_implicit() {
    type Nodes = <Iso2<5> as lobattox::Shape<2>>::Nodes;
    let mesh = square(3, 2, true, Some(bending_2d()));
    let implicit = Space::<2, Iso2<5>>::new(&mesh, Numbering::None);
    let declared = Space::<2, Iso2<5>>::builder(&mesh)
        .quadrature::<Nodes>(GeoStorage::Store)
        .build();
    let u = seeded(implicit.n_dup() * 2, 3);
    assert_eq!(implicit.quadrature_values::<Nodes, 2>(&u), u);
    assert_eq!(
        implicit.quadrature_weights::<Nodes>(),
        declared.quadrature_weights::<Nodes>()
    );
    assert_eq!(
        implicit.quadrature_xyz::<Nodes>(),
        declared.quadrature_xyz::<Nodes>()
    );
    let f = |x: [f64; 2]| [x[0] * x[1], x[1].sin()];
    assert_eq!(
        implicit.local_load::<2>(f),
        declared.local_load_with::<Nodes, 2>(f)
    );
    let k = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(1.0),
    };
    let (mut a, mut b) = (vec![0.0; u.len()], vec![0.0; u.len()]);
    implicit.mlt_add::<2>(&k, 1.0, &u, 0.0, &mut a);
    implicit.mlt_add_mixed::<Nodes, 2, 2, _>(&implicit, &k, 1.0, &u, 0.0, &mut b);
    assert_eq!(a, b);
}

/// Interpolated polynomials of degree `≤ N − 1` per direction are exact at
/// the quadrature points, and over-integrated L² norms are exact: anisotropic
/// shape, anisotropic and mixed-family rule.
#[test]
fn values_and_l2_norms_are_exact() {
    type Q = Quad3<Gl<3>, Gll<4>, Gl<2>>;
    let mesh = cube([2, 2, 1], false, None); // [0,1]×[0,2]×[0,1.5]
    let space = Space::<3, Gll3<2, 4, 3>>::builder(&mesh)
        .quadrature::<Q>(GeoStorage::Recompute)
        .build();
    let p = |x: [f64; 3]| (1.0 + x[0]) * (x[1] * x[1] * x[1] - x[1]) * (0.5 + x[2] * x[2]);
    let u = space.sample::<1>(|x| [p(x)]);
    let (v, x, w) = (
        space.quadrature_values::<Q, 1>(&u),
        space.quadrature_xyz::<Q>(),
        space.quadrature_weights::<Q>(),
    );
    for (vi, xi) in v.iter().zip(&x) {
        assert!((vi - p(*xi)).abs() < 1e-13, "{vi} vs {}", p(*xi));
    }
    // ∫ p over the box: degrees (1, 3, 2) are within (5, 5, 3).
    // ∫(1+x) = 1.5 on [0,1], ∫(y³−y) = 2 on [0,2], ∫(½+z²) = 1.875 on [0,1.5].
    let ex = 1.5 * (2.0f64.powi(4) / 4.0 - 2.0) * (0.75 + 1.5f64.powi(3) / 3.0);
    let i: f64 = v.iter().zip(&w).map(|(v, w)| v * w).sum();
    assert!((i - ex).abs() < 1e-12 * ex.abs(), "∫ p = {i}, exact {ex}");
}

/// Circle: the core cells are affine, the arms mapped; with the exact map the
/// area is π R² to quadrature accuracy, against the Q2 interpolation of the
/// converted mesh. The circle numbering is continuous.
#[test]
fn circle_exact_geometry() {
    type Q = Quad2<Gl<8>, Gl<8>>;
    let (center, radius) = ([0.3, -0.2], 1.5);
    let circle = Circle::new(center, radius, 7, None, Some(true));
    let space = Space::<2, Iso2<4>>::builder(&circle)
        .numbering(Numbering::Continuous)
        .quadrature::<Q>(GeoStorage::Recompute)
        .build();
    let n_core = circle.n_x * circle.n_x;
    assert_eq!(
        kinds(&space),
        [
            n_core,
            0,
            0,
            circle.get_elements_by_dimension(2).len() - n_core
        ]
    );
    let exact = std::f64::consts::PI * radius * radius;
    let area: f64 = space.quadrature_weights::<Q>().iter().sum();
    assert!(
        (area - exact).abs() < 1e-11 * exact,
        "exact geometry: {area} vs {exact}"
    );
    // The unstructured conversion keeps only the Q2 nodes.
    let q2 = Space::<2, Iso2<4>>::builder(&circle.get_unstructured_mesh())
        .quadrature::<Q>(GeoStorage::Recompute)
        .build();
    let area_q2: f64 = q2.quadrature_weights::<Q>().iter().sum();
    assert!(
        (area_q2 - exact).abs() > 1e-6 * exact,
        "Q2 interpolation: {area_q2}"
    );
    // Boundary of the exact geometry: all nodes within the disk, the outer ones on it.
    let x = space.sample::<2>(|x| x);
    let n = Space::<2, Iso2<4>>::NLOC;
    let r_max = (0..space.n_dup())
        .map(|i| {
            let (e, k) = (i / n, i % n);
            ((x[e * 2 * n + k] - center[0]).powi(2) + (x[e * 2 * n + n + k] - center[1]).powi(2))
                .sqrt()
        })
        .fold(0.0, f64::max);
    assert!((r_max - radius).abs() < 1e-12, "outer radius {r_max}");
    // Continuity of a sampled field across the shared nodes.
    let mut v = space.sample::<1>(|x| [x[0].sin() * x[1]]);
    let before = v.clone();
    space.c_sum_shared::<1>(&mut v);
    let m = space.c_multiplicity();
    for ((a, b), k) in v.iter().zip(&before).zip(&m) {
        assert!((a / k - b).abs() < 1e-12);
    }
}

/// Parametric `Square` with an attached deformation: mapped elements stored
/// from the exact map; anisotropic continuous numbering from the structured
/// mesh.
#[test]
fn deformed_square_is_mapped() {
    type Q = Quad2<Gl<5>, Gl<3>>;
    let sq = Square::new([0.0, 0.0], 2.0, 1.0, 4, 2, Some(bending_2d()), None);
    let space = Space::<2, lobattox::Gll2<5, 3>>::builder(&sq)
        .numbering(Numbering::Continuous)
        .quadrature::<Q>(GeoStorage::Recompute)
        .build();
    assert_eq!(kinds(&space), [0, 0, 0, 8]);
    // Points: images of the reference quadrature points by the exact map.
    let x = space.quadrature_xyz::<Q>();
    let w = space.quadrature_weights::<Q>();
    assert_eq!((x.len(), w.len()), (8 * 15, 8 * 15));
    // Area of the deformed rectangle: det Df integrated over [0,2]×[0,1].
    // det Df = (1 + 0.05 y) − 0.4 x (0.1 cos y + 0.05 x).
    let fine = |n: usize| {
        let h = (2.0 / n as f64, 1.0 / n as f64);
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..n {
                let (xc, yc) = ((i as f64 + 0.5) * h.0, (j as f64 + 0.5) * h.1);
                s += (1.0 + 0.05 * yc - 0.4 * xc * (0.1 * yc.cos() + 0.05 * xc)) * h.0 * h.1;
            }
        }
        s
    };
    let area: f64 = w.iter().sum();
    let reference = fine(2000);
    assert!((area - reference).abs() < 1e-6, "{area} vs {reference}");
}

#[test]
#[should_panic(expected = "was not declared")]
fn undeclared_quadrature_panics() {
    let space = Space::<2, Iso2<3>>::new(&square(2, 2, false, None), Numbering::None);
    let _ = space.quadrature_weights::<G34>();
}

#[test]
#[should_panic(expected = "declared twice")]
fn quadrature_declared_twice_panics() {
    let mesh = square(2, 2, false, None);
    let _ = Space::<2, Iso2<3>>::builder(&mesh)
        .quadrature::<G34>(GeoStorage::Store)
        .quadrature::<G34>(GeoStorage::Recompute)
        .build();
}

/// The engine on mapped elements (stored exact geometry) matches the engine
/// on the same elements with recomputed geometry of their exact Q1 nodes, for
/// a bilinear deformation (the exact map IS Q1).
#[test]
fn engine_on_mapped_elements() {
    let mapped = Square::new([0.0, 0.0], 2.0, 1.0, 3, 2, Some(bilinear_2d()), None);
    let mut applied = mapped.get_unstructured_mesh();
    applied.apply_deformation();
    let (sm, sa) = (
        Space::<2, Iso2<5>>::new(&mapped, Numbering::Continuous),
        Space::<2, Iso2<5>>::new(&applied, Numbering::Continuous),
    );
    assert_eq!(kinds(&sm), [0, 0, 0, 6]);
    assert_eq!(kinds(&sa), [0, 6, 0, 0]);
    let k = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(0.7),
    };
    let u = sm.sample::<2>(|x| [x[0].sin() * x[1], x[0] * x[0] - x[1]]);
    let (mut a, mut b) = (vec![0.0; u.len()], vec![0.0; u.len()]);
    sm.mlt_add::<2>(&k, 1.0, &u, 0.0, &mut a);
    sa.mlt_add::<2>(&k, 1.0, &u, 0.0, &mut b);
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(diff < 1e-12 * scale, "max diff {diff:e}");
}

/// Cube parametric mesh, deformed: mapped 3D elements.
#[test]
fn deformed_cube_is_mapped() {
    type Q = Quad3<Gl<4>, Gl<4>, Gl<4>>;
    let cube = Cube::new([0.0; 3], 1.0, 1.0, 1.0, 2, 2, 2, Some(bending_3d()), None);
    let space = Space::<3, Iso3<3>>::builder(&cube)
        .quadrature::<Q>(GeoStorage::Recompute)
        .build();
    assert_eq!(kinds(&space), [0, 0, 0, 8]);
    let vol: f64 = space.quadrature_weights::<Q>().iter().sum();
    assert!(vol > 0.5 && vol < 2.0, "volume {vol}");
    let nodal: f64 = space.local_mass(&Constant(1.0)).iter().sum();
    assert!((vol - nodal).abs() < 1e-3, "{vol} vs nodal {nodal}");
}
