mod common;

use ondomathx_mesh::mesh::square::Square;
use ondomathx_mesh::mesh::{Mesh, UnstructuredMesh};
use common::*;
use ondomathx_sem::fields::Constant;
use ondomathx_sem::{GeoKind, Gll2, Gll3, Iso2, Iso3, Numbering, Shape, Space};

// ── Helpers ────────────────────────────────────────────────────────────

/// Number of copies of each duplicated node.
fn multiplicity<const D: usize, S: Shape<D>>(space: &Space<D, S>) -> Vec<f64> {
    let mut ones = vec![1.0; space.num_dg_dofs()];
    space.sum_shared::<1>(&mut ones);
    ones
}

/// Number of distinct nodes, recovered from the multiplicities: Σ 1/mult.
fn distinct_nodes(mult: &[f64]) -> f64 {
    mult.iter().map(|m| 1.0 / m).sum()
}

/// Σ_global mass = Σ_dup mass / mult, with mass = 1 / inverse_mass_dg.
fn total_mass<const D: usize, S: Shape<D>>(space: &Space<D, S>) -> f64 {
    let inv = space
        .inverse_mass(&Constant(1.0))
        .diagonal()
        .unwrap()
        .to_vec();
    let mult = multiplicity(space);
    inv.iter().zip(&mult).map(|(i, m)| 1.0 / (i * m)).sum()
}

/// Averaging over the copies of a continuous field gives the field back.
fn check_continuity<const D: usize, S: Shape<D>>(
    space: &Space<D, S>,
    f: impl Fn([f64; D]) -> [f64; 2] + Sync,
) {
    let v = space.sample::<2>(f);
    let mut avg = v.clone();
    space.sum_shared::<2>(&mut avg);
    let mult = multiplicity(space);
    let nloc = S::NLOC;
    for (i, (a, b)) in avg.iter().zip(&v).enumerate() {
        let (e, node) = (i / (2 * nloc), i % nloc);
        let a = a / mult[e * nloc + node];
        assert!(
            (a - b).abs() <= 1e-12 * b.abs().max(1.0),
            "entry {i}: {a} vs {b}"
        );
    }
}

/// The Jacobian at every GLL node matches the mesh's own `get_element_grad_def`.
fn check_jacobian<const D: usize, S: Shape<D>>(space: &Space<D, S>, mesh: &UnstructuredMesh) {
    for e in 0..space.num_elements() {
        for node in 0..S::NLOC {
            let mut uvw = [0.0; 3];
            uvw[..D].copy_from_slice(&space.loc_dof_coordinates(node));
            let mut g = vec![0.0; D * D];
            mesh.get_element_grad_def(space.mesh_element(e), &uvw, &mut g);
            let j = space.map_and_jacobian(e, node).1;
            for a in 0..D {
                for b in 0..D {
                    assert!(
                        (j[(a, b)] - g[a * D + b]).abs() <= 1e-12,
                        "e={e} node={node} J[{a}][{b}] = {} vs mesh {}",
                        j[(a, b)],
                        g[a * D + b]
                    );
                }
            }
        }
    }
}

fn kind_counts<const D: usize, S: Shape<D>>(space: &Space<D, S>) -> [usize; 4] {
    GeoKind::ALL.map(|k| space.elements(k).len())
}

/// Shoelace area of the quadrangles of a 2D Q1 mesh.
fn polygon_area(mesh: &UnstructuredMesh) -> f64 {
    let nodes = mesh.get_nodes();
    mesh.get_elements_by_dimension(2)
        .iter()
        .map(|&m| {
            let idx = &mesh.get_elements()[m].indexes;
            let p: Vec<[f64; 3]> = idx[..4].iter().map(|&i| nodes[i]).collect();
            0.5 * (0..4)
                .map(|k| p[k][0] * p[(k + 1) % 4][1] - p[(k + 1) % 4][0] * p[k][1])
                .sum::<f64>()
                .abs()
        })
        .sum()
}

// ── 2D ─────────────────────────────────────────────────────────────────

fn check_square<const NP: usize>() {
    let p = NP - 1;
    for q2 in [false, true] {
        let mesh = square(3, 2, q2, None);
        let space = Space::<2, Iso2<NP>>::new(&mesh, Numbering::Continuous);
        assert_eq!(space.num_elements(), 6);
        assert_eq!(
            space.num_dofs(),
            (3 * p + 1) * (2 * p + 1),
            "NP={NP} q2={q2}"
        );
        assert!((distinct_nodes(&multiplicity(&space)) - space.num_dofs() as f64).abs() < 1e-9);
        assert_eq!(
            kind_counts(&space),
            [6, 0, 0, 0],
            "straight Square is affine"
        );
        assert!((total_mass(&space) - 2.0).abs() < 1e-12, "area");
        check_continuity(&space, |x| [x[0].sin() + x[1], x[0] * x[1]]);
        check_jacobian(&space, &mesh);
    }
}

#[test]
fn square_np2() {
    check_square::<2>();
}
#[test]
fn square_np3() {
    check_square::<3>();
}
#[test]
fn square_np5() {
    check_square::<5>();
}
#[test]
fn square_np9() {
    check_square::<9>();
}

#[test]
fn deformed_q1_square() {
    let mesh = square(4, 3, false, Some(bending_2d()));
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    assert_eq!(kind_counts(&space), [0, 12, 0, 0]);
    // det J of a Q1 map is degree (1, 1): GLL with NP = 4 integrates it exactly.
    let (m, a) = (total_mass(&space), polygon_area(&mesh));
    assert!((m - a).abs() < 1e-12, "mass {m} vs area {a}");
    check_continuity(&space, |x| [x[0].sin() + x[1], x[0] * x[1]]);
    check_jacobian(&space, &mesh);
}

#[test]
fn deformed_q2_square() {
    let mesh = square(4, 3, true, Some(bending_2d()));
    let space = Space::<2, Iso2<5>>::new(&mesh, Numbering::Continuous);
    assert_eq!(kind_counts(&space), [0, 0, 12, 0]);
    check_continuity(&space, |x| [x[0].sin() + x[1], x[0] * x[1]]);
    check_jacobian(&space, &mesh);
}

#[test]
fn q2_with_straight_edges_is_demoted_to_q1() {
    let mesh = square(4, 3, true, Some(bilinear_2d()));
    let space = Space::<2, Iso2<3>>::new(&mesh, Numbering::Continuous);
    assert_eq!(kind_counts(&space), [0, 12, 0, 0]);
    check_jacobian(&space, &mesh);
}

#[test]
fn storage_permutation_and_labels() {
    let mesh = square(3, 2, false, Some(bending_2d()));
    let space = Space::<2, Iso2<3>>::new(&mesh, Numbering::Continuous);
    let mut seen: Vec<usize> = (0..space.num_elements()).map(|e| space.mesh_element(e)).collect();
    seen.sort();
    assert_eq!(seen, mesh.get_elements_by_dimension(2));
    for e in 0..space.num_elements() {
        assert_eq!(
            space.get_label(e),
            mesh.get_elements()[space.mesh_element(e)].label
        );
    }
}

/// An attached (unapplied) deformation makes every element mapped: the exact
/// map is stored, the Jacobians are the mesh's, continuity holds.
#[test]
fn attached_deformation_is_mapped() {
    let mesh =
        Square::new([0.0, 0.0], 1.0, 1.0, 3, 2, Some(bending_2d()), None).get_unstructured_mesh();
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    assert_eq!(kind_counts(&space), [0, 0, 0, 6]);
    check_jacobian(&space, &mesh);
    check_continuity(&space, |x| [x[0].sin() + x[1], x[0] * x[1]]);
    // A rotation (affine map) keeps the affine fast path.
    let mut rotated = Square::new([0.0, 0.0], 1.0, 1.0, 3, 2, None, None).get_unstructured_mesh();
    rotated.compose_deformation(rotation_2d());
    let space = Space::<2, Iso2<3>>::new(&rotated, Numbering::Continuous);
    assert_eq!(kind_counts(&space), [6, 0, 0, 0]);
    check_jacobian(&space, &rotated);
}

#[test]
#[should_panic(expected = "differ from D")]
fn wrong_dimension_panics() {
    let mesh = square(2, 2, false, None);
    let _ = Space::<3, Iso3<3>>::new(&mesh, Numbering::Continuous);
}

// ── 3D ─────────────────────────────────────────────────────────────────

fn check_cube<const NP: usize>() {
    let p = NP - 1;
    for q2 in [false, true] {
        let mesh = cube([2, 3, 2], q2, None);
        let space = Space::<3, Iso3<NP>>::new(&mesh, Numbering::Continuous);
        assert_eq!(space.num_elements(), 12);
        assert_eq!(
            space.num_dofs(),
            (2 * p + 1) * (3 * p + 1) * (2 * p + 1),
            "NP={NP} q2={q2}"
        );
        assert!((distinct_nodes(&multiplicity(&space)) - space.num_dofs() as f64).abs() < 1e-9);
        assert_eq!(
            kind_counts(&space),
            [12, 0, 0, 0],
            "straight Cube is affine"
        );
        assert!((total_mass(&space) - 3.0).abs() < 1e-12, "volume");
        check_continuity(&space, |x| [x[0].sin() + x[1] * x[2], x[0] * x[1] - x[2]]);
        check_jacobian(&space, &mesh);
    }
}

#[test]
fn cube_np2() {
    check_cube::<2>();
}
#[test]
fn cube_np4() {
    check_cube::<4>();
}
#[test]
fn cube_np7() {
    check_cube::<7>();
}

#[test]
fn deformed_cubes() {
    for (q2, counts) in [(false, [0, 12, 0, 0]), (true, [0, 0, 12, 0])] {
        let mesh = cube([2, 3, 2], q2, Some(bending_3d()));
        let space = Space::<3, Iso3<4>>::new(&mesh, Numbering::Continuous);
        assert_eq!(kind_counts(&space), counts, "q2={q2}");
        check_continuity(&space, |x| [x[0].sin() + x[1] * x[2], x[0] * x[1] - x[2]]);
        check_jacobian(&space, &mesh);
    }
}

// ── Small algebra ──────────────────────────────────────────────────────

/// Returns `(det via lobattox, test matrix)` after checking `J·cof(J)ᵀ = det(J)·I`.
fn check_cofactor<const D: usize>() -> (f64, nalgebra::SMatrix<f64, D, D>) {
    let m = nalgebra::SMatrix::<f64, D, D>::from_fn(|a, b| {
        ((3 * a + 7 * b + 1) as f64).sin() + if a == b { 2.0 } else { 0.0 }
    });
    let d = ondomathx_sem::det(&m);
    let should_be_identity = m * ondomathx_sem::cofactor(&m).transpose() / d;
    assert!((should_be_identity - nalgebra::SMatrix::<f64, D, D>::identity()).amax() < 1e-14);
    (d, m)
}

#[test]
fn cofactor_and_det() {
    // nalgebra's determinant needs concrete sizes (see `ondomathx_sem::cofactor`).
    let (d2, m2) = check_cofactor::<2>();
    assert!((d2 - m2.determinant()).abs() < 1e-14);
    let (d3, m3) = check_cofactor::<3>();
    assert!((d3 - m3.determinant()).abs() < 1e-13);
}

// ── Anisotropic shapes ─────────────────────────────────────────────────

/// Σ_dup local mass (no numbering needed) and the node Jacobians.
fn check_anisotropic<const D: usize, S: Shape<D>>(
    straight: &UnstructuredMesh,
    measure: f64,
    deformed: &UnstructuredMesh,
) {
    let n = &S::N[..D];
    let space = Space::<D, S>::new(straight, Numbering::None);
    assert_eq!(space.num_dg_dofs(), space.num_elements() * n.iter().product::<usize>());
    let m: f64 = space.local_mass(&Constant(1.0)).iter().sum();
    assert!((m - measure).abs() < 1e-12, "N={n:?}: measure {m}");
    for (d, &nd) in n.iter().enumerate() {
        assert_eq!(space.points_1d(d).len(), nd);
        assert!((space.weights_1d(d).iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }
    for node in 0..S::NLOC {
        let id = Space::<D, S>::multi_index(node);
        let x = space.loc_dof_coordinates(node);
        for d in 0..D {
            assert_eq!(x[d], space.points_1d(d)[id[d]]);
        }
    }
    check_jacobian(&Space::<D, S>::new(deformed, Numbering::None), deformed);
}

#[test]
fn anisotropic_shapes() {
    let (s, d) = (
        square(3, 2, false, None),
        square(3, 2, true, Some(bending_2d())),
    );
    check_anisotropic::<2, Gll2<3, 5>>(&s, 2.0, &d);
    check_anisotropic::<2, Gll2<6, 2>>(&s, 2.0, &d);
    let (s, d) = (
        cube([2, 3, 2], false, None),
        cube([2, 1, 2], true, Some(bending_3d())),
    );
    check_anisotropic::<3, Gll3<2, 4, 3>>(&s, 3.0, &d);
    check_anisotropic::<3, Gll3<5, 3, 4>>(&s, 3.0, &d);
}

/// An `UnstructuredMesh` numbers only isotropic layouts (`ondomathx-mesh` `DoFSet`).
#[test]
#[should_panic(expected = "same count")]
fn anisotropic_continuous_on_unstructured_panics() {
    let _ = Space::<2, Gll2<3, 4>>::new(&square(2, 2, false, None), Numbering::Continuous);
}

/// Structured meshes number anisotropic layouts (`Square::get_numbering`).
#[test]
fn anisotropic_continuous_on_square_and_cube() {
    let sq = Square::new([0.0, 0.0], 2.0, 1.0, 4, 3, Some(bending_2d()), None);
    let space = Space::<2, Gll2<3, 5>>::new(&sq, Numbering::Continuous);
    assert_eq!(space.num_dofs(), (4 * 2 + 1) * (3 * 4 + 1));
    check_continuity(&space, |x| [x[0].sin() + x[1], x[0] * x[1]]);
    assert!((distinct_nodes(&multiplicity(&space)) - space.num_dofs() as f64).abs() < 1e-9);
    let cube = ondomathx_mesh::mesh::cube::Cube::new([0.0; 3], 1.0, 2.0, 1.0, 2, 3, 2, None, None);
    let space = Space::<3, Gll3<2, 4, 3>>::new(&cube, Numbering::Continuous);
    assert_eq!(space.num_dofs(), (2 + 1) * (3 * 3 + 1) * (2 * 2 + 1));
    check_continuity(&space, |x| [x[0] * x[2] + x[1], x[0] - x[1] * x[2]]);
}

/// The DG ↔ global transfers on a curved mesh, two components: a shared
/// sum is `collect_dg` then `scatter_cg` (bit for bit), and averaging a
/// continuous field then scattering it gives the field back.
#[test]
fn transfers_between_dg_and_global() {
    let mesh = square(3, 2, true, Some(bending_2d()));
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let n_glob = space.num_dofs();
    let k: Vec<f64> = (0..2 * space.num_dg_dofs()).map(|i| ((i as f64) * 0.37).sin()).collect();

    let mut summed = k.clone();
    space.sum_shared::<2>(&mut summed);
    let mut cg = vec![0.0; 2 * n_glob];
    space.collect_dg::<2>(&k, &mut cg);
    let mut back = vec![f64::NAN; k.len()];
    space.scatter_cg::<2>(&cg, &mut back);
    assert_eq!(summed, back);

    let u = space.sample::<2>(|x| [x[0] * x[1], x[1].cos()]);
    space.collect_avg_dg::<2>(&u, &mut cg);
    space.scatter_cg::<2>(&cg, &mut back);
    for (a, b) in u.iter().zip(&back) {
        assert!((a - b).abs() <= 1e-15 * a.abs().max(1.0), "{a} vs {b}");
    }
}
