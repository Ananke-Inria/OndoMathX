mod common;

use common::*;
use lobattox::fields::{Constant, PerLabel};
use lobattox::kernels::{AcousticStiffness, IsotropicElastic};
use lobattox::{
    dot, stable_dt, Iso2, Iso3, Numbering, PointKernel, Shape, Space, VtkField, VtkWriter,
};
use nalgebra::DMatrix;

/// Largest generalized eigenvalue of `(K, M)` by a dense solve: the continuous
/// basis is one indicator vector per global node (the shared-DOF sum of a unit
/// vector at its first copy), times `C` components.
fn dense_lambda_max<const D: usize, S: Shape<D>, const C: usize, K: PointKernel<D, C>>(
    space: &Space<D, S>,
    k: &K,
    m_local: &[f64],
) -> f64 {
    dense_lambda_max_of::<D, S, C>(
        space,
        |b, y| space.mlt_add::<C, K>(k, 1.0, b, 0.0, y),
        m_local,
    )
}

/// Same, for any element-local operator `local(x, y)` (`y = A_loc x`, with
/// `⟨b_i, A b_j⟩ = dot(b_i, A_loc b_j)` for continuous `b_i`, `b_j`).
fn dense_lambda_max_of<const D: usize, S: Shape<D>, const C: usize>(
    space: &Space<D, S>,
    local: impl Fn(&[f64], &mut [f64]),
    m_local: &[f64],
) -> f64 {
    let nloc = S::NLOC;
    let n = space.n_dup();
    // Representatives: the first copy of each global node.
    let mut seen = vec![false; n];
    let mut basis_1 = Vec::new();
    for i in 0..n {
        if seen[i] {
            continue;
        }
        let mut b = vec![0.0; n];
        b[i] = 1.0;
        space.c_sum_shared::<1>(&mut b);
        for (j, &v) in b.iter().enumerate() {
            if v != 0.0 {
                seen[j] = true;
            }
        }
        basis_1.push(b);
    }
    assert_eq!(basis_1.len(), space.c_n_global());
    let basis: Vec<Vec<f64>> = basis_1
        .iter()
        .flat_map(|b| {
            (0..C).map(move |c| {
                let mut v = vec![0.0; n * C];
                for e in 0..space.n_elem() {
                    for node in 0..nloc {
                        v[e * C * nloc + c * nloc + node] = b[e * nloc + node];
                    }
                }
                v
            })
        })
        .collect();
    let nb = basis.len();
    let kb: Vec<Vec<f64>> = basis
        .iter()
        .map(|b| {
            let mut y = vec![0.0; n * C];
            local(b, &mut y);
            y
        })
        .collect();
    let mdiag: Vec<f64> = basis
        .iter()
        .map(|b| space.mass_inner::<C>(m_local, b, b))
        .collect();
    let a = DMatrix::from_fn(nb, nb, |i, j| {
        dot(&basis[i], &kb[j]) / (mdiag[i] * mdiag[j]).sqrt()
    });
    let a = (&a + a.transpose()) * 0.5;
    a.symmetric_eigen().eigenvalues.max()
}

#[test]
fn local_mass_and_mass_inner() {
    // f = (x, y(1 + 0.3x)) on [0,2]×[0,1]: area ∫ (1 + 0.3x) = 2.6.
    let mesh = square(4, 3, false, Some(bilinear_2d()));
    let space = Space::<2, Iso2<3>>::new(&mesh, Numbering::Continuous);
    let m = space.local_mass(&Constant(1.0));
    assert!((m.iter().sum::<f64>() - 2.6).abs() < 1e-12);
    // L² norm of u = x on the straight square: ∫ x² = 8/3 (exact for P ≥ 2).
    let space = Space::<2, Iso2<3>>::new(&square(4, 3, false, None), Numbering::Continuous);
    let m = space.local_mass(&Constant(1.0));
    let u = space.sample::<1>(|x| [x[0]]);
    assert!((space.mass_inner::<1>(&m, &u, &u) - 8.0 / 3.0).abs() < 1e-12);
    // Two components add up; inverse_mass is 1/Σ_shared(local_mass).
    let u2 = space.sample::<2>(|x| [x[0], 1.0]);
    assert!((space.mass_inner::<2>(&m, &u2, &u2) - (8.0 / 3.0 + 2.0)).abs() < 1e-12);
    let mut ms = m.clone();
    space.c_sum_shared::<1>(&mut ms);
    let inv = space
        .c_inverse_mass(&Constant(1.0))
        .diagonal()
        .unwrap()
        .to_vec();
    assert!(inv
        .iter()
        .zip(&ms)
        .all(|(i, s)| (i * s - 1.0).abs() < 1e-14));
    let mult = space.c_multiplicity();
    assert_eq!(
        mult.iter().map(|k| 1.0 / k).sum::<f64>().round() as usize,
        space.c_n_global()
    );
}

#[test]
fn apply_and_scale_by() {
    let mesh = square(3, 2, true, Some(bending_2d()));
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let k = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(1.0),
    };
    let u = space.sample::<2>(|x| [x[0] * x[1], x[1].sin()]);
    let mut expected = vec![0.0; u.len()];
    space.mlt_add::<2, _>(&k, 1.0, &u, 0.0, &mut expected);
    space.c_sum_shared::<2>(&mut expected);
    let mut out = vec![f64::NAN; u.len()];
    space.c_apply::<2, _>(&k, &u, &mut out);
    assert_eq!(out, expected);

    let diag: Vec<f64> = (0..space.n_dup()).map(|i| 1.0 + i as f64).collect();
    let mut v = u.clone();
    space.scale_by::<2>(&diag, &mut v);
    let nloc = Space::<2, Iso2<4>>::NLOC;
    for (i, (a, b)) in v.iter().zip(&u).enumerate() {
        let (e, node) = (i / (2 * nloc), i % nloc);
        assert_eq!(*a, b * diag[e * nloc + node]);
    }
}

#[test]
fn max_eigenvalue_matches_dense() {
    // Acoustics, two materials, curved Q2.
    let mesh = square(2, 2, true, Some(bending_2d()));
    let mut mesh = mesh;
    let nodes = mesh.get_nodes().to_vec();
    mesh.set_labels(|_, el| if nodes[el.indexes[0]][0] < 1.0 { 1 } else { 2 });
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let rho = PerLabel::from_pairs(&[(1, 1.0), (2, 3.0)]);
    let k = AcousticStiffness { rho };
    let mass = Constant(0.5);
    let exact = dense_lambda_max(&space, &k, &space.local_mass(&mass));
    let est = space.c_max_eigenvalue::<1, _, _>(&k, &mass, 20_000, 1e-14);
    assert!(
        est.lambda <= exact * (1.0 + 1e-12),
        "{} > {exact}",
        est.lambda
    );
    assert!(
        (est.lambda - exact).abs() < 1e-8 * exact,
        "{} vs {exact}",
        est.lambda
    );

    // Elasticity, C = 2.
    let space = Space::<2, Iso2<3>>::new(
        &square(2, 1, false, Some(bilinear_2d())),
        Numbering::Continuous,
    );
    let k = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(1.0),
    };
    let mass = Constant(1.5);
    let exact = dense_lambda_max(&space, &k, &space.local_mass(&mass));
    let est = space.c_max_eigenvalue::<2, _, _>(&k, &mass, 20_000, 1e-14);
    assert!(
        (est.lambda - exact).abs() < 1e-8 * exact,
        "{} vs {exact}",
        est.lambda
    );

    assert!((stable_dt(4.0, 0.95) - 0.95).abs() < 1e-15);
}

#[test]
fn vtk_writer_counts() {
    let dir = std::env::temp_dir().join(format!("lobattox-vtk-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let space = Space::<2, Iso2<3>>::new(&square(2, 1, false, None), Numbering::Continuous);
    let u = space.sample::<1>(|x| [x[0]]);
    let v = space.sample::<2>(|x| [x[0], x[1]]);
    let w = space.sample::<3>(|_| [1.0, 2.0, 3.0]);
    let path = dir.join("square.vtk");
    VtkWriter::new(&space)
        .write(
            &path,
            &[
                VtkField {
                    name: "p",
                    components: 1,
                    data: &u,
                },
                VtkField {
                    name: "u",
                    components: 2,
                    data: &v,
                },
                VtkField {
                    name: "w",
                    components: 3,
                    data: &w,
                },
            ],
        )
        .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    // 2 elements × 9 points, 2 × 4 sub-quads.
    assert!(text.contains("POINTS 18 double"));
    assert!(text.contains("CELLS 8 40"));
    assert!(text.contains("CELL_TYPES 8"));
    assert!(text.contains("SCALARS p double 1"));
    assert!(text.contains("VECTORS u double"));
    assert!(text.contains("SCALARS w_2 double 1"));
    assert_eq!(text.lines().filter(|l| *l == "9").count(), 8);

    let space = Space::<3, Iso3<3>>::new(&cube([1, 1, 1], false, None), Numbering::Continuous);
    let path = dir.join("cube.vtk");
    let u = space.sample::<3>(|x| x);
    VtkWriter::new(&space)
        .write(
            &path,
            &[VtkField {
                name: "u",
                components: 3,
                data: &u,
            }],
        )
        .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("POINTS 27 double"));
    assert!(text.contains("CELLS 8 72"));
    assert!(text.contains("8 0 1 4 3 9 10 13 12"));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The article's operator `M⁻¹(R + λ Bᵀ (M^P)⁻¹ B)` (displacement Q3, pressure
/// Q1 with a 2-point Gauss–Legendre mass, `b_h` on the displacement GLL
/// rule, λ = 1e3) through the closure power iteration, against a dense solve.
#[test]
fn power_iteration_on_a_composite_operator() {
    use lobattox::kernels::{Divergence, DivergenceTranspose};
    use lobattox::{mlt_add_mixed, power_iteration, GeoStorage, Gl, Gll, Quad2};
    type DispRule = Quad2<Gll<4>, Gll<4>>;
    type PresMass = Quad2<Gl<2>, Gl<2>>;
    let mesh = square(2, 2, true, Some(bending_2d()));
    let su = Space::<2, Iso2<4>>::builder(&mesh)
        .numbering(Numbering::Continuous)
        .quadrature::<DispRule>(GeoStorage::Store)
        .build();
    let sp = Space::<2, Iso2<2>>::builder(&mesh)
        .quadrature::<PresMass>(GeoStorage::Store)
        .build();
    let lambda = 1e3;
    let r = IsotropicElastic {
        lambda: Constant(0.0),
        mu: Constant(1.0),
    };
    let rho = Constant(1.3);
    let inv_mp = sp.inverse_mass_with::<PresMass, _>(&Constant(1.0));
    let div = Divergence {
        coef: Constant(1.0),
    };
    let div_t = DivergenceTranspose {
        coef: Constant(1.0),
    };
    // Element-local A x = R x + λ Bᵀ (M^P)⁻¹ B x.
    let local = |x: &[f64], y: &mut [f64]| {
        su.mlt_add::<2, _>(&r, 1.0, x, 0.0, y);
        let mut p = vec![0.0; sp.n_dup()];
        mlt_add_mixed::<DispRule, 2, 1, 2, _, _, _>(&su, &sp, &div, 1.0, x, 0.0, &mut p);
        inv_mp.apply::<1>(&mut p);
        mlt_add_mixed::<DispRule, 1, 2, 2, _, _, _>(&sp, &su, &div_t, lambda, &p, 1.0, y);
    };
    let m = su.local_mass(&rho);
    let inv_m = su.c_inverse_mass(&rho);
    let exact = dense_lambda_max_of::<2, _, 2>(&su, local, &m);
    let est = power_iteration(
        su.c_start_vector::<2>(),
        |x, y| {
            local(x, y);
            su.c_sum_shared::<2>(y);
            inv_m.apply::<2>(y);
        },
        |a, b| su.mass_inner::<2>(&m, a, b),
        50_000,
        1e-14,
    );
    assert!(
        est.lambda <= exact * (1.0 + 1e-12),
        "{} > {exact}",
        est.lambda
    );
    assert!(
        (est.lambda - exact).abs() < 1e-8 * exact,
        "{} vs {exact} ({} iterations)",
        est.lambda,
        est.iterations
    );
    // λ dominates: the spectrum is far above the elastic one.
    let elastic = su
        .c_max_eigenvalue::<2, _, _>(&r, &rho, 50_000, 1e-12)
        .lambda;
    assert!(exact > 10.0 * elastic, "{exact} vs elastic {elastic}");
}
