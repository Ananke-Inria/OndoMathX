//! Mixed products (`mlt_add_mixed`), the divergence pair, and the inverse
//! mass of a quadrature (`Space::inverse_mass_with`).

mod common;

use anamesh::mesh::square::Square;
use anamesh::mesh::Mesh;
use common::*;
use lobatto::collocation::Gauss;
use lobatto::tables::{derivative_matrix_dyn, interpolation_row, points_weights_dyn};
use lobatto::utilities::barycentric_weights;
use lobattox::fields::Constant;
use lobattox::kernels::{
    AcousticStiffness, Divergence, DivergenceTranspose, IsotropicElastic, Mass,
};
use lobattox::{
    cofactor, det, GeoStorage, Gl, Gll, Gll2, Iso2, Iso3, Numbering, PointIn, PointKernel,
    PointOut, Quad2, Quad3, Quadrature, Shape, Space,
};
use nalgebra::{SMatrix, SVector};

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
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / scale
}

/// A mixed kernel using everything: `CU = 2` → `CV = 1`.
struct Everything;
impl<const D: usize> PointKernel<D, 2, 1> for Everything {
    const USES_VALUE: bool = true;
    const USES_GRAD: bool = true;
    const USES_XYZ: bool = true;
    const USES_LABEL: bool = true;
    const OUTPUTS_FLUX: bool = true;
    const OUTPUTS_SOURCE: bool = true;
    fn apply(&self, i: &PointIn<D, 2>, o: &mut PointOut<D, 1>) {
        o.source[0] = i.value[0] * i.xyz[0] + 0.3 * i.grad[(1, 0)] * (1 + i.label) as f64;
        o.flux[(0, 0)] = i.value[1] + i.xyz[D - 1];
        o.flux[(0, D - 1)] += i.grad[(0, D - 1)] - 0.5 * i.grad[(1, 0)];
    }
}

/// 1D basis values and derivatives at the points, `[point][node]`.
type Basis1d = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// 1D basis tables of `n` GLL nodes at the points `y`: values and derivatives.
fn basis_1d(n: usize, y: &[f64]) -> Basis1d {
    let (x, _) = points_weights_dyn(n, Gauss::Lobatto);
    let bw = barycentric_weights(&x);
    let v = y
        .iter()
        .map(|&yj| {
            let mut r = vec![0.0; n];
            interpolation_row(&x, &bw, yj, &mut r);
            r
        })
        .collect();
    (v, derivative_matrix_dyn(&x, y))
}

/// Brute force: tensor basis values / reference gradients from lobatto,
/// geometry from the mesh's own map at the quadrature points.
fn reference<
    const D: usize,
    SU: Shape<D>,
    SV: Shape<D>,
    Q: Quadrature<D>,
    const CU: usize,
    const CV: usize,
>(
    mesh: &impl Mesh,
    space: &Space<D, SU>,
    kernel: &impl PointKernel<D, CU, CV>,
    u: &[f64],
) -> Vec<f64> {
    let fam = Q::families();
    let pw: Vec<(Vec<f64>, Vec<f64>)> = (0..D)
        .map(|d| points_weights_dyn(Q::M[d], fam[d]))
        .collect();
    let bu: Vec<_> = (0..D).map(|d| basis_1d(SU::N[d], &pw[d].0)).collect();
    let bv: Vec<_> = (0..D).map(|d| basis_1d(SV::N[d], &pw[d].0)).collect();
    let idx = |k: usize, n: &[usize]| -> [usize; D] {
        let mut k = k;
        core::array::from_fn(|d| {
            let r = k % n[d];
            k /= n[d];
            r
        })
    };
    // φ_i(ξ_q) and ∇ξ φ_i(ξ_q) for a shape with counts `n`, tables `b`.
    let phi = |b: &Vec<Basis1d>, n: &[usize], i: usize, q: usize| -> (f64, [f64; D]) {
        let (ii, qq) = (idx(i, n), idx(q, &Q::M));
        let val: f64 = (0..D).map(|d| b[d].0[qq[d]][ii[d]]).product();
        let grad = core::array::from_fn(|g| {
            (0..D)
                .map(|d| {
                    if d == g {
                        b[d].1[qq[d]][ii[d]]
                    } else {
                        b[d].0[qq[d]][ii[d]]
                    }
                })
                .product()
        });
        (val, grad)
    };
    let (nu, nv) = (SU::NLOC, SV::NLOC);
    let mut y = vec![0.0; space.n_elem() * CV * nv];
    for e in 0..space.n_elem() {
        let m = space.mesh_element(e);
        for q in 0..Q::NQ {
            let qq = idx(q, &Q::M);
            let mut uvw = [0.0; 3];
            let mut wq = 1.0;
            for d in 0..D {
                uvw[d] = pw[d].0[qq[d]];
                wq *= pw[d].1[qq[d]];
            }
            let (mut x, mut g) = ([0.0; 3], [0.0; 9]);
            mesh.get_element_def(m, &uvw, &mut x);
            mesh.get_element_grad_def(m, &uvw, &mut g);
            let j = SMatrix::<f64, D, D>::from_fn(|a, b| g[a * D + b]);
            let (dj, cof) = (det(&j), cofactor(&j));
            let jinv_t = cof / dj;
            let mut input = PointIn::<D, CU> {
                value: SVector::zeros(),
                grad: SMatrix::zeros(),
                xyz: SVector::from_fn(|a, _| x[a]),
                label: space.label(e),
                dof: e * Q::NQ + q,
            };
            let mut gu = SMatrix::<f64, CU, D>::zeros();
            for i in 0..nu {
                let (pv, pg) = phi(&bu, &SU::N, i, q);
                for c in 0..CU {
                    let ui = u[(e * CU + c) * nu + i];
                    input.value[c] += ui * pv;
                    for d in 0..D {
                        gu[(c, d)] += ui * pg[d];
                    }
                }
            }
            input.grad = gu * jinv_t.transpose();
            let mut out = PointOut::<D, CV> {
                flux: SMatrix::zeros(),
                source: SVector::zeros(),
            };
            kernel.apply(&input, &mut out);
            let ws = wq * dj.abs();
            for i in 0..nv {
                let (pv, pg) = phi(&bv, &SV::N, i, q);
                let gx = jinv_t * SVector::<f64, D>::from(pg);
                for c in 0..CV {
                    let g: f64 = (0..D).map(|a| out.flux[(c, a)] * gx[a]).sum();
                    y[(e * CV + c) * nv + i] += ws * (out.source[c] * pv + g);
                }
            }
        }
    }
    y
}

fn check_vs_reference<const D: usize, SU: Shape<D>, SV: Shape<D>, Q>(mesh: &impl Mesh, what: &str)
where
    Q: lobattox::Sweep<D, SU> + lobattox::Sweep<D, SV>,
{
    let su = Space::<D, SU>::builder(mesh)
        .quadrature::<Q>(GeoStorage::Recompute)
        .build();
    let sv = Space::<D, SV>::new(mesh, Numbering::None);
    let u = seeded(su.n_dup() * 2, 5);
    let mut v = vec![f64::NAN; sv.n_dup()];
    su.mlt_add_mixed::<Q, 2, 1, SV>(&sv, &Everything, 1.0, &u, 0.0, &mut v);
    let r = reference::<D, SU, SV, Q, 2, 1>(mesh, &su, &Everything, &u);
    let err = max_rel(&v, &r);
    assert!(err < 1e-12, "{what}: max rel diff {err:e}");
}

#[test]
fn mixed_matches_reference_2d() {
    type Q = Quad2<Gl<5>, Gll<4>>;
    for q2 in [false, true] {
        let mesh = square(5, 3, q2, Some(partial_2d()));
        check_vs_reference::<2, Iso2<4>, Gll2<3, 2>, Q>(&mesh, &format!("partial q2={q2}"));
        check_vs_reference::<2, Iso2<3>, Iso2<3>, Q>(&mesh, &format!("same shape q2={q2}"));
    }
    // Mapped elements (exact map of an attached deformation).
    let sq = Square::new([0.0, 0.0], 2.0, 1.0, 3, 2, Some(bending_2d()), None);
    check_vs_reference::<2, Iso2<4>, Gll2<3, 2>, Q>(&sq, "mapped");
}

#[test]
fn mixed_matches_reference_3d() {
    type Q = Quad3<Gl<3>, Gl<4>, Gll<3>>;
    for q2 in [false, true] {
        let mesh = cube([2, 1, 2], q2, Some(partial_3d()));
        check_vs_reference::<3, Iso3<3>, Iso3<2>, Q>(&mesh, &format!("3D partial q2={q2}"));
    }
}

/// U = V and Q = the nodes: the mixed product is `mlt_add`.
#[test]
fn same_space_on_its_nodes_is_mlt_add() {
    type Nodes = Quad2<Gll<5>, Gll<5>>;
    let mesh = square(4, 3, true, Some(bending_2d()));
    let s = Space::<2, Iso2<5>>::builder(&mesh)
        .numbering(Numbering::Continuous)
        .quadrature::<Nodes>(GeoStorage::Store)
        .build();
    let ka = AcousticStiffness { rho: Constant(2.0) };
    let u = seeded(s.n_dup(), 1);
    let (mut a, mut b) = (vec![0.0; u.len()], vec![0.0; u.len()]);
    s.mlt_add::<1>(&ka, 1.0, &u, 0.0, &mut a);
    s.mlt_add_mixed::<Nodes, 1, 1, _>(&s, &ka, 1.0, &u, 0.0, &mut b);
    assert!(max_rel(&b, &a) < 1e-13);
    let ke = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(0.7),
    };
    let u = seeded(s.n_dup() * 2, 2);
    let (mut a, mut b) = (vec![0.0; u.len()], vec![0.0; u.len()]);
    s.mlt_add::<2>(&ke, 1.0, &u, 0.0, &mut a);
    s.mlt_add_mixed::<Nodes, 2, 2, _>(&s, &ke, 1.0, &u, 0.0, &mut b);
    assert!(max_rel(&b, &a) < 1e-13);
}

type DispRule = Quad2<Gll<5>, Gll<5>>;

fn disp_pres(mesh: &impl Mesh) -> (Space<2, Iso2<5>>, Space<2, Iso2<3>>) {
    (
        Space::builder(mesh)
            .numbering(Numbering::Continuous)
            .quadrature::<DispRule>(GeoStorage::Store)
            .build(),
        Space::new(mesh, Numbering::None),
    )
}

/// `⟨B u, p⟩ = ⟨u, Bᵀ p⟩` (element-local operators), on curved elements.
#[test]
fn divergence_pair_is_adjoint() {
    let mesh = square(3, 2, true, Some(bending_2d()));
    let (su, sp) = disp_pres(&mesh);
    let u = seeded(su.n_dup() * 2, 3);
    let p = seeded(sp.n_dup(), 4);
    let (mut bu, mut btp) = (vec![0.0; p.len()], vec![0.0; u.len()]);
    let c = Constant(1.7);
    su.mlt_add_mixed::<DispRule, 2, 1, _>(&sp, &Divergence { coef: c }, 1.0, &u, 0.0, &mut bu);
    sp.mlt_add_mixed::<DispRule, 1, 2, _>(
        &su,
        &DivergenceTranspose { coef: c },
        1.0,
        &p,
        0.0,
        &mut btp,
    );
    let (a, b) = (dot(&bu, &p), dot(&u, &btp));
    assert!((a - b).abs() < 1e-12 * a.abs().max(1.0), "{a} vs {b}");
}

/// `Σ_i p_i (B u)_i = ∫ p div u`, exact for polynomials on an affine mesh:
/// `u = (x², x y)`, `div u = 3x`, `p = 1 + y` on `[0, 2] × [0, 1]`: 9.
#[test]
fn divergence_is_exact_on_polynomials() {
    let mesh = square(4, 2, false, None);
    let (su, sp) = disp_pres(&mesh);
    let u = su.sample::<2>(|x| [x[0] * x[0], x[0] * x[1]]);
    let p = sp.sample::<1>(|x| [1.0 + x[1]]);
    let mut bu = vec![0.0; p.len()];
    su.mlt_add_mixed::<DispRule, 2, 1, _>(
        &sp,
        &Divergence {
            coef: Constant(1.0),
        },
        1.0,
        &u,
        0.0,
        &mut bu,
    );
    let i = dot(&bu, &p);
    assert!((i - 9.0).abs() < 1e-12, "∫ p div u = {i}");
    // Divergence-free field: rotation.
    let u = su.sample::<2>(|x| [-x[1], x[0]]);
    su.mlt_add_mixed::<DispRule, 2, 1, _>(
        &sp,
        &Divergence {
            coef: Constant(1.0),
        },
        1.0,
        &u,
        0.0,
        &mut bu,
    );
    assert!(bu.iter().all(|v| v.abs() < 1e-13), "div of a rotation");
}

/// `M⁻¹ (M p) = p` with `M` the mass of a Gauss–Legendre rule (mixed `Mass`
/// kernel) and `M⁻¹` its tensor inverse, on Q1, Q2 and mapped elements.
#[test]
fn inverse_mass_with_inverts_the_quadrature_mass() {
    type G3 = Quad2<Gl<3>, Gl<3>>;
    let check = |mesh: &dyn Fn() -> Space<2, Iso2<3>>, what: &str| {
        let s = mesh();
        let c = Constant(2.5);
        let p = seeded(s.n_dup() * 2, 6);
        let mut mp = vec![0.0; p.len()];
        s.mlt_add_mixed::<G3, 2, 2, _>(&s, &Mass { coef: c }, 1.0, &p, 0.0, &mut mp);
        let im = s.inverse_mass_with::<G3>(&c);
        assert!(im.diagonal().is_none());
        im.apply::<2>(&mut mp);
        let err = max_rel(&mp, &p);
        assert!(err < 1e-12, "{what}: {err:e}");
    };
    for q2 in [false, true] {
        let mesh = square(3, 2, q2, Some(bending_2d()));
        check(
            &|| {
                Space::builder(&mesh)
                    .quadrature::<G3>(GeoStorage::Recompute)
                    .build()
            },
            &format!("q2={q2}"),
        );
    }
    let sq = Square::new([0.0, 0.0], 2.0, 1.0, 3, 2, Some(bending_2d()), None);
    check(
        &|| {
            Space::builder(&sq)
                .quadrature::<G3>(GeoStorage::Recompute)
                .build()
        },
        "mapped",
    );
}

#[test]
fn inverse_mass_with_3d() {
    type G = Quad3<Gl<3>, Gl<3>, Gl<3>>;
    let mesh = cube([2, 1, 2], true, Some(bending_3d()));
    let s = Space::<3, Iso3<3>>::builder(&mesh)
        .quadrature::<G>(GeoStorage::Store)
        .build();
    let p = seeded(s.n_dup(), 7);
    let mut mp = vec![0.0; p.len()];
    s.mlt_add_mixed::<G, 1, 1, _>(
        &s,
        &Mass {
            coef: Constant(1.0),
        },
        1.0,
        &p,
        0.0,
        &mut mp,
    );
    s.inverse_mass_with::<G>(&Constant(1.0)).apply::<1>(&mut mp);
    assert!(max_rel(&mp, &p) < 1e-12);
}

/// On the space's own GLL rule the tensor inverse is the diagonal one.
#[test]
fn inverse_mass_with_gll_rule_is_diagonal() {
    type Nodes = Quad2<Gll<4>, Gll<4>>;
    let mesh = square(3, 2, true, Some(bending_2d()));
    let s = Space::<2, Iso2<4>>::builder(&mesh)
        .quadrature::<Nodes>(GeoStorage::Store)
        .build();
    let c = Constant(0.8);
    let v = seeded(s.n_dup() * 3, 8);
    let (mut a, mut b) = (v.clone(), v);
    s.inverse_mass_with::<Nodes>(&c).apply::<3>(&mut a);
    s.inverse_mass(&c).apply::<3>(&mut b);
    assert!(max_rel(&a, &b) < 1e-14);
}

#[test]
#[should_panic(expected = "different meshes")]
fn different_meshes_panic() {
    let a = Space::<2, Iso2<3>>::builder(&square(2, 2, false, None))
        .quadrature::<DispRule>(GeoStorage::Store)
        .build();
    let b = Space::<2, Iso2<3>>::new(&square(3, 2, false, None), Numbering::None);
    let u = vec![0.0; a.n_dup()];
    let mut v = vec![0.0; b.n_dup()];
    a.mlt_add_mixed::<DispRule, 1, 1, _>(
        &b,
        &Mass {
            coef: Constant(1.0),
        },
        1.0,
        &u,
        0.0,
        &mut v,
    );
}

#[test]
#[should_panic(expected = "declared on neither space")]
fn undeclared_mixed_quadrature_panics() {
    let mesh = square(2, 2, false, None);
    let a = Space::<2, Iso2<3>>::new(&mesh, Numbering::None);
    let u = vec![0.0; a.n_dup()];
    let mut v = u.clone();
    a.mlt_add_mixed::<DispRule, 1, 1, _>(
        &a,
        &Mass {
            coef: Constant(1.0),
        },
        1.0,
        &u,
        0.0,
        &mut v,
    );
}

#[test]
#[should_panic(expected = "must have the space's counts")]
fn inverse_mass_with_wrong_counts_panics() {
    type G = Quad2<Gl<4>, Gl<3>>;
    let s = Space::<2, Iso2<3>>::builder(&square(2, 2, false, None))
        .quadrature::<G>(GeoStorage::Store)
        .build();
    let _ = s.inverse_mass_with::<G>(&Constant(1.0));
}
