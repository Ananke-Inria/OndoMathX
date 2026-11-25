mod common;

use ondomathx_mesh::mesh::UnstructuredMesh;
use common::*;
use lobatto::tables::derivative_matrix_dyn;
use ondomathx_sem::fields::Constant;
use ondomathx_sem::{
    cofactor, det, GeoKind, Gll2, Gll3, Iso2, Iso3, Numbering, PointIn, PointKernel, PointOut,
    Shape, Space,
};
use nalgebra::{SMatrix, SVector};

// ── Kernels ────────────────────────────────────────────────────────────

/// `a(u, v) = ∫ ∇u·∇v`.
struct Laplace;
impl<const D: usize> PointKernel<D, 1> for Laplace {
    const USES_GRAD: bool = true;
    const OUTPUTS_FLUX: bool = true;
    fn apply(&self, i: &PointIn<D, 1>, o: &mut PointOut<D, 1>) {
        o.flux = i.grad;
    }
}

/// `a(u, v) = ∫ ρ u v`: no flux, so no derivative kernel runs.
struct Mass(f64);
impl<const D: usize> PointKernel<D, 1> for Mass {
    const USES_VALUE: bool = true;
    const OUTPUTS_SOURCE: bool = true;
    fn apply(&self, i: &PointIn<D, 1>, o: &mut PointOut<D, 1>) {
        o.source = i.value * self.0;
    }
}

/// Uses every input: `a(u, v) = ∫ k(x) ∇u·∇v + (r[label] + q[dof]) u v`,
/// with `k(x) = 1 + 0.5·x_0·x_{D−1}`, plus a non-symmetric advection-like
/// flux term `0.3·u·e_0`.
struct Everything {
    r: Vec<f64>,
    q: Vec<f64>,
}
impl<const D: usize> PointKernel<D, 1> for Everything {
    const USES_VALUE: bool = true;
    const USES_GRAD: bool = true;
    const USES_XYZ: bool = true;
    const USES_LABEL: bool = true;
    const USES_DOF: bool = true;
    const OUTPUTS_FLUX: bool = true;
    const OUTPUTS_SOURCE: bool = true;
    fn apply(&self, i: &PointIn<D, 1>, o: &mut PointOut<D, 1>) {
        let k = 1.0 + 0.5 * i.xyz[0] * i.xyz[D - 1];
        o.flux = i.grad * k;
        o.flux[(0, 0)] += 0.3 * i.value[0];
        o.source[0] = (self.r[i.label % self.r.len()] + self.q[i.dof]) * i.value[0];
    }
}

/// Isotropic linear elasticity, `σ = λ tr(ε) I + 2μ ε`.
struct Elastic {
    lambda: f64,
    mu: f64,
}
impl<const D: usize> PointKernel<D, D> for Elastic {
    const USES_GRAD: bool = true;
    const OUTPUTS_FLUX: bool = true;
    fn apply(&self, i: &PointIn<D, D>, o: &mut PointOut<D, D>) {
        let eps = (i.grad + i.grad.transpose()) * 0.5;
        o.flux =
            eps * (2.0 * self.mu) + SMatrix::<f64, D, D>::identity() * (self.lambda * eps.trace());
    }
}

// ── Reference: element matrices by brute force ─────────────────────────

/// `y = A u` element by element, without batching or sum factorization: at
/// each point `q`, `∇ᵤu(q)` from the full 1D derivative sums (per-direction
/// GLL differentiation matrices), `J` from `Space::map_and_jacobian`, and the
/// test functions' physical gradients `∇φ_i(q) = J⁻ᵀ ∇ᵤφ_i(q)` applied one by
/// one.
fn reference<const D: usize, S: Shape<D>, const C: usize, K: PointKernel<D, C>>(
    space: &Space<D, S>,
    kernel: &K,
    u: &[f64],
) -> Vec<f64> {
    let nloc = S::NLOC;
    let d1: Vec<Vec<Vec<f64>>> = (0..D)
        .map(|b| derivative_matrix_dyn(space.points_1d(b), space.points_1d(b)))
        .collect();
    let id = Space::<D, S>::multi_index;
    // ∂φ_i/∂ξ_b at node q.
    let dphi = |i: usize, q: usize, b: usize| -> f64 {
        let (ii, qq) = (id(i), id(q));
        if (0..D).any(|d| d != b && ii[d] != qq[d]) {
            0.0
        } else {
            d1[b][qq[b]][ii[b]]
        }
    };
    let mut y = vec![0.0; u.len()];
    for e in 0..space.n_elem() {
        let ue = &u[e * C * nloc..(e + 1) * C * nloc];
        let ye = &mut y[e * C * nloc..(e + 1) * C * nloc];
        for q in 0..nloc {
            let (x, j) = space.map_and_jacobian(e, q);
            let (dj, cof) = (det(&j), cofactor(&j));
            let jinv_t = cof / dj; // J⁻ᵀ
            let mut gu = SMatrix::<f64, C, D>::zeros();
            for c in 0..C {
                for b in 0..D {
                    gu[(c, b)] = (0..nloc).map(|i| ue[c * nloc + i] * dphi(i, q, b)).sum();
                }
            }
            let input = PointIn::<D, C> {
                value: SVector::from_fn(|c, _| ue[c * nloc + q]),
                grad: gu * jinv_t.transpose(),
                xyz: x,
                label: space.label(e),
                dof: e * nloc + q,
            };
            let mut out = PointOut::<D, C> {
                flux: SMatrix::zeros(),
                source: SVector::zeros(),
            };
            kernel.apply(&input, &mut out);
            let wdet = space.weight(q) * dj.abs();
            for c in 0..C {
                ye[c * nloc + q] += wdet * out.source[c];
                for i in 0..nloc {
                    let gphi_u = SVector::<f64, D>::from_fn(|b, _| dphi(i, q, b));
                    let gphi_x = jinv_t * gphi_u;
                    let g: f64 = (0..D).map(|a| out.flux[(c, a)] * gphi_x[a]).sum();
                    ye[c * nloc + i] += wdet * g;
                }
            }
        }
    }
    y
}

// ── Helpers ────────────────────────────────────────────────────────────

fn seeded(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f64::from_bits((s >> 12) | 0x3FF0000000000000) * 2.0 - 3.0
        })
        .collect()
}

fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * scale,
            "{what}: entry {i}: {x:e} vs {y:e} (scale {scale:e})"
        );
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn apply<const D: usize, S: Shape<D>, const C: usize, K: PointKernel<D, C>>(
    space: &Space<D, S>,
    k: &K,
    u: &[f64],
) -> Vec<f64> {
    let mut v = vec![f64::NAN; u.len()];
    space.mlt_add(k, 1.0, u, 0.0, &mut v);
    v
}

fn everything<const D: usize, S: Shape<D>>(space: &Space<D, S>) -> Everything {
    Everything {
        r: vec![0.5, 1.5, 2.5],
        q: seeded(space.n_dup(), 7).iter().map(|x| x.abs()).collect(),
    }
}

/// Anisotropic shapes have no continuous numbering yet; the engine does not
/// use it anyway.
fn check_vs_reference<const D: usize, S: Shape<D>>(mesh: &UnstructuredMesh, what: &str) {
    let numbering = if S::ISOTROPIC {
        Numbering::Continuous
    } else {
        Numbering::None
    };
    let space = Space::<D, S>::new(mesh, numbering);
    let n = &S::N[..D];
    let k = everything(&space);
    let u = seeded(space.n_dup(), 1);
    assert_close(
        &apply(&space, &k, &u),
        &reference(&space, &k, &u),
        1e-12,
        &format!("{what} Everything N={n:?}"),
    );
    let el = Elastic {
        lambda: 2.0,
        mu: 0.7,
    };
    let u = seeded(space.n_dup() * D, 2);
    let mut v = vec![f64::NAN; u.len()];
    space.mlt_add::<D>(&el, 1.0, &u, 0.0, &mut v);
    assert_close(
        &v,
        &reference::<D, S, D, _>(&space, &el, &u),
        1e-12,
        &format!("{what} Elastic N={n:?}"),
    );
}

// ── Engine vs reference ────────────────────────────────────────────────

/// Two groups per mesh (affine + Q1, affine + Q2); 5×3 elements, so groups
/// are not multiples of the batch width and the last batches are padded.
fn check_2d<const NP: usize>() {
    for q2 in [false, true] {
        let mesh = square(5, 3, q2, Some(partial_2d()));
        let space = Space::<2, Iso2<NP>>::new(&mesh, Numbering::Continuous);
        let kinds = GeoKind::ALL.map(|k| space.elements(k).len());
        assert_eq!(
            kinds,
            if q2 { [6, 0, 9, 0] } else { [6, 9, 0, 0] },
            "q2={q2}"
        );
        check_vs_reference::<2, Iso2<NP>>(&mesh, &format!("2D partial q2={q2}"));
    }
    check_vs_reference::<2, Iso2<NP>>(&square(4, 3, true, Some(bending_2d())), "2D bending Q2");
}

/// Anisotropic shapes, on the same meshes (affine + Q1, affine + Q2, Q2 only).
fn check_2d_aniso<S: Shape<2>>() {
    for q2 in [false, true] {
        check_vs_reference::<2, S>(
            &square(5, 3, q2, Some(partial_2d())),
            &format!("2D partial q2={q2}"),
        );
    }
    check_vs_reference::<2, S>(&square(4, 3, true, Some(bending_2d())), "2D bending Q2");
}

#[test]
fn engine_2d_anisotropic() {
    check_2d_aniso::<Gll2<3, 5>>();
    check_2d_aniso::<Gll2<2, 4>>();
    check_2d_aniso::<Gll2<6, 4>>();
}

#[test]
fn engine_2d_np2() {
    check_2d::<2>();
}
#[test]
fn engine_2d_np3() {
    check_2d::<3>();
}
#[test]
fn engine_2d_np4_hc() {
    check_2d::<4>();
}
#[test]
fn engine_2d_np5_hc() {
    check_2d::<5>();
}
#[test]
fn engine_2d_np7_hc() {
    check_2d::<7>();
}
#[test]
fn engine_2d_np9() {
    check_2d::<9>();
}

fn check_3d<const NP: usize>() {
    for q2 in [false, true] {
        let mesh = cube([2, 3, 2], q2, Some(partial_3d()));
        let space = Space::<3, Iso3<NP>>::new(&mesh, Numbering::Continuous);
        assert!(!space.elements(GeoKind::Affine).is_empty(), "q2={q2}");
        assert!(
            space.elements(GeoKind::Affine).len() < space.n_elem(),
            "q2={q2}"
        );
        check_vs_reference::<3, Iso3<NP>>(&mesh, &format!("3D partial q2={q2}"));
    }
}

#[test]
fn engine_3d_anisotropic() {
    for q2 in [false, true] {
        let mesh = cube([2, 3, 2], q2, Some(partial_3d()));
        check_vs_reference::<3, Gll3<2, 4, 3>>(&mesh, &format!("3D partial q2={q2}"));
        check_vs_reference::<3, Gll3<5, 3, 4>>(&mesh, &format!("3D partial q2={q2}"));
        check_vs_reference::<3, Gll3<4, 4, 6>>(&mesh, &format!("3D partial q2={q2}"));
    }
}

#[test]
fn engine_3d_np2() {
    check_3d::<2>();
}
#[test]
fn engine_3d_np3() {
    check_3d::<3>();
}
#[test]
fn engine_3d_np5_hc() {
    check_3d::<5>();
}
#[test]
fn engine_3d_np8() {
    check_3d::<8>();
}

// ── Engine properties ──────────────────────────────────────────────────

#[test]
fn batch_width_does_not_change_the_result() {
    let mesh = square(5, 3, true, Some(partial_2d()));
    let space = Space::<2, Iso2<5>>::new(&mesh, Numbering::Continuous);
    let k = everything(&space);
    let u = seeded(space.n_dup(), 3);
    let run = |w: usize| {
        let mut v = vec![0.0; u.len()];
        match w {
            1 => space.mlt_add_batched::<1, 1>(&k, 1.0, &u, 0.0, &mut v),
            3 => space.mlt_add_batched::<1, 3>(&k, 1.0, &u, 0.0, &mut v),
            4 => space.mlt_add_batched::<1, 4>(&k, 1.0, &u, 0.0, &mut v),
            _ => space.mlt_add_batched::<1, 8>(&k, 1.0, &u, 0.0, &mut v),
        }
        v
    };
    let v4 = run(4);
    for w in [1, 3, 8] {
        assert_eq!(run(w), v4, "W={w} differs from W=4");
    }
}

#[test]
fn alpha_beta_semantics() {
    let mesh = square(3, 2, false, Some(bending_2d()));
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let u = seeded(space.n_dup(), 4);
    let v0 = seeded(space.n_dup(), 5);
    let au = apply(&space, &Laplace, &u);
    let mut v = v0.clone();
    space.mlt_add(&Laplace, 2.0, &u, 0.5, &mut v);
    let expected: Vec<f64> = v0.iter().zip(&au).map(|(a, b)| 0.5 * a + 2.0 * b).collect();
    assert_close(&v, &expected, 1e-14, "β·v + α·A·u");
    // β = 0 never reads v: NaN input must not leak.
    let mut v = vec![f64::NAN; space.n_dup()];
    space.mlt_add(&Laplace, 1.0, &u, 0.0, &mut v);
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn laplace_is_symmetric_and_kills_constants() {
    for mesh in [
        square(4, 3, true, Some(bending_2d())),
        square(5, 3, false, Some(partial_2d())),
    ] {
        let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
        let (u, w) = (seeded(space.n_dup(), 8), seeded(space.n_dup(), 9));
        let (au, aw) = (apply(&space, &Laplace, &u), apply(&space, &Laplace, &w));
        let (a, b) = (dot(&w, &au), dot(&u, &aw));
        assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0), "{a} vs {b}");
        let a1 = apply(&space, &Laplace, &vec![1.0; space.n_dup()]);
        assert!(a1.iter().all(|x| x.abs() < 1e-11), "A·1 ≠ 0");
    }
    let mesh = cube([2, 2, 2], true, Some(bending_3d()));
    let space = Space::<3, Iso3<3>>::new(&mesh, Numbering::Continuous);
    let a1 = apply(&space, &Laplace, &vec![1.0; space.n_dup()]);
    assert!(a1.iter().all(|x| x.abs() < 1e-11), "3D A·1 ≠ 0");
}

#[test]
fn laplace_energy_of_linear_fields() {
    // Affine meshes: |∇u|² is constant, GLL integrates it exactly.
    let space = Space::<2, Iso2<3>>::new(&square(3, 2, false, None), Numbering::Continuous);
    let u = space.sample::<1>(|x| [x[0] + 2.0 * x[1]]);
    let e = dot(&u, &apply(&space, &Laplace, &u));
    assert!((e - 2.0 * 5.0).abs() < 1e-12, "2D energy {e}");
    let space = Space::<3, Iso3<2>>::new(&cube([2, 2, 1], false, None), Numbering::Continuous);
    let u = space.sample::<1>(|x| [x[0] - x[1] + 3.0 * x[2]]);
    let e = dot(&u, &apply(&space, &Laplace, &u));
    assert!((e - 3.0 * 11.0).abs() < 1e-12, "3D energy {e}");
}

#[test]
fn mass_operator_matches_inverse_mass() {
    let mesh = square(4, 3, true, Some(bending_2d()));
    let space = Space::<2, Iso2<5>>::new(&mesh, Numbering::Continuous);
    let mut m = apply(&space, &Mass(1.0), &vec![1.0; space.n_dup()]);
    space.c_sum_shared::<1>(&mut m);
    let inv = space
        .c_inverse_mass(&Constant(1.0))
        .diagonal()
        .unwrap()
        .to_vec();
    let expected: Vec<f64> = inv.iter().map(|x| 1.0 / x).collect();
    assert_close(&m, &expected, 1e-14, "Σ_shared(M·1) vs 1/inverse_mass");
}

#[test]
fn elastic_rigid_modes_and_energy() {
    // Q1 geometry: rigid fields are interpolated exactly for every P.
    let el = Elastic {
        lambda: 2.0,
        mu: 0.7,
    };
    fn rigid_2d<S: Shape<2>>(space: &Space<2, S>, el: &Elastic) {
        for f in [
            |_x: [f64; 2]| [1.0, 0.0],
            |_x: [f64; 2]| [0.0, 1.0],
            |x: [f64; 2]| [-x[1], x[0]],
        ] {
            let u = space.sample::<2>(f);
            let mut v = vec![0.0; u.len()];
            space.mlt_add::<2>(el, 1.0, &u, 0.0, &mut v);
            assert!(
                v.iter().all(|x| x.abs() < 1e-11),
                "N={:?}: rigid mode not in kernel",
                S::N
            );
        }
    }
    let mesh = square(4, 3, false, Some(bending_2d()));
    rigid_2d(&Space::<2, Iso2<2>>::new(&mesh, Numbering::Continuous), &el);
    rigid_2d(&Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous), &el);
    rigid_2d(&Space::<2, Iso2<6>>::new(&mesh, Numbering::Continuous), &el);

    let space = Space::<3, Iso3<3>>::new(
        &cube([2, 2, 2], false, Some(bending_3d())),
        Numbering::Continuous,
    );
    for f in [
        |_x: [f64; 3]| [0.0, 0.0, 1.0],
        |x: [f64; 3]| [-x[1], x[0], 0.0],
        |x: [f64; 3]| [x[2], 0.0, -x[0]],
    ] {
        let u = space.sample::<3>(f);
        let mut v = vec![0.0; u.len()];
        space.mlt_add::<3>(&el, 1.0, &u, 0.0, &mut v);
        assert!(
            v.iter().all(|x| x.abs() < 1e-11),
            "3D rigid mode not in kernel"
        );
    }

    // u = E x with E symmetric on an affine square: energy = area·(λ tr(E)² + 2μ E:E).
    let space = Space::<2, Iso2<3>>::new(&square(3, 2, false, None), Numbering::Continuous);
    let u = space.sample::<2>(|x| [0.3 * x[0] + 0.2 * x[1], 0.2 * x[0] - 0.1 * x[1]]);
    let mut v = vec![0.0; u.len()];
    space.mlt_add::<2>(&el, 1.0, &u, 0.0, &mut v);
    let exact = 2.0 * (2.0 * 0.2f64.powi(2) + 2.0 * 0.7 * (0.09 + 0.01 + 2.0 * 0.04));
    let e = dot(&u, &v);
    assert!((e - exact).abs() < 1e-12, "elastic energy {e} vs {exact}");
}
