//! Linear algebra on plain vectors (slices), shared by `ondomathx-tetrix`
//! and `ondomathx-sem`: dot products and norms, the power iteration for the
//! spectral radius of `M⁻¹K`, and a fused update for explicit schemes.
//!
//! Every sum is **reproducible**: the vector is cut into fixed chunks of
//! [`CHUNK`] entries, each summed in order, and the partial sums added in
//! order. The result depends neither on the number of threads nor on the
//! run (a plain parallel sum may differ in its last bits between runs).

use rayon::prelude::*;

/// Number of entries summed sequentially per parallel task in the
/// reductions of this module (fixed, so the summation order is fixed).
pub const CHUNK: usize = 4096;

/// Reproducible parallel sum of `f(i)` over `0..n`.
fn sum_by_chunks(n: usize, f: impl Fn(usize) -> f64 + Sync) -> f64 {
    let partial: Vec<f64> = (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|k| (k * CHUNK..((k + 1) * CHUNK).min(n)).map(&f).sum())
        .collect();
    partial.iter().sum()
}

/// Dot product `Σ a_i b_i` (parallel, reproducible).
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector lengths");
    sum_by_chunks(a.len(), |i| a[i] * b[i])
}

/// Mass-weighted L² norm `√(uᵀ M u)` for a diagonal mass `M` (one value per
/// entry of `u`).
pub fn mass_norm(u: &[f64], mass: &[f64]) -> f64 {
    assert_eq!(u.len(), mass.len(), "vector lengths");
    sum_by_chunks(u.len(), |i| mass[i] * u[i] * u[i]).sqrt()
}

/// Maximum norm `max |v_i|`.
pub fn infinite_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// A deterministic, non-trivial start vector of length `size` for
/// [`power_iteration`] (TetriX's initialisation).
pub fn start_vector(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| (i % 7 + 3 * (i % 3) + (i % 2) + i % 20) as f64)
        .collect()
}

/// Result of [`power_iteration`].
#[derive(Clone, Copy, Debug)]
pub struct EigenEstimate {
    /// Rayleigh quotient of the last iterate: a lower bound of `λ_max`.
    pub lambda: f64,
    /// Iterations performed.
    pub n_iter: usize,
    /// Relative change of `lambda` at the last iteration.
    pub residue: f64,
}

/// Largest eigenvalue of `M⁻¹K` (`K` symmetric, `M` symmetric positive
/// definite), i.e. the spectral radius that bounds the leapfrog step
/// `dt < 2 / √λ_max`, by power iteration.
///
/// - `solve(v)` applies `M⁻¹` in place (`v ← M⁻¹ v`);
/// - `mlt_add(u, v)` writes `v = K u` (`v` is overwritten);
/// - `inner(a, b)` is the `M` inner product `⟨a, b⟩_M`;
/// - `x0` is the start vector (e.g. [`start_vector`]); its length is the
///   size of the problem.
///
/// Each iterate is `M`-normalised and the estimate is the Rayleigh quotient
/// `⟨x, M⁻¹K x⟩_M = ⟨x, K x⟩`, which converges from below. Stops when its
/// relative change is below `tol`, or after `n_iter_max` iterations.
pub fn power_iteration(
    mut solve: impl FnMut(&mut [f64]),
    mut mlt_add: impl FnMut(&[f64], &mut [f64]),
    inner: impl Fn(&[f64], &[f64]) -> f64,
    mut x: Vec<f64>,
    tol: f64,
    n_iter_max: usize,
) -> EigenEstimate {
    let norm = inner(&x, &x).sqrt();
    assert!(norm > 0.0, "power iteration: zero start vector");
    x.par_iter_mut().for_each(|v| *v /= norm);
    let mut y = vec![0.0; x.len()];
    let mut est = EigenEstimate {
        lambda: 0.0,
        n_iter: 0,
        residue: f64::INFINITY,
    };
    for it in 1..=n_iter_max {
        mlt_add(&x, &mut y);
        solve(&mut y);
        let lambda = inner(&x, &y);
        est.residue = (lambda - est.lambda).abs() / lambda.abs();
        est.lambda = lambda;
        est.n_iter = it;
        if est.residue < tol {
            break;
        }
        let norm = inner(&y, &y).sqrt();
        y.par_iter_mut().for_each(|v| *v /= norm);
        std::mem::swap(&mut x, &mut y);
    }
    est
}

/// `out ← a·x + b·out + c·(diag ⊙ y) + d·z` in one pass, for `C`-component
/// vectors stored per element as `[elem][comp][node]` with `nloc` nodes per
/// element, `diag` one value per node shared by the components. The last
/// term is `z = Some((d, z))`, or absent with `None`; `b = 0` never reads
/// `out`.
///
/// With `diag` the inverse lumped mass it fuses `M⁻¹` into the update of an
/// explicit scheme: the leapfrog step
/// `uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹K uⁿ + dt² g(tⁿ) M⁻¹f`, written into `u_prev`
/// from `ku = K uⁿ` (assembled) and `minv_f = M⁻¹f`, is
/// `axpby_diag::<C>(nloc, 2.0, &u_now, -1.0, &mut u_prev, -dt * dt, inv_m, &ku,
/// Some((dt * dt * g, &minv_f)))`, the same floating-point operations as
/// `M⁻¹` applied to `ku` and then the update.
#[allow(clippy::too_many_arguments)]
pub fn axpby_diag<const C: usize>(
    nloc: usize,
    a: f64,
    x: &[f64],
    b: f64,
    out: &mut [f64],
    c: f64,
    diag: &[f64],
    y: &[f64],
    z: Option<(f64, &[f64])>,
) {
    let block = C * nloc;
    assert!(nloc > 0 && diag.len().is_multiple_of(nloc), "diagonal length");
    let n = diag.len() * C;
    assert_eq!(x.len(), n, "x length");
    assert_eq!(y.len(), n, "y length");
    assert_eq!(out.len(), n, "out length");
    if let Some((_, z)) = z {
        assert_eq!(z.len(), n, "z length");
    }
    out.par_chunks_mut(block).enumerate().for_each(|(e, ob)| {
        let dg = &diag[e * nloc..(e + 1) * nloc];
        for (k, o) in ob.chunks_exact_mut(nloc).enumerate() {
            let r = e * block + k * nloc..e * block + (k + 1) * nloc;
            let (xc, yc) = (&x[r.clone()], &y[r.clone()]);
            let it = o.iter_mut().zip(xc).zip(yc).zip(dg);
            match (b == 0.0, z) {
                (true, None) => it.for_each(|(((o, &xv), &yv), &dv)| *o = a * xv + c * (dv * yv)),
                (false, None) => {
                    it.for_each(|(((o, &xv), &yv), &dv)| *o = a * xv + b * *o + c * (dv * yv))
                }
                (true, Some((d, z))) => it.zip(&z[r]).for_each(|((((o, &xv), &yv), &dv), &zv)| {
                    *o = a * xv + c * (dv * yv) + d * zv
                }),
                (false, Some((d, z))) => it.zip(&z[r]).for_each(|((((o, &xv), &yv), &dv), &zv)| {
                    *o = a * xv + b * *o + c * (dv * yv) + d * zv
                }),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sums do not depend on the number of threads: bit-identical on 1
    /// and 8 threads, for a size that is not a multiple of `CHUNK`.
    #[test]
    fn sums_are_reproducible_across_thread_counts() {
        let n = 3 * CHUNK + 17;
        let a: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
        let b: Vec<f64> = (0..n).map(|i| 1.0 + ((i as f64) * 0.11).cos()).collect();
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| (dot(&a, &b), mass_norm(&a, &b)))
        };
        assert_eq!(run(1), run(8));
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive).abs() < 1e-12 * naive.abs().max(1.0));
    }

    /// On a diagonal `K` with `M = I` the estimate tends to the largest entry,
    /// from below.
    #[test]
    fn power_iteration_finds_the_largest_eigenvalue() {
        let k: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let est = power_iteration(
            |_v| {},
            |u, v| v.iter_mut().zip(u).zip(&k).for_each(|((v, &u), &d)| *v = d * u),
            dot,
            start_vector(k.len()),
            1e-12,
            10_000,
        );
        assert!(est.lambda <= 50.0 + 1e-9 && est.lambda > 50.0 - 1e-6, "{est:?}");
        assert!(est.residue < 1e-12);
    }

    /// `axpby_diag` on two components (the diagonal shared by both) against
    /// the expression written out, with and without the fourth term, and
    /// `b = 0` never reading `out`.
    #[test]
    fn axpby_diag_matches_the_written_expression() {
        let (nloc, n_elem) = (3, 4);
        let n = 2 * nloc * n_elem;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 - i as f64 * 0.25).collect();
        let z: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let diag: Vec<f64> = (0..nloc * n_elem).map(|i| 2.0 + i as f64).collect();
        let d_of = |i: usize| diag[(i / (2 * nloc)) * nloc + i % nloc];
        let out0: Vec<f64> = (0..n).map(|i| (i as f64).sqrt()).collect();
        for b in [0.0, -1.0] {
            for with_z in [false, true] {
                let zz = with_z.then_some((0.25, z.as_slice()));
                let mut out = if b == 0.0 { vec![f64::NAN; n] } else { out0.clone() };
                axpby_diag::<2>(nloc, 2.0, &x, b, &mut out, -0.5, &diag, &y, zz);
                for i in 0..n {
                    let mut e = 2.0 * x[i] + -0.5 * (d_of(i) * y[i]);
                    if b != 0.0 {
                        e = 2.0 * x[i] + b * out0[i] + -0.5 * (d_of(i) * y[i]);
                    }
                    if with_z {
                        e += 0.25 * z[i];
                    }
                    assert_eq!(out[i], e, "i = {i}, b = {b}, z = {with_z}");
                }
            }
        }
    }
}
