//! Building blocks for explicit time stepping, which itself lives in the
//! examples (e.g. `examples/acoustic_wave_square.rs`).
//!
//! With the diagonal GLL mass, a leapfrog step for `M ü + K u = F` is
//! ```text
//! a    = M⁻¹ (F − K uⁿ)                      // c_apply + InverseMass::apply
//! uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ + dt² a
//! ```
//! with `dt = stable_dt(λ_max(M⁻¹K), 0.95)`. For continuous vectors the
//! duplicated storage needs no assembly to evaluate energies:
//! `⟨K u, w⟩ = dot(K_loc u, w)` with `K_loc u` from [`Space::mlt_add`], and
//! `⟨u, w⟩_M = mass_inner(local_mass, u, w)`.

use rayon::prelude::*;

use crate::engine::PointKernel;
use crate::fields::Field;
use crate::shape::Shape;
use crate::space::Space;

/// Largest stable leapfrog step for `M ü + K u = 0`: `safety · 2 / √λ_max`,
/// with `λ_max` the largest eigenvalue of `M⁻¹K`.
pub fn stable_dt(lambda_max: f64, safety: f64) -> f64 {
    assert!(lambda_max > 0.0, "λ_max must be positive ({lambda_max})");
    safety * 2.0 / lambda_max.sqrt()
}

/// Plain dot product of two duplicated vectors (parallel).
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector lengths");
    a.par_iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Result of [`Space::c_max_eigenvalue`].
#[derive(Clone, Copy, Debug)]
pub struct EigenEstimate {
    /// Rayleigh quotient of the last iterate: a lower bound of `λ_max`.
    pub lambda: f64,
    /// Iterations performed.
    pub iterations: usize,
    /// Relative change of `lambda` at the last iteration.
    pub rel_change: f64,
}

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// Assembled operator of the continuous space: `out = Σ_shared(K_loc u)`,
    /// i.e. [`Space::mlt_add`] with `α = 1`, `β = 0` followed by
    /// [`Space::c_sum_shared`]. `out` is overwritten.
    pub fn c_apply<const C: usize, K: PointKernel<D, C>>(
        &self,
        kernel: &K,
        u: &[f64],
        out: &mut [f64],
    ) {
        self.mlt_add::<C, K>(kernel, 1.0, u, 0.0, out);
        self.c_sum_shared::<C>(out);
    }

    /// `v[e][c][node] *= diag[e·NLOC + node]` for every component `c`: applies
    /// a one-value-per-node diagonal to a `C`-component vector.
    pub fn scale_by<const C: usize>(&self, diag: &[f64], v: &mut [f64]) {
        let nloc = Self::NLOC;
        assert_eq!(diag.len(), self.n_dup(), "diagonal length");
        assert_eq!(v.len(), self.n_dup() * C, "vector length");
        v.par_chunks_mut(C * nloc)
            .zip(diag.par_chunks(nloc))
            .for_each(|(block, d)| {
                for comp in block.chunks_exact_mut(nloc) {
                    for (x, s) in comp.iter_mut().zip(d) {
                        *x *= s;
                    }
                }
            });
    }

    /// `Σ_dup m · a·b` over the duplicated nodes and the `C` components. With
    /// `m` from [`Space::local_mass`] this is the GLL mass product `⟨a, b⟩_M` of
    /// the discontinuous space, and of the continuous space for continuous
    /// `a`, `b` (`local_mass(&Constant(1.0))` gives the L² product). Never uses
    /// the numbering.
    pub fn mass_inner<const C: usize>(&self, m: &[f64], a: &[f64], b: &[f64]) -> f64 {
        let nloc = Self::NLOC;
        assert_eq!(m.len(), self.n_dup(), "mass length");
        assert_eq!(a.len(), self.n_dup() * C, "a length");
        assert_eq!(b.len(), self.n_dup() * C, "b length");
        a.par_chunks(C * nloc)
            .zip(b.par_chunks(C * nloc))
            .zip(m.par_chunks(nloc))
            .map(|((ab, bb), mb)| {
                let mut s = 0.0;
                for (ac, bc) in ab.chunks_exact(nloc).zip(bb.chunks_exact(nloc)) {
                    for ((x, y), w) in ac.iter().zip(bc).zip(mb) {
                        s += w * x * y;
                    }
                }
                s
            })
            .sum()
    }

    /// Fixed pseudo-random continuous `C`-component vector (a hash of the copy
    /// index, averaged over the copies): rough, so rich in the high modes,
    /// and reproducible. The start of [`Space::c_max_eigenvalue`]. Panics
    /// without the continuous numbering.
    pub fn c_start_vector<const C: usize>(&self) -> Vec<f64> {
        let n = self.n_dup() * C;
        let nloc = Self::NLOC;
        let mut x: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut h = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
                h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                h ^= h >> 31;
                (h >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        self.c_sum_shared::<C>(&mut x);
        let mult = self.c_multiplicity();
        x.par_chunks_mut(nloc).enumerate().for_each(|(k, comp)| {
            let e = k / C;
            for (v, mu) in comp.iter_mut().zip(&mult[e * nloc..(e + 1) * nloc]) {
                *v /= mu;
            }
        });
        x
    }

    /// Largest eigenvalue of `M⁻¹K` on the continuous space by power iteration,
    /// `K` the operator of `kernel` and `M` the lumped mass of coefficient
    /// `mass`: [`power_iteration`] from [`Space::c_start_vector`], with
    /// `x ↦ M⁻¹ Σ_shared(K_loc x)` and the GLL mass product. Panics without the
    /// continuous numbering.
    pub fn c_max_eigenvalue<const C: usize, K: PointKernel<D, C>, F: Field<D>>(
        &self,
        kernel: &K,
        mass: &F,
        max_iter: usize,
        rtol: f64,
    ) -> EigenEstimate {
        let m = self.local_mass(mass);
        let inv_m = self.c_inverse_mass(mass);
        power_iteration(
            self.c_start_vector::<C>(),
            |x, y| {
                self.c_apply::<C, K>(kernel, x, y);
                inv_m.apply::<C>(y);
            },
            |a, b| self.mass_inner::<C>(&m, a, b),
            max_iter,
            rtol,
        )
    }
}

/// Largest eigenvalue of `M⁻¹A` (`A` symmetric, `M` symmetric positive
/// definite) by power iteration.
///
/// - `apply(x, y)` writes `y = M⁻¹A x` (any composite operator, e.g.
///   `M⁻¹(R + λ Bᵀ (M^P)⁻¹ B)`); `y` is overwritten.
/// - `inner(a, b)` is the `M` inner product `⟨a, b⟩_M`.
///
/// Each iterate is `M`-normalized; the estimate is the Rayleigh quotient
/// `⟨x, M⁻¹A x⟩_M = ⟨x, A x⟩` (a lower bound of `λ_max`, monotone for
/// symmetric `A`). Stops when its relative change drops below `rtol`, or after
/// `max_iter` iterations. Same role as TetriX `lal::power_iteration`, which
/// takes `A` and `M⁻¹` separately and estimates with the ∞-norm ratio.
pub fn power_iteration(
    mut x: Vec<f64>,
    mut apply: impl FnMut(&[f64], &mut [f64]),
    inner: impl Fn(&[f64], &[f64]) -> f64,
    max_iter: usize,
    rtol: f64,
) -> EigenEstimate {
    let norm = inner(&x, &x).sqrt();
    assert!(norm > 0.0, "power iteration: zero start vector");
    x.par_iter_mut().for_each(|v| *v /= norm);
    let mut y = vec![0.0; x.len()];
    let mut est = EigenEstimate {
        lambda: 0.0,
        iterations: 0,
        rel_change: f64::INFINITY,
    };
    for it in 1..=max_iter {
        apply(&x, &mut y);
        let lambda = inner(&x, &y);
        est.rel_change = (lambda - est.lambda).abs() / lambda.abs();
        est.lambda = lambda;
        est.iterations = it;
        if est.rel_change < rtol {
            break;
        }
        let norm = inner(&y, &y).sqrt();
        y.par_iter_mut().for_each(|v| *v /= norm);
        std::mem::swap(&mut x, &mut y);
    }
    est
}
