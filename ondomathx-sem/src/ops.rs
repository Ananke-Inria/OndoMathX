//! Building blocks for explicit time stepping, which itself lives in the
//! examples (e.g. `examples/acoustic_wave_square.rs`).
//!
//! With the diagonal GLL mass, a leapfrog step for `M ü + K u = F` is
//! ```text
//! a    = M⁻¹ (F − K uⁿ)                      // mlt_assembled + InverseMass::apply
//! uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ + dt² a
//! ```
//! with `dt = 0.95 · 2 / √λ_max(M⁻¹K)` (95 % of the leapfrog limit, `λ_max`
//! from [`Space::max_eigenvalue`] or `ondomathx_toolbox::lal::power_iteration`);
//! [`Space::axpby_diag`] does `M⁻¹` and the update in one pass, after
//! `mlt_assembled`. For continuous vectors the
//! duplicated storage needs no assembly to evaluate energies:
//! `⟨K u, w⟩ = dot(K_loc u, w)` with `K_loc u` from [`Space::mlt_dg`] and
//! `dot` from `ondomathx_toolbox::lal`, and
//! `⟨u, w⟩_M = mass_inner(local_mass, u, w)`.

use ondomathx_toolbox::lal::{self, EigenEstimate};
use rayon::prelude::*;

use crate::engine::PointKernel;
use crate::fields::Field;
use crate::shape::Shape;
use crate::space::Space;

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// Assembled operator of the continuous space: `out = Σ_shared(K_loc u)`,
    /// i.e. [`Space::mlt_dg`] with `α = 1` followed by
    /// [`Space::sum_shared`]. `out` is overwritten.
    pub fn mlt_assembled<const C: usize>(
        &self,
        kernel: &impl PointKernel<D, C>,
        u: &[f64],
        out: &mut [f64],
    ) {
        self.mlt_dg::<C>(kernel, u, out, 1.0);
        self.sum_shared::<C>(out);
    }

    /// `v[e][c][node] *= diag[e·NLOC + node]` for every component `c`: applies
    /// a one-value-per-node diagonal to a `C`-component vector.
    pub fn scale_by<const C: usize>(&self, diag: &[f64], v: &mut [f64]) {
        let nloc = Self::NLOC;
        assert_eq!(diag.len(), self.num_dg_dofs(), "diagonal length");
        assert_eq!(v.len(), self.num_dg_dofs() * C, "vector length");
        scale_nodes::<C>(nloc, diag, v);
    }

    /// `out ← a·x + b·out + c·(diag ⊙ y) + d·z` in one pass, for `C`-component
    /// vectors of this space, `diag` one value per duplicated node shared by
    /// the components (as in [`Space::scale_by`]): `ondomathx_toolbox::lal::axpby_diag`
    /// with this space's `NLOC`. With `diag` the inverse mass
    /// ([`crate::InverseMass::diagonal`]) it fuses `M⁻¹` into the leapfrog
    /// update `uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹K uⁿ + dt² g(tⁿ) M⁻¹f`, written into
    /// `u_prev` from `ku = K uⁿ` and `minv_f = M⁻¹f`:
    /// `axpby_diag(2.0, &u_now, -1.0, &mut u_prev, -dt * dt, inv_m, &ku,
    /// Some((dt * dt * g, &minv_f)))`. `b = 0` never reads `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn axpby_diag<const C: usize>(
        &self,
        a: f64,
        x: &[f64],
        b: f64,
        out: &mut [f64],
        c: f64,
        diag: &[f64],
        y: &[f64],
        z: Option<(f64, &[f64])>,
    ) {
        assert_eq!(diag.len(), self.num_dg_dofs(), "diagonal length");
        lal::axpby_diag::<C>(Self::NLOC, a, x, b, out, c, diag, y, z);
    }

    /// `Σ_dup m · a·b` over the duplicated nodes and the `C` components. With
    /// `m` from [`Space::local_mass`] this is the GLL mass product `⟨a, b⟩_M` of
    /// the discontinuous space, and of the continuous space for continuous
    /// `a`, `b` (`local_mass(&Constant(1.0))` gives the L² product). Never uses
    /// the numbering.
    pub fn mass_inner<const C: usize>(&self, m: &[f64], a: &[f64], b: &[f64]) -> f64 {
        let nloc = Self::NLOC;
        assert_eq!(m.len(), self.num_dg_dofs(), "mass length");
        assert_eq!(a.len(), self.num_dg_dofs() * C, "a length");
        assert_eq!(b.len(), self.num_dg_dofs() * C, "b length");
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
            .collect::<Vec<f64>>()
            .iter()
            .sum() // per-element sums added in element order: reproducible
    }

    /// Fixed pseudo-random continuous `C`-component vector (a hash of the copy
    /// index, averaged over the copies): rough, so rich in the high modes,
    /// and reproducible. The start of [`Space::max_eigenvalue`]. Panics
    /// without the continuous numbering.
    pub fn start_vector<const C: usize>(&self) -> Vec<f64> {
        let n = self.num_dg_dofs() * C;
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
        self.sum_shared::<C>(&mut x);
        let mult = self.multiplicity_dg();
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
    /// `mass`: `ondomathx_toolbox::lal::power_iteration` from
    /// [`Space::start_vector`], with `K x = Σ_shared(K_loc x)`, the inverse
    /// lumped mass and the GLL mass product. Panics without the continuous
    /// numbering.
    pub fn max_eigenvalue<const C: usize>(
        &self,
        kernel: &impl PointKernel<D, C>,
        mass: &impl Field<D>,
        max_iter: usize,
        rtol: f64,
    ) -> EigenEstimate {
        let m = self.local_mass(mass);
        let inv_m = self.inverse_mass(mass);
        lal::power_iteration(
            |y| inv_m.apply::<C>(y),
            |x, y| self.mlt_assembled::<C>(kernel, x, y),
            |a, b| self.mass_inner::<C>(&m, a, b),
            self.start_vector::<C>(),
            rtol,
            max_iter,
        )
    }
}

/// `v[e][c][node] *= diag[e·nloc + node]` for every component: the body of
/// [`Space::scale_by`] and of the diagonal [`crate::InverseMass::apply`].
pub(crate) fn scale_nodes<const C: usize>(nloc: usize, diag: &[f64], v: &mut [f64]) {
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
