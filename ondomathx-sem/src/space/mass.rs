//! Masses and loads: element-local lumped mass, loads, inverse masses
//! ([`InverseMass`]).

use std::any::type_name;
use std::marker::PhantomData;

use rayon::prelude::*;

use crate::engine::PointIn;
use crate::fields::Field;
use crate::quadrature::Sweep;
use crate::shape::Shape;

use super::Space;

/// Element-local tensor application of `M_K⁻¹ = B W⁻¹ Bᵀ` to one component
/// block: `(1/W at the points of the element, block, scratch)`.
type TensorApply = Box<dyn Fn(&[f64], &mut [f64], &mut [f64]) + Send + Sync>;

enum InverseMassKind {
    /// One value per duplicated node.
    Diagonal(Vec<f64>),
    /// `B W⁻¹ Bᵀ` per element ([`Space::inverse_mass_with`]).
    Tensor {
        inv_w: Vec<f64>,
        nq: usize,
        scratch: usize,
        apply: TensorApply,
    },
}

/// Inverse mass operator of a [`Space`], built by [`Space::inverse_mass`]
/// (discontinuous GLL lumped mass), [`Space::c_inverse_mass`] (continuous
/// lumped mass) or [`Space::inverse_mass_with`] (discontinuous mass of a
/// quadrature, tensor form), applied with [`InverseMass::apply`].
pub struct InverseMass<const D: usize, S: Shape<D>> {
    kind: InverseMassKind,
    _shape: PhantomData<S>,
}

impl<const D: usize, S: Shape<D>> InverseMass<D, S> {
    /// Inverts a one-value-per-node mass. Panics on a non-positive value.
    fn from_mass(mut m: Vec<f64>) -> Self {
        m.par_iter_mut().for_each(|x| {
            assert!(*x > 0.0, "non-positive mass coefficient {x}");
            *x = 1.0 / *x;
        });
        InverseMass {
            kind: InverseMassKind::Diagonal(m),
            _shape: PhantomData,
        }
    }

    /// `v ← M⁻¹ v` for a `C`-component vector (`n_elem · C · NLOC` entries).
    pub fn apply<const C: usize>(&self, v: &mut [f64]) {
        let nloc = S::NLOC;
        match &self.kind {
            InverseMassKind::Diagonal(diag) => {
                assert_eq!(v.len(), diag.len() * C, "vector length");
                crate::ops::scale_nodes::<C>(nloc, diag, v);
            }
            InverseMassKind::Tensor {
                inv_w,
                nq,
                scratch,
                apply,
            } => {
                assert_eq!(v.len() / (C * nloc), inv_w.len() / nq, "vector length");
                v.par_chunks_mut(C * nloc).enumerate().for_each_init(
                    || vec![0.0; *scratch],
                    |tmp, (e, block)| {
                        for comp in block.chunks_exact_mut(nloc) {
                            apply(&inv_w[e * nq..(e + 1) * nq], comp, tmp);
                        }
                    },
                );
            }
        }
    }

    /// The diagonal of `M⁻¹`, one value per duplicated node, when `M⁻¹` is
    /// diagonal (`None` for [`Space::inverse_mass_with`]).
    pub fn diagonal(&self) -> Option<&[f64]> {
        match &self.kind {
            InverseMassKind::Diagonal(d) => Some(d),
            InverseMassKind::Tensor { .. } => None,
        }
    }
}

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// Element-local lumped mass `c · w · |det J|`, one value per duplicated
    /// node (before the shared-DOF sum).
    ///
    /// For `C`-component vectors `a`, `b`, [`Space::mass_inner`]`(m, a, b)` with
    /// this `m` is the GLL mass product `⟨a, b⟩_M` of the discontinuous space,
    /// and of the continuous space when `a`, `b` are continuous; no assembly.
    pub fn local_mass(&self, coef: &impl Field<D>) -> Vec<f64> {
        let nloc = Self::NLOC;
        let mut m = vec![0.0; self.n_dup()];
        m.par_chunks_mut(nloc).enumerate().for_each(|(e, block)| {
            let mut p = PointIn::<D, 1>::zeros();
            p.label = self.labels[e];
            let kind = self.kind(e);
            for (node, out) in block.iter_mut().enumerate() {
                let (x, det_j, _) = self.geo_at::<true>(&self.nodes, kind, e, node);
                p.xyz = x;
                p.dof = e * nloc + node;
                *out = coef.eval(&p) * self.node_weights[node] * det_j.abs();
            }
        });
        m
    }

    /// Element-local load vector of a volume source, `F_i = ∫ f·φ_i` with the
    /// GLL rule (the nodes): `w_i·|det J_i|·f(x_i)`, `C` components, one entry
    /// per duplicated DoF (before the shared-DOF sum).
    ///
    /// For a source `f(x)·g(t)`, `F` is computed once; with the lumped mass
    /// `M⁻¹ Σ_shared F` is the nodal value of `f` divided by the mass
    /// coefficient, and a leapfrog step adds `dt²·g(tⁿ)·M⁻¹F`.
    pub fn local_load<const C: usize>(&self, f: impl Fn([f64; D]) -> [f64; C] + Sync) -> Vec<f64> {
        self.local_load_with::<S::Nodes, C>(f)
    }

    /// Element-local load vector `F_i = ∫ f·φ_i` integrated with the
    /// quadrature `Q` (declared, e.g. over-integrated or the `L²` projection
    /// load of a non-polynomial source, or the node rule `S::Nodes`):
    /// `Σ_q w_q·|det J_q|·f(x_q)·φ_i(x_q)`.
    pub fn local_load_with<Q: Sweep<D, S>, const C: usize>(
        &self,
        f: impl Fn([f64; D]) -> [f64; C] + Sync,
    ) -> Vec<f64> {
        let nloc = Self::NLOC;
        let (q, t) = self.rule::<Q>();
        let mut v = vec![0.0; self.n_dup() * C];
        v.par_chunks_mut(C * nloc).enumerate().for_each_init(
            || (vec![0.0; C * Q::NQ], vec![0.0; Q::TMP]),
            |(fq, tmp), (e, block)| {
                let kind = self.kind(e);
                for p in 0..Q::NQ {
                    let (x, det_j, _) = self.geo_at::<true>(q.geo, kind, e, p);
                    let ws = q.weights[p] * det_j.abs();
                    let val = f(core::array::from_fn(|a| x[a]));
                    for c in 0..C {
                        fq[c * Q::NQ + p] = ws * val[c];
                    }
                }
                for c in 0..C {
                    Q::interpolate_t::<1>(
                        t,
                        &fq[c * Q::NQ..(c + 1) * Q::NQ],
                        &mut block[c * nloc..(c + 1) * nloc],
                        tmp,
                    );
                }
            },
        );
        v
    }

    /// Inverse of the element-local lumped mass `c · w · |det J|`: the mass of
    /// the discontinuous space (block diagonal, here diagonal). Never uses the
    /// numbering; see [`Space::c_inverse_mass`] for the continuous mass.
    ///
    /// Panics if a mass coefficient is not positive.
    pub fn inverse_mass(&self, coef: &impl Field<D>) -> InverseMass<D, S> {
        InverseMass::from_mass(self.local_mass(coef))
    }

    /// Inverse of the assembled lumped mass `Σ_shared c · w · |det J|` of the
    /// continuous space (equal across the copies of a node, shared by all
    /// components): `ρ` for elastodynamics, `Inverse(κ)` for acoustics.
    ///
    /// Panics if a mass coefficient is not positive, or without the continuous
    /// numbering.
    pub fn c_inverse_mass(&self, coef: &impl Field<D>) -> InverseMass<D, S> {
        let mut m = self.local_mass(coef);
        self.c_sum_shared::<1>(&mut m);
        InverseMass::from_mass(m)
    }

    /// Inverse of the element-local mass `∫ c u v` integrated with the
    /// quadrature `Q` (the discontinuous mass, e.g. the pressure mass
    /// `M^P` with a Gauss–Legendre rule), in tensor form: with `I` the
    /// interpolation nodes → points (square and invertible: `Q` has the
    /// space's counts per direction) and `B = I⁻¹` the interpolation back,
    /// `M_K = Iᵀ W_K I` and `M_K⁻¹ = B W_K⁻¹ Bᵀ`, `W_K = c·w·|det J|` at the
    /// points. Two sum-factorized passes and a diagonal per element and
    /// component, also on non-affine elements.
    ///
    /// Panics if `Q` is neither declared nor the node rule, if its counts
    /// differ from the space's, or on a non-positive `c·w·|det J|`.
    pub fn inverse_mass_with<Q: Sweep<D, S>>(&self, coef: &impl Field<D>) -> InverseMass<D, S> {
        assert!(
            (0..D).all(|d| Q::M[d] == S::N[d]),
            "inverse_mass_with: {} must have the space's counts {:?} per direction",
            type_name::<Q>(),
            &S::N[..D]
        );
        let (q, _) = self.rule::<Q>();
        let nq = Q::NQ;
        let mut inv_w = vec![0.0; self.n_elem * nq];
        inv_w.par_chunks_mut(nq).enumerate().for_each(|(e, block)| {
            let kind = self.kind(e);
            let mut p = PointIn::<D, 1>::zeros();
            p.label = self.labels[e];
            for (k, out) in block.iter_mut().enumerate() {
                let (x, det_j, _) = self.geo_at::<true>(q.geo, kind, e, k);
                p.xyz = x;
                p.dof = e * nq + k;
                let m = coef.eval(&p) * q.weights[k] * det_j.abs();
                assert!(m > 0.0, "non-positive mass coefficient {m}");
                *out = 1.0 / m;
            }
        });
        let t = Q::tables();
        let apply: TensorApply = Box::new(move |iw: &[f64], v: &mut [f64], tmp: &mut [f64]| {
            let (a, rest) = tmp.split_at_mut(iw.len());
            Q::back_t::<1>(&t, v, a, rest);
            for (x, w) in a.iter_mut().zip(iw) {
                *x *= w;
            }
            Q::back::<1>(&t, a, v, rest);
        });
        InverseMass {
            kind: InverseMassKind::Tensor {
                inv_w,
                nq,
                scratch: nq + Q::TMP,
                apply,
            },
            _shape: PhantomData,
        }
    }
}
