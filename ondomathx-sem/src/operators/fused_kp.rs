//! Fused pressure stiffness `K^P = Bᵀ (M^P)⁻¹ B`.

use std::marker::PhantomData;

use nalgebra::SMatrix;

use crate::element::{Batch, ElementOperator, PointSet};
use crate::quadrature::{Quadrature, Sweep};
use crate::shape::Shape;
use crate::space::Space;

/// The pressure stiffness `K^P = Bᵀ (M^P)⁻¹ B` of the displacement–pressure
/// formulation, applied element by element in one pass (the pressure is
/// discontinuous, so `K^P` is element-local and no global pressure vector is
/// formed):
///
/// - `B`: `b(u, q) = ∫ q div u`, integrated with the quadrature `QB`
///   (option 2 of the article: the displacement GLL rule, `Gll<k+1>`),
///   declared on either space;
/// - `(M^P)⁻¹`: the inverse of the pressure mass `∫ c p q` with the
///   quadrature `QM` (Gauss–Legendre with the pressure counts, `Gl<k−1>`),
///   declared on the pressure space, in tensor form `B_P W⁻¹ B_Pᵀ`
///   (as `Space::inverse_mass_with_dg`);
/// - `Bᵀ`: `bᵀ(p, v) = ∫ p div v`, same quadrature `QB`.
///
/// Trial and test spaces are the displacement space (`D` components):
/// `disp.mlt_add_element_dg(&disp, &kp, &u, &mut y, λ)`
/// adds `λ K^P u` to `y`. Same result as `mlt_mixed_dg` with `Divergence`,
/// `InverseMass::apply` of `inverse_mass_with_dg`, `mlt_mixed_dg` with
/// `DivergenceTranspose` (to rounding).
pub struct FusedKp<'a, const D: usize, SU, SP, QB, QM>
where
    SU: Shape<D>,
    SP: Shape<D>,
    QB: Sweep<D, SU> + Sweep<D, SP>,
    QM: Sweep<D, SP>,
{
    b_points: PointSet<'a, D>,
    /// `1 / (c·w·|det J|)` at the points of `QM`, `[elem][q]`.
    inv_w: Vec<f64>,
    tbu: <QB as Sweep<D, SU>>::Tables,
    tbp: <QB as Sweep<D, SP>>::Tables,
    tm: <QM as Sweep<D, SP>>::Tables,
    _types: PhantomData<Types<SU, SP, QB, QM>>,
}

/// Marker of the type parameters (the tables only name them through
/// projections).
type Types<SU, SP, QB, QM> = fn() -> (SU, SP, QB, QM);

impl<'a, const D: usize, SU, SP, QB, QM> FusedKp<'a, D, SU, SP, QB, QM>
where
    SU: Shape<D>,
    SP: Shape<D>,
    QB: Sweep<D, SU> + Sweep<D, SP>,
    QM: Sweep<D, SP>,
{
    /// `K^P` between the displacement space `disp` and the (discontinuous)
    /// pressure space `pres`, pressure mass coefficient `mass_coef`.
    ///
    /// Panics if the spaces come from different meshes, if `QB` is declared on
    /// neither space, if `QM` is not declared on `pres` or does not have the
    /// pressure counts, or on a non-positive mass.
    pub fn new(disp: &'a Space<D, SU>, pres: &'a Space<D, SP>, mass_coef: f64) -> Self {
        assert!(
            disp.same_mesh(pres),
            "FusedKp: spaces built from different meshes"
        );
        assert!(
            (0..D).all(|d| <QM as Quadrature<D>>::M[d] == SP::N[d]),
            "FusedKp: the pressure mass rule must have the pressure counts per direction"
        );
        let b_points = disp
            .quadrature_points::<QB>()
            .or_else(|| pres.quadrature_points::<QB>())
            .expect("FusedKp: the b_h quadrature is declared on neither space");
        let inv_w = pres
            .quadrature_weights::<QM>()
            .into_iter()
            .map(|w| {
                let m = mass_coef * w;
                assert!(m > 0.0, "FusedKp: non-positive pressure mass {m}");
                1.0 / m
            })
            .collect();
        FusedKp {
            b_points,
            inv_w,
            tbu: <QB as Sweep<D, SU>>::tables(),
            tbp: <QB as Sweep<D, SP>>::tables(),
            tm: <QM as Sweep<D, SP>>::tables(),
            _types: PhantomData,
        }
    }
}

/// Per-task scratch of [`FusedKp`].
pub struct FusedKpScratch<const D: usize> {
    /// Reference gradients of `u`, then pulled-back fluxes, at the `QB` points.
    g: Vec<f64>,
    /// `det J` and `cof J` at the `QB` points, `[q][lane]`.
    det: Vec<f64>,
    cof: Vec<SMatrix<f64, D, D>>,
    /// Weighted divergence, then pressure values, at the `QB` points.
    sq: Vec<f64>,
    /// Pressure at its nodes.
    p: Vec<f64>,
    /// Pressure at the `QM` points.
    pm: Vec<f64>,
    /// One transposed sweep on the displacement shape.
    yt: Vec<f64>,
    tmp: Vec<f64>,
}

impl<const D: usize, SU, SP, QB, QM> ElementOperator<D, SU, SU, D, D>
    for FusedKp<'_, D, SU, SP, QB, QM>
where
    SU: Shape<D>,
    SP: Shape<D>,
    QB: Sweep<D, SU> + Sweep<D, SP>,
    QM: Sweep<D, SP>,
{
    type Scratch = FusedKpScratch<D>;

    fn scratch<const W: usize>(&self) -> FusedKpScratch<D> {
        let nb = <QB as Quadrature<D>>::NQ * W;
        let tmp = <QB as Sweep<D, SU>>::TMP
            .max(<QB as Sweep<D, SP>>::TMP)
            .max(<QM as Sweep<D, SP>>::TMP);
        FusedKpScratch {
            g: vec![0.0; D * D * nb],
            det: vec![0.0; nb],
            cof: vec![SMatrix::zeros(); nb],
            sq: vec![0.0; nb],
            p: vec![0.0; SP::NLOC * W],
            pm: vec![0.0; <QM as Quadrature<D>>::NQ * W],
            yt: vec![0.0; SU::NLOC * W],
            tmp: vec![0.0; tmp * W],
        }
    }

    fn apply<const W: usize>(
        &self,
        batch: &Batch<'_, D, SU, SU>,
        u: &[f64],
        v: &mut [f64],
        s: &mut FusedKpScratch<D>,
    ) {
        let (nb, nm) = (<QB as Quadrature<D>>::NQ, <QM as Quadrature<D>>::NQ);
        let (nuw, nbw) = (SU::NLOC * W, nb * W);

        // B u: reference gradients at the QB points, then the weighted
        // divergence Σ_c Σ_d ∂_d u_c cof_cd / det · w |det| = w sign(det) Σ g:cof.
        for c in 0..D {
            for d in 0..D {
                let out = &mut s.g[(c * D + d) * nbw..(c * D + d + 1) * nbw];
                <QB as Sweep<D, SU>>::gradient_interpolation::<W>(
                    &self.tbu,
                    d,
                    &u[c * nuw..(c + 1) * nuw],
                    out,
                    &mut s.tmp,
                );
            }
        }
        for q in 0..nb {
            let w = self.b_points.weight(q);
            for lane in 0..W {
                let i = q * W + lane;
                let (_, det_j, cof) = batch.geometry(&self.b_points, lane, q);
                s.det[i] = det_j;
                s.cof[i] = cof;
                let mut div = 0.0;
                for c in 0..D {
                    for d in 0..D {
                        div += s.g[(c * D + d) * nbw + i] * cof[(c, d)];
                    }
                }
                s.sq[i] = w * det_j.signum() * div;
            }
        }
        <QB as Sweep<D, SP>>::transpose_interpolation::<W>(&self.tbp, &s.sq, &mut s.p, &mut s.tmp);

        // (M^P)⁻¹ p = B_P W⁻¹ B_Pᵀ p, per element.
        <QM as Sweep<D, SP>>::transpose_back_interpolation::<W>(&self.tm, &s.p, &mut s.pm, &mut s.tmp);
        for m in 0..nm {
            for lane in 0..W {
                s.pm[m * W + lane] *= self.inv_w[batch.element(lane) * nm + m];
            }
        }
        <QM as Sweep<D, SP>>::back_interpolation::<W>(&self.tm, &s.pm, &mut s.p, &mut s.tmp);

        // Bᵀ p: p at the QB points, pulled-back flux p I → w sign(det) p cof,
        // tested with the transposed gradients of the displacement shape.
        <QB as Sweep<D, SP>>::interpolation::<W>(&self.tbp, &s.p, &mut s.sq, &mut s.tmp);
        for q in 0..nb {
            let w = self.b_points.weight(q);
            for lane in 0..W {
                let i = q * W + lane;
                let a = w * s.det[i].signum() * s.sq[i];
                for c in 0..D {
                    for d in 0..D {
                        s.g[(c * D + d) * nbw + i] = a * s.cof[i][(c, d)];
                    }
                }
            }
        }
        for c in 0..D {
            let vc = &mut v[c * nuw..(c + 1) * nuw];
            vc.fill(0.0);
            for d in 0..D {
                let h = &s.g[(c * D + d) * nbw..(c * D + d + 1) * nbw];
                <QB as Sweep<D, SU>>::transpose_gradient_interpolation::<W>(&self.tbu, d, h, &mut s.yt, &mut s.tmp);
                for (y, t) in vc.iter_mut().zip(&s.yt) {
                    *y += t;
                }
            }
        }
    }
}
