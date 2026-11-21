//! `K^P` for Q4 displacement / Q2 pressure with `b_h` by Green's formula
//! (option 3 of the inf-sup article), port of OndoMathX_gitlab
//! `cornea/Sources/Utility_Cornea/GradDiv.h` (`MltAddGradDiv`, 2D and 3D).

use crate::element::{Batch, ElementOperator};
use crate::geometry::{cofactor, det};
use crate::quadrature::{Gl, Gll, Quad2, Quad3, Sweep};
use crate::shape::{Iso2, Iso3, Shape};
use crate::space::Space;

/// Q2 pressure nodes (GLL 3 points) → Q4 displacement nodes (GLL 5 points).
type Nodes2 = Quad2<Gll<5>, Gll<5>>;
type Nodes3 = Quad3<Gll<5>, Gll<5>, Gll<5>>;

/// `K^P = Bᵀ (M^P)⁻¹ B` for the displacement space Q4 (`Iso2<5>` / `Iso3<5>`)
/// and an element-local Q2 pressure, as hard-coded in cornea's `GradDiv.h`:
///
/// - `b(u, q) = ∫_∂K (u·n) q − ∫_K u·∇q` on each element (Green's formula),
///   both integrals at the GLL 5 nodes: the boundary term with the
///   precomputed `bndy` (`BndyMatrix`: at each node of each reference face,
///   `cof(J)·n̂` times the 1D face weights), the volume term with
///   `−w·cof(J)ᵀ u` tested with the reference gradients of the Q2 basis;
/// - `(M^P)⁻¹`: the lumped (GLL 3, Simpson) pressure mass at the Q2 nodes,
///   which are the displacement nodes `0, 2, 4` per direction (`InvMass`:
///   `1 / (c·w₃·|det J|)` there);
/// - `Bᵀ` by the same formula, transposed.
///
/// cornea's hard-coded Q2 → Q4 interpolation functions are lobatto's
/// rectangular kernels at the fixed sizes 3 → 5. Run with
/// `mlt_add_element(&disp, &disp, &op, λ, u, β, y)`.
pub struct GradDivQ4Q2<const D: usize> {
    /// `cof(J)·n̂·w_face` summed over the faces through each node, `[elem][node]`.
    bndy: Vec<[f64; D]>,
    /// `1 / (c·w₃·|det J|)` at the Q2 nodes, `[elem][q2 node]`.
    inv_mass: Vec<f64>,
    /// GLL 5 tensor weights.
    w: Vec<f64>,
}

impl<const D: usize> GradDivQ4Q2<D> {
    const NU: usize = 5usize.pow(D as u32);
    const NP: usize = 3usize.pow(D as u32);

    fn build<S: Shape<D>>(disp: &Space<D, S>, mass_coef: f64) -> Self {
        assert!(S::NLOC == Self::NU, "GradDivQ4Q2: the displacement space must be Q4");
        let n_elem = disp.n_elem();
        let w1 = disp.weights_1d(0).to_vec();
        let w3 = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];
        let mut bndy = vec![[0.0; D]; n_elem * Self::NU];
        let mut inv_mass = vec![0.0; n_elem * Self::NP];
        for e in 0..n_elem {
            for node in 0..Self::NU {
                let id = Space::<D, S>::multi_index(node);
                let cof = cofactor(&disp.map_and_jacobian(e, node).1);
                for d in 0..D {
                    for (end, sign) in [(0usize, -1.0), (4, 1.0)] {
                        if id[d] != end {
                            continue;
                        }
                        // Face ξ_d = end / 4: outward reference normal sign·e_d,
                        // weight = product of the 1D weights of the other directions.
                        let wf: f64 = (0..D).filter(|&a| a != d).map(|a| w1[id[a]]).product();
                        for (a, b) in bndy[e * Self::NU + node].iter_mut().enumerate() {
                            *b += sign * cof[(a, d)] * wf;
                        }
                    }
                }
            }
            for k in 0..Self::NP {
                let mut node = 0;
                let mut stride = 1;
                let mut wk = 1.0;
                let mut kk = k;
                for _ in 0..D {
                    let i = kk % 3;
                    kk /= 3;
                    node += 2 * i * stride;
                    stride *= 5;
                    wk *= w3[i];
                }
                let dj = det(&disp.map_and_jacobian(e, node).1);
                let m = mass_coef * wk * dj.abs();
                assert!(m > 0.0, "GradDivQ4Q2: non-positive pressure mass {m}");
                inv_mass[e * Self::NP + k] = 1.0 / m;
            }
        }
        let w = (0..Self::NU).map(|n| disp.weight(n)).collect();
        GradDivQ4Q2 { bndy, inv_mass, w }
    }

    /// One batch: `B` (Green), `mass(p, tmp)` (the inverse pressure mass on
    /// the Q2 pressure `[node][lane]`, in place), `Bᵀ`; `Q` the GLL 5 rule seen
    /// from the Q2 pressure shape `SP`.
    #[allow(clippy::too_many_arguments)]
    fn apply_with<SU: Shape<D>, SP: Shape<D>, Q: Sweep<D, SP>, const W: usize>(
        &self,
        t: &Q::Tables,
        batch: &Batch<'_, D, SU, SU>,
        u: &[f64],
        v: &mut [f64],
        s: &mut GradDivScratch,
        mass: &mut dyn FnMut(&mut [f64], &mut [f64]),
    ) {
        let nu = Self::NU;
        let nuw = nu * W;
        let nodes = batch.u_space.node_points();

        // Boundary term `bndy·u` and volume term `−w cof(J)ᵀ u` at the Q4 nodes.
        for node in 0..nu {
            for lane in 0..W {
                let e = batch.element(lane);
                let i = node * W + lane;
                let b = &self.bndy[e * nu + node];
                let (_, _, cof) = batch.geometry(&nodes, lane, node);
                let mut bu = 0.0;
                for c in 0..D {
                    bu += b[c] * u[c * nuw + i];
                }
                s.pb[i] = bu;
                for d in 0..D {
                    // (cofᵀ u)_d
                    let mut x = 0.0;
                    for c in 0..D {
                        x += cof[(c, d)] * u[c * nuw + i];
                    }
                    s.px[d * nuw + i] = -self.w[node] * x;
                }
            }
        }
        // To the Q2 pressure: Iᵀ (boundary) + Σ_d G_dᵀ (volume).
        Q::interpolate_t::<W>(t, &s.pb, &mut s.p, &mut s.tmp);
        for d in 0..D {
            Q::gradient_t::<W>(t, d, &s.px[d * nuw..(d + 1) * nuw], &mut s.pt, &mut s.tmp);
            for (p, x) in s.p.iter_mut().zip(&s.pt) {
                *p += x;
            }
        }
        // Inverse pressure mass.
        mass(&mut s.p, &mut s.tmp);
        // Back to the displacement: `bndy·p − w cof(J) ∇ξ p` at the Q4 nodes.
        Q::interpolate::<W>(t, &s.p, &mut s.pb, &mut s.tmp);
        for d in 0..D {
            Q::gradient::<W>(t, d, &s.p, &mut s.px[d * nuw..(d + 1) * nuw], &mut s.tmp);
        }
        for node in 0..nu {
            for lane in 0..W {
                let e = batch.element(lane);
                let i = node * W + lane;
                let b = &self.bndy[e * nu + node];
                let (_, _, cof) = batch.geometry(&nodes, lane, node);
                for c in 0..D {
                    let mut g = 0.0;
                    for d in 0..D {
                        g += cof[(c, d)] * s.px[d * nuw + i];
                    }
                    v[c * nuw + i] = b[c] * s.pb[i] - self.w[node] * g;
                }
            }
        }
    }
}

impl<const D: usize> GradDivQ4Q2<D> {
    /// The lumped (GLL 3) inverse pressure mass on one batch.
    fn lumped<SU: Shape<D>, const W: usize>(&self, batch: &Batch<'_, D, SU, SU>, p: &mut [f64]) {
        let np = Self::NP;
        for k in 0..np {
            for lane in 0..W {
                p[k * W + lane] *= self.inv_mass[batch.element(lane) * np + k];
            }
        }
    }
}

impl GradDivQ4Q2<2> {
    /// The operator on the Q4 displacement space `disp`, pressure mass
    /// coefficient `mass_coef` (1 in cornea).
    pub fn new(disp: &Space<2, Iso2<5>>, mass_coef: f64) -> Self {
        Self::build(disp, mass_coef)
    }
}

impl GradDivQ4Q2<3> {
    /// The operator on the Q4 displacement space `disp`, pressure mass
    /// coefficient `mass_coef` (1 in cornea).
    pub fn new(disp: &Space<3, Iso3<5>>, mass_coef: f64) -> Self {
        Self::build(disp, mass_coef)
    }
}

/// Per-task scratch of [`GradDivQ4Q2`].
pub struct GradDivScratch {
    /// Boundary term, then pressure values, at the Q4 nodes.
    pb: Vec<f64>,
    /// Volume terms, then pressure reference gradients, at the Q4 nodes.
    px: Vec<f64>,
    /// Pressure at the Q2 nodes.
    p: Vec<f64>,
    /// One transposed sweep at the Q2 nodes.
    pt: Vec<f64>,
    tmp: Vec<f64>,
}

fn scratch<const D: usize, const W: usize>(tmp: usize) -> GradDivScratch {
    let (nu, np) = (5usize.pow(D as u32), 3usize.pow(D as u32));
    GradDivScratch {
        pb: vec![0.0; nu * W],
        px: vec![0.0; D * nu * W],
        p: vec![0.0; np * W],
        pt: vec![0.0; np * W],
        tmp: vec![0.0; tmp * W],
    }
}

impl ElementOperator<2, Iso2<5>, Iso2<5>, 2, 2> for GradDivQ4Q2<2> {
    type Scratch = (GradDivScratch, <Nodes2 as Sweep<2, Iso2<3>>>::Tables);

    fn scratch<const W: usize>(&self) -> Self::Scratch {
        (scratch::<2, W>(<Nodes2 as Sweep<2, Iso2<3>>>::TMP), <Nodes2 as Sweep<2, Iso2<3>>>::tables())
    }

    fn apply<const W: usize>(&self, batch: &Batch<'_, 2, Iso2<5>, Iso2<5>>, u: &[f64], v: &mut [f64], s: &mut Self::Scratch) {
        let (sc, t) = s;
        let mut mass = |p: &mut [f64], _: &mut [f64]| self.lumped::<Iso2<5>, W>(batch, p);
        self.apply_with::<Iso2<5>, Iso2<3>, Nodes2, W>(t, batch, u, v, sc, &mut mass);
    }
}

impl ElementOperator<3, Iso3<5>, Iso3<5>, 3, 3> for GradDivQ4Q2<3> {
    type Scratch = (GradDivScratch, <Nodes3 as Sweep<3, Iso3<3>>>::Tables);

    fn scratch<const W: usize>(&self) -> Self::Scratch {
        (scratch::<3, W>(<Nodes3 as Sweep<3, Iso3<3>>>::TMP), <Nodes3 as Sweep<3, Iso3<3>>>::tables())
    }

    fn apply<const W: usize>(&self, batch: &Batch<'_, 3, Iso3<5>, Iso3<5>>, u: &[f64], v: &mut [f64], s: &mut Self::Scratch) {
        let (sc, t) = s;
        let mut mass = |p: &mut [f64], _: &mut [f64]| self.lumped::<Iso3<5>, W>(batch, p);
        self.apply_with::<Iso3<5>, Iso3<3>, Nodes3, W>(t, batch, u, v, sc, &mut mass);
    }
}

// ── B by Green's formula, Gauss–Legendre pressure mass ───────────────────

/// The pressure mass rule of the revised article: 3 Gauss–Legendre points
/// per direction (`k − 1` for `k = 4`).
pub type PressureMass2 = Quad2<Gl<3>, Gl<3>>;
/// See [`PressureMass2`].
pub type PressureMass3 = Quad3<Gl<3>, Gl<3>, Gl<3>>;

/// `K^P = Bᵀ (M^P)⁻¹ B` for Q4 displacement / Q2 pressure with `B` (and `Bᵀ`)
/// of [`GradDivQ4Q2`] (Green's formula, hard-coded as in cornea) and the
/// pressure mass of the revised article, `∫ c p q` with 3 Gauss–Legendre
/// points per direction, inverted per element in tensor form
/// (`B_P W⁻¹ B_Pᵀ`, as `FusedKp` / `Space::inverse_mass_with`).
///
/// The rule [`PressureMass2`] / [`PressureMass3`] must be declared on the
/// displacement space (its geometry at those points). Run with
/// `mlt_add_element(&disp, &disp, &op, λ, u, β, y)`.
pub struct FusedKpQ4Q2<const D: usize> {
    b: GradDivQ4Q2<D>,
    /// `1 / (c·w·|det J|)` at the Gauss–Legendre points, `[elem][q]`.
    inv_w: Vec<f64>,
}

impl<const D: usize> FusedKpQ4Q2<D> {
    fn from_weights(b: GradDivQ4Q2<D>, weights: Vec<f64>, mass_coef: f64) -> Self {
        let inv_w = weights
            .into_iter()
            .map(|w| {
                let m = mass_coef * w;
                assert!(m > 0.0, "FusedKpQ4Q2: non-positive pressure mass {m}");
                1.0 / m
            })
            .collect();
        FusedKpQ4Q2 { b, inv_w }
    }

    /// `(M^P)⁻¹` on one batch: `B_P (W⁻¹ (B_Pᵀ p))`, `B_P` the interpolation
    /// Gauss–Legendre → Q2 nodes.
    fn tensor<SU: Shape<D>, SP: Shape<D>, QM: Sweep<D, SP>, const W: usize>(
        &self,
        tm: &QM::Tables,
        batch: &Batch<'_, D, SU, SU>,
        p: &mut [f64],
        pm: &mut [f64],
        tmp: &mut [f64],
    ) {
        let nm = QM::NQ;
        QM::back_t::<W>(tm, p, pm, tmp);
        for m in 0..nm {
            for lane in 0..W {
                pm[m * W + lane] *= self.inv_w[batch.element(lane) * nm + m];
            }
        }
        QM::back::<W>(tm, pm, p, tmp);
    }
}

impl FusedKpQ4Q2<2> {
    /// The operator on the Q4 displacement space `disp` (with
    /// [`PressureMass2`] declared), pressure mass coefficient `mass_coef`.
    pub fn new(disp: &Space<2, Iso2<5>>, mass_coef: f64) -> Self {
        Self::from_weights(GradDivQ4Q2::<2>::new(disp, mass_coef), disp.quadrature_weights::<PressureMass2>(), mass_coef)
    }
}

impl FusedKpQ4Q2<3> {
    /// The operator on the Q4 displacement space `disp` (with
    /// [`PressureMass3`] declared), pressure mass coefficient `mass_coef`.
    pub fn new(disp: &Space<3, Iso3<5>>, mass_coef: f64) -> Self {
        Self::from_weights(GradDivQ4Q2::<3>::new(disp, mass_coef), disp.quadrature_weights::<PressureMass3>(), mass_coef)
    }
}

impl ElementOperator<2, Iso2<5>, Iso2<5>, 2, 2> for FusedKpQ4Q2<2> {
    type Scratch = (
        GradDivScratch,
        <Nodes2 as Sweep<2, Iso2<3>>>::Tables,
        <PressureMass2 as Sweep<2, Iso2<3>>>::Tables,
        Vec<f64>,
    );

    fn scratch<const W: usize>(&self) -> Self::Scratch {
        let tmp = <Nodes2 as Sweep<2, Iso2<3>>>::TMP.max(<PressureMass2 as Sweep<2, Iso2<3>>>::TMP);
        (
            scratch::<2, W>(tmp),
            <Nodes2 as Sweep<2, Iso2<3>>>::tables(),
            <PressureMass2 as Sweep<2, Iso2<3>>>::tables(),
            vec![0.0; 9 * W],
        )
    }

    fn apply<const W: usize>(&self, batch: &Batch<'_, 2, Iso2<5>, Iso2<5>>, u: &[f64], v: &mut [f64], s: &mut Self::Scratch) {
        let (sc, t, tm, pm) = s;
        let mut mass =
            |p: &mut [f64], tmp: &mut [f64]| self.tensor::<Iso2<5>, Iso2<3>, PressureMass2, W>(tm, batch, p, pm, tmp);
        self.b.apply_with::<Iso2<5>, Iso2<3>, Nodes2, W>(t, batch, u, v, sc, &mut mass);
    }
}

impl ElementOperator<3, Iso3<5>, Iso3<5>, 3, 3> for FusedKpQ4Q2<3> {
    type Scratch = (
        GradDivScratch,
        <Nodes3 as Sweep<3, Iso3<3>>>::Tables,
        <PressureMass3 as Sweep<3, Iso3<3>>>::Tables,
        Vec<f64>,
    );

    fn scratch<const W: usize>(&self) -> Self::Scratch {
        let tmp = <Nodes3 as Sweep<3, Iso3<3>>>::TMP.max(<PressureMass3 as Sweep<3, Iso3<3>>>::TMP);
        (
            scratch::<3, W>(tmp),
            <Nodes3 as Sweep<3, Iso3<3>>>::tables(),
            <PressureMass3 as Sweep<3, Iso3<3>>>::tables(),
            vec![0.0; 27 * W],
        )
    }

    fn apply<const W: usize>(&self, batch: &Batch<'_, 3, Iso3<5>, Iso3<5>>, u: &[f64], v: &mut [f64], s: &mut Self::Scratch) {
        let (sc, t, tm, pm) = s;
        let mut mass =
            |p: &mut [f64], tmp: &mut [f64]| self.tensor::<Iso3<5>, Iso3<3>, PressureMass3, W>(tm, batch, p, pm, tmp);
        self.b.apply_with::<Iso3<5>, Iso3<3>, Nodes3, W>(t, batch, u, v, sc, &mut mass);
    }
}
