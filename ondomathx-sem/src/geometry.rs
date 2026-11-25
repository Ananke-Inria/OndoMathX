//! Element maps of tensor-product Q1/Q2 geometric elements.
//!
//! Geometric nodes are stored in **tensor order** (`k = k_0 + n1·k_1 + n1²·k_2`,
//! `n1 = 2` for Q1, `3` for Q2), node `k` sitting at reference point
//! `(k_0, k_1, k_2) / (n1 − 1)`. The map `x(ξ) = Σ_k X_k φ_k(ξ)` is evaluated
//! from a `GeoTab`: the Anamesh shape functions of the element
//! (`GeometricElement::tabulate`) at a fixed point set, columns in tensor
//! order. No shape function is written here.

use anamesh::geometric_elements::GeometricElement;
use nalgebra::{SMatrix, SVector};

/// Tolerance of the affine / Q1 detection, in reference coordinates.
pub(crate) const SHAPE_TOL: f64 = 1e-10;

/// Cofactor matrix of a 2×2 or 3×3 matrix, `cof(J) = det(J) · J⁻ᵀ`.
///
/// Written out for `D = 2, 3` instead of nalgebra's `determinant` /
/// `try_inverse`, which on a generic `const D` require a
/// `Const<D>: DimMin<Const<D>, Output = Const<D>>` bound that would then have
/// to be repeated by every generic user of the space.
pub fn cofactor<const D: usize>(m: &SMatrix<f64, D, D>) -> SMatrix<f64, D, D> {
    let mut c = SMatrix::<f64, D, D>::zeros();
    match D {
        2 => {
            c[(0, 0)] = m[(1, 1)];
            c[(0, 1)] = -m[(1, 0)];
            c[(1, 0)] = -m[(0, 1)];
            c[(1, 1)] = m[(0, 0)];
        }
        3 => {
            for a in 0..3 {
                for b in 0..3 {
                    let (a1, a2) = ((a + 1) % 3, (a + 2) % 3);
                    let (b1, b2) = ((b + 1) % 3, (b + 2) % 3);
                    c[(a, b)] = m[(a1, b1)] * m[(a2, b2)] - m[(a1, b2)] * m[(a2, b1)];
                }
            }
        }
        _ => panic!("cofactor: D must be 2 or 3 (D = {D})"),
    }
    c
}

/// Determinant of a 2×2 or 3×3 matrix (expansion along the first row).
pub fn det<const D: usize>(m: &SMatrix<f64, D, D>) -> f64 {
    let c = cofactor(m);
    (0..D).map(|b| m[(0, b)] * c[(0, b)]).sum()
}

/// Digits of `k` in base `base`, least significant first.
pub(crate) fn digits<const D: usize>(mut k: usize, base: usize) -> [usize; D] {
    let mut out = [0; D];
    for o in out.iter_mut() {
        *o = k % base;
        k /= base;
    }
    out
}

/// Values and reference gradients of the shape functions of one geometric
/// element at a fixed point set, columns in tensor node order:
/// `v[p·nk + k] = φ_k(ξ_p)`, `g[p·nk + k] = ∇φ_k(ξ_p)`.
pub(crate) struct GeoTab<const D: usize> {
    nk: usize,
    v: Vec<f64>,
    g: Vec<[f64; D]>,
}

impl<const D: usize> GeoTab<D> {
    /// Tabulates `ge` (Anamesh) at `points`; `perm` is its
    /// [`tensor_permutation`].
    pub(crate) fn new(ge: &GeometricElement, perm: &[usize], points: &[[f64; 3]]) -> Self {
        let n = ge.nshape_functions;
        assert_eq!(perm.len(), n, "permutation length");
        let (values, gradients) = ge.tabulate(points);
        let mut v = Vec::with_capacity(points.len() * n);
        let mut g = Vec::with_capacity(points.len() * n);
        for p in 0..points.len() {
            for &i in perm {
                v.push(values[p * n + i]);
                g.push(core::array::from_fn(|d| gradients[p * n + i][d]));
            }
        }
        GeoTab { nk: n, v, g }
    }

    /// Image `x(ξ_p)` and Jacobian `J[a][b] = ∂x_a/∂ξ_b` at point `p` of an
    /// element with tensor-ordered `nodes`.
    #[inline]
    pub(crate) fn eval(
        &self,
        nodes: &[[f64; D]],
        p: usize,
    ) -> (SVector<f64, D>, SMatrix<f64, D, D>) {
        let (v, g) = (
            &self.v[p * self.nk..(p + 1) * self.nk],
            &self.g[p * self.nk..(p + 1) * self.nk],
        );
        let mut x = SVector::<f64, D>::zeros();
        let mut jac = SMatrix::<f64, D, D>::zeros();
        for ((node, &phi), dphi) in nodes.iter().zip(v).zip(g) {
            for a in 0..D {
                x[a] += node[a] * phi;
                for b in 0..D {
                    jac[(a, b)] += node[a] * dphi[b];
                }
            }
        }
        (x, jac)
    }
}

/// Permutation `perm[t] = i` from tensor order to the Anamesh shape-function
/// order of `ge`, found by evaluating the shape functions at the tensor
/// reference nodes (`φ_i(ξ_t) = δ_it`). Panics if they are not nodal there.
pub(crate) fn tensor_permutation<const D: usize>(ge: &GeometricElement, n1: usize) -> Vec<usize> {
    let n = n1.pow(D as u32);
    assert_eq!(
        ge.nshape_functions, n,
        "unexpected number of shape functions"
    );
    let mut perm = Vec::with_capacity(n);
    for t in 0..n {
        let kd = digits::<D>(t, n1);
        let mut uvw = [0.0; 3];
        for d in 0..D {
            uvw[d] = kd[d] as f64 / (n1 - 1) as f64;
        }
        let i = (0..n)
            .find(|&i| ((ge.shape_functions[i])(&uvw) - 1.0).abs() < 1e-12)
            .expect("shape functions are not nodal at the tensor reference nodes");
        perm.push(i);
    }
    let mut seen = vec![false; n];
    for &i in &perm {
        assert!(!seen[i], "tensor permutation is not a bijection");
        seen[i] = true;
    }
    perm
}

/// How the geometry of a point set (the nodes, or a quadrature) is evaluated
/// for the affine/Q1/Q2 groups. Mapped elements are always stored; affine
/// elements always keep their single Jacobian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeoStorage {
    /// Recomputed at each use from the geometric nodes and the Anamesh shape
    /// functions tabulated at the points.
    Recompute,
    /// `x`, `det J` and `cof J` stored at every point of every element.
    Store,
}

/// Stored geometry of one group at the points of a set, index `k·np + p`
/// (`k` the element index within its group, `np` the number of points).
pub(crate) struct StoredGeo<const D: usize> {
    pub(crate) x: Vec<SVector<f64, D>>,
    pub(crate) det: Vec<f64>,
    pub(crate) cof: Vec<SMatrix<f64, D, D>>,
}

impl<const D: usize> StoredGeo<D> {
    /// `(x, det J, cof J)` at index `i = k·np + p`.
    #[inline(always)]
    pub(crate) fn at(&self, i: usize) -> (SVector<f64, D>, f64, SMatrix<f64, D, D>) {
        (self.x[i], self.det[i], self.cof[i])
    }
}

/// Geometry of the Q1 or Q2 group at the points of a set.
pub(crate) enum GroupGeo<const D: usize> {
    /// The shape functions tabulated at the points; the map is evaluated
    /// from the element's nodes at each use ([`GeoStorage::Recompute`]).
    Tabulated(GeoTab<D>),
    /// `x`, `det J` and `cof J` at every point ([`GeoStorage::Store`]).
    Stored(StoredGeo<D>),
}

/// Geometry of a set of reference points (the nodes, or a quadrature). A
/// group is `None` when the space has no element of that kind; affine
/// elements need nothing per point (one Jacobian per element).
pub(crate) struct PointGeo<const D: usize> {
    /// Reference points in `[0, 1]^D` (trailing zeros), tensor order.
    pub(crate) ref_points: Vec<[f64; 3]>,
    pub(crate) q1: Option<GroupGeo<D>>,
    pub(crate) q2: Option<GroupGeo<D>>,
    /// Always stored (from the mesh's own map).
    pub(crate) mapped: Option<StoredGeo<D>>,
}

/// Inverse of [`cofactor`]: `J` from `cof J` and `det J`
/// (`cof(cof J) = det(J)^(D−2) · J`).
pub(crate) fn jacobian_from_cofactor<const D: usize>(
    cof: &SMatrix<f64, D, D>,
    det_j: f64,
) -> SMatrix<f64, D, D> {
    let c = cofactor(cof);
    if D == 2 {
        c
    } else {
        c / det_j
    }
}

/// Geometric classification of one element.
pub(crate) enum ElementMap<const D: usize> {
    /// `x(ξ) = x0 + J ξ`.
    Affine(SVector<f64, D>, SMatrix<f64, D, D>),
    /// Bilinear/trilinear: the `2^D` corners in tensor order.
    Q1(Vec<[f64; D]>),
    /// Biquadratic/triquadratic: the `3^D` nodes in tensor order.
    Q2(Vec<[f64; D]>),
    /// Given by the mesh's own map (analytic or deformed), not affine.
    Mapped,
}

/// Classifies an element from its `n1^D` tensor-ordered nodes.
///
/// The affine frame `(x0, J)` is built from corner 0 and its `D` neighbouring
/// corners. Deviations are measured in reference coordinates, `|J⁻¹ (X_k − y_k)|∞`,
/// where `y_k` is the affine image (affine test) or, for Q2, the Q1
/// interpolation of the corners (Q1 test). Panics on a degenerate frame.
/// `q1_at_q2` is the Q1 table at the `3^D` tensor reference nodes of a Q2
/// element (used only when `n1 = 3`).
pub(crate) fn classify<const D: usize>(
    nodes: Vec<[f64; D]>,
    n1: usize,
    q1_at_q2: &GeoTab<D>,
) -> ElementMap<D> {
    let corner = |c: [usize; D]| -> usize {
        let mut k = 0;
        for d in (0..D).rev() {
            k = k * n1 + c[d] * (n1 - 1);
        }
        k
    };
    let x0 = SVector::<f64, D>::from(nodes[0]);
    let jac = SMatrix::<f64, D, D>::from_fn(|a, b| {
        let mut c = [0; D];
        c[b] = 1;
        nodes[corner(c)][a] - x0[a]
    });
    let det_j = det(&jac);
    assert!(
        det_j.abs() > 1e-14 * jac.norm().powi(D as i32),
        "degenerate geometric element (det J = {det_j:e} at corner 0)"
    );
    let inv = cofactor(&jac).transpose() / det_j;
    let deviation = |y: &dyn Fn(usize) -> SVector<f64, D>| -> f64 {
        (0..nodes.len())
            .map(|k| (inv * (SVector::<f64, D>::from(nodes[k]) - y(k))).amax())
            .fold(0.0, f64::max)
    };

    let affine_image = |k: usize| {
        let kd = digits::<D>(k, n1);
        let xi = SVector::<f64, D>::from_fn(|d, _| kd[d] as f64 / (n1 - 1) as f64);
        x0 + jac * xi
    };
    if deviation(&affine_image) <= SHAPE_TOL {
        return ElementMap::Affine(x0, jac);
    }
    if n1 == 2 {
        return ElementMap::Q1(nodes);
    }

    let corners: Vec<[f64; D]> = (0..1usize << D)
        .map(|c| nodes[corner(digits::<D>(c, 2))])
        .collect();
    let q1_image = |k: usize| q1_at_q2.eval(&corners, k).0;
    if deviation(&q1_image) <= SHAPE_TOL {
        ElementMap::Q1(corners)
    } else {
        ElementMap::Q2(nodes)
    }
}
