//! Quadrature markers: Gauss–Legendre or Gauss–Lobatto rules, family and point
//! count per direction, as types.
//!
//! A rule of one direction is [`Gl<M>`] or [`Gll<M>`] (aliases of
//! [`Rule<F, M>`]); a tensor rule is [`Quad2<A, B>`] or [`Quad3<A, B, C>`],
//! e.g. `Quad3<Gl<3>, Gl<3>, Gll<5>>`. A space declares its quadratures at
//! construction ([`crate::SpaceBuilder::quadrature`]) and operators name them
//! by type.
//!
//! Same stable-Rust rule as the shape markers: counts are const parameters of
//! the implementations (`impl … for Quad3<Rule<F0, M0>, …>`), so everything
//! that needs them as consts is a method of the (quadrature, shape) pair
//! ([`Sweep`]). The 1D data comes from `lobatto::tables`.

use std::marker::PhantomData;

use lobatto::collocation::Gauss;
use lobatto::collocation_hc::*;
use lobatto::tables::{derivative_matrix, interpolation_matrix};

use crate::shape::{Gll2, Gll3, Shape};

/// Point family of a 1D rule.
pub trait Family: Send + Sync + 'static {
    const GAUSS: Gauss;
}

/// Gauss–Legendre points (open).
pub struct Legendre;
/// Gauss–Lobatto points (with the endpoints).
pub struct Lobatto;

impl Family for Legendre {
    const GAUSS: Gauss = Gauss::Legendre;
}
impl Family for Lobatto {
    const GAUSS: Gauss = Gauss::Lobatto;
}

/// A 1D rule: `M` points of family `F` on `[0, 1]`.
pub struct Rule<F, const M: usize>(PhantomData<F>);
/// `M`-point Gauss–Legendre rule (exact to degree `2M − 1`).
pub type Gl<const M: usize> = Rule<Legendre, M>;
/// `M`-point Gauss–Lobatto rule (exact to degree `2M − 3`).
pub type Gll<const M: usize> = Rule<Lobatto, M>;

/// 2D tensor rule.
pub struct Quad2<A, B>(PhantomData<(A, B)>);
/// 3D tensor rule.
pub struct Quad3<A, B, C>(PhantomData<(A, B, C)>);

/// A tensor quadrature rule in dimension `D`.
pub trait Quadrature<const D: usize>: Send + Sync + 'static {
    /// Points per direction; entries `d ≥ D` are 1.
    const M: [usize; 3];
    /// Points per element, `M[0]·…·M[D−1]`.
    const NQ: usize;
    /// Family per direction (entries `d ≥ D` unused).
    fn families() -> [Gauss; 3];
}

impl<F0: Family, F1: Family, const M0: usize, const M1: usize> Quadrature<2>
    for Quad2<Rule<F0, M0>, Rule<F1, M1>>
{
    const M: [usize; 3] = [M0, M1, 1];
    const NQ: usize = M0 * M1;
    fn families() -> [Gauss; 3] {
        [F0::GAUSS, F1::GAUSS, Gauss::Lobatto]
    }
}

impl<F0: Family, F1: Family, F2: Family, const M0: usize, const M1: usize, const M2: usize>
    Quadrature<3> for Quad3<Rule<F0, M0>, Rule<F1, M1>, Rule<F2, M2>>
{
    const M: [usize; 3] = [M0, M1, M2];
    const NQ: usize = M0 * M1 * M2;
    fn families() -> [Gauss; 3] {
        [F0::GAUSS, F1::GAUSS, F2::GAUSS]
    }
}

/// Node → point operators between the GLL nodes of the shape `S` and the
/// points of the quadrature, by sum factorization (one lobatto rectangular
/// sweep per direction).
///
/// With `I` the interpolation (nodes → points), `G_b` the same with the
/// derivative matrix in direction `b`, and `B` the interpolation back (points
/// → nodes, the Lagrange basis on the quadrature points; `B = I⁻¹` when the
/// counts match): `interpolate` = `I`, `interpolate_t` = `Iᵀ`, `gradient` =
/// `G_b`, `gradient_t` = `G_bᵀ`, `back` = `B`, `back_t` = `Bᵀ`. Layouts are
/// lobatto's batched `[point][lane]`; outputs are overwritten; `tmp` holds at
/// least `TMP · W` entries.
///
/// A direction whose rule is the shape's own nodes (Lobatto, `M = N`) has an
/// exact identity interpolation: its pass is skipped (bitwise the same
/// result), so on the node rule `gradient(b)` is the single `D_b` pass.
pub trait Sweep<const D: usize, S: Shape<D>>: Quadrature<D> {
    /// 1D matrices (fixed-size arrays): interpolation, derivative, back.
    type Tables: Send + Sync + 'static;
    /// Scratch per lane.
    const TMP: usize;

    /// Builds the tables from `lobatto::tables`.
    fn tables() -> Self::Tables;
    /// Nodes → points: values.
    fn interpolate<const W: usize>(
        t: &Self::Tables,
        input: &[f64],
        output: &mut [f64],
        tmp: &mut [f64],
    );
    /// Points → nodes: transpose of [`Sweep::interpolate`].
    fn interpolate_t<const W: usize>(
        t: &Self::Tables,
        input: &[f64],
        output: &mut [f64],
        tmp: &mut [f64],
    );
    /// Nodes → points: reference derivative `∂/∂ξ_b`.
    fn gradient<const W: usize>(
        t: &Self::Tables,
        b: usize,
        input: &[f64],
        output: &mut [f64],
        tmp: &mut [f64],
    );
    /// Points → nodes: transpose of [`Sweep::gradient`].
    fn gradient_t<const W: usize>(
        t: &Self::Tables,
        b: usize,
        input: &[f64],
        output: &mut [f64],
        tmp: &mut [f64],
    );
    /// Points → nodes: interpolation back.
    fn back<const W: usize>(t: &Self::Tables, input: &[f64], output: &mut [f64], tmp: &mut [f64]);
    /// Nodes → points: transpose of [`Sweep::back`].
    fn back_t<const W: usize>(t: &Self::Tables, input: &[f64], output: &mut [f64], tmp: &mut [f64]);
}

/// 1D matrices of one direction: `N` nodes, `M` points.
pub struct Tables1<const N: usize, const M: usize> {
    /// Interpolation `L_i(y_j)`, nodes → points.
    pub interp: [[f64; N]; M],
    /// Derivative `L_i'(y_j)`, nodes → points.
    pub deriv: [[f64; N]; M],
    /// Interpolation back `ℓ_j(ξ_i)` (Lagrange basis on the points), points → nodes.
    pub back: [[f64; M]; N],
}

impl<const N: usize, const M: usize> Tables1<N, M> {
    fn new(family: Gauss) -> Self {
        Tables1 {
            interp: interpolation_matrix::<N, M>(Gauss::Lobatto, family),
            deriv: derivative_matrix::<N, M>(Gauss::Lobatto, family),
            back: interpolation_matrix::<M, N>(family, Gauss::Lobatto),
        }
    }
}

/// Whether the 1D interpolation of direction `N` nodes → `M` points of `F` is
/// the identity: lobatto returns it exactly (unit rows) when the points are
/// the nodes, so its pass can be skipped (bitwise the same result). Constant
/// after monomorphization: the branches on it fold.
#[inline(always)]
fn identity<F: Family>(n: usize, m: usize) -> bool {
    matches!(F::GAUSS, Gauss::Lobatto) && n == m
}

#[inline(always)]
fn id2<F0: Family, F1: Family>(n0: usize, m0: usize, n1: usize, m1: usize) -> [bool; 2] {
    [identity::<F0>(n0, m0), identity::<F1>(n1, m1)]
}

#[inline(always)]
fn id3<F0: Family, F1: Family, F2: Family>(n: [usize; 3], m: [usize; 3]) -> [bool; 3] {
    [
        identity::<F0>(n[0], m[0]),
        identity::<F1>(n[1], m[1]),
        identity::<F2>(n[2], m[2]),
    ]
}

/// `(A0, A1) → (B0, A1) → (B0, B1)`; a pass with `skip[d]` (its matrix is the
/// identity, `A_d = B_d`) is not run.
#[inline(always)]
fn fwd2<const A0: usize, const B0: usize, const A1: usize, const B1: usize, const W: usize>(
    m0: &[[f64; A0]; B0],
    m1: &[[f64; A1]; B1],
    skip: [bool; 2],
    input: &[f64],
    output: &mut [f64],
    tmp: &mut [f64],
) {
    match skip {
        [false, false] => {
            let t = &mut tmp[..B0 * A1 * W];
            dx_qnn_qmn::<A0, B0, A1, W>(m0, input, t);
            dy_qnn_qnm::<A1, B1, B0, W>(m1, t, output);
        }
        [false, true] => dx_qnn_qmn::<A0, B0, A1, W>(m0, input, output),
        [true, false] => dy_qnn_qnm::<A1, B1, B0, W>(m1, input, output),
        [true, true] => {
            let n = A0 * A1 * W;
            output[..n].copy_from_slice(&input[..n]);
        }
    }
}

/// Transpose of [`fwd2`]: `(B0, B1) → (B0, A1) → (A0, A1)`.
#[inline(always)]
fn trans2<const A0: usize, const B0: usize, const A1: usize, const B1: usize, const W: usize>(
    m0: &[[f64; A0]; B0],
    m1: &[[f64; A1]; B1],
    skip: [bool; 2],
    input: &[f64],
    output: &mut [f64],
    tmp: &mut [f64],
) {
    match skip {
        [false, false] => {
            let t = &mut tmp[..B0 * A1 * W];
            dy_t_qnm_qnn::<A1, B1, B0, W>(m1, input, t);
            dx_t_qmn_qnn::<A0, B0, A1, W>(m0, t, output);
        }
        [false, true] => dx_t_qmn_qnn::<A0, B0, A1, W>(m0, input, output),
        [true, false] => dy_t_qnm_qnn::<A1, B1, B0, W>(m1, input, output),
        [true, true] => {
            let n = B0 * B1 * W;
            output[..n].copy_from_slice(&input[..n]);
        }
    }
}

/// `(A0, A1, A2) → (B0, A1, A2) → (B0, B1, A2) → (B0, B1, B2)`; a pass with
/// `skip[d]` is not run (the next one reads what the previous one read).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fwd3<
    const A0: usize,
    const B0: usize,
    const A1: usize,
    const B1: usize,
    const A2: usize,
    const B2: usize,
    const W: usize,
>(
    m0: &[[f64; A0]; B0],
    m1: &[[f64; A1]; B1],
    m2: &[[f64; A2]; B2],
    skip: [bool; 3],
    input: &[f64],
    output: &mut [f64],
    tmp: &mut [f64],
) {
    let (t1, t2) = tmp.split_at_mut(B0 * A1 * A2 * W);
    let t2 = &mut t2[..B0 * B1 * A2 * W];
    // The last pass run writes `output`; the others write t1 (x), t2 (y).
    let last = match skip {
        [_, _, false] => 2,
        [_, false, true] => 1,
        [false, true, true] => 0,
        [true, true, true] => {
            let n = A0 * A1 * A2 * W;
            output[..n].copy_from_slice(&input[..n]);
            return;
        }
    };
    let mut cur: &[f64] = input;
    if !skip[0] {
        if last == 0 {
            dx_qnnn_qmnn::<A0, B0, A1, A2, W>(m0, cur, output);
            return;
        }
        dx_qnnn_qmnn::<A0, B0, A1, A2, W>(m0, cur, t1);
        cur = t1;
    }
    if !skip[1] {
        if last == 1 {
            dy_qnnn_qnmn::<A1, B1, B0, A2, W>(m1, cur, output);
            return;
        }
        dy_qnnn_qnmn::<A1, B1, B0, A2, W>(m1, cur, t2);
        cur = t2;
    }
    dz_qnnn_qnnm::<A2, B2, B0, B1, W>(m2, cur, output);
}

/// Transpose of [`fwd3`]: `(B0, B1, B2) → (B0, B1, A2) → (B0, A1, A2) → (A0, A1, A2)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn trans3<
    const A0: usize,
    const B0: usize,
    const A1: usize,
    const B1: usize,
    const A2: usize,
    const B2: usize,
    const W: usize,
>(
    m0: &[[f64; A0]; B0],
    m1: &[[f64; A1]; B1],
    m2: &[[f64; A2]; B2],
    skip: [bool; 3],
    input: &[f64],
    output: &mut [f64],
    tmp: &mut [f64],
) {
    let (t1, t2) = tmp.split_at_mut(B0 * A1 * A2 * W);
    let t2 = &mut t2[..B0 * B1 * A2 * W];
    // The last pass run writes `output`; the others write t2 (z), t1 (y).
    let last = match skip {
        [false, _, _] => 0,
        [true, false, _] => 1,
        [true, true, false] => 2,
        [true, true, true] => {
            let n = B0 * B1 * B2 * W;
            output[..n].copy_from_slice(&input[..n]);
            return;
        }
    };
    let mut cur: &[f64] = input;
    if !skip[2] {
        if last == 2 {
            dz_t_qnnm_qnnn::<A2, B2, B0, B1, W>(m2, cur, output);
            return;
        }
        dz_t_qnnm_qnnn::<A2, B2, B0, B1, W>(m2, cur, t2);
        cur = t2;
    }
    if !skip[1] {
        if last == 1 {
            dy_t_qnmn_qnnn::<A1, B1, B0, A2, W>(m1, cur, output);
            return;
        }
        dy_t_qnmn_qnnn::<A1, B1, B0, A2, W>(m1, cur, t1);
        cur = t1;
    }
    dx_t_qmnn_qnnn::<A0, B0, A1, A2, W>(m0, cur, output);
}

const fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

impl<
        F0: Family,
        F1: Family,
        const N0: usize,
        const N1: usize,
        const M0: usize,
        const M1: usize,
    > Sweep<2, Gll2<N0, N1>> for Quad2<Rule<F0, M0>, Rule<F1, M1>>
{
    type Tables = (Tables1<N0, M0>, Tables1<N1, M1>);
    const TMP: usize = max(M0 * N1, N0 * M1);

    fn tables() -> Self::Tables {
        (Tables1::new(F0::GAUSS), Tables1::new(F1::GAUSS))
    }
    fn interpolate<const W: usize>(t: &Self::Tables, i: &[f64], o: &mut [f64], tmp: &mut [f64]) {
        fwd2::<N0, M0, N1, M1, W>(
            &t.0.interp,
            &t.1.interp,
            id2::<F0, F1>(N0, M0, N1, M1),
            i,
            o,
            tmp,
        );
    }
    fn interpolate_t<const W: usize>(t: &Self::Tables, i: &[f64], o: &mut [f64], tmp: &mut [f64]) {
        trans2::<N0, M0, N1, M1, W>(
            &t.0.interp,
            &t.1.interp,
            id2::<F0, F1>(N0, M0, N1, M1),
            i,
            o,
            tmp,
        );
    }
    fn gradient<const W: usize>(
        t: &Self::Tables,
        b: usize,
        i: &[f64],
        o: &mut [f64],
        tmp: &mut [f64],
    ) {
        let [s0, s1] = id2::<F0, F1>(N0, M0, N1, M1);
        match b {
            0 => fwd2::<N0, M0, N1, M1, W>(&t.0.deriv, &t.1.interp, [false, s1], i, o, tmp),
            _ => fwd2::<N0, M0, N1, M1, W>(&t.0.interp, &t.1.deriv, [s0, false], i, o, tmp),
        }
    }
    fn gradient_t<const W: usize>(
        t: &Self::Tables,
        b: usize,
        i: &[f64],
        o: &mut [f64],
        tmp: &mut [f64],
    ) {
        let [s0, s1] = id2::<F0, F1>(N0, M0, N1, M1);
        match b {
            0 => trans2::<N0, M0, N1, M1, W>(&t.0.deriv, &t.1.interp, [false, s1], i, o, tmp),
            _ => trans2::<N0, M0, N1, M1, W>(&t.0.interp, &t.1.deriv, [s0, false], i, o, tmp),
        }
    }
    fn back<const W: usize>(t: &Self::Tables, i: &[f64], o: &mut [f64], tmp: &mut [f64]) {
        fwd2::<M0, N0, M1, N1, W>(
            &t.0.back,
            &t.1.back,
            id2::<F0, F1>(N0, M0, N1, M1),
            i,
            o,
            tmp,
        );
    }
    fn back_t<const W: usize>(t: &Self::Tables, i: &[f64], o: &mut [f64], tmp: &mut [f64]) {
        trans2::<M0, N0, M1, N1, W>(
            &t.0.back,
            &t.1.back,
            id2::<F0, F1>(N0, M0, N1, M1),
            i,
            o,
            tmp,
        );
    }
}

impl<
        F0: Family,
        F1: Family,
        F2: Family,
        const N0: usize,
        const N1: usize,
        const N2: usize,
        const M0: usize,
        const M1: usize,
        const M2: usize,
    > Sweep<3, Gll3<N0, N1, N2>> for Quad3<Rule<F0, M0>, Rule<F1, M1>, Rule<F2, M2>>
{
    type Tables = (Tables1<N0, M0>, Tables1<N1, M1>, Tables1<N2, M2>);
    const TMP: usize = max(M0 * N1 * N2 + M0 * M1 * N2, N0 * M1 * M2 + N0 * N1 * M2);

    fn tables() -> Self::Tables {
        (
            Tables1::new(F0::GAUSS),
            Tables1::new(F1::GAUSS),
            Tables1::new(F2::GAUSS),
        )
    }
    fn interpolate<const W: usize>(t: &Self::Tables, i: &[f64], o: &mut [f64], tmp: &mut [f64]) {
        let id = id3::<F0, F1, F2>([N0, N1, N2], [M0, M1, M2]);
        fwd3::<N0, M0, N1, M1, N2, M2, W>(&t.0.interp, &t.1.interp, &t.2.interp, id, i, o, tmp);
    }
    fn interpolate_t<const W: usize>(t: &Self::Tables, i: &[f64], o: &mut [f64], tmp: &mut [f64]) {
        let id = id3::<F0, F1, F2>([N0, N1, N2], [M0, M1, M2]);
        trans3::<N0, M0, N1, M1, N2, M2, W>(&t.0.interp, &t.1.interp, &t.2.interp, id, i, o, tmp);
    }
    fn gradient<const W: usize>(
        t: &Self::Tables,
        b: usize,
        i: &[f64],
        o: &mut [f64],
        tmp: &mut [f64],
    ) {
        let [s0, s1, s2] = id3::<F0, F1, F2>([N0, N1, N2], [M0, M1, M2]);
        match b {
            0 => fwd3::<N0, M0, N1, M1, N2, M2, W>(
                &t.0.deriv,
                &t.1.interp,
                &t.2.interp,
                [false, s1, s2],
                i,
                o,
                tmp,
            ),
            1 => fwd3::<N0, M0, N1, M1, N2, M2, W>(
                &t.0.interp,
                &t.1.deriv,
                &t.2.interp,
                [s0, false, s2],
                i,
                o,
                tmp,
            ),
            _ => fwd3::<N0, M0, N1, M1, N2, M2, W>(
                &t.0.interp,
                &t.1.interp,
                &t.2.deriv,
                [s0, s1, false],
                i,
                o,
                tmp,
            ),
        }
    }
    fn gradient_t<const W: usize>(
        t: &Self::Tables,
        b: usize,
        i: &[f64],
        o: &mut [f64],
        tmp: &mut [f64],
    ) {
        let [s0, s1, s2] = id3::<F0, F1, F2>([N0, N1, N2], [M0, M1, M2]);
        match b {
            0 => trans3::<N0, M0, N1, M1, N2, M2, W>(
                &t.0.deriv,
                &t.1.interp,
                &t.2.interp,
                [false, s1, s2],
                i,
                o,
                tmp,
            ),
            1 => trans3::<N0, M0, N1, M1, N2, M2, W>(
                &t.0.interp,
                &t.1.deriv,
                &t.2.interp,
                [s0, false, s2],
                i,
                o,
                tmp,
            ),
            _ => trans3::<N0, M0, N1, M1, N2, M2, W>(
                &t.0.interp,
                &t.1.interp,
                &t.2.deriv,
                [s0, s1, false],
                i,
                o,
                tmp,
            ),
        }
    }
    fn back<const W: usize>(t: &Self::Tables, i: &[f64], o: &mut [f64], tmp: &mut [f64]) {
        let id = id3::<F0, F1, F2>([N0, N1, N2], [M0, M1, M2]);
        fwd3::<M0, N0, M1, N1, M2, N2, W>(&t.0.back, &t.1.back, &t.2.back, id, i, o, tmp);
    }
    fn back_t<const W: usize>(t: &Self::Tables, i: &[f64], o: &mut [f64], tmp: &mut [f64]) {
        let id = id3::<F0, F1, F2>([N0, N1, N2], [M0, M1, M2]);
        trans3::<M0, N0, M1, N1, M2, N2, W>(&t.0.back, &t.1.back, &t.2.back, id, i, o, tmp);
    }
}
