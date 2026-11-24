//! Shape markers: the GLL point counts per direction of a space, as types.
//!
//! `Space<D, S>` takes a marker `S: Shape<D>`: [`Gll2<N0, N1>`] (2D),
//! [`Gll3<N0, N1, N2>`] (3D), or the isotropic aliases [`Iso2<N>`] /
//! [`Iso3<N>`]. Counts may differ per direction (anisotropic spaces).
//!
//! On stable Rust an associated const of a marker cannot be a const argument
//! (`kernel::<{ S::N0 }>()` is rejected), whereas inside
//! `impl<const N0, …> Shape<3> for Gll3<N0, …>` the counts are ordinary const
//! parameters. So everything that needs the counts as consts is a method of
//! the marker: the 1D tables (fixed-size arrays) and the reference derivative
//! kernels. Callers only see `usize` associated consts and slices.
//!
//! The 1D data comes from `lobatto::tables`; nothing is computed here.

use lobatto::collocation::Gauss;
use lobatto::collocation_hc::*;
use lobatto::tables::{derivative_matrix, points_weights};

/// GLL point counts per direction of a space (`N[d] ≥ 2` for `d < D`).
pub trait Shape<const D: usize>: Send + Sync + 'static {
    /// Points per direction; entries `d ≥ D` are 1.
    const N: [usize; 3];
    /// Nodes per element and component, `N[0]·…·N[D−1]`.
    const NLOC: usize;
    /// Whether all directions have the same count.
    const ISOTROPIC: bool;
    /// Per-direction 1D data (GLL points, weights, differentiation matrices).
    type Tables: Send + Sync;

    /// Builds the tables from `lobatto::tables`.
    fn tables() -> Self::Tables;
    /// GLL points on `[0, 1]` of direction `d`.
    fn points_1d(t: &Self::Tables, d: usize) -> &[f64];
    /// GLL weights of direction `d` (they sum to 1).
    fn weights_1d(t: &Self::Tables, d: usize) -> &[f64];
    /// Reference derivative `∂/∂ξ_dir` (or its transpose) on a batch of `W`
    /// elements in lobatto's batched layout (node `g`, lane `l` at `g·W + l`);
    /// `output` is overwritten.
    fn ref_derivative<const W: usize>(
        t: &Self::Tables,
        dir: usize,
        transpose: bool,
        input: &[f64],
        output: &mut [f64],
    );
}

/// GLL data of one direction with `N` points.
#[derive(Clone, Debug)]
pub struct Gll1<const N: usize> {
    /// Points on `[0, 1]`, increasing.
    pub xi: [f64; N],
    /// Weights.
    pub w: [f64; N],
    /// Differentiation matrix, `d[j][i] = L_i'(xi[j])`.
    pub d: [[f64; N]; N],
}

impl<const N: usize> Gll1<N> {
    fn new() -> Self {
        let (xi, w) = points_weights::<N>(Gauss::Lobatto);
        Gll1 {
            xi,
            w,
            d: derivative_matrix::<N, N>(Gauss::Lobatto, Gauss::Lobatto),
        }
    }
}

/// 2D shape: `N0 × N1` GLL points.
pub struct Gll2<const N0: usize, const N1: usize>;
/// 3D shape: `N0 × N1 × N2` GLL points.
pub struct Gll3<const N0: usize, const N1: usize, const N2: usize>;
/// Isotropic 2D shape, `N` points per direction (degree `N − 1`).
pub type Iso2<const N: usize> = Gll2<N, N>;
/// Isotropic 3D shape, `N` points per direction (degree `N − 1`).
pub type Iso3<const N: usize> = Gll3<N, N, N>;

/// Applies the hardcoded lobatto kernel `$f` of each `(dir, transpose)` case.
macro_rules! hc {
    ($dir:ident, $t:ident, $input:ident, $output:ident, $w:ident; $($d:literal, $tr:literal => $f:ident),* $(,)?) => {
        match ($dir, $t) {
            $(($d, $tr) => $f::<$w>($input, $output),)*
            _ => unreachable!(),
        }
    };
}

impl<const N0: usize, const N1: usize> Shape<2> for Gll2<N0, N1> {
    const N: [usize; 3] = [N0, N1, 1];
    const NLOC: usize = N0 * N1;
    const ISOTROPIC: bool = N0 == N1;
    type Tables = (Gll1<N0>, Gll1<N1>);

    fn tables() -> Self::Tables {
        (Gll1::new(), Gll1::new())
    }

    fn points_1d(t: &Self::Tables, d: usize) -> &[f64] {
        match d {
            0 => &t.0.xi,
            1 => &t.1.xi,
            _ => panic!("direction {d} out of range in 2D"),
        }
    }

    fn weights_1d(t: &Self::Tables, d: usize) -> &[f64] {
        match d {
            0 => &t.0.w,
            1 => &t.1.w,
            _ => panic!("direction {d} out of range in 2D"),
        }
    }

    /// Hardcoded lobatto kernels for isotropic `N = 4..7`, generic rectangular
    /// kernels (square matrix per direction) otherwise. The choice folds at
    /// compile time.
    fn ref_derivative<const W: usize>(
        t: &Self::Tables,
        dir: usize,
        transpose: bool,
        input: &[f64],
        output: &mut [f64],
    ) {
        if N0 == N1 {
            match N0 {
                4 => {
                    return hc!(dir, transpose, input, output, W;
                    0, false => dx_q33_q33, 1, false => dy_q33_q33,
                    0, true => dx_t_q33_q33, 1, true => dy_t_q33_q33)
                }
                5 => {
                    return hc!(dir, transpose, input, output, W;
                    0, false => dx_q44_q44, 1, false => dy_q44_q44,
                    0, true => dx_t_q44_q44, 1, true => dy_t_q44_q44)
                }
                6 => {
                    return hc!(dir, transpose, input, output, W;
                    0, false => dx_q55_q55, 1, false => dy_q55_q55,
                    0, true => dx_t_q55_q55, 1, true => dy_t_q55_q55)
                }
                7 => {
                    return hc!(dir, transpose, input, output, W;
                    0, false => dx_q66_q66, 1, false => dy_q66_q66,
                    0, true => dx_t_q66_q66, 1, true => dy_t_q66_q66)
                }
                _ => {}
            }
        }
        match (dir, transpose) {
            (0, false) => dx_qnn_qmn::<N0, N0, N1, W>(&t.0.d, input, output),
            (1, false) => dy_qnn_qnm::<N1, N1, N0, W>(&t.1.d, input, output),
            (0, true) => dx_t_qmn_qnn::<N0, N0, N1, W>(&t.0.d, input, output),
            (1, true) => dy_t_qnm_qnn::<N1, N1, N0, W>(&t.1.d, input, output),
            _ => unreachable!(),
        }
    }
}

impl<const N0: usize, const N1: usize, const N2: usize> Shape<3> for Gll3<N0, N1, N2> {
    const N: [usize; 3] = [N0, N1, N2];
    const NLOC: usize = N0 * N1 * N2;
    const ISOTROPIC: bool = N0 == N1 && N1 == N2;
    type Tables = (Gll1<N0>, Gll1<N1>, Gll1<N2>);

    fn tables() -> Self::Tables {
        (Gll1::new(), Gll1::new(), Gll1::new())
    }

    fn points_1d(t: &Self::Tables, d: usize) -> &[f64] {
        match d {
            0 => &t.0.xi,
            1 => &t.1.xi,
            2 => &t.2.xi,
            _ => panic!("direction {d} out of range in 3D"),
        }
    }

    fn weights_1d(t: &Self::Tables, d: usize) -> &[f64] {
        match d {
            0 => &t.0.w,
            1 => &t.1.w,
            2 => &t.2.w,
            _ => panic!("direction {d} out of range in 3D"),
        }
    }

    /// Hardcoded lobatto kernels for isotropic `N = 4..7`, generic rectangular
    /// kernels (square matrix per direction) otherwise. The choice folds at
    /// compile time.
    fn ref_derivative<const W: usize>(
        t: &Self::Tables,
        dir: usize,
        transpose: bool,
        input: &[f64],
        output: &mut [f64],
    ) {
        if N0 == N1 && N1 == N2 {
            match N0 {
                4 => {
                    return hc!(dir, transpose, input, output, W;
                    0, false => dx_q333_q333, 1, false => dy_q333_q333, 2, false => dz_q333_q333,
                    0, true => dx_t_q333_q333, 1, true => dy_t_q333_q333, 2, true => dz_t_q333_q333)
                }
                5 => {
                    return hc!(dir, transpose, input, output, W;
                    0, false => dx_q444_q444, 1, false => dy_q444_q444, 2, false => dz_q444_q444,
                    0, true => dx_t_q444_q444, 1, true => dy_t_q444_q444, 2, true => dz_t_q444_q444)
                }
                6 => {
                    return hc!(dir, transpose, input, output, W;
                    0, false => dx_q555_q555, 1, false => dy_q555_q555, 2, false => dz_q555_q555,
                    0, true => dx_t_q555_q555, 1, true => dy_t_q555_q555, 2, true => dz_t_q555_q555)
                }
                7 => {
                    return hc!(dir, transpose, input, output, W;
                    0, false => dx_q666_q666, 1, false => dy_q666_q666, 2, false => dz_q666_q666,
                    0, true => dx_t_q666_q666, 1, true => dy_t_q666_q666, 2, true => dz_t_q666_q666)
                }
                _ => {}
            }
        }
        match (dir, transpose) {
            (0, false) => dx_qnnn_qmnn::<N0, N0, N1, N2, W>(&t.0.d, input, output),
            (1, false) => dy_qnnn_qnmn::<N1, N1, N0, N2, W>(&t.1.d, input, output),
            (2, false) => dz_qnnn_qnnm::<N2, N2, N0, N1, W>(&t.2.d, input, output),
            (0, true) => dx_t_qmnn_qnnn::<N0, N0, N1, N2, W>(&t.0.d, input, output),
            (1, true) => dy_t_qnmn_qnnn::<N1, N1, N0, N2, W>(&t.1.d, input, output),
            (2, true) => dz_t_qnnm_qnnn::<N2, N2, N0, N1, W>(&t.2.d, input, output),
            _ => unreachable!(),
        }
    }
}
