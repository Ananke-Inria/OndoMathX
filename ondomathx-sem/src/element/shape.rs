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
//! the marker (the 1D tables, as fixed-size arrays) or of a (quadrature,
//! shape) pair ([`Sweep`]: interpolations and reference derivatives, on the
//! node rule [`Shape::Nodes`] for the same-space products). Callers only see
//! `usize` associated consts and slices.
//!
//! The 1D data comes from `lobatto::tables`; nothing is computed here.

use lobatto::collocation::Gauss;
use lobatto::tables::points_weights;

use crate::element::quadrature::{Gll, Quad2, Quad3, Sweep};

/// GLL point counts per direction of a space (`N[d] ≥ 2` for `d < D`).
pub trait Shape<const D: usize>: Sized + Send + Sync + 'static {
    /// Points per direction; entries `d ≥ D` are 1.
    const N: [usize; 3];
    /// Nodes per element and component, `N[0]·…·N[D−1]`.
    const NLOC: usize;
    /// Whether all directions have the same count.
    const ISOTROPIC: bool;
    /// Per-direction 1D data (GLL points and weights).
    type Tables: Send + Sync;
    /// The rule of the nodes themselves (Gauss–Lobatto, the shape's counts):
    /// the quadrature of the same-space products.
    type Nodes: Sweep<D, Self>;

    /// Builds the tables from `lobatto::tables`.
    fn tables() -> Self::Tables;
    /// GLL points on `[0, 1]` of direction `d`.
    fn points_1d(t: &Self::Tables, d: usize) -> &[f64];
    /// GLL weights of direction `d` (they sum to 1).
    fn weights_1d(t: &Self::Tables, d: usize) -> &[f64];
}

/// GLL data of one direction with `N` points.
#[derive(Clone, Debug)]
pub struct Gll1<const N: usize> {
    /// Points on `[0, 1]`, increasing.
    pub xi: [f64; N],
    /// Weights.
    pub w: [f64; N],
}

impl<const N: usize> Gll1<N> {
    fn new() -> Self {
        let (xi, w) = points_weights::<N>(Gauss::Lobatto);
        Gll1 { xi, w }
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

impl<const N0: usize, const N1: usize> Shape<2> for Gll2<N0, N1> {
    const N: [usize; 3] = [N0, N1, 1];
    const NLOC: usize = N0 * N1;
    const ISOTROPIC: bool = N0 == N1;
    type Tables = (Gll1<N0>, Gll1<N1>);
    type Nodes = Quad2<Gll<N0>, Gll<N1>>;

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
}

impl<const N0: usize, const N1: usize, const N2: usize> Shape<3> for Gll3<N0, N1, N2> {
    const N: [usize; 3] = [N0, N1, N2];
    const NLOC: usize = N0 * N1 * N2;
    const ISOTROPIC: bool = N0 == N1 && N1 == N2;
    type Tables = (Gll1<N0>, Gll1<N1>, Gll1<N2>);
    type Nodes = Quad3<Gll<N0>, Gll<N1>, Gll<N2>>;

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
}
