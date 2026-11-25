//! Scalar parameter fields (density, moduli, …), monomorphized per regularity.
//!
//! A field is evaluated at a GLL point from the kernel input. Its `NEEDS_*`
//! flags tell the engine which inputs to provide, so a [`Constant`] costs
//! nothing and a [`Function`] triggers the computation of `xyz`.

use crate::engine::PointIn;
use crate::shape::Shape;
use crate::space::Space;

/// A scalar field evaluated at GLL points.
pub trait Field<const D: usize>: Sync {
    const NEEDS_XYZ: bool = false;
    const NEEDS_LABEL: bool = false;
    const NEEDS_DOF: bool = false;

    /// Value at the point described by `p` (only the fields whose `NEEDS_*`
    /// flag is set are meaningful).
    fn eval<const C: usize>(&self, p: &PointIn<D, C>) -> f64;
}

/// The same value everywhere.
#[derive(Clone, Copy, Debug)]
pub struct Constant(pub f64);

impl<const D: usize> Field<D> for Constant {
    #[inline]
    fn eval<const C: usize>(&self, _p: &PointIn<D, C>) -> f64 {
        self.0
    }
}

/// One value per mesh label, indexed by the bare `usize` label.
///
/// Labels not given to [`PerLabel::from_pairs`] hold `NaN`, so a material
/// forgotten in the table shows up as `NaN` in the results.
#[derive(Clone, Debug)]
pub struct PerLabel(pub Vec<f64>);

impl PerLabel {
    /// Table from `(label, value)` pairs; missing labels are `NaN`.
    pub fn from_pairs(pairs: &[(usize, f64)]) -> Self {
        let n = pairs.iter().map(|&(l, _)| l + 1).max().unwrap_or(0);
        let mut table = vec![f64::NAN; n];
        for &(l, v) in pairs {
            table[l] = v;
        }
        PerLabel(table)
    }
}

impl<const D: usize> Field<D> for PerLabel {
    const NEEDS_LABEL: bool = true;
    #[inline]
    fn eval<const C: usize>(&self, p: &PointIn<D, C>) -> f64 {
        self.0[p.label]
    }
}

/// `f(params, xyz)`, evaluated on the fly (same argument order as `ondomathx-mesh`'s
/// `Diffeomorphism`). `xyz` has `D` entries, recomputed from the element map.
#[derive(Clone, Debug)]
pub struct Function {
    pub f: fn(&[f64], &[f64]) -> f64,
    pub params: Vec<f64>,
}

impl<const D: usize> Field<D> for Function {
    const NEEDS_XYZ: bool = true;
    #[inline]
    fn eval<const C: usize>(&self, p: &PointIn<D, C>) -> f64 {
        (self.f)(&self.params, p.xyz.as_slice())
    }
}

/// One value per duplicated node (`n_dup` entries, indexed by `e·NLOC + node`).
#[derive(Clone, Debug)]
pub struct PointData(pub Vec<f64>);

impl PointData {
    /// Samples `f(xyz)` once at every duplicated node of `space`.
    pub fn sample<const D: usize, S: Shape<D>>(
        space: &Space<D, S>,
        f: impl Fn([f64; D]) -> f64 + Sync,
    ) -> Self {
        PointData(space.sample::<1>(|x| [f(x)]))
    }
}

impl<const D: usize> Field<D> for PointData {
    const NEEDS_DOF: bool = true;
    #[inline]
    fn eval<const C: usize>(&self, p: &PointIn<D, C>) -> f64 {
        self.0[p.dof]
    }
}

/// `1 / F`, e.g. the acoustic mass coefficient `1/κ` from a bulk modulus field.
#[derive(Clone, Debug)]
pub struct Inverse<F>(pub F);

impl<const D: usize, F: Field<D>> Field<D> for Inverse<F> {
    const NEEDS_XYZ: bool = F::NEEDS_XYZ;
    const NEEDS_LABEL: bool = F::NEEDS_LABEL;
    const NEEDS_DOF: bool = F::NEEDS_DOF;
    #[inline]
    fn eval<const C: usize>(&self, p: &PointIn<D, C>) -> f64 {
        1.0 / self.0.eval(p)
    }
}
