//! Parameter fields (density, moduli, anisotropy matrices, …) shared by the
//! OndoMathX spaces: the value of a coefficient at a point.
//!
//! The kind of value is the type `T`: a scalar `f64`, a vector `[f64; N]`,
//! a matrix `[[f64; J]; I]` (row `i`, column `j`). The regularity is the
//! field's type, after the C++ `OndoMathX::Field::Regularity`:
//!
//! | Type | C++ regularity | Value at a point |
//! |---|---|---|
//! | [`Constant`] | `Constant` | the same everywhere |
//! | [`PerLabel`] | `PiecewiseConstant` | one value per mesh label |
//! | [`Function`] | `C0` | `f(xyz)` |
//! | [`LabelFunction`] | `PiecewiseC0` | `f(xyz, label)` |
//! | [`PointData`] | `PointData` | one value per point index |
//!
//! Each type's `NEEDS_*` flags tell the space which data of the point to
//! provide ([`Point`]), so a [`Constant`]
//! costs nothing and only a function needs the coordinates (the C++
//! `if constexpr` on the regularity). A field only gives its value: the
//! algebra (scaling, `A·∇u`, …) is the kernels'.

/// The point at which a field is evaluated, as a space provides it: its
/// physical coordinates, the label of its element, and its index (e.g. the
/// DG index of a node). Only the data a field's `NEEDS_*` flags ask for
/// need to be meaningful.
pub trait Point {
    /// Physical coordinates (`D` entries).
    fn xyz(&self) -> &[f64];
    /// Mesh label of the element.
    fn label(&self) -> usize;
    /// Index of the point (the entry of a [`PointData`] table).
    fn index(&self) -> usize;
}

/// A plain [`Point`]: coordinates, label, index.
#[derive(Clone, Copy, Debug)]
pub struct At<'a> {
    pub xyz: &'a [f64],
    pub label: usize,
    pub index: usize,
}

impl Point for At<'_> {
    fn xyz(&self) -> &[f64] {
        self.xyz
    }
    fn label(&self) -> usize {
        self.label
    }
    fn index(&self) -> usize {
        self.index
    }
}

/// A field of values of type `T`, evaluated at points.
pub trait Field<T>: Sync {
    /// Whether [`Field::eval`] reads the coordinates of the point.
    const NEEDS_XYZ: bool = false;
    /// Whether it reads the label.
    const NEEDS_LABEL: bool = false;
    /// Whether it reads the index.
    const NEEDS_INDEX: bool = false;

    /// The value at `p`.
    fn eval(&self, p: &impl Point) -> T;
}

impl<T, F: Field<T>> Field<T> for &F {
    const NEEDS_XYZ: bool = F::NEEDS_XYZ;
    const NEEDS_LABEL: bool = F::NEEDS_LABEL;
    const NEEDS_INDEX: bool = F::NEEDS_INDEX;
    #[inline]
    fn eval(&self, p: &impl Point) -> T {
        (**self).eval(p)
    }
}

/// The same value everywhere.
#[derive(Clone, Copy, Debug)]
pub struct Constant<T>(pub T);

impl<T: Copy + Sync> Field<T> for Constant<T> {
    #[inline]
    fn eval(&self, _p: &impl Point) -> T {
        self.0
    }
}

/// One value per mesh label (piecewise constant). A label without a value
/// panics when evaluated, naming it.
#[derive(Clone, Debug)]
pub struct PerLabel<T>(pub Vec<Option<T>>);

impl<T: Copy> PerLabel<T> {
    /// The table from `(label, value)` pairs.
    pub fn from_pairs(pairs: &[(usize, T)]) -> Self {
        let n = pairs.iter().map(|&(l, _)| l + 1).max().unwrap_or(0);
        let mut table = vec![None; n];
        for &(l, v) in pairs {
            table[l] = Some(v);
        }
        PerLabel(table)
    }
}

impl<T: Copy + Sync> Field<T> for PerLabel<T> {
    const NEEDS_LABEL: bool = true;
    #[inline]
    fn eval(&self, p: &impl Point) -> T {
        let l = p.label();
        self.0
            .get(l)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("PerLabel: no value for label {l}"))
    }
}

/// `f(xyz)` (a continuous field), evaluated on the fly.
#[derive(Clone, Copy, Debug)]
pub struct Function<F>(pub F);

impl<T, F: Fn(&[f64]) -> T + Sync> Field<T> for Function<F> {
    const NEEDS_XYZ: bool = true;
    #[inline]
    fn eval(&self, p: &impl Point) -> T {
        (self.0)(p.xyz())
    }
}

/// `f(xyz, label)` (continuous on each labelled region), evaluated on the
/// fly.
#[derive(Clone, Copy, Debug)]
pub struct LabelFunction<F>(pub F);

impl<T, F: Fn(&[f64], usize) -> T + Sync> Field<T> for LabelFunction<F> {
    const NEEDS_XYZ: bool = true;
    const NEEDS_LABEL: bool = true;
    #[inline]
    fn eval(&self, p: &impl Point) -> T {
        (self.0)(p.xyz(), p.label())
    }
}

/// One value per point index (e.g. sampled once at every DG node).
#[derive(Clone, Debug)]
pub struct PointData<T>(pub Vec<T>);

impl<T: Copy + Sync> Field<T> for PointData<T> {
    const NEEDS_INDEX: bool = true;
    #[inline]
    fn eval(&self, p: &impl Point) -> T {
        self.0[p.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(xyz: &[f64], label: usize, index: usize) -> At<'_> {
        At { xyz, label, index }
    }

    /// The flags of a field, as a tuple.
    fn flags<T, F: Field<T>>(_f: &F) -> (bool, bool, bool) {
        (F::NEEDS_XYZ, F::NEEDS_LABEL, F::NEEDS_INDEX)
    }

    #[test]
    fn scalar_fields_of_every_regularity() {
        let p = at(&[1.0, 2.0, 3.0], 2, 5);
        assert_eq!(Constant(4.0).eval(&p), 4.0);
        assert_eq!(PerLabel::from_pairs(&[(0, 1.0), (2, 7.0)]).eval(&p), 7.0);
        assert_eq!(Function(|x: &[f64]| x[0] + x[2]).eval(&p), 4.0);
        assert_eq!(LabelFunction(|x: &[f64], l: usize| x[1] * l as f64).eval(&p), 4.0);
        assert_eq!(PointData(vec![0.0, 0.0, 0.0, 0.0, 0.0, 9.0]).eval(&p), 9.0);
    }

    /// Vectors and matrices are just other value types.
    #[test]
    fn vector_and_matrix_fields() {
        let p = at(&[0.5, -1.0], 1, 0);
        let a: [[f64; 2]; 2] = [[2.0, 0.5], [0.5, 1.0]];
        assert_eq!(Constant(a).eval(&p), a);
        let per = PerLabel::from_pairs(&[(1, [[1.0, 0.0], [0.0, 3.0]])]);
        assert_eq!(per.eval(&p), [[1.0, 0.0], [0.0, 3.0]]);
        let rot = Function(|x: &[f64]| [[x[0], -x[1]], [x[1], x[0]]]);
        assert_eq!(rot.eval(&p), [[0.5, 1.0], [-1.0, 0.5]]);
        let v = LabelFunction(|x: &[f64], l: usize| [x[0], x[1], l as f64]);
        assert_eq!(v.eval(&p), [0.5, -1.0, 1.0]);
        assert_eq!(PointData(vec![[1.0, 2.0, 3.0]]).eval(&p), [1.0, 2.0, 3.0]);
    }

    /// Each regularity asks only for the point data it reads.
    #[test]
    fn the_flags_follow_the_regularity() {
        let f = |x: &[f64]| x[0];
        let g = |x: &[f64], l: usize| x[0] + l as f64;
        assert_eq!(flags::<f64, _>(&Constant(1.0)), (false, false, false));
        assert_eq!(flags::<f64, _>(&PerLabel::from_pairs(&[(0, 1.0)])), (false, true, false));
        assert_eq!(flags::<f64, _>(&Function(f)), (true, false, false));
        assert_eq!(flags::<f64, _>(&LabelFunction(g)), (true, true, false));
        assert_eq!(flags::<f64, _>(&PointData(vec![1.0])), (false, false, true));
    }

    #[test]
    #[should_panic(expected = "no value for label 1")]
    fn a_missing_label_panics() {
        PerLabel::from_pairs(&[(0, 1.0), (2, 2.0)]).eval(&at(&[0.0], 1, 0));
    }
}
