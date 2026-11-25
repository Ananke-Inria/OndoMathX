//! Geometry evaluation: `(x, det J, cof J)` at a point of a point set, by
//! geometric kind.

use nalgebra::{SMatrix, SVector};

use crate::geometry::{cofactor, det, jacobian_from_cofactor, GroupGeo, PointGeo};
use crate::shape::Shape;

use super::{GeoKind, Space};

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// Image `x` and Jacobian `J[a][b] = ∂x_a/∂u_b` at local node `node` of
    /// storage element `e` (from the stored cofactor when the node geometry
    /// is stored).
    pub fn map_and_jacobian(&self, e: usize, node: usize) -> (SVector<f64, D>, SMatrix<f64, D, D>) {
        let kind = self.kind(e);
        let from_stored = |(x, det_j, cof): (SVector<f64, D>, f64, SMatrix<f64, D, D>)| {
            (x, jacobian_from_cofactor(&cof, det_j))
        };
        match kind {
            GeoKind::Affine => {
                let (x0, j) = &self.affine[e - self.ranges.affine.start];
                (x0 + j * SVector::<f64, D>::from(self.ref_point(node)), *j)
            }
            GeoKind::Mapped => from_stored(self.mapped_at(&self.nodes, e, node)),
            GeoKind::Q1 | GeoKind::Q2 => match self.nodal_group(&self.nodes, kind) {
                GroupGeo::Stored(st) => {
                    from_stored(st.at(self.point_index(&self.nodes, kind, e, node)))
                }
                GroupGeo::Tabulated(tab) => tab.eval(self.geometric_nodes(kind, e), node),
            },
        }
    }

    /// The tensor-ordered geometric nodes of storage element `e`, of kind Q1
    /// (`2^D` nodes) or Q2 (`3^D`).
    #[inline(always)]
    pub(super) fn geometric_nodes(&self, kind: GeoKind, e: usize) -> &[[f64; D]] {
        let (nodes, n) = match kind {
            GeoKind::Q1 => (&self.q1_nodes, 1 << D),
            GeoKind::Q2 => (&self.q2_nodes, 3usize.pow(D as u32)),
            _ => unreachable!("only Q1/Q2 elements have geometric nodes"),
        };
        let k = e - self.ranges.of(kind).start;
        &nodes[k * n..(k + 1) * n]
    }

    /// The geometry of the Q1 or Q2 group at the point set `pg`.
    #[inline(always)]
    fn nodal_group<'a>(&self, pg: &'a PointGeo<D>, kind: GeoKind) -> &'a GroupGeo<D> {
        let group = match kind {
            GeoKind::Q1 => &pg.q1,
            GeoKind::Q2 => &pg.q2,
            _ => unreachable!("only Q1/Q2 groups"),
        };
        group.as_ref().expect("geometry of a non-empty group")
    }

    /// Index of point `p` of storage element `e` in the stored geometry of
    /// its group at `pg`.
    #[inline(always)]
    fn point_index(&self, pg: &PointGeo<D>, kind: GeoKind, e: usize, p: usize) -> usize {
        (e - self.ranges.of(kind).start) * pg.ref_points.len() + p
    }

    /// Stored `(x, det J, cof J)` of mapped element `e` at point `p` of `pg`.
    #[inline(always)]
    fn mapped_at(
        &self,
        pg: &PointGeo<D>,
        e: usize,
        p: usize,
    ) -> (SVector<f64, D>, f64, SMatrix<f64, D, D>) {
        let st = pg.mapped.as_ref().expect("geometry of the mapped group");
        st.at(self.point_index(pg, GeoKind::Mapped, e, p))
    }

    /// `(x, det J, cof J)` at point `p` of the point set `pg`, element `e` of
    /// kind `kind` (the image only if `XYZ`, for affine elements).
    #[inline(always)]
    pub(crate) fn geo_at<const XYZ: bool>(
        &self,
        pg: &PointGeo<D>,
        kind: GeoKind,
        e: usize,
        p: usize,
    ) -> (SVector<f64, D>, f64, SMatrix<f64, D, D>) {
        match kind {
            GeoKind::Affine => {
                let (x0, j) = &self.affine[e - self.ranges.affine.start];
                let x = if XYZ {
                    x0 + j * SVector::<f64, D>::from_fn(|d, _| pg.ref_points[p][d])
                } else {
                    SVector::zeros()
                };
                (x, det(j), cofactor(j))
            }
            GeoKind::Mapped => self.mapped_at(pg, e, p),
            GeoKind::Q1 | GeoKind::Q2 => match self.nodal_group(pg, kind) {
                GroupGeo::Stored(st) => st.at(self.point_index(pg, kind, e, p)),
                GroupGeo::Tabulated(tab) => {
                    let (x, j) = tab.eval(self.geometric_nodes(kind, e), p);
                    (x, det(&j), cofactor(&j))
                }
            },
        }
    }
}
