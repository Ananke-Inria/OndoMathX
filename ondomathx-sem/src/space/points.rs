//! Point sets and quadratures: the nodes, the declared rules and the node
//! rule `S::Nodes`, their sweep tables, and values at their points.

use std::any::{type_name, Any, TypeId};

use rayon::prelude::*;

use crate::element::PointSet;
use crate::quadrature::Sweep;
use crate::shape::Shape;

use super::{QuadEntry, Space};

impl<const D: usize, S: Shape<D>> Space<D, S> {
    /// The declared quadrature `Q`, if any.
    pub(crate) fn find_entry<Q: 'static>(&self) -> Option<&QuadEntry<D>> {
        self.quads.iter().find(|q| q.id == TypeId::of::<Q>())
    }

    /// The sweep tables of `Q` stored in the space: the node rule's, or a
    /// declared quadrature's; `None` otherwise.
    pub(crate) fn sweep_tables<Q: Sweep<D, S>>(&self) -> Option<&Q::Tables> {
        if TypeId::of::<Q>() == TypeId::of::<S::Nodes>() {
            return (&self.node_sweep as &dyn Any).downcast_ref::<Q::Tables>();
        }
        self.find_entry::<Q>()
            .and_then(|q| q.sweep.downcast_ref::<Q::Tables>())
    }

    /// The nodes as a point set (GLL weights, node geometry), for
    /// [`crate::ElementOperator`]s.
    pub fn node_points(&self) -> PointSet<'_, D> {
        PointSet {
            geo: &self.nodes,
            weights: &self.node_weights,
        }
    }

    /// The quadrature `Q` as a point set: a declared one, or the node rule
    /// `S::Nodes` (always available, the node geometry); `None` otherwise.
    pub fn quadrature_points<Q: 'static>(&self) -> Option<PointSet<'_, D>> {
        match self.find_entry::<Q>() {
            Some(q) => Some(PointSet {
                geo: &q.geo,
                weights: &q.weights,
            }),
            None if TypeId::of::<Q>() == TypeId::of::<S::Nodes>() => Some(self.node_points()),
            None => None,
        }
    }

    /// The quadrature `Q` (declared, or the node rule) and its sweep tables;
    /// panics if it is neither.
    pub(super) fn rule<Q: Sweep<D, S>>(&self) -> (PointSet<'_, D>, &Q::Tables) {
        let set = self.quadrature_points::<Q>().unwrap_or_else(|| {
            panic!(
                "quadrature {} was not declared (SpaceBuilder::quadrature)",
                type_name::<Q>()
            )
        });
        let tables = self
            .sweep_tables::<Q>()
            .expect("sweep tables of the quadrature");
        (set, tables)
    }

    /// `w_q · |det J|` at every point of the quadrature `Q` (declared, or the
    /// node rule), `[elem][q]` (`n_elem · NQ` entries): `∫ f = Σ w f(x_q)`.
    pub fn quadrature_weights<Q: Sweep<D, S>>(&self) -> Vec<f64> {
        let (q, _) = self.rule::<Q>();
        let mut out = vec![0.0; self.n_elem * Q::NQ];
        out.par_chunks_mut(Q::NQ)
            .enumerate()
            .for_each(|(e, block)| {
                let kind = self.kind(e);
                for (p, w) in block.iter_mut().enumerate() {
                    *w = q.weights[p] * self.geo_at::<false>(q.geo, kind, e, p).1.abs();
                }
            });
        out
    }

    /// Physical points of the quadrature `Q`, `[elem][q]`.
    pub fn quadrature_xyz<Q: Sweep<D, S>>(&self) -> Vec<[f64; D]> {
        let (q, _) = self.rule::<Q>();
        let mut out = vec![[0.0; D]; self.n_elem * Q::NQ];
        out.par_chunks_mut(Q::NQ)
            .enumerate()
            .for_each(|(e, block)| {
                let kind = self.kind(e);
                for (p, x) in block.iter_mut().enumerate() {
                    let xp = self.geo_at::<true>(q.geo, kind, e, p).0;
                    *x = core::array::from_fn(|a| xp[a]);
                }
            });
        out
    }

    /// Values of the `C`-component vector `u` at the points of the quadrature
    /// `Q` (Lagrange interpolation from the GLL nodes, by sum factorization),
    /// `[elem][comp][q]` (`n_elem · C · NQ` entries).
    pub fn quadrature_values<Q: Sweep<D, S>, const C: usize>(&self, u: &[f64]) -> Vec<f64> {
        let nloc = Self::NLOC;
        assert_eq!(u.len(), self.n_dup() * C, "vector length");
        let (_, t) = self.rule::<Q>();
        let mut out = vec![0.0; self.n_elem * C * Q::NQ];
        out.par_chunks_mut(C * Q::NQ).enumerate().for_each_init(
            || vec![0.0; Q::TMP],
            |tmp, (e, block)| {
                for c in 0..C {
                    let input = &u[(e * C + c) * nloc..(e * C + c + 1) * nloc];
                    Q::interpolate::<1>(t, input, &mut block[c * Q::NQ..(c + 1) * Q::NQ], tmp);
                }
            },
        );
        out
    }
}
