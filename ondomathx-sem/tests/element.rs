//! Hand-written element operators (`ElementOperator`, `mlt_element_dg`).

mod common;

use common::*;
use ondomathx_toolbox::field::Constant;
use ondomathx_sem::kernels::Mass;
use ondomathx_sem::{
    Batch, ElementOperator, GeoKind, GeoStorage, Gl, Iso2, Numbering, Quad2, Shape, Space,
};

fn seeded(n: usize, seed: u64) -> Vec<f64> {
    let mut s = 0x9E37_79B9_7F4A_7C15_u64 ^ seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// Nodal mass `c·w·|det J|·u`, written by hand with the node geometry.
struct NodalMass {
    c: f64,
}
impl<const D: usize, S: Shape<D>> ElementOperator<D, S, S, 1, 1> for NodalMass {
    type Scratch = ();
    fn scratch<const W: usize>(&self) {}
    fn apply<const W: usize>(
        &self,
        batch: &Batch<'_, D, S, S>,
        u: &[f64],
        v: &mut [f64],
        _: &mut (),
    ) {
        let nodes = batch.u_space.node_points();
        for node in 0..S::NLOC {
            for lane in 0..W {
                let (_, det_j, _) = batch.geometry(&nodes, lane, node);
                v[node * W + lane] =
                    nodes.weight(node) * det_j.abs() * (u[node * W + lane] * self.c);
            }
        }
    }
}

#[test]
fn hand_written_mass_equals_the_kernel_engine() {
    let mesh = square(5, 3, true, Some(partial_2d()));
    let space = Space::<2, Iso2<4>>::new(&mesh, Numbering::Continuous);
    let u = seeded(space.num_dg_dofs(), 1);
    let (mut a, mut b) = (vec![0.0; u.len()], vec![0.0; u.len()]);
    space.mlt_dg::<1>(
        &Mass {
            coef: Constant(1.7),
        },
        &u,
        &mut a,
        1.0,
    );
    space.mlt_element_dg(&space, &NodalMass { c: 1.7 }, &u, &mut b, 1.0);
    assert_eq!(a, b);
}

/// Per-element data (indexed by `Batch::element`), labels, padded lanes and
/// the α/β update.
struct PerElement {
    scale: Vec<f64>,
}
impl<const D: usize, S: Shape<D>> ElementOperator<D, S, S, 2, 1> for PerElement {
    type Scratch = ();
    fn scratch<const W: usize>(&self) {}
    fn apply<const W: usize>(
        &self,
        batch: &Batch<'_, D, S, S>,
        u: &[f64],
        v: &mut [f64],
        _: &mut (),
    ) {
        let n = S::NLOC;
        for lane in 0..W {
            let e = batch.element(lane);
            assert!(
                lane < batch.n_valid || e == batch.first + batch.n_valid - 1,
                "padded lane"
            );
            if lane >= batch.n_valid {
                assert!(
                    u[lane] == 0.0 && u[n * W + lane] == 0.0,
                    "padded input is zero"
                );
            }
            for node in 0..n {
                // component 1 of u, times the element's factor, plus its label
                v[node * W + lane] =
                    self.scale[e] * u[(n + node) * W + lane] + batch.label(lane) as f64;
            }
        }
    }
}

#[test]
fn per_element_data_labels_and_update() {
    let mut mesh = square(5, 3, false, Some(partial_2d()));
    let nodes = ondomathx_mesh::mesh::UnstructuredMesh::get_nodes(&mesh).to_vec();
    mesh.set_labels(|_, el| if nodes[el.indexes[0]][1] < 0.5 { 3 } else { 7 });
    let space = Space::<2, Iso2<3>>::new(&mesh, Numbering::None);
    assert!(!space.elements(GeoKind::Affine).is_empty() && !space.elements(GeoKind::Q1).is_empty());
    let n = Space::<2, Iso2<3>>::NLOC;
    let op = PerElement {
        scale: (0..space.num_elements()).map(|e| 1.0 + e as f64).collect(),
    };
    let u = seeded(space.num_dg_dofs() * 2, 2);
    let v0 = seeded(space.num_dg_dofs(), 3);
    let mut v = v0.clone();
    v.iter_mut().for_each(|x| *x *= 0.5);
    space.mlt_add_element_dg(&space, &op, &u, &mut v, 2.0);
    for e in 0..space.num_elements() {
        for node in 0..n {
            let y = op.scale[e] * u[(2 * e + 1) * n + node] + space.get_label(e) as f64;
            let expected = 0.5 * v0[e * n + node] + 2.0 * y;
            assert_eq!(v[e * n + node], expected, "e {e} node {node}");
        }
    }
}

/// Element areas `Σ_q w_q |det J|` from a declared quadrature found through
/// the batch, written at node 0.
struct Area<Q>(std::marker::PhantomData<Q>);
impl<const D: usize, S: Shape<D>, Q: 'static + Sync> ElementOperator<D, S, S, 1, 1> for Area<Q> {
    type Scratch = ();
    const READS_INPUT: bool = false;
    fn scratch<const W: usize>(&self) {}
    fn apply<const W: usize>(
        &self,
        batch: &Batch<'_, D, S, S>,
        _: &[f64],
        v: &mut [f64],
        _: &mut (),
    ) {
        let q = batch.quadrature::<Q>();
        v.fill(0.0);
        for (lane, v0) in v[..W].iter_mut().enumerate() {
            *v0 = (0..q.len())
                .map(|p| q.weight(p) * batch.geometry(&q, lane, p).1.abs())
                .sum();
        }
    }
}

#[test]
fn quadrature_through_the_batch() {
    type G = Quad2<Gl<3>, Gl<3>>;
    // f = (x, y(1 + 0.3x)) on [0,2]×[0,1]: area 2.6, det J of degree (1, 1).
    let mesh = square(4, 3, false, Some(bilinear_2d()));
    let space = Space::<2, Iso2<3>>::builder(&mesh)
        .quadrature::<G>(GeoStorage::Recompute)
        .build();
    let mut v = vec![0.0; space.num_dg_dofs()];
    space.mlt_element_dg(
        &space,
        &Area::<G>(Default::default()),
        &v.clone(),
        &mut v,
        1.0,
    );
    let area: f64 = v.iter().sum();
    assert!((area - 2.6).abs() < 1e-12, "{area}");
}

#[test]
#[should_panic(expected = "different meshes")]
fn element_operator_on_different_meshes_panics() {
    let a = Space::<2, Iso2<3>>::new(&square(2, 2, false, None), Numbering::None);
    let b = Space::<2, Iso2<3>>::new(&square(3, 2, false, None), Numbering::None);
    let u = vec![0.0; a.num_dg_dofs()];
    let mut v = vec![0.0; b.num_dg_dofs()];
    a.mlt_element_dg(&b, &NodalMass { c: 1.0 }, &u, &mut v, 1.0);
}
