//! Homogeneous Dirichlet masks (`Space::boundary_mask`) and the error
//! kernels (`L2Error`, `H1Error`).

mod common;

use std::f64::consts::PI;

use ondomathx_mesh::mesh::circle::Circle;
use ondomathx_mesh::mesh::cube::Cube;
use ondomathx_mesh::mesh::square::Square;
use common::*;
use ondomathx_sem::fields::Constant;
use ondomathx_sem::kernels::{AcousticStiffness, H1Error, L2Error};
use ondomathx_sem::{GeoStorage, Gl, Iso2, Iso3, Numbering, Quad2, Shape, Space};

/// Mask value at every node against a geometric predicate of the boundary.
fn check_mask<const D: usize, S: Shape<D>>(
    space: &Space<D, S>,
    mask: &[f64],
    on: impl Fn([f64; D]) -> bool,
) {
    let n = S::NLOC;
    for e in 0..space.num_elements() {
        for node in 0..n {
            let x = space.map_and_jacobian(e, node).0;
            let x: [f64; D] = core::array::from_fn(|a| x[a]);
            let expected = if on(x) { 0.0 } else { 1.0 };
            assert_eq!(
                mask[e * n + node],
                expected,
                "element {e} node {node} at {x:?}"
            );
        }
    }
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

#[test]
fn square_mask_whole_boundary_and_labels() {
    let sq = Square::new([0.0, 0.0], 2.0, 1.0, 3, 2, None, None);
    let space = Space::<2, Iso2<4>>::new(&sq, Numbering::Continuous);
    let mask = space.boundary_mask(&sq, None);
    check_mask(&space, &mask, |x| {
        near(x[0], 0.0) || near(x[0], 2.0) || near(x[1], 0.0) || near(x[1], 1.0)
    });
    // Distinct boundary DoFs: (3·3 + 1)(2·3 + 1) − (3·3 − 1)(2·3 − 1) = 30.
    let mult = space.multiplicity_dg();
    let zeros: f64 = mask
        .iter()
        .zip(&mult)
        .filter(|(m, _)| **m == 0.0)
        .map(|(_, k)| 1.0 / k)
        .sum();
    assert!((zeros - 30.0).abs() < 1e-9, "{zeros}");
    // All labels = the whole boundary; label 0 = the bottom edge.
    assert_eq!(space.boundary_mask(&sq, Some(&[0, 1, 2, 3])), mask);
    // Without the left label (3), the left side is free except its corners.
    let three = space.boundary_mask(&sq, Some(&[0, 1, 2]));
    check_mask(&space, &three, |x| {
        near(x[0], 2.0) || near(x[1], 0.0) || near(x[1], 1.0)
    });
    let bottom = space.boundary_mask(&sq, Some(&[0]));
    check_mask(&space, &bottom, |x| near(x[1], 0.0));
}

#[test]
fn unstructured_cube_and_circle_masks() {
    // Deformed unstructured mesh: the boundary is the image of the box's.
    let mesh = square(4, 3, true, Some(bending_2d()));
    let space = Space::<2, Iso2<3>>::new(&mesh, Numbering::Continuous);
    let topo = space.boundary_mask(&mesh, None);
    assert_eq!(space.boundary_mask(&mesh, Some(&[0, 1, 2, 3])), topo);
    assert!(topo.iter().filter(|m| **m == 0.0).count() > 0);

    let cube = Cube::new([0.0; 3], 1.0, 2.0, 1.0, 2, 3, 2, None, None);
    let space = Space::<3, Iso3<3>>::new(&cube, Numbering::Continuous);
    let mask = space.boundary_mask(&cube, None);
    check_mask(&space, &mask, |x| {
        near(x[0], 0.0)
            || near(x[0], 1.0)
            || near(x[1], 0.0)
            || near(x[1], 2.0)
            || near(x[2], 0.0)
            || near(x[2], 1.0)
    });

    // Circle with exact geometry: the mask is the circle.
    let (c, r) = ([0.3, -0.2], 1.5);
    let circle = Circle::new(c, r, 7, None, Some(true));
    let space = Space::<2, Iso2<4>>::new(&circle, Numbering::Continuous);
    let mask = space.boundary_mask(&circle, None);
    check_mask(&space, &mask, |x| {
        near(((x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2)).sqrt(), r)
    });
}

/// `M⁻¹K` masked on `sin πx sin πy` (a Dirichlet eigenmode of `[0, 1]²`)
/// gives `2π² u` up to the space error.
#[test]
fn dirichlet_eigenmode() {
    let sq = Square::new([0.0, 0.0], 1.0, 1.0, 6, 6, None, None);
    let space = Space::<2, Iso2<7>>::new(&sq, Numbering::Continuous);
    let mask = space.boundary_mask(&sq, None);
    let k = AcousticStiffness { rho: Constant(1.0) };
    let u = space.sample::<1>(|x| [(PI * x[0]).sin() * (PI * x[1]).sin()]);
    let mut y = vec![0.0; u.len()];
    space.mlt_assembled::<1>(&k, &u, &mut y);
    space.inverse_mass(&Constant(1.0)).apply::<1>(&mut y);
    for (yi, m) in y.iter_mut().zip(&mask) {
        *yi *= m;
    }
    let err = y
        .iter()
        .zip(&u)
        .map(|(a, b)| (a - 2.0 * PI * PI * b).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-5 * 2.0 * PI * PI, "max error {err:e}");
}

/// `∫ |u_h − u|²` and `∫ |∇u_h − ∇u|²` by an error kernel summed over a
/// scalar test space: exact integrals for `u = x y` on `[0, 2] × [0, 1]`.
#[test]
fn error_kernels_integrate_the_squared_error() {
    type Q = Quad2<Gl<4>, Gl<4>>;
    let mesh = square(4, 2, false, None);
    let space = Space::<2, Iso2<3>>::builder(&mesh)
        .quadrature::<Q>(GeoStorage::Recompute)
        .build();
    let l2 = L2Error {
        exact: |x: [f64; 2]| [x[0] * x[1]],
    };
    let h1 = H1Error {
        exact_grad: |x: [f64; 2]| [[x[1], x[0]]],
    };
    let total = |u: &[f64], h1_norm: bool| -> f64 {
        let mut s = vec![0.0; space.num_dg_dofs()];
        if h1_norm {
            space.mlt_mixed_dg::<Q, 1, 1, _>(&space, &h1, u, &mut s, 1.0);
        } else {
            space.mlt_mixed_dg::<Q, 1, 1, _>(&space, &l2, u, &mut s, 1.0);
        }
        s.iter().sum()
    };
    let zero = vec![0.0; space.num_dg_dofs()];
    assert!((total(&zero, false) - 8.0 / 9.0).abs() < 1e-13);
    assert!((total(&zero, true) - 10.0 / 3.0).abs() < 1e-13);
    let u = space.sample::<1>(|x| [x[0] * x[1]]);
    assert!(total(&u, false) < 1e-26 && total(&u, true) < 1e-24);
}
