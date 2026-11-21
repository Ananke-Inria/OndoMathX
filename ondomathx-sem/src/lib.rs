//! # lobattox
//!
//! Matrix-free spectral elements on quadrangle and hexahedron meshes:
//! Gauss–Lobatto (GLL) Lagrange elements of arbitrary order, operator
//! application without assembly, and the building blocks of explicit time
//! stepping for wave problems (acoustics, elastodynamics); the time loops
//! themselves live in the examples.
//!
//! Meshes come from [`anamesh`]; 1D GLL data and the batched sum-factorized
//! derivative kernels come from [`lobatto`].

pub mod element;
pub mod engine;
pub mod fields;
pub mod geometry;
pub mod kernels;
pub mod mixed;
pub mod operators;
pub mod ops;
pub mod quadrature;
pub mod shape;
pub mod space;
pub mod vizir;
pub mod vtk;

pub use engine::{MixedKernel, PointIn, PointKernel, PointOut};
pub use element::{mlt_add_element, Batch, ElementOperator, PointSet};
pub use mixed::{mlt_add_mixed, Pointwise};
pub use geometry::{cofactor, det};
pub use ops::{dot, power_iteration, stable_dt, EigenEstimate};
pub use shape::{Gll2, Gll3, Iso2, Iso3, Shape};
pub use geometry::GeoStorage;
pub use quadrature::{Gl, Gll, Quad2, Quad3, Quadrature, Rule, Sweep};
pub use space::{GeoKind, InverseMass, Numbering, Space, SpaceBuilder};
pub use vizir::VizirWriter;
pub use vtk::{VtkField, VtkWriter};
