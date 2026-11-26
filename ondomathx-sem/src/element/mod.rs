//! The reference element and what acts on one element at a time:
//!
//! * [`shape`]      — GLL point counts per direction (type markers) and their
//!   1D tables.
//! * [`quadrature`] — quadrature rules (type markers) and the node ↔ point
//!   operators by sum factorization ([`quadrature::Sweep`]).
//! * [`operator`]   — element operators ([`operator::ElementOperator`]) and
//!   the driver of the element-local products.

pub mod operator;
pub mod quadrature;
pub mod shape;
