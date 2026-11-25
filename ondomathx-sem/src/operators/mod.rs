//! Hand-written element operators ([`crate::ElementOperator`]) for specific
//! formulations, written with the public API only (as user code would be).
//!
//! - [`FusedKp`]: the pressure stiffness `K^P = Bᵀ (M^P)⁻¹ B` of the
//!   displacement–pressure formulation of nearly incompressible elasticity
//!   (inf-sup article, Ramiche–Imperiale), fused per element: no global
//!   pressure vector.
//! - [`FusedKpQ4Q2`]: the same `K^P` for Q4 displacement / Q2 pressure with
//!   `b_h` by Green's formula (option 3, hard-coded as in cornea's
//!   `GradDiv.h`, 2D and 3D) and the Gauss–Legendre pressure mass of the
//!   revised article (tensor inverse).

mod fused_kp;
mod grad_div;

pub use fused_kp::FusedKp;
pub use grad_div::{FusedKpQ4Q2, FusedKpQ4Q2Scratch, PressureMass2, PressureMass3};
