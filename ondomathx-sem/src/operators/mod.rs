//! Hand-written element operators ([`crate::ElementOperator`]) for specific
//! formulations, written with the public API only (as user code would be).
//!
//! - [`FusedKp`]: the pressure stiffness `K^P = Bᵀ (M^P)⁻¹ B` of the
//!   displacement–pressure formulation of nearly incompressible elasticity
//!   (inf-sup article, Ramiche–Imperiale), fused per element: no global
//!   pressure vector.
//! - [`GradDivQ4Q2`]: the same `K^P` for Q4 displacement / Q2 pressure with
//!   `b_h` by Green's formula (option 3) and the lumped GLL 3 pressure mass,
//!   hard-coded as in cornea's `GradDiv.h` (2D and 3D).
//! - [`FusedKpQ4Q2`]: `B`, `Bᵀ` of [`GradDivQ4Q2`] with the Gauss–Legendre
//!   pressure mass of the revised article (tensor inverse).

mod fused_kp;
mod grad_div;

pub use fused_kp::FusedKp;
pub use grad_div::{FusedKpQ4Q2, GradDivQ4Q2, GradDivScratch, PressureMass2, PressureMass3};
