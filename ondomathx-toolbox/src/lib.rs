//! Shared tools of the OndoMathX crates (`ondomathx-tetrix`,
//! `ondomathx-sem`): code that does not depend on a particular
//! finite-element space.
//!
//! * [`lal`]     — linear algebra on plain vectors: reproducible dot
//!   products and norms, the power iteration for the spectral radius of
//!   `M⁻¹K`, the fused update `axpby_diag` of explicit schemes.
//! * [`signals`] — source signals: Ricker wavelet, Gaussian, C∞ bump.
//!
//! Tools that need meshes go behind the `mesh` feature, so that
//! `ondomathx-mesh` stays independent of this crate.

pub mod lal;
pub mod signals;
