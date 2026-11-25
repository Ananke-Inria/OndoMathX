//! Shared tools of the OndoMathX crates (`ondomathx-tetrix`,
//! `ondomathx-sem`): code that does not depend on a particular
//! finite-element space.
//!
//! * [`lal`]     — linear algebra on plain vectors: reproducible dot
//!   products and norms, the power iteration for the spectral radius of
//!   `M⁻¹K`, the fused update `axpby_diag` of explicit schemes.
//! * [`numbering`] — [`numbering::DgMap`], the map between per-element (DG)
//!   storage and the global DoFs: assembly, averages, scatter, shared-DoF
//!   sum, inverse lumped mass.
//! * [`signals`] — source signals: Ricker wavelet, Gaussian, C∞ bump.
//!
//! Tools that need meshes go behind the `mesh` feature, so that
//! `ondomathx-mesh` stays independent of this crate.

pub mod lal;
pub mod numbering;
pub mod signals;
