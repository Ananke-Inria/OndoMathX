# ondomathx-toolbox

Shared tools of the OndoMathX crates (`ondomathx-tetrix`, `ondomathx-sem`): code that does not depend on a particular finite-element space.

- **`lal`** — linear algebra on plain vectors: reproducible dot products and norms (the same result on any number of threads), the power iteration for the spectral radius of `M⁻¹K` (the leapfrog stability limit), and `axpby_diag`, which fuses the inverse lumped mass into the update of an explicit scheme.
- **`numbering`** — `DgMap`, the map between per-element (DG) storage and the global DoFs of a continuous space: assembly (`collect_dg`), averages, scatter, the in-place shared-DoF sum, and the inverse lumped mass in one pass.
- **`signals`** — source signals: the Ricker wavelet, the Gaussian, a C∞ bump with compact support.
- **`io_vtk`, `io_vizir`** — the file formats of the writers: legacy ASCII VTK unstructured grids (ParaView) and Vizir high-order solutions (`HOSolAt…`), the values supplied by functions so that each space keeps its own storage.

Tools that need meshes go behind the `mesh` feature, which brings in [`ondomathx-mesh`](https://crates.io/crates/ondomathx-mesh); without it the crate depends on no other member of the workspace.

Part of the [OndoMathX](https://github.com/Ananke-Inria/OndoMathX) workspace.

## License

Licensed under either of Apache License 2.0 or MIT license, at your option.
