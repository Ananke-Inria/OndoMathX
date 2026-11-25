# ondomathx-tetrix

Finite elements on tetrahedral meshes (formerly TetriX), with meshes from [`ondomathx-mesh`](https://crates.io/crates/ondomathx-mesh).

- **Elements** (`finite_element`) — tetrahedral elements with 4, 5, 10, 14 and 15 nodes, including P2 Lagrange elements enriched with face and interior bubble functions, with their quadrature rules and local stiffness operators.
- **P2 space** (`tetrix_p2_space`) — an isoparametric finite-element space on a tetrahedral mesh with duplicated (DG) storage, element coloring for race-free parallel application, global numbering, mass and stiffness (applied element by element, or assembled as a sparse matrix).
- **Shared tools** — the norms, the power iteration for the spectral radius of M⁻¹K and the source signals come from [`ondomathx-toolbox`](https://crates.io/crates/ondomathx-toolbox) (`lal`, `signals`).
- **Output** — VTK (quadratic tetrahedra) and Vizir high-order solutions: writer objects (`io_vtk::VtkWriter`, `io_vizir::VizirWriter`) that format the geometry once and take several fields per snapshot, on the file formats of [`ondomathx-toolbox`](https://crates.io/crates/ondomathx-toolbox) shared with `ondomathx-sem`.

The binary `TetriX` runs a scalar acoustic wave with a leapfrog scheme on the `Heart.mesh` asset of `ondomathx-mesh`:

```sh
cargo run --release --bin TetriX
```

Part of the [OndoMathX](https://github.com/Ananke-Inria/OndoMathX) workspace.

## License

Licensed under either of Apache License 2.0 or MIT license, at your option.
