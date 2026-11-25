# OndoMathX

Finite-element crates of the Ananke team (Inria), in one Cargo workspace:

| Crate | Role |
|---|---|
| [`ondomathx-mesh`](ondomathx-mesh) | Meshes, geometric maps, labels, DOF numbering (formerly Anamesh) |
| [`ondomathx-sem`](ondomathx-sem) | Matrix-free spectral elements on quadrangle and hexahedron meshes (formerly LobattoX) |
| [`ondomathx-tetrix`](ondomathx-tetrix) | Finite elements on tetrahedra (formerly TetriX) |
| [`ondomathx-toolbox`](ondomathx-toolbox) | Shared tools |

The 1D Gauss–Lobatto data and tensor-grid numerics come from the [Lobatto](https://github.com/Ananke-Inria/Lobatto) crates (`lobatto` on crates.io). OndoMathX depends on Lobatto, never the reverse.

```sh
cargo test --workspace --release
```

Licensed under either of Apache License 2.0 or MIT license, at your option.
