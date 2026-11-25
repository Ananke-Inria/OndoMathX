# ondomathx-sem

Matrix-free spectral elements on quadrangle and hexahedron meshes (formerly LobattoX): Gauss–Lobatto (GLL) Lagrange elements of arbitrary order, operators applied without assembly, and the building blocks of explicit time stepping for wave problems.

- **Spaces** — GLL elements of order 1 to 8 in 2D and 3D, isotropic or with a different point count per direction, discontinuous storage with an optional continuous numbering. Meshes come from [`ondomathx-mesh`](https://crates.io/crates/ondomathx-mesh): affine, bilinear/trilinear, biquadratic/triquadratic and exactly mapped (curved) elements.
- **Operators** — a pointwise kernel (`PointKernel`) describes the physics; the crate applies it matrix-free on batches of elements, in parallel. Kernels for acoustics, isotropic elasticity, masses, divergence pairs and error norms. Mixed products between two spaces at any declared Gauss–Legendre or Gauss–Lobatto quadrature, and hand-written element operators (`ElementOperator`).
- **Time stepping blocks** — lumped and tensor inverse masses, power iteration for the stable time step, homogeneous Dirichlet masks, volume sources.
- **Output** — VTK (tessellated elements) and Vizir high-order solutions.

The 1D Gauss–Lobatto data and sum-factorised kernels come from [`lobatto`](https://crates.io/crates/lobatto); the DG ↔ global map, the linear algebra (power iteration, fused updates) and the source signals from [`ondomathx-toolbox`](https://crates.io/crates/ondomathx-toolbox), shared with [`ondomathx-tetrix`](https://crates.io/crates/ondomathx-tetrix).

## Examples

One file per case, each a complete simulation (mesh, space, physics, time step, source, leapfrog loop, output in `out/<example>/`):

```sh
cargo run --release --example acoustic_wave_square
cargo run --release --example elastic_wave_cube
cargo run --release --example nearly_incompressible_wave_square
```

Also `acoustic_wave_cube`, `acoustic_wave_circle`, `acoustic_wave_2d_mesh`, `elastic_wave_square` and `nearly_incompressible_wave_cube`.

Part of the [OndoMathX](https://github.com/Ananke-Inria/OndoMathX) workspace.

## License

Licensed under either of Apache License 2.0 or MIT license, at your option.
