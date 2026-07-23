# Anamesh

Rust library for **finite element mesh handling**: from a mesh file or a parametric
description down to a global DOF numbering ready for FE assembly. Covers reference
elements, geometric elements + shape functions, mesh I/O, topology, smooth deformation
of mesh through the use of diffeomorphism, graph coloring for parallelism, and conforming DOF numbering.

Part of the **OndoMathX** workspace (siblings: `LobattoX`, `QuadratiX`, `TetriX`).
Anamesh is its own git repository — commit here, not from the workspace root.

## Commands

Run all `cargo` commands from the crate root (`Anamesh/`). Tests load mesh files via
**relative paths** like `./assets/Potatoe.msh`, so the working directory matters.

```bash
cargo build
cargo test                    # all tests (integration tests live in tests/)
cargo test --test topology    # one integration-test file
cargo run --example example   # the example in examples/example.rs
cargo fmt
cargo clippy
```

Tests that write output put it under `tests/results/` (gitignored except `.gitkeep`) —
don't commit generated `.msh` files.

## Module map (`src/`)

- `reference_elements.rs` — combinatorial + geometric structure of reference cells
  (`VERTEX`, `EDGE`, `TRIANGLE`, `QUADRANGLE`, `TETRAHEDRON`, `HEXAHEDRON`). Owns the
  edge/face orientation maps (`uvw_edge_transform`, `uvw_face_transform`) that the DOF
  numbering depends on.
- `geometric_elements.rs` — reference cell + Lagrange shape functions/gradients =
  `GeometricElement`; also stores the Gmsh `msh_id`. Implemented: EDGE2/EDGE3,
  TRIANGLE3, QUADRANGLE4/QUADRANGLE9, TETRAHEDRON4, HEXAHEDRON8.
- `diffeomorphism.rs` — the `Diffeomorphism` struct (smooth map R^inner → R^outer via
  `diffeo`/`grad_diffeo` fn pointers). `compose` for chain rule, `test_grad` validates
  the Jacobian against finite differences. Attach to a mesh for curved/analytic domains.
- `dof_set.rs` — `DoFSet`: layout of DOFs on a reference element (positions in
  reference coords), classified into vertex/edge/face/interior. Input to
  `UnstructuredMesh::get_numbering`.
- `topology.rs` — `Edge`/`Face` with canonical sorted-index keys so a shared entity
  hashes the same from either neighbor; `get_relative_orientation` feeds DOF numbering.
- `mesh.rs` — the `Mesh` trait (node/element access, `get_element_grad_def`, coloring,
  `write_msh`, default `get_unstructured_mesh`).
- `mesh/square.rs` — `Square`, structured 2D quad mesh, hardcoded 2×2 checkerboard
  4-coloring, 4 labelled boundary edge groups. Optional deformation.
- `mesh/circle.rs` — `Circle`, structured disk mesh (square core + curved arms via a
  deformation).
- `mesh/unstructured_mesh.rs` — `UnstructuredMesh`, general mixed-element mesh.
  Computes indexing (by dim/label), full topology (neighbour_faces/edges/vertexes and
  their inverses), and a greedy dual-graph coloring (share-a-vertex adjacency) for
  race-free parallel assembly. Constructors: `new_from_points_and_elements`,
  `new_from_msh` (Gmsh 2.2), `new_from_mesh` (Medit).
- `macros.rs` — internal macros.

## Key workflow: global DOF numbering

`UnstructuredMesh::get_numbering(&dof_set) -> (numbering, n_dof)` where
`numbering[element][local_dof]` is the global index. Iterates highest-dim elements
first, assigns fresh indices to vertex/edge/face DOFs on first encounter (reusing via
precomputed orientation), always fresh for volume interiors, then backfills lower-dim
boundary elements from their neighbors. Conforming, orientation-aware, works for P1
and Q2, supports mixed-dimension meshes.

## Conventions

- Element `.msh`/`.mesh` I/O currently supports **P1 elements only** (Edge2,
  Triangle3, Quadrangle4, Tetrahedron4, Hexahedron8). Writing is Gmsh 2.2 only.
- Q2 (`QUADRANGLE9`) and P2 (`EDGE3`) shape functions + DOF classification exist but
  are **not wired into the parsers** yet — a known gap (see `TODO.md`).
- Diffeormphisms use to represent curved domains analytically without storing extra
  nodes; when attached, deformed coords are used in `write_msh` and
  `get_element_grad_def`. `apply_deformation` bakes it permanently into the node list.
- Edition 2021. Dependencies are minimal: `serde`/`serde_json` (mostly future use),
  `itertools` (formatting index lists on `.msh` write).

## Notes

- The deformation module is **`diffeomorphism`** exposing **`Diffeomorphism`**. Its
  `diffeo`/`grad_diffeo` are plain `fn` pointers with signature `(params, xyz, f_xyz)`;
  per-map constants travel through the `params: Vec<f64>` field since `fn`s can't capture.
  Coordinates are 3-component (`[f64; 3]`) even for 2D meshes.
- Both `Square` and `Circle` are implemented parametric meshes (`src/mesh/`).

Open work items are in `TODO.md` (P2 wiring, more parametric meshes: Cylinder/Ball/
Cube/…, subdomain & boundary strategy incl. `get_parent_face`, refine/split/renumber).
