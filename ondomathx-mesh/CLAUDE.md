# Anamesh

Rust library for **finite element mesh handling**: from a mesh file or a parametric
description down to a global DOF numbering ready for FE assembly. Covers reference
elements, geometric elements + shape functions, mesh I/O, topology, smooth deformation
of meshes through diffeomorphisms, graph coloring for parallelism, and conforming DOF
numbering.

Part of the **OndoMathX** workspace (siblings: `LobattoX`, `QuadratiX`, `TetriX`).
Anamesh is its own git repository — commit here, not from the workspace root.
`README.md` is user-facing only; this file holds everything for development:
conventions, contracts, design decisions, and open work.

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
  TRIANGLE3/TRIANGLE6, QUADRANGLE4/QUADRANGLE9, TETRAHEDRON4/TETRAHEDRON10,
  HEXAHEDRON8/HEXAHEDRON27.
- `diffeomorphism.rs` — the `Diffeomorphism` struct (smooth map R^inner → R^outer via
  `diffeo`/`grad_diffeo` fn pointers). `compose` for chaining maps, `grad_eval`
  applies the chain rule, `compose_grad_def` left-multiplies an element Jacobian by
  Df (the shared chain rule of all `get_element_grad_def` impls), `test_grad`
  validates against finite differences. Attach to a mesh for curved/analytic domains.
- `dof_set.rs` — `DoFSet`: layout of DOFs on a reference element (positions in
  reference coords), classified into vertex/edge/face/interior. Input to
  `UnstructuredMesh::get_numbering`.
- `topology.rs` — `Edge`/`Face` with canonical sorted-index keys so a shared entity
  hashes the same from either neighbor; `get_relative_orientation` feeds DOF
  numbering. Faces use 4 indices with a repeated last index signalling a triangle.
- `mesh.rs` — the `Mesh` trait (node/element access, `get_element_grad_def`,
  coloring, `write_msh`/`write_mesh`, default `get_unstructured_mesh`).
- `mesh/square.rs` — `Square`, structured 2D quad mesh, hardcoded 2×2 checkerboard
  4-coloring, 4 labelled boundary edge groups (0 bottom, 1 right, 2 top, 3 left; top
  and left run reversed). Optional deformation, optional `q2` flag.
- `mesh/cube.rs` — `Cube`, structured 3D hex mesh mirroring `Square`: hardcoded
  2×2×2 checkerboard 8-coloring (color = parities of i, j, k), 6 labelled
  boundary-face groups in hex face-table normal order (0 bottom z=0, 1 front y=0,
  2 left x=0, 3 right x=lx, 4 back y=ly, 5 top z=lz; corners ordered outward),
  optional deformation, optional `q2` flag (→ HEXAHEDRON27/QUADRANGLE9).
- `mesh/circle.rs` — `Circle`, structured disk mesh (square core + curved arms via
  the built-in circle map), hardcoded 4-coloring, optional extra deformation,
  optional `q2` flag.
- `mesh/unstructured_mesh.rs` — `UnstructuredMesh`, general mixed-element mesh.
  Computes indexing (by dim/label), full topology (neighbour_faces/edges/vertexes and
  their inverses), and a greedy dual-graph coloring (share-a-vertex adjacency) for
  race-free parallel assembly; rejects dangling boundary elements at construction.
  Constructors: `new_from_points_and_elements`, `new_from_msh` (Gmsh 2.x ASCII),
  `new_from_mesh` (Medit).
- `mesh/refine.rs` — `UnstructuredMesh::uniform_refine[_by(n)]`, n ≤ 100: new mesh
  + `parent_to_children` map. Children generated in reference coordinates (tensor
  split for edge/quad/hex, principal lattice for triangles, Freudenthal-Kuhn for
  tetrahedra), numbered by feeding the fine lattice as a `DoFSet` to
  `get_numbering`; node positions evaluated through the parent maps, so Q2
  curvature, labels and the attached deformation are preserved.
- `macros.rs` — float-comparison and small-vector macros (`F64_EPSILON = 1e-10`).

## Contracts

- **`get_element_grad_def(el, uvw, grad_def)`** OVERWRITES `grad_def` with the
  Jacobian ∂x/∂u of the COMPLETE element map (undeformed map composed with any
  attached deformation), row-major, exact shape geo_dim × element reference dim —
  quad cell in 2D → 2×2, boundary edge → 2×1 (the tangent). Each mesh type
  implements it; the deformation chain rule is shared through
  `Diffeomorphism::compose_grad_def`. FD-validated in `tests/grad_def.rs`.
- **`grad_diffeo(params, xyz, grad)`** WRITES the Jacobian Df row-major
  `outer_dim × inner_dim`; it never reads or multiplies the buffer. All chain-rule
  products are done by the library (`grad_eval`, `compose_grad_def`).
- **Tolerances are reference-space only**: the `eq!`/`leq!`/… macros and
  `F64_EPSILON` are valid only on reference coordinates (everything O(1)); never
  apply them to physical coordinates.
- **Canonical internal node ordering**, derived from the topology tables: vertices
  first (`indexes[..nvertexes]` are always the corners), then edge nodes in
  edge-table order, then face nodes, then interior. File-format deviations (Gmsh
  TET10/HEX27, Medit HexahedraQ2) are quarantined in permutation tables in the
  readers/writers, with inverse pairs guaranteeing round-trips.
- **Coordinates are always `[f64; 3]`** (also `uvw`); lower dimensions embed with
  trailing zeros. Diffeomorphism `params: Vec<f64>` carries per-map constants since
  plain `fn` pointers cannot capture.
- **Coloring** is defined only for elements of the highest topological dimension;
  all indices exchanged through the coloring API are global element indices.

## Key workflow: global DOF numbering

`UnstructuredMesh::get_numbering(&dof_set) -> (numbering, n_dof)` where
`numbering[element][local_dof]` is the global index. Iterates highest-dim elements
first, assigns fresh indices to vertex/edge/face DOFs on first encounter (reusing via
precomputed orientation), always fresh for volume interiors, then backfills lower-dim
boundary elements from their neighbors. Conforming, orientation-aware, works for P1
and Q2, supports mixed-dimension meshes.

## Design decisions (interview 2026-07-23)

- **Purpose**: mesh layer for the OndoMathX solvers — optimize for what they need
  (quad/hex meshes, orientation-aware numbering, coloring), not general meshing.
- **Eager construction**: a constructed `UnstructuredMesh` is always fully
  queryable (indexing + topology + coloring computed up front).
- **Panic = bug, Result = I/O**: asserts for programming errors; file parsing must
  return proper errors (rewrite pending, see open work).
- **Pre-1.0**: break the API freely; the only consumers are sibling crates.
- **Founding principle**: plain `fn`-pointer tables + `params`, no capturing
  closures, no trait objects, no generics over `Fn`. Linear algebra stays
  hand-rolled on row-major `&[f64]` slices (≤3×3) — **no nalgebra** (tried and
  removed once; column-major + heap/dispatch costs don't pay off at these sizes).
- **`Mesh::get_element` returns by value** — that's what lets parametric meshes
  synthesize elements on demand; `get_element_as_ref` is the hot-loop path on
  `UnstructuredMesh`. Labels stay bare `usize`.
- **Geometric order**: Q2 now, Q3later with handwritten tables; the
  multi-node-per-edge ordering and table-production conventions are decided when the
  first Q3 element is implemented. Field order is already unbounded via `DoFSet`
  (arbitrary positions, e.g. GLL points).
- **Cell zoo**: PRISM and PYRAMID (Gmsh 6, 7) planned as geometry/topology/I/O-only
  citizens. Mixed-cell numbering (a "DoF family" with one DoFSet per cell type and
  interface-compatibility validation) is deferred until a hybrid mesh needs it; the
  `DoFSet` per-face symmetry asserts get relaxed only as part of that work.
- **Dimension inference**: coordinate-scan heuristic (nonzero y ⇒ 2D, nonzero z ⇒
  3D) stays the default; planned: optional explicit `geometric_dimension` and loud
  panics on unsupported (geo, topo) pairs.
- **Boundary model**: labelled lower-dim elements are THE boundary representation;
  connect to volumes on demand via `get_parent_face`/`get_parent_edge` (to add):

  ```rust
  /// For a boundary element (dim = topologic_dimension - 1),
  /// returns (volume_element_index, local_face_index_in_that_element).
  pub fn get_parent_face(&self, boundary_el: usize) -> (usize, usize) {
      let i_face = self.neighbour_faces[boundary_el][0];
      let [vol_el, local_face] = self.faces_neighbour[i_face][0];
      (vol_el, local_face)
  }
  ```

- **Skeleton numbering**: deferred; sketched as
  `get_skeleton_numbering(face_dof_set)` numbering DOFs on every mesh face
  (hybrid/mortar/HDG-type methods), reusing the face tables and orientation
  machinery. Nothing built today may conflict with it.
- **Coloring**: greedy largest-degree-first on the share-a-vertex graph is enough;
  parametric meshes keep hardcoded optimal 4-colorings as fast paths.
- **I/O scope frozen**: Gmsh 2.x ASCII (read; writes always 2.2) + Medit ASCII;
  no Gmsh 4.x, no binary formats until a concrete workflow requires it.
- **Round-trip invariants** (documented contract, enforced by tests): `.msh`
  write→read is the identity (nodes, element order, labels, connectivity, internal
  node ordering); `.mesh` write→read preserves the mesh up to element reordering
  (grouped by type; always written as dimension 3 with z = 0 and vertex refs 0).

## Status

- P1 and P2/Q2 elements working end-to-end: shape functions, mesh I/O (with Gmsh and
  Medit node-ordering permutations), topology, coloring, DOF numbering.
- `get_element_grad_def` contract unified and deformation composition fixed
  (2026-07-23): overwrite semantics, exact shapes, `compose_grad_def` shared chain
  rule; regression + FD coverage in `tests/grad_def.rs`.
- Parametric meshes: `Square`, `Circle` and `Cube` (all with `q2` option; `Cube`
  added 2026-07-25).
- Uniform refinement (2026-07-24): factor n for all cell types, Q2-preserving,
  conforming via the `get_numbering` machinery; extensive coverage + visual
  outputs in `tests/refine.rs`.
- `Circle` arm map replaced by an O-grid map (2026-07-28): the old
  angle-preserving radial blend `r_new = r + t·(R−r)` had a zero radial
  derivative at the four points where the diagonals meet the circle, so the Q2
  elements touching them had det J = 0 at that corner (and `uniform_refine_by(3)`
  produced inverted children there). The new `arm_map` blends each center ray
  linearly between its inner-square boundary point and the circle point at
  angle driven by the transverse coordinate (θ = s·π/4 in the arm frame);
  Jacobian nonzero on the closed arms, boundary vertices uniformly spaced in
  angle, and the `get_node_ref` pre-warp became unnecessary and was removed.

## Known asset issues (found 2026-07-24)

- **Dangling boundary elements are now rejected at construction** (resolved
  2026-07-24). `assets/Potatoe.msh` contained 19 edge elements whose vertex pair
  was not an edge of any quad (geometry polylines meshed at a different
  resolution than the cells); `get_numbering` — and therefore `uniform_refine` —
  panicked on them. The invariant *every lower-dimensional element is the trace
  of an element of the highest topological dimension* is now enforced by
  `check_dangling_elements` in `new_from_points_and_elements` (loud panic), and
  the 19 edges were stripped from the asset (now 116 edges + 816 quads + 12
  points; 901 nodes unchanged).
- **`assets/Fractals.msh` is Gmsh 2.1** (resolved 2026-07-24): `new_from_msh` now
  accepts any ASCII 2.x version (the `$Nodes`/`$Elements` structure is identical
  across 2.0–2.2) and rejects binary files (file-type ≠ 0). The writer still
  emits 2.2. Fractals is the triangle asset with a conforming boundary, used in
  `tests/refine.rs`.

## Open work (roadmap order)

1. **`get_parent_face` / `get_parent_edge`** — the boundary-model gap (see sketch
   above).
2. **Parser rewrite** — `MeshError` with file path + line number instead of unwrap
   panics, `split_whitespace` + `\r` robustness, single-pass reading, and full
   semantic validation (index ranges, counts vs headers, tag consistency).
3. **VERTEX1** — geometric element for Gmsh point elements (msh id 15), read/write,
   so labelled points survive I/O and can be numbered (the dim-0 numbering path
   already exists).
4. **Explicit `geometric_dimension` override** + loud panics on unsupported
   (geo, topo) pairs.
5. **Prism + pyramid** — reference/geometric elements, topology, I/O (numbering
   deferred, see design decisions).
6. **Drop `serde`/`serde_json`** — unused; re-add only with an actual serialization
   design.
7. **Docs/tests** — document the tolerance rule and round-trip invariants in-code;
   extend round-trip tests to every supported element type in both formats.

Later, unordered: Cylinder, Ball/Sphere, Annulus, Triangle domain, Ventricle,
structured tri/tet meshes of square and cube; Split (tri/tet → quad/hex); DOF
renumbering for locality; Skeleton numbering; mixed-cell DoF family; Q3 elements.
