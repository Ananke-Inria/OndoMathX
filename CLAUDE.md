# OndoMathX workspace

Cargo workspace of the Ananke team's (Inria) finite-element crates, repository
`Ananke-Inria/OndoMathX`, public. Each crate keeps its own CLAUDE.md (design
record); this file holds only the workspace rules.

| Crate | Formerly | Design record |
|---|---|---|
| `ondomathx-mesh` | Anamesh | `ondomathx-mesh/CLAUDE.md` |
| `ondomathx-sem` | LobattoX | `ondomathx-sem/CLAUDE.md` |
| `ondomathx-tetrix` | TetriX | none yet |
| `ondomathx-toolbox` | new | — (see "API convergence" below) |

Rules:
- **OndoMathX depends on Lobatto, never the reverse.** The Lobatto crates
  (`lobatto`, `lobatto-grid`, `lobatto-lowrank`; repository
  `Ananke-Inria/Lobatto`) come from crates.io.
- Dependencies between members are written `path` + `version`, so that each
  crate can be published; publish in the order mesh → toolbox → sem → tetrix.
- `ondomathx-mesh` depends on no other member; tools that need meshes go
  behind the toolbox's `mesh` feature.
- One `Cargo.lock` and one `target/` at the root. Commit from the root.
- History: the three former repositories were imported with their history
  (2026-11-25, `git filter-repo --to-subdirectory-filter`); `Thorax.mesh` was
  removed from the imported history. The former repositories (Anamesh,
  LobattoX, TetriX) are deleted once the migration is complete.

Commands: `cargo test --workspace --release`, `cargo clippy --workspace --all-targets`.

## API convergence of `ondomathx-tetrix` and `ondomathx-sem` (decided 2026-11-25)

Both crates store vectors per element (DG storage) and use the spaces the
same way; their APIs converge on **TetriX's** (the user's reference), and
`ondomathx-toolbox` holds what does not depend on a particular space. Every
choice is approved by the user before it is made.

Decisions (all approved):
- **D1** counts: TetriX names in both, `num_elements()`, `num_dofs()`
  (unique DoFs), `num_loc_dofs()`, plus `num_dg_dofs()` (duplicated size).
- **D2** numbering: one `ondomathx_toolbox::numbering::DgMap` (`glob2dg` as
  CSR, `dg2glob`, multiplicity, and the pruned shared groups for
  `sum_shared`), `u32` indices, built by `DgMap::new(loc2glob, nloc, n_dof)`
  from a flat table in storage order or by `DgMap::from_numbering(&numbering,
  n_dof, &elements)` from the result of `UnstructuredMesh::get_numbering` or
  `Mesh::get_tensor_numbering` (no `mesh` feature needed). Both spaces
  replace their own maps with it in step 3.
- **D3** transfers: `collect_dg`, `collect_avg_dg`, `scatter_cg` (TetriX
  names) plus `sum_shared` (in place) and `multiplicity`, in both.
- **D4** mass: **sem's model**, the fastest (user's choice after the
  benchmark): the shared-DoF sum `sum_shared`, then `axpby_diag` with the
  inverse lumped mass per copy (a DG-size vector). TetriX's time loop
  switches to it in step 3; `mlt_inverse_mass` (TetriX's sum and divide in
  one pass) stays available in `DgMap`. The fused sum+divide+update pass is
  dropped.
- **D5** operators: TetriX's argument order, `mlt_<op>(u, v, α)` overwrites
  `v`, `mlt_add_<op>(u, v, α)` does `v ← v + αAu`; sem's generic engine
  becomes `mlt(kernel, u, v, α)` / `mlt_add(kernel, u, v, α)`.
- **D6** slices (`&[f64]`, `&mut [f64]`) in both. Done in TetriX's space
  (2026-11-25, step 1).
- **D7** accessors: TetriX's names where both crates have the concept
  (`get_label`, `loc_dof_coordinates`, …); sem's `sample(f)` added to TetriX.
- **D8** element verbs: TetriX's pairs (`gradient_interpolation` /
  `transpose_gradient_interpolation`, …) in sem's `Sweep`. **D9** TetriX on
  nalgebra 0.34.
- **D10** power iteration: TetriX's closures (`solve`, `mlt_add`) with an M
  inner product, sem's Rayleigh estimate, reproducible sums. Done (step 1).
- **D11** signals: TetriX's conventions, `ricker(t, f0, t0)` normalised
  (peak +1, `ricker_delay(f0) = 1/f0`), `gaussian(r, σ²)`, sem's `bump`.
  Done (step 1): sem's examples use f0 = 7 and 14 (were s0 = 500 and 2000,
  f0 ≈ 7.12 and 14.2) with t0 = 0.2, and the normalised amplitude (their
  solutions are about −1/(2·s0) times the former ones).
- Toolbox I/O: the file formats only (`io_vtk`, `io_vizir`), each crate
  keeping a thin adapter. No re-export of toolbox items by sem or TetriX.

Order of work (each step proposed in detail and approved first):
1. **Done 2026-11-25**: `ondomathx_toolbox::lal` (`dot`, `mass_norm`,
   `infinite_norm`, `start_vector`, `power_iteration` → `EigenEstimate
   { lambda, n_iter, residue }`, `axpby_diag::<C>(nloc, …)`; sums in fixed
   chunks of `CHUNK` = 4096, reproducible on any number of threads) and
   `signals` (`ricker`, `ricker_delay`, `gaussian`, `bump`). TetriX's `lal`
   module removed; its driver: ρ = 5.078259e5 in 9 iterations (was 17),
   dt unchanged to 7 digits. sem's examples: same dt and iteration counts,
   and the acoustic cube is now identical from run to run.
2. **Done 2026-11-25**: `ondomathx_toolbox::numbering::DgMap` (D2, D3
   operations: `collect_dg`, `collect_avg_dg`, `scatter_cg`, `sum_shared`,
   `mlt_inverse_mass`, `multiplicity`; DG layout `[elem][comp][node]`, global
   layout `[dof][comp]`; copies summed in increasing DG index, bit-identical
   to TetriX's and sem's current sums) and the D4 benchmark,
   `ondomathx-toolbox/benches/d4_mass.rs` (path-only dev-dependencies on sem
   and TetriX). Post-`K u` time of a leapfrog step, ms, M4 Pro, 10 threads
   (1 / 14 threads in the bench's output): (a) sum + `axpby_diag`, (b)
   `mlt_inverse_mass` + update, (c) fused:
   - sem acoustic cube 48³ Q4, 7.2 M DoFs (`K u` 9.7): (a) 4.8, (b) 4.8,
     (c) 8.2; 1 thread (a) 15.7, (b) 18.3, (c) 42.6.
   - sem elastic cube 32³ Q4, 6.4 M DoFs, C = 3 (`K u` 7.7): (a) 4.3,
     (b) 4.6, (c) 7.1.
   - TetriX Heart, 222k DoFs (`K u` 0.33): (a) 0.37, (b) 0.39, (c) 0.38;
     14 threads (thread-overhead bound): (a) 0.61, (b) 0.54, (c) 0.47.
   `DgMap` against the crates' current maps, bit-identical: `sum_shared`
   3–17 % faster than sem's `c_sum_shared`, `mlt_inverse_mass` 3 % (1
   thread) to 36 % (14 threads) faster than TetriX's. Memory: 199 MB on the
   acoustic cube (one DG vector 111 MB; sem's shared groups ≈ 110 MB), 59 MB
   on the elastic cube, 10 MB on the Heart (TetriX's own tables ≈ 25 MB).
3. In three parts, each approved and checked before the next:
   - **3a, done 2026-11-25**: both spaces own a `DgMap`. TetriX: it replaces
     `loc2glob` (`Vec<Vec<usize>>`), `glob2dg`, `dg2glob`; `collect_dg`,
     `collect_avg_dg`, `scatter_cg`, `mlt_inverse_mass`, `loc2glob(e, i)`
     keep their names; new `sum_shared`, `num_dg_dofs`, `multiplicity`,
     `dg_map`. Its driver uses the D4 model (`mlt_stiffness_dg`,
     `sum_shared`, `lal::axpby_diag` with `scatter_cg(1/mass)`): ‖u‖_M
     unchanged to 7 digits at all 29 outputs, the last snapshot within
     5.6e-13 of the former one (rounding, over 10 904 steps), time loop
     13.8 → 12.1 s. sem: the continuous numbering is a `DgMap` (see
     `ondomathx-sem/CLAUDE.md`); new `collect_dg`, `collect_avg_dg`,
     `scatter_cg`; the eight examples bit-identical.
   - **3b, done 2026-11-25**: TetriX on nalgebra 0.34 and nalgebra-sparse
     0.11 (benches): one nalgebra (0.34.2) in the workspace; the driver's 30
     VTK files bit-identical to 3a, 51 TetriX tests pass, the
     `tetrix_p2_space` bench runs again (it read `Heart.mesh` through a
     stale `CARGO_MANIFEST_DIR` until the build cache was cleared).
   - **3c, done 2026-11-25**: the renaming, approved as a name-by-name
     table (in `ondomathx-sem/CLAUDE.md`, "API names since 0.2.0"), with
     three rules decided then: TetriX's `_dg` suffix in sem for
     element-local operations on DG vectors (`mlt_dg`, `mlt_add_dg`,
     `mlt_mixed_dg`, `mlt_element_dg`, …, `inverse_mass_dg`,
     `inverse_mass_with_dg`, `multiplicity_dg`), plain names for assembled
     ones (`inverse_mass`, `mlt_assembled`, `boundary_mask`, `start_vector`,
     `max_eigenvalue`, `multiplicity(i_glob)`); D5's β replaced by the pair
     overwrite / accumulate (the two β = 0.5 tests scale `v` first). TetriX
     gained `mlt_add_stiffness_dg` and `sample::<C>` (tested on
     `Cube.mesh`). sem's eight examples bit-identical, 393 workspace tests
     pass. `ondomathx-sem` is 0.2.0.

Versions (decided 2026-11-25): `ondomathx-toolbox` 0.1.1 (0.1.0 on crates.io
is the empty crate; sem and TetriX require `0.1.1`), `ondomathx-sem` 0.2.0,
`ondomathx-tetrix` 0.2.0 (breaking: `lal` removed, slices, writers),
`ondomathx-mesh` unchanged at 0.1.0 (not republished). `cargo package -p
ondomathx-toolbox -p ondomathx-sem -p ondomathx-tetrix` verifies the three
together. Publish order: toolbox → sem → tetrix.
4. **Done 2026-11-25**: the file formats in the toolbox, `io_vtk::VtkGrid`
   (header, points, one cell type, formatted once; `write` / `write_to` of
   point fields given by `value(point, comp)`: `SCALARS`, `VECTORS` for
   `dim` components, else one `SCALARS` per component) and
   `io_vizir::HoSolution` (`HOSolAt<tag>` header once; rows from
   `value(row, node, comp)`). One text format: shortest exact numbers,
   `# vtk DataFile Version 3.0`, each crate's name as title, VTK values with
   `|v| < 1e-40` written as `0` (TetriX's rule, now in both). Adapters: sem's
   `VtkWriter` / `VizirWriter` unchanged in API; TetriX gained the same
   writer objects (`io_vtk::VtkWriter`, `io_vizir::VizirWriter`, global
   vectors `[dof][comp]`), its `write`, `write_vtk`, `write_solution` are
   wrappers (one scalar `SOLUTION`, as before). Checks: sem's `.sol` files
   and TetriX's `.sol` files byte-identical; sem's VTK files differ only in
   the title and the flushed values (54k values in `acoustic_wave_2d_mesh`,
   113k in `acoustic_wave_cube`), TetriX's in the version and title lines
   and the points' exact digits (≤ 3.8e-14 relative). 397 workspace tests.

Fields (decided 2026-11-26, after the C++ `Sources/Field` of
`~/Documents/OndoMathX_gitlab/core`): `ondomathx_toolbox::field`, **values
only** (no operator `Eval`/`AdjointEval` layer; the algebra stays in the
kernels). `Field<T>` with `NEEDS_XYZ`, `NEEDS_LABEL`, `NEEDS_INDEX` and
`eval(&impl Point) -> T`; the point through the trait `Point` (`xyz`,
`label`, `index`; `At` a plain one). One type per C++ regularity, generic
over the value: `Constant<T>`, `PerLabel<T>` (a missing label panics),
`Function<F>` (`Fn(&[f64]) -> T`), `LabelFunction<F>` (`Fn(&[f64], usize) ->
T`, the C++ `PiecewiseC0`), `PointData<T>` (no `Inverse`: removed at the user's request, 1/κ is written with the other types; a general `Map(field, g)` if a need appears).
Values now: `f64`, `[f64; N]`, `[[f64; J]; I]` (row `i`). The C++
isotropic, 4th/3rd-order and Neo-Hookean tensors are not ported.
**Adopted 2026-11-26**: sem's `fields.rs` removed, its kernels and masses take
`Field<f64>` (`PointIn` implements `Point`, `dof` as the index; `Function`
is a closure, a missing `PerLabel` label panics instead of giving NaN; the
eight examples bit-identical). TetriX: `assemble_stiffness`, `assemble_mass`
(density), `mlt_stiffness`, `mlt_stiffness_dg_seq`, `mlt_stiffness_dg`,
`mlt_add_stiffness_dg` take a coefficient field first, **constant only** (a
field needing xyz, label or index is refused at compile time); it multiplies
α (its driver's 60 output files bit-identical). Versions for the next
publication: toolbox 0.1.2, sem 0.3.0, tetrix 0.3.0 (both require toolbox
0.1.2).

TetriX layout (2026-11-26): `src/`, `benches/`, `tests/`, `examples/`,
`tools/` side by side. The former `TetriX` binary (`src/main.rs`) is
`examples/acoustic_wave_heart.rs` (output in `out/acoustic_wave_heart/`,
git-ignored, as sem's examples; its 60 output files bit-identical); the
`gen_mat_*` code generators of the `finite_element/` tables are in `tools/`,
declared `[[example]]` (`cargo run --example gen_mat_10_4`), so no binary is
installed with the crate.

Vizir output fixed (2026-11-26): the `.sol` field-type line was the component
count (`1 3` for a 3D displacement), which libMeshb reads as a symmetric
matrix (types: 1 scalar, 2 vector, 3 symmetric matrix, 4 matrix), so Vizir
read 6 values per node and the data were misaligned (`elastic_wave_cube`;
the 2D displacements were wrong too). `io_vizir` now writes `1 1` for one
component, `1 2` for 2 or 3 (2D vectors padded with `z = 0`: the files are in
dimension 3), `C 1 … 1` otherwise; checked with `vizir4 -checksurf` (the
reconstructed `.solb` sizes match 1 and 3 values per node). New
`io_vizir::write_sols(path, files)`: every example writes `<name>.sols` (its
snapshots, **absolute paths**; the nearly incompressible ones also
`<name>_pressure.sols`), TetriX's example also `heart.mesh`, and the run ends
with `vizir4 -in <absolute mesh> -sols <absolute .sols> -checksurf` (Vizir adds
the solutions on the boundary faces; after the C++
convention, `Scripts/ViZiR_README.md`). No `default.vizir` (removed at the
user's request; `Checksurf 1` in it would make Vizir add the solutions on
the boundary faces, which the `.sol` files do not hold). Not checked on
screen (no screen access from the session; `vizir4 -movie` without a saved
`vizir.state` shows only the mesh).
**Corrected the same day** (against the ViZiR 4 user guide and the installed
`vizir4`, Oct 2026 build): `-checksurf` after `-in … -sols …` is ignored
("the surface reconstruction is not active"; it is a mode of its own, guide
11.3, or the `Checksurf` keyword of `default.vizir`, 7.10), so the surfaces
showed no solution. `-sols` does not open the first file (it tries
`<mesh>.sol`), and a relative path in a `.sols` is resolved against the
working directory (tested: bare names fail from the parent directory, load
from the files' one). Now (user's choice): `.sols` files list **bare file
names**, the 3D examples (heart, the three cubes) write `default.vizir` with
`Checksurf 1` next to their output, and the runs end with `cd <dir> && vizir4
-in <mesh> -sol <first>.00000.sol -sols <name>.sols`. Heart mesh: 1788 of its
boundary faces are missing from its `Triangles`; Vizir adds them.

Common module structure (2026-11-26; the rest of each crate to be sorted
later): `io` (`io::io_vtk`, `io::io_vizir`), `element` (the reference element
and what acts on one element), `space` (finite-element spaces).
- sem: `element::{operator, shape, quadrature}` (were `element.rs`,
  `shape.rs`, `quadrature.rs`), `io::{io_vtk, io_vizir}` (were `vtk.rs`,
  `vizir.rs`), `space` unchanged; the crate-root re-exports (`Space`,
  `ElementOperator`, `Sweep`, `Iso2`, `VtkWriter`, …) keep their paths. Still
  at the root: `engine`, `mixed`, `kernels`, `operators`, `geometry`, `ops`.
- TetriX: `element` (was `finite_element`, every `tet_*` module),
  `io::{io_vtk, io_vizir}`, `space::p2_plus` (the type `P2PlusSpace`, was
  `tetrix_p2_space::TetriXP2Space`); no root re-exports, so
  its paths changed (`ondomathx_tetrix::space::p2_plus::P2PlusSpace`).

The convergence plan (D1–D11) is complete. Not decided: `Field` coefficients for TetriX; the names of the mass
builders (TetriX `assemble_mass(&mut mass, α)` global-size, sem
`local_mass(coef)` per copy, `inverse_mass(coef)`); test meshes in the
toolbox.
