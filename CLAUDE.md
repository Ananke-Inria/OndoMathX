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

The convergence plan (D1–D11) is complete. Not decided: `Field` coefficients for TetriX; the names of the mass
builders (TetriX `assemble_mass(&mut mass, α)` global-size, sem
`local_mass(coef)` per copy, `inverse_mass(coef)`); test meshes in the
toolbox.
