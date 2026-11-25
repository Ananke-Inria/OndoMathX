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
  CSR, `dg2glob`, multiplicity) built from a `loc2glob` table, owned by both
  spaces; its constructor from a mesh and a DoF set behind the `mesh`
  feature; sem's pruned shared groups stay a fast path of `sum_shared`.
- **D3** transfers: `collect_dg`, `collect_avg_dg`, `scatter_cg` (TetriX
  names) plus `sum_shared` (in place) and `multiplicity`, in both.
- **D4** mass: TetriX's model (global-size mass vector, `mlt_inverse_mass`
  summing the copies and dividing in one pass) **after a benchmark** on
  sem's large cubes. The benchmark also compares, for TetriX: sum+divide then
  update; sum then `axpby_diag`; one fused sum+divide+update pass.
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
2. `DgMap` and the D4 benchmark.
3. The renaming (D1, D3, D5, D7, D8) and nalgebra 0.34 (D9): breaking,
   `ondomathx-sem` 0.2.0.
4. The I/O formats.
