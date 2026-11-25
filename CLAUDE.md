# OndoMathX workspace

Cargo workspace of the Ananke team's (Inria) finite-element crates, repository
`Ananke-Inria/OndoMathX`, public. Each crate keeps its own CLAUDE.md (design
record); this file holds only the workspace rules.

| Crate | Formerly | Design record |
|---|---|---|
| `ondomathx-mesh` | Anamesh | `ondomathx-mesh/CLAUDE.md` |
| `ondomathx-sem` | LobattoX | `ondomathx-sem/CLAUDE.md` |
| `ondomathx-tetrix` | TetriX | none yet |
| `ondomathx-toolbox` | new (empty) | — |

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
