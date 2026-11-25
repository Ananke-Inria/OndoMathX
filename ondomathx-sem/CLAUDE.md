# ondomathx-sem (formerly LobattoX)

Rust library for **matrix-free spectral elements on quadrangle and hexahedron
meshes**: Gauss–Lobatto (GLL) Lagrange elements of arbitrary order, operator
application without assembly, and explicit time stepping for wave problems
(acoustics, elastodynamics). Successor of the C++ `OndoMathX_gitlab/core`
LobattoX module (`Sources/FEM/LobattoX`, operator engine
`Operators/Core/AbstractMltAdd.h`), with TetriX's storage and threading model.

Crate `ondomathx-sem` of the **OndoMathX** workspace (repository
`Ananke-Inria/OndoMathX`). Dependencies:
- `ondomathx-mesh` (`../ondomathx-mesh`, formerly Anamesh): meshes, geometric maps, labels, DOF numbering.
- `lobatto` (crates.io, `0.1.4`; source in the Lobatto repository, formerly QuadratiX): all 1D
  data (`lobatto::tables`) and the sum-factorized batched kernels.

Until 2026-11-25 this was the separate repository LobattoX; its history was
imported here. Commit from the workspace root. Older entries below keep the old
names (LobattoX, Anamesh, QuadratiX, `lobattox::`).
This file holds the design record: decisions, contracts, lessons from the
predecessors, and the ordered roadmap.

## Commands

```bash
cargo build
cargo test --release          # tests in release mode (QuadratiX convention)
cargo clippy --tests          # zero warnings
cargo fmt
```

## Design decisions (interview 2026-11-19)

### Method and storage
- **CG spectral elements, duplicated storage.** The method is continuous Galerkin
  (SEM) with GLL collocation (diagonal mass). Vectors are stored per element, as in
  DG: a DOF shared by k elements has k copies. A product is element-local
  (race-free, no coloring) followed by a **shared-DOF sum** that restores
  continuity. There are no interface fluxes; DG is not a goal.
- **Layout `[elem][comp][node]`.** Element `e` owns the contiguous block
  `e·C·NLOC .. (e+1)·C·NLOC` (`C` components, `NLOC = (P+1)^D`). Inside it there is
  one contiguous slice per component, with nodes in **tensor lexicographic order**,
  `i` fastest: `node = i + (P+1)·j + (P+1)²·k`, with `i` along reference `u`.
- **`ondomathx-mesh` numbering is used at setup only.** A `DoFSet` built from the GLL points
  in tensor order on `[0,1]^D` makes `get_numbering` return tensor-ordered local
  indices, because `DoFSet` keeps the order of `pos_list`. No permutation table
  is needed. The numbering is consumed to build the shared groups, then dropped.
- **Shared-DOF groups (CSR).** For each global DOF shared by ≥ 2 elements, the
  list of its duplicated indices: `offsets: Vec<usize>`, `indices: Vec<usize>`.
  Interior DOFs (most of them at high order) do not appear. This is TetriX's
  `glob2dg` pruned to the interface.
- **Elements are reordered by geometric type.** Storage holds contiguous groups:
  affine, general Q1, general Q2. Each group gets its own monomorphized loop. A
  permutation `mesh_elem[e]` maps storage to mesh order (labels, I/O).
- **Minimum geometry.** Only the geometric nodes of each element are kept (copied
  contiguously in storage order); the Jacobian is recomputed at each GLL point
  inside the product. Affine elements store a single Jacobian. There is no
  per-point metric storage. The C++ `FESpaceT::ComputeGeometricInfo` cache is
  deliberately not reproduced.
- **Inverse mass: one duplicated-size vector**, holding the assembled
  (shared-summed) and inverted `ρ·w·detJ`. It is equal across the copies of a DOF
  and shared by all components. `M⁻¹` is then a pointwise product.

### Order, dimension, kernels
- **Const generics `<D, NP>`** (D = 2 or 3, NP = number of GLL points per
  direction = P + 1), with a runtime `match` on the degree P at the top level
  picking the monomorphized instance (`1 => <_, 2>`, …, `8 => <_, 9>`). NP, not
  P, is the parameter because stable Rust cannot write `P + 1` as a const
  argument (array sizes `[[f64; NP]; NP]`, lobatto's generic kernels
  `dx_qnnn_qnnn::<NP, W>`); P = NP − 1 is derived (decided 2026-11-19).
  **Superseded by step 8 (2026-11-20):** `Space<D, S>` with a shape marker
  (see "Shape markers").
- **Stable Rust only.** Array sizes like `(P+1)^D` cannot come from const
  generics on stable, so loop bounds are const while buffers are `&mut [f64]`
  slices carved from **per-task scratch** allocated once per rayon task
  (`for_each_init`). No allocation in hot loops.
- **Batched computation over W elements.** Storage stays `[elem][comp][node]`.
  Each task gathers W (4 or 8) consecutive elements of the same type into a
  `[node][lane]` scratch buffer (lobatto's batched layout `g·M·W + lane`), runs
  the batched derivative kernels, evaluates geometry and the point kernel
  lane-wise, and transposes back. The transpose is `O((P+1)^D)` per element and the
  derivatives are `O((P+1)^(D+1))`. The last batch of a group is padded with zero
  lanes, which are never scattered.
- **nalgebra for small fixed-size algebra, for now** (revised 2026-11-19; the
  interview had chosen hand-rolled `[[f64; D]; D]`). Jacobians, `det`, cofactor
  and inverse, gradients and stress tensors use `SMatrix`/`SVector` (stack
  allocated, const-sized). Same version as lobatto (`nalgebra = "0.34"`) so that
  only one copy is built. Efficiency is to be measured later against
  hand-rolled ≤3×3 code (see Roadmap, later items).
  - The batched derivative kernels and the per-task scratch stay plain `f64`
    slices in lobatto's layout; nalgebra is not used for them. No `DMatrix` in
    hot loops, so `CollocationBasis` is not used for operator application.
  - **LobattoX computes no 1D data** (decided 2026-11-20): points, weights,
    differentiation and interpolation matrices all come from `lobatto::tables`
    (the shape markers' `Tables` only store them as arrays).
  - `ondomathx-mesh` works with `[f64; D]`; conversion happens at the interface.
  - **`det` and `cofactor` are written out for D = 2, 3** (`geometry.rs`, on
    `SMatrix`), not nalgebra's `determinant`/`try_inverse`. On a generic
    `const D` those require `Const<D>: DimMin<Const<D>, Output = Const<D>>`,
    and that bound would leak into every generic user of `Space<D, NP>` (it did
    in the first test draft). Inverse = `cof(J)ᵀ / det J`. The cofactor is the
    one the push-forward needs anyway.

### Operator engine (successor of `AbstractMltAdd`)
- **One generic engine plus a `PointKernel` trait.** The engine does the gather,
  reference gradients, geometry, push-forward, weighting, transposed gradients
  and scatter. The physics is a small struct implementing a pointwise map.
  Associated consts (`USES_VALUE`, `USES_GRAD`, `USES_XYZ`, `USES_LABEL`,
  `USES_DOF`) switch buffers and work on and off at compile time, like the
  `constexpr bool req_*` flags of the C++ engine.
- Gradient convention: `grad[c][d] = ∂u_c/∂x_d` (row = component), as in the C++
  `Buffer_UD[u][d]`.
- Push-forward with the cofactor matrix, as in C++: `∇ₓu = cof(J)·∇ᵤu / detJ`.
  On the test side the result is multiplied by `w·cof(J)ᵀ/detJ`, so one division
  per point.
- Semantics `v ← β·v + α·A·u`, as in the C++ `MltAdd` (`β = 0` overwrites).
  The output is element-local; the shared-DOF sum is a separate call, so that
  products can be chained before summing.
- **As implemented (step 4, `engine.rs`):**
  - `trait PointKernel<const D, const C>: Sync` with `apply(&PointIn, &mut PointOut)`.
    Since 2026-11-24 `PointKernel<const D, const CU, const CV = CU>` (stable
    const default): one trait for same-space and mixed kernels, replacing
    `MixedKernel` and its blanket impl; `PointKernel<D, C>` code is unchanged.
    - `PointIn { value: SVector<C>, grad: SMatrix<C, D>, xyz: SVector<D>, label, dof }`,
      where `dof = e·NLOC + node`. Unused fields are zero.
    - `PointOut { flux: SMatrix<C, D>, source: SVector<C> }`, zeroed before
      each call. The form is `a(u, v) = ∫ G(u) : ∇v + S(u)·v`.
    - Flags: `USES_VALUE/GRAD/XYZ/LABEL/DOF`, plus `OUTPUTS_FLUX` and
      `OUTPUTS_SOURCE`, all defaulting to `false`. Without flux, no
      transposed-derivative kernel runs (mass-type operators); without
      `USES_GRAD`, no forward one.
  - Entry point: `Space::mlt_add(kernel, α, u, β, v)` with W = 4;
    `mlt_add_batched::<C, W, K>` for another width. The results are
    bitwise-identical for any W (tested).
  - **One product path** (since 2026-11-24): `mlt_add` is the mixed product
    on the space's own node rule `S::Nodes` (`Pointwise` run by the element
    driver); the former dedicated engine loop is gone. See "Performance" in
    the roadmap for the measurements.
  - Pull-back: `H = sign(det J)·w·G·cof(J)`, then `y = Σ_b D_bᵀ H_b + w·|det J|·S`.
    The division by `det J` happens only in the push-forward.
  - Parallelism: each geometric group is split into W-element chunks of `v`
    (`par_chunks_mut` + `for_each_init` scratch). Writes are disjoint, so there
    is no `unsafe`. `Pointwise` dispatches each batch on its kind to a const
    generic, so each group gets its own monomorphized point loop.
  - Padded lanes reuse the last valid element's geometry (finite values). Their
    input is zero and their output is not scattered.
  - Derivatives: `Sweep::gradient` / `gradient_t`. On the node rule the
    interpolations are identities (skipped) and each derivative is one pass:
    lobatto hc kernels for isotropic N = 4..7 (`shape::hc_2d` / `hc_3d`),
    generic rectangular kernels otherwise; the choice folds at compile time.
    (Until 2026-11-24: `Shape::ref_derivative`, removed with the engine loop.)
  - Geometry (since step 8): affine `det`/`cof` are hoisted per lane. Q1/Q2
    maps are evaluated per point from full tables of the **`ondomathx-mesh` shape
    functions** at the nodes (`GeometricElement::tabulate`, `GeoTab` in
    `geometry.rs`, columns in tensor node order), `x = Σ_k X_k φ_k`; the same
    tables serve `Space::map_and_jacobian`. LobattoX writes no shape function.

### Physics (v1)
- **Second-order formulations, one field.**
  - Acoustics, pressure `p`: `(1/κ) p̈ − div((1/ρ) ∇p) = f`.
  - Elastodynamics, displacement `u`: `ρ ü − div σ(u) = f`, with isotropic
    `σ = λ tr(ε) I + 2μ ε`.
  - Anisotropic `Cijkl`, nonlinear laws and mixed first-order formulations are
    later, as new kernels.
- **Parameter fields**, generic and monomorphized (C++ `Field::Regularity`):
  - `Constant`.
  - `PerLabel`: a table indexed by the bare `usize` mesh label.
  - `Function`: an `ondomathx-mesh`-style `fn(xyz, &params) -> value` evaluated on the fly,
    with xyz recomputed from the element nodes.
  - `PointData`: one value per duplicated DOF. `PointData::sample(fn)` fills it
    once from a function.

  The user picks between `Function` and sampled `PointData`.
- **As implemented (step 5, `fields.rs`, `kernels.rs`):**
  - `trait Field<const D>: Sync` with `eval(&PointIn<D, C>) -> f64` and flags
    `NEEDS_XYZ/LABEL/DOF`. A kernel's `USES_*` flags are the OR of its fields'
    flags (associated consts), so `Constant` costs no geometry and `Function`
    switches on `xyz`.
  - Fields: `Constant(f64)`; `PerLabel(Vec<f64>)` with `from_pairs` (labels
    not given are `NaN`, so a forgotten material shows up in the results);
    `Function { f, params }`; `PointData(Vec<f64>)` with
    `PointData::sample(&space, f)`; `Inverse(F)` = `1/F`, for the acoustic
    mass `1/κ`.
  - **`Function` takes `f(params, xyz)`**, the argument order of `ondomathx-mesh`'s
    `Diffeomorphism` (`fn(&[f64] params, &[f64] xyz, …)`). The interview note
    said `fn(xyz, &params)`; the `ondomathx-mesh` order was kept for consistency.
  - Kernels: `AcousticStiffness { rho }` (flux `∇p/ρ`),
    `IsotropicElastic { lambda, mu }` (flux `σ(u)`, `C = D`), and `Mass { coef }`
    (any `C`, no flux, so no derivative kernel runs).
  - `Space::c_inverse_mass(&impl Field)` (continuous space; `inverse_mass` for
    the element-local mass, since step 7) replaces the duplicated-vector
    version: `c_inverse_mass(&rho)` for elastodynamics,
    `c_inverse_mass(&Inverse(kappa))` for acoustics.
- **Time integration: leapfrog only, and not in the library** (decided
  2026-11-19). The library provides the building blocks (`ops.rs`); the time
  loops live in the examples (each example writes its own loop).
  `uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt²·M⁻¹·Σ_shared(K_loc uⁿ) + dt²·M⁻¹F(tⁿ)`.
  `dt = 0.95·2/√ρ(M⁻¹K)`, estimated by power iteration
  (`ondomathx_toolbox::lal::power_iteration`, formerly TetriX `lal.rs` and
  this crate's `ops.rs`).
  - Building blocks (names since step 7): `Space::c_apply` (`mlt_add` with
    β = 0, then `c_sum_shared`), `InverseMass::apply::<C>` (from
    `c_inverse_mass`), `Space::scale_by::<C>` (one-value-per-node diagonal on C
    components), `Space::local_mass` (unassembled `c·w·|det J|`),
    `Space::mass_inner::<C>` (`Σ m·a·b`, the M product for continuous vectors),
    `Space::c_multiplicity`, `Space::c_max_eigenvalue` (the toolbox's power
    iteration on M⁻¹K, returns `ondomathx_toolbox::lal::EigenEstimate`
    `{ lambda, n_iter, residue }`), `Space::axpby_diag::<C>` (the toolbox's
    `lal::axpby_diag` with this space's `NLOC`:
    `out ← a·x + b·out + c·(diag ⊙ y) + d·z` in one pass, the last term
    optional: `M⁻¹` fused into the leapfrog update, used by the leapfrog
    examples). `mass_inner` adds its per-element sums in element order
    (reproducible, like the toolbox's sums: the CFL step no longer varies
    between runs). Since 2026-11-25 `dot`, `power_iteration` and
    `EigenEstimate` are in `ondomathx_toolbox::lal` (not re-exported), and the
    examples take `ricker`, `gaussian`, `bump` from
    `ondomathx_toolbox::signals`. No `stable_dt` (removed 2026-11-25): the
    examples compute `dt = 0.95 · 2 / √λ_max` themselves.
  - Energies need no assembly: `⟨Ku, w⟩ = dot(K_loc u, w)` and
    `⟨u, w⟩_M = mass_inner(local_mass, u, w)` for continuous `u`, `w`.
  - Power iteration start: a hashed pseudo-random vector, averaged over the
    copies (continuous, rich in high modes, reproducible).
- **Boundary conditions: homogeneous Neumann (free surface)**, natural, no
  code; **homogeneous Dirichlet** (since step 12) by the mask
  `c_boundary_mask` applied to the update.
- **Sources.**
  - Volume `f(x)·g(t)`: because the GLL mass is diagonal, `M⁻¹F` is the nodal
    value of `f` divided by the mass coefficient.
  - Point source: `F_i = φ_i(x_s)·g(t)` on one element containing `x_s`, followed
    by the shared-DOF sum. The location comes from `Mesh::locate` (to add to
    `ondomathx-mesh`). Until then, the user gives `(element, uvw)`.

### Output
- **Vizir** (done, step 17, `vizir.rs`): a port of the C++ `IO_Vizir.h`: `HOSolAt{Quadrilaterals,Hexahedra}Q{1,2}`
  with `NodesPositions` set to the GLL points, degree P, and the mesh written by
  `ondomathx-mesh` `write_mesh`. Duplicated storage is exactly Vizir's per-element layout,
  so no averaging is needed.
- **VTK** (done, `vtk.rs`): legacy ASCII on **tessellated** elements, with P^D linear sub-cells per
  element on the GLL points (VTK_QUAD 9 / VTK_HEXAHEDRON 12; connectivity as in
  the C++ `GetInterpolatedElementConnection`). Points are duplicated per element,
  values are written straight from storage, and xyz is recomputed from the
  geometry.
  - `VtkWriter::new(&space)` formats points and cells once; `write(path,
    &[VtkField])` formats only the data, in parallel over elements. 1 component
    → `SCALARS`, `D` → `VECTORS` (padded to 3 in 2D), otherwise one `SCALARS`
    per component.
- Snapshots only in v1: no receivers and no energy files.

### Scope and organisation
- A library plus integration tests and examples. No CLI or config-file driver.
- `ondomathx-mesh` from the same workspace, by path and version (`path =
  "../ondomathx-mesh", version = "0.1.0"`, for publishing); `lobatto = "0.1.4"`
  from crates.io (QuadratiX rule: version-only dependencies between published
  crates).
- **Validation**:
  - Operator unit tests against a dense assembled matrix on tiny meshes
    (symmetry; kernel = constants for acoustics, rigid modes for elasticity;
    exactness of the shared-DOF sum).
  - Conservation of the staggered discrete energy
    `½‖(uⁿ⁺¹−uⁿ)/dt‖²_M + ½⟨K uⁿ⁺¹, uⁿ⟩`, which is exact below CFL. The energy at
    integer steps only oscillates.
  - Analytic Neumann eigenmodes (cosines) on `Square`/`Cube`, straight, deformed
    and `Circle`: rate P+1 in space and 2 in time. Power iteration needs a
    non-constant start vector, because the constant mode is in `ker K`.
    Done in the examples for `Square`/`Cube` (see roadmap step 6); `Circle`
    (Bessel modes) not done.

### Work in sibling crates
- **lobatto (QuadratiX)** — lobatto 0.1.4, published 2026-11-20:
  - Generic batched kernels in `collocation_hc`, same layout as the hc kernels
    (node `g` of lane `e` at `g·M·N + e`): `dx_qn_qn`, `dx_qnn_qnn`,
    `dy_qnn_qnn`, `dx_qnnn_qnnn`, `dy_qnnn_qnnn`, `dz_qnnn_qnnn` and the `_t_`
    transposes, all `<const NP, const N>(d: &[[f64; NP]; NP], input, output)`
    with `d[j][i] = L_i'(ξ_j)` on `[0,1]`. Output is overwritten, as in the hc
    kernels. The stride is a runtime argument (no `NP*NP` const argument on
    stable); it costs nothing (dz as fast as dx).
  - Tested against `CollocationBasis` for NP = 2..9, N = 1, 4, 8 (relative 1e-12)
    and against the hc Q4/Q6 kernels (1e-13); not bit-identical to hc (different
    summation order).
  - Criterion groups `generic_vs_hc_3d` and `generic_3d` in
    `benches/collocation_hc.rs`. 3D, N = 4/8: generic is 1.15–1.45× slower than
    hc (dx 1.15–1.35×, dz/dz_t 1.3–1.45×), because hc uses the antisymmetry of D.
    Generic cost ≈ 0.18 (NP=2), 0.27 (3), 0.35 (4), 0.44 (5), 0.54 (6),
    0.63 (7), 0.75–0.88 (8), 0.81 (9) ns/dof per derivative.
  - **Kernel choice in LobattoX:** hc kernels for P = 3..6, generic kernels for
    P = 1, 2, 7, 8, selected at compile time in `Sweep::gradient` /
    `gradient_t` when every direction is the node rule (`NP = P+1`).
  - `lobatto::tables` (added 2026-11-20): `points_weights<N>(Gauss)`,
    `interpolation_matrix<NI, NO>(from, to)`, `derivative_matrix<NI, NO>(from,
    to)` (fixed-size, `m[j][i]`), and their run-time-size versions
    (`points_weights_dyn`, `interpolation_row`, `derivative_matrix_dyn`; the
    fixed-size ones wrap them). Interpolation is an exact unit row when a target
    point equals a source point. `CollocationBasis` is built on them
    (bit-identical, `eval` now exact at the nodes).
  - Generic rectangular batched kernels (added 2026-11-20): `dx_qn_qm`,
    `dx_qnn_qmn`, `dy_qnn_qnm`, `dx_qnnn_qmnn`, `dy_qnnn_qnmn`, `dz_qnnn_qnnm`
    and their `_t_` transposes, `NI → NO` points along one direction, every count
    const (anisotropic shapes). Const loop counts are up to 2× faster than
    runtime ones for NP ≤ 5. The square generic kernels share their core
    (`_fibre`/`_sweep`), bit-identical to before, 1–3 % slower in a
    single-thread microbenchmark.
  - **LobattoX builds no 1D data:** the shape markers take the GLL points,
    weights and D from `lobatto::tables`. Interpolation at arbitrary points
    (point sources) will use `lobatto::tables::interpolation_row`.
- **`ondomathx-mesh`**: `Mesh::locate(x) -> (element, uvw)` (bounding boxes + Newton on
  the element map), reusable by TetriX.

## Design decisions: mixed spaces and quadratures (2026-11-20)

Motivation: the inf-sup article (Z. Ramiche, S. Imperiale): displacement in
continuous GLL `Q_k`, pressure in discontinuous `Q_{k−2}`, `M Ü + R U +
λ B*(M^P)⁻¹B U = F`, `M^P` by a `k−1` point Gauss–Legendre rule (block diagonal).
The divergence operator `b_h` must come from a generic mixed product (as the
C++ `AbstractMltAdd` with `QuadratureOpt` `Default`/`Trial`/`Other`), not from a
dedicated implementation. This section revises "Method and storage" and
"Order, dimension, kernels" where they differ.

### Spaces
- **Every space is discontinuous** (duplicated storage `[elem][comp][node]`, as
  now) **plus an optional continuous numbering.** The notion of a "continuous
  space" disappears: continuity is the numbering.
  - The numbering is requested at construction (the space does not keep the
    mesh): `Space::new(&mesh, Numbering::Continuous | Numbering::None)`;
    `None` skips `get_numbering` entirely.
  - **Naming rule: functions prefixed `c_` use the continuous numbering and
    panic without it** (`c_sum_shared`, `c_multiplicity`, `c_n_global`,
    `c_apply`, `c_inverse_mass`, `c_max_eigenvalue`); **unprefixed functions
    are element-local and never touch it** (`mlt_add`, `local_mass`,
    `mass_inner`, `inverse_mass`, `sample`, `scale_by`). So `inverse_mass` is
    the discontinuous (element-local) mass and `c_inverse_mass` the assembled
    continuous one (decided 2026-11-20).
  - For now it is continuous everywhere. Later: continuous per label
    (interfaces, faults) or periodic, built from a `DoFSet` plus a rule.
- **DoFs are only at GLL points** (strong choice, kept). `NP ≥ 2` in every
  direction: Q0 (`k = 2` pressure) is not supported.
- **Anisotropic spaces**: the point count may differ per direction.

### Shape markers (types)
- `Space<const D, S: Shape<D>>`, with marker types carrying the GLL counts:
  `Q2<N0, N1>` (2D), `Q3<N0, N1, N2>` (3D); isotropic aliases (`Iso2<N>`,
  `Iso3<N>`) for the usual case. `D` stays a const parameter (nalgebra
  `SMatrix<D, D>`).
- **Stable-Rust constraint (checked on rustc 1.97):** an associated const of a
  marker cannot be a const argument (`kernel::<{ S::N0 }>()` → "generic
  parameters may not be used in const operations"). Inside
  `impl<const N0, const N1, const N2> Shape<3> for Q3<N0, N1, N2>` the counts
  are ordinary const parameters. Therefore **everything that needs the counts
  as consts is a trait method implemented on the marker**: the 1D tables (as
  arrays), the sum-factorized derivative and interpolation sweeps (choice of
  the hc or generic lobatto kernels in one place, replacing the
  `match (D, NP)` in `engine.rs`). The engine calls `S::grad::<W>(…)` and never
  names the counts. Associated consts (`NLOC`, counts as `usize`) are fine for
  loop bounds and slice lengths.
- `gll.rs` / `Gll1d<NP>` is subsumed by the per-direction tables of the
  marker implementation (`Gll1<N>` in `shape.rs`); removed in step 8.
- **Anisotropic + continuous** is not supported by `ondomathx-mesh`'s `DoFSet`
  numbering (it needs a point set symmetric under the reference element's
  edge/face orientations). It will be an `ondomathx-mesh` feature for geometries that
  can provide an explicit numbering (structured `Square`/`Cube` already have
  `get_numbering(nlocdofx, nlocdofy[, nlocdofz])`), as in the C++ code.
  **Done in step 9** through `ondomathx-mesh` `Mesh::get_tensor_numbering`: a space
  built from a `Square`/`Cube` numbers anisotropic shapes; an
  `UnstructuredMesh` (and `Circle`) panics on anisotropic counts.

### Meshes and geometry (step 9, decided 2026-11-20)
- `Space::builder(&impl Mesh)` / `Space::new(&impl Mesh, numbering)`: any
  `ondomathx-mesh` mesh (`UnstructuredMesh`, `Square`, `Cube`, `Circle`), borrowed only
  during construction. The numbering is `Mesh::get_tensor_numbering(S::N)`.
- **The mesh decides the geometry, per element** (`Mesh::is_mapped_element`,
  added to `ondomathx-mesh`): false → the map is the interpolation of the element's
  affine/Q1/Q2 geometric nodes (groups Affine/Q1/Q2 as before); true (analytic
  map, attached deformation) → group `GeoKind::Mapped`, geometry from the
  mesh's own map (`Mesh::get_element_def` and `get_element_grad_def`, added to
  `ondomathx-mesh`) stored at every point set at construction. A mapped element whose
  map is affine (tested on the 3^D reference nodes) joins the affine group.
  This replaces both the former panic on attached deformations and the
  proposed `.geometry(Isoparametric | Exact)` switch ("isoparametric" was the
  wrong word: GLL order ≠ geometric order).

### Quadratures
- **Gauss–Legendre or Gauss–Lobatto, any count, family and count per
  direction.** Marker types per direction: `Gl<N>`, `Gll<N>`, combined as
  `Quad2<A, B>`, `Quad3<A, B, C>` (e.g. `Quad3<Gl<3>, Gl<3>, Gll<5>>`). Enums
  cannot be const parameters on stable, hence types. Same rule as shapes: the
  rectangular sweeps between a shape and a quadrature are trait methods on the
  marker pair.
- **Chosen per operator, from a fixed list given at construction**:
  `Space::new(mesh, quadratures)`. Entry 0 is always the space's own GLL nodes
  (implicit). An operator call names its quadrature by type; the type is
  checked against the list at the start of the call. Since 2026-11-24 the
  implicit entry has a type, `S::Nodes` (`Shape::Nodes`), accepted wherever
  a declared quadrature is (`mlt_add_mixed`, `local_load_with`,
  `quadrature_*`, `inverse_mass_with`, `Batch::quadrature`) with the node
  geometry; a declared `S::Nodes` entry takes precedence.
- **Geometry on each quadrature: stored or recomputed**, a run-time choice per
  entry (branch once per batch of W elements, both paths compiled). Stored =
  `cof(J)` and `det J` at every point (kernel-agnostic; 10 values per point in
  3D). Affine elements keep a single Jacobian in both modes.

### Mixed operators
- `mlt_add(u_space, v_space, q, kernel, α, u, β, v)`, free function: `q` is a
  quadrature of `u_space` or `v_space`; values and gradients are interpolated
  from each space's GLL nodes to the points of `q` with the lobatto
  rectangular kernels, and back with their transposes. The current
  `Space::mlt_add` (same space, quadrature = nodes) stays as the fast path.
  (Since 2026-11-24 it is this product on the node rule, at the same speed.)
- Same-mesh guarantee: spaces built from the same mesh have the same element
  order (classification is deterministic); a mixed call checks a mesh
  fingerprint (O(1)).
- **Inverse mass is an operator** (step 7): `InverseMass<D, NP>`, built by
  `inverse_mass(coef)` / `c_inverse_mass(coef)`, applied by `im.apply::<C>(v)`;
  `im.diagonal()` gives the diagonal when it exists (fused leapfrog update,
  sources):
  - quadrature = the space's nodes: diagonal, as now;
  - discontinuous space, Gauss–Legendre with the same counts: tensor trick
    `M_K⁻¹ = I⁻¹ W_J⁻¹ I⁻ᵀ`, `I⁻¹` = `interpolation_matrix(Legendre → Lobatto)`
    (two sum-factorized passes and a diagonal, also on non-affine elements);
  - anything else panics (no iterative solver).
- **`ElementOperator` hook** for hard-coded local operators (successor of
  cornea `GradDiv.h`, which hard-codes D = 2, 3, P = 4, Q2 pressure, Green's
  formula, fused `B (M^P)⁻¹ Bᵀ` per element): the driver (geometric groups,
  W-batching, gather/scatter, parallel chunks, scratch) is shared, the element
  operator receives a gathered batch, the element geometry and scratch, and
  writes the local output. The generic engine is one implementation. No face
  concept is introduced for it.

### Other decisions
- `max_eigenvalue` takes closures (operator application, mass solve, M inner
  product), as TetriX `lal::power_iteration` and C++ `LAL::PowerIteration`.
- The examples use a relative tolerance `1e-3` and at most 5000 power
  iterations (1e-6 until 2026-11-22). The test is the relative change of the
  Rayleigh quotient, which converges from below, so the estimate is low:
  dt grows by at most 1.3 % against 1e-6 (square 9.926e-4 → 1.006e-3, cube
  4.588e-3 → 4.638e-3, the others ≤ 0.2 %), inside the 5 % safety margin;
  iterations 701 → 12 (square), 756 → 35 (cube), 18–35 elsewhere.
- Shared toolbox `ondomathx-toolbox` (option (a): it depends on `ondomathx-mesh`,
  behind a cargo feature; `ondomathx-mesh` stays independent). Its contents
  and the API convergence with `ondomathx-tetrix` are decided in the
  workspace CLAUDE.md, "API convergence" (2026-11-25).

## Defaults not covered by the interview (revisit freely)
- The space is built from any `&impl Mesh` (since step 9) and does not borrow
  it after construction. Attached deformations are supported (mapped
  elements, step 9).
- Affine detection is numerical: a Q1 parallelogram/parallelepiped, or a Q2 whose
  nodes are an affine image of the reference nodes. A Q2 whose mid nodes are the
  Q1 interpolation of its corners is demoted to the Q1 group (straight but
  non-affine elements, e.g. a bilinear deformation). Deviations are measured in
  reference coordinates, `|J⁻¹(X_k − y_k)|∞ ≤ 1e-10`, with `J` the frame of
  corner 0 (`ondomathx-mesh` rule). A degenerate frame panics.
- Geometric nodes are copied in **tensor order** (`k_0 + n1·k_1 + n1²·k_2`,
  `n1 = 2` or `3`): the classification builds the affine frame from the
  corners, and the geometry tables have tensor-ordered columns. The
  permutation from `ondomathx-mesh`'s node order is found by evaluating its shape
  functions at the tensor reference nodes, with no table hard-coded.
- Shared-DOF entries are component-free (`e·NLOC + node`); component `c` of a
  `C`-component vector is at `(i / NLOC)·C·NLOC + c·NLOC + i % NLOC`.
- `inverse_mass` takes a `Field` (since step 5; a duplicated vector goes through
  `PointData`). It uses `|det J|`.
- Runtime order range: P = 1..=8.
- Lane width W = 4 by default, as a const generic.
- Point kernels are written per point (scalar) and called in a `for lane in 0..W`
  loop over `[comp][lane]` buffers, relying on inlining and vectorization. If
  benchmarks show no vectorization, switch to a lane-array signature.
- The shared-DOF sum is parallel over groups. The groups partition the shared
  duplicated indices, so the writes are disjoint. It is implemented with the
  TetriX `usize`-pointer pattern and a SAFETY comment.

## Lessons from the predecessors
- C++ `AbstractMltAdd`:
  - Kept: the pointwise structure (gather → gradient → geometry → parameter
    evaluation → adjoint → scatter), compile-time switching of unused work,
    cofactor push-forward, the affine fast path, and `Check_zero` (skip elements
    whose local input is zero; worth keeping for sources).
  - Dropped: the E1..E3/F1..F3 DiffOp triples, the L/M parameter chains, coloring
    of the volume loop, and the full `loc2glob` table.
- TetriX:
  - Kept: duplicated storage with `par_chunks_mut` per element, the leapfrog loop
    with buffer swap, the power iteration, and the single-pass shared-DOF update.
  - Dropped: the CG vectors (mass, sources and output are all duplicated-size).
    nalgebra is kept for now, limited to small fixed-size algebra.
- QuadratiX:
  - Kept: per-task scratch, no internal grain-size thresholds (the caller controls
    threads through the Rayon pool), reference tests for bit-exactness when
    optimizing.

## Status
- Design agreed (2026-11-19). The sketch files of the initial upload
  (`fields.rs`, `spaces.rs`, `generic_mlt_add.rs`, `test.rs`; their notes,
  pipeline of `generic_mlt_add` and DoFSet remarks, are covered by the design
  above) were deleted on 2026-11-21, never committed.

## Roadmap (order)
1. ~~**Crate setup**~~: done (2026-11-19). Path dependencies `anamesh`, `lobatto`;
   `nalgebra = "0.34"` (one copy, shared with lobatto), `rayon = "1"`.
2. ~~**lobatto additions**~~: done, lobatto 0.1.4 (generic batched kernels; see
   "Work in sibling crates"). 0.1.4 published (2026-11-20); LobattoX uses the
   version dependency and builds no 1D data itself.
3. ~~**`Space<D, NP>`**~~: done (2026-11-19), in `gll.rs` (`Gll1d<NP>`),
   `geometry.rs` (tensor-order element maps, classification, `det`/`cofactor`)
   and `space.rs`.
   - API: `Space::new(&UnstructuredMesh)`, `n_elem`/`n_dup`/`n_global`,
     `elements(kind)`, `kind(e)`, `mesh_element(e)`, `label(e)`,
     `map_and_jacobian(e, node)` (scalar, setup/tests), `weight(node)`,
     `sum_shared::<C>`, `sample::<C>(f)`, `inverse_mass(rho)`.
   - Tests (`tests/space.rs`, 15):
     - Global counts on Square and Cube, Q1 and Q2 geometry, NP = 2..9.
     - Multiplicities, continuity after `sum_shared`, total mass = area/volume.
     - Jacobian vs `get_element_grad_def` at every GLL node, on straight and
       deformed meshes.
     - Classification: affine, Q1, Q2, and the Q2 → Q1 demotion.
     - The panics, and `cofactor`/`det`.
4. ~~**Engine**~~: done (2026-11-19), `engine.rs` (see "Operator engine, as
   implemented").
   - Tests (`tests/engine.rs`, 16; shared meshes in `tests/common/mod.rs`):
     - The engine against a brute-force element-local reference
       (`Space::map_and_jacobian`, full 1D derivative sums, `J⁻ᵀ` test
       gradients). Two kernels: one using every input (`xyz`, label, dof,
       value, a non-symmetric flux term), and isotropic elasticity. Run in 2D
       with NP = 2, 3, 4, 5, 7, 9 and in 3D with NP = 2, 3, 5, 8, on meshes with
       two geometric groups each and padded last batches.
     - Bitwise W-independence (W = 1, 3, 4, 8) and α/β semantics (β = 0 never
       reads `v`).
     - Laplace: symmetry, `A·1 = 0`, exact energy of linear fields.
     - Mass operator vs `inverse_mass`.
     - Elastic rigid modes (2D and 3D, deformed Q1) and exact elastic energy.
     - Mutation check: dropping the transpose in the push-forward fails 12/16.
   - Rough timing, 14 threads, scratch crate (Laplace / elastic, ns per DOF
     component):
     - P = 4, 8000 hexes: affine 0.9 / 1.0, Q1 2.1 / 1.6.
     - P = 2, 27000 hexes: affine 0.9 / 0.8, Q1 2.1 / 1.4.
     - Q1 costs about 2.3× affine for Laplace: per-point geometry dominates.
5. ~~**Kernels**~~: done (2026-11-19), `fields.rs` and `kernels.rs` (see "Physics,
   as implemented").
   - Tests (`tests/kernels.rs`, 7):
     - Kernel flags follow the fields (compile-time `const { assert!(..) }`);
       `PerLabel::from_pairs`.
     - `Function` vs sampled `PointData`: stiffness, mass and inverse mass agree
       to 1e-13.
     - `inverse_mass` with `Inverse(κ)` and `PerLabel`, against the `Mass`
       kernel; total mass `Σ ρ·area`.
     - Two-material exact energies, acoustic (`1 + 1/4`) and elastic.
     - Acoustic constants and symmetry with a `Function` density on a curved
       3D mesh; elastic rigid modes with mixed `Constant`/`PerLabel` moduli.
6. ~~**Time stepping**~~: done (2026-11-19) as library building blocks
   (`ops.rs`, see "Time integration") plus two examples; tests in
   `tests/ops.rs` (4): `local_mass`/`mass_inner` exact areas and norms,
   `apply` = `mlt_add` + `sum_shared`, `scale_by`, `max_eigenvalue` against a
   dense generalized eigensolve (acoustic two-material curved Q2, elastic),
   VTK counts and hex connectivity.
   - `examples/acoustic_modes.rs`, `examples/elastic_modes.rs` (`study` by
     default, `bench [--vtk DIR] [--every N] [--steps N]`), with shared code in
     `examples/common/`. **Removed 2026-11-22** (not in the spirit of the
     examples: one simple standalone file per case); the measurements below
     are kept as a record. Meshes: straight,
     bump-deformed Q1, curved Q2; the bump `ε·L·Π sin(πx_d/L_d)·(…)` vanishes on
     the boundary, so the box and its eigenmodes are exact. Elastic modes:
     divergence-free shear `(cos ax sin ay, −sin ax cos ay)`, traction-free on a
     box with sides multiple of π/a (3D: sum of the (x,y) and (y,z) modes).
   - Space study: CFL step, error against the dispersion-corrected reference
     `cos(ω̃t)`, `sin(ω̃dt/2) = ωdt/2` (the Taylor start
     `u¹ = u⁰ − dt²/2·M⁻¹Ku⁰` makes the scheme exact on a discrete eigenmode up
     to this shift). **Since step 9 the error is the true relative L² norm**,
     over-integrated (`examples/common::rel_l2_error`, 10 GL points per
     direction in 2D, 8 in 3D); the former GLL-node norm superconverged
     (rates ≈ 4 for P = 2, ≈ 6 for P = 3, 4). Measured rates on the finest
     pair: acoustics P = 2: 3.07–3.10, P = 4: 4.99–5.02 (all geometries);
     elasticity P = 2: 3.87–3.88, P = 4: 5.33–5.47 (still pre-asymptotic: the
     eigenvalue error, O(h^{2P}), dominates the phase on these meshes); 3D
     P = 3: 4.5–4.8 (two refinements only).
   - Time study (P = 6, curved Q2): rate 2.00. The final time is 1.3 periods: at
     whole periods `cos` is at an extremum and the rate reads 4.
   - Staggered energy drift ≤ 2e-14 (round-off) in every run.
   - Bench (14 threads, P = 4, curved Q2, 500 steps, energy off; ns per
     duplicated dof and step, VTK every 100 steps = 6 snapshots):
     - acoustic 2D 819k dofs: 1.41 ns, output 23 ms/snapshot (19 % of total);
     - acoustic 3D 1.54M dofs: 4.7 ns, 40 ms/snapshot (6 %);
     - elastic 2D 1.64M dofs: 1.0 ns, 29 ms/snapshot (16 %);
     - elastic 3D 3.07M dofs: 2.2 ns, 44 ms/snapshot (7 %).
     3D acoustics costs more per dof than 3D elasticity: the per-point Q2
     geometry is amortized over 1 component instead of 3. 3D snapshots are
     ~150 MB (ASCII).
Article first (Ramiche–Imperiale, decided 2026-11-20): mixed spaces and
quadratures (see "Design decisions: mixed spaces and quadratures"), in order;
each step tested before the next:
7. ~~Optional continuous numbering; inverse mass as an operator~~: done
   (2026-11-20). `Numbering { None, Continuous }` argument of `Space::new`,
   `c_` prefix rule, `InverseMass` (`apply`, `diagonal`); all call sites moved
   to the `c_` names (results unchanged: same CFL steps and power-iteration
   counts in the examples). Tests `tests/numbering.rs` (9): storage and
   element-local operators independent of the numbering (bitwise), the five
   `c_` panics, `inverse_mass` = `1/local_mass` and `c_inverse_mass` =
   `1/Σ_shared local_mass` (bitwise), `M⁻¹(Mu) = u` for both spaces,
   `apply` = `scale_by` (bitwise).
8. ~~Shape markers `Space<D, S>`, anisotropic counts; kernel selection and 1D
   tables move into the marker impls; `gll.rs` removed~~: done (2026-11-20).
   - `shape.rs`: `Shape<D>` (`N`, `NLOC`, `ISOTROPIC`, `Tables`,
     `points_1d`, `weights_1d`, `ref_derivative::<W>` — removed 2026-11-24,
     with `Gll1::d`), `Gll2<N0, N1>`,
     `Gll3<N0, N1, N2>`, `Iso2<N>`, `Iso3<N>`, per-direction `Gll1<N>` from
     `lobatto::tables`. `Space::multi_index`, `ref_point`, `points_1d`,
     `weights_1d` replace `gll()`. VTK tessellates per direction.
   - Geometry from Anamesh tables (`GeometricElement::tabulate`, added to
     Anamesh): `GeoTables`, `lagrange_1d` and the scalar map are gone; the Q1
     test of the classification uses the Anamesh Q1 element at the Q2 nodes.
     Round-off-level changes only: the examples give the same CFL steps and
     power-iteration counts.
   - Engine A/B, full tables vs the former 1D products (14 threads, P = 4,
     8000 hexes; Laplace / elastic ns per dof): Q1 1.7 / 1.0 vs 2.3 / 1.3,
     Q2 4.2–4.6 / 1.8 vs 5.2 / 2.2 (the 2× of an isolated single-element
     benchmark shrinks in the engine). Affine 0.7–0.8 / 0.7.
   - Tests (55 in total): engine against the brute-force reference on
     `Gll2<3, 5>`, `Gll2<2, 4>`, `Gll2<6, 4>`, `Gll3<2, 4, 3>`, `Gll3<5, 3, 4>`,
     `Gll3<4, 4, 6>` (affine/Q1/Q2 groups, padded batches); anisotropic
     measure, node Jacobians vs `get_element_grad_def`, multi-index/points;
     anisotropic + `Numbering::Continuous` panics.
9. ~~Quadrature list at construction, geometry stored or recomputed per entry,
   exact geometry from the mesh~~: done (2026-11-20).
   - Anamesh additions (validated): `Mesh::get_element_def`,
     `is_mapped_element`, `get_tensor_numbering` (see Anamesh/CLAUDE.md).
   - `quadrature.rs`: `Legendre`/`Lobatto` families, `Rule<F, M>`, `Gl<M>`,
     `Gll<M>`, `Quad2`/`Quad3`, trait `Quadrature<D>` (`M`, `NQ`, families),
     trait `Sweep<D, S>` on (quadrature, shape) pairs: node → point
     interpolation with `lobatto::tables::interpolation_matrix` and the
     rectangular kernels (values only so far; derivatives and transposes come
     with the mixed product, step 10).
   - `SpaceBuilder`: `.numbering(…)` (default `None`), `.node_geometry(…)`,
     `.quadrature::<Q>(GeoStorage)` (one call per rule, duplicates panic),
     `.build()`. An entry stores its points, weights, `PointGeo` and the typed
     sweep tables (`Box<dyn Any>`, found by `TypeId`; an undeclared `Q`
     panics). Since 2026-11-24 `Pointwise` borrows them (and the node rule's,
     `Space::node_sweep`) instead of building them at every call. `GeoStorage::Store` keeps `x`, `det J`, `cof J` per point
     (`StoredGeo`; `map_and_jacobian` rebuilds `J = cof(cof J)/det^(D−2)`).
   - Outputs: `quadrature_weights::<Q>()` (`w·|det J|`), `quadrature_xyz::<Q>()`,
     `quadrature_values::<Q, C>(u)` (`[elem][comp][q]`, W = 1 sweeps): true L²
     errors and integrals by plain sums.
   - Engine: geometry through `Space::geo_at` (stored or tables), group
     `MAPPED`. Results unchanged (same CFL steps, same timings with
     `Recompute`).
   - **Stored node geometry is the default** (decided 2026-11-20;
     `node_geometry(GeoStorage::Store)`, 14 threads,
     P = 4, 8000 hexes, Laplace / elastic ns per dof): Q1 0.9–1.0 / 0.9 (vs
     1.8–1.9 / 1.1 recomputed), Q2 0.9–1.0 / 0.9 (vs 3.8–4.5 / 1.9–2.1),
     affine 0.7 / 0.7. Memory 13 values per point in 3D (≈ 104 MB here).
   - Tests: `tests/quadrature.rs` (10): exact integration (GL and GLL),
     `Store` = `Recompute` bitwise (weights, points, values, `mlt_add`,
     `local_mass`), the node quadrature is the identity (bitwise), exact
     values and integrals on an anisotropic shape with a mixed-family rule,
     `Circle` (affine core + mapped arms, area π R² to 1e-11 vs > 1e-6 for the
     Q2 conversion, boundary nodes on the circle, continuity), deformed
     `Square`/`Cube` mapped, engine on mapped = engine on the applied Q1
     deformation, undeclared/duplicate quadrature panics. `tests/space.rs`:
     attached deformation is mapped (rotation stays affine), anisotropic
     continuous numbering on `Square`/`Cube`, anisotropic on unstructured
     panics.
10. ~~Mixed product; tensor inverse mass of a quadrature~~: done (2026-11-20).
    - `MixedKernel<D, CU, CV>` (same flags as `PointKernel`; blanket impl
      for every `PointKernel<D, C>` as `<D, C, C>`; merged into
      `PointKernel<D, CU, CV = CU>` on 2026-11-24). Kernels `Divergence`
      (`b(u, q) = ∫ c q div u`, `D` → 1) and `DivergenceTranspose`
      (`bᵀ(p, v) = ∫ c p div v = ∫ (c p I) : ∇v`, 1 → `D`: the gradient falls
      on the test function).
    - `mlt_add_mixed::<Q, CU, CV, …>(&u_space, &v_space, &kernel, α, u, β, v)`
      (`mixed.rs`): `Q` declared on either space; spaces from the same mesh
      (storage fingerprint, FNV of the element order and groups). Values and
      reference gradients of `u` at the points of `Q` (`Sweep::interpolate`,
      `Sweep::gradient`), geometry from `Q`'s entry, kernel, pull-back,
      tested with `Sweep::interpolate_t` / `gradient_t` on the test shape;
      W = 4 batches. In a mixed product `PointIn::dof = e·NQ + q`.
    - `Sweep` gained `interpolate_t`, `gradient(b)`, `gradient_t(b)`, `back`,
      `back_t` (`Tables1<N, M>`: interpolation, derivative, back matrices
      from `lobatto::tables`; private `fwd2/3`, `trans2/3`).
    - `Space::inverse_mass_with::<Q, F>(coef)`: element-local mass of the
      quadrature `Q` (same counts as the space), `M_K⁻¹ = B W_K⁻¹ Bᵀ`,
      `B = I⁻¹` the interpolation points → nodes; `InverseMass` has a tensor
      variant (`diagonal()` is `None`).
    - Timing (14 threads, 8000 Q1 hexes, displacement P = 4, pressure P = 2):
      Laplace `mlt_add` 1.11 ns/dof vs mixed on the nodes 1.97 (the
      identity sweeps are not skipped); `B` 0.83, `Bᵀ` 0.69 ns per
      displacement dof; tensor `M^P⁻¹` (GL 3) 0.67 ns per pressure dof.
    - Tests `tests/mixed.rs` (11): against a brute-force reference (lobatto
      1D bases, geometry from the mesh's own map at the points) in 2D (U ≠ V
      shapes incl. anisotropic, affine/Q1/Q2/mapped) and 3D; U = V on its
      nodes = `mlt_add`; `⟨B u, p⟩ = ⟨u, Bᵀ p⟩`; `∫ p div u` exact on
      polynomials, rotation divergence-free; `M⁻¹(M p) = p` with the
      Gauss–Legendre mass (Q1, Q2, mapped, 3D); tensor inverse on the GLL rule
      = diagonal inverse; three panics. Mutation (push-forward without
      transpose) fails 4/11.
11. ~~`max_eigenvalue` on closures~~: done (2026-11-20). `ops::power_iteration(x0,
    apply, inner, max_iter, rtol)`: `apply(x, y)` writes `y = M⁻¹A x` (any
    composite operator), `inner` the `M` product; Rayleigh quotient of
    `M`-normalized iterates (TetriX `lal::power_iteration` takes `A` and `M⁻¹`
    separately with the ∞-norm ratio: both are toolbox candidates).
    `Space::c_start_vector::<C>()` (continuous hashed start) and
    `c_max_eigenvalue` as a wrapper: same CFL steps and iteration counts in
    the examples. Test: `M⁻¹(R + λ Bᵀ (M^P)⁻¹ B)` (Q3/Q1, λ = 1e3, curved Q2)
    against a dense generalized eigensolve, 1e-8.
12. ~~Homogeneous Dirichlet and error kernels~~: done (2026-11-20). The article
    example itself (`b_h` option 2 as three global operators, λ = 1e3, k = 4)
    is redone by the user (decided 2026-11-20), with the suggestions: a
    manufactured solution away from the boundary or homogeneous Dirichlet,
    non-zero pressure, λ ∈ {1, 1e3, 1e6}, pressure errors, distorted and
    curved (`Circle`, exact geometry) meshes, true over-integrated errors.
    - `Space::c_boundary_mask(&mesh, labels)`: `0` at every copy of a
      boundary node, `1` elsewhere; `None` = the topological boundary
      (Anamesh `get_boundary_faces`), `Some(labels)` = facets of the labelled
      boundary elements (Anamesh `get_parent_face`, added for this); local
      nodes on a facet by `ReferenceElement::is_on_edge` / `is_on_face`, then
      propagated to every copy by `c_sum_shared`. With the diagonal mass,
      multiplying the update (and the power-iteration closure) by the mask is
      exactly the scheme on `V_h ∩ H¹₀`.
    - Error kernels `L2Error { exact }` and `H1Error { exact_grad }`
      (`MixedKernel<D, C, 1>`, source = pointwise squared error): with
      `mlt_add_mixed` on an over-integration rule and any scalar test space,
      the sum of the output entries is the integral (partition of unity). No
      `quadrature_gradients` needed. (Plain `mlt_add` with the stiffness on
      `I_h u − u_h` measures the distance to the interpolant, which
      superconverges.)
    - Tests `tests/dirichlet.rs` (4): masks vs the geometric boundary
      (`Square` all/three/one labels, deformed unstructured, `Cube`,
      `Circle` exact geometry), boundary DoF count; masked `M⁻¹K` on the
      Dirichlet mode `sin πx sin πy` gives `2π² u` (1e-5); error kernels
      integrate `∫ (xy)² = 8/9`, `∫ |∇(xy)|² = 10/3`, zero on a representable
      field.
13. ~~`ElementOperator` hook~~: done (2026-11-20), `element.rs`.
    - `ElementOperator<D, SU, SV, CU, CV>` (`Scratch`, `READS_INPUT`,
      `scratch::<W>()`, `apply::<W>(batch, u, v, scratch)`: gathered
      `[c][node][lane]` in, local result out, overwritten); `Batch` (`u_space`,
      `v_space`, `kind`, `first`, `n_valid`, `element(lane)`, `label(lane)`,
      `geometry(set, lane, p) -> (x, det J, cof J)`, `quadrature::<Q>()`);
      `PointSet` (`Space::node_points`, `Space::quadrature_points::<Q>`:
      weights, reference points, geometry).
    - `mlt_add_element(&u_space, &v_space, &op, α, u, β, v)`: the shared
      plumbing (groups, W = 4 batches, parallel chunks, scratch, gather,
      scatter, checks). Operators can carry per-element data indexed by
      `Batch::element`.
    - `mlt_add_mixed` is now `Pointwise<K, Q, SU, SV>` run by
      `mlt_add_element`: outputs bit-identical (hash of 2D/3D, all kinds,
      stored/recomputed, five kernels, before/after), same speed. The
      same-space `mlt_add` keeps its own path (hc kernels) — until
      2026-11-24, when it became `Pointwise` on the node rule.
    - Tests `tests/element.rs` (4): hand-written nodal mass = `Mass` kernel
      engine (bitwise), per-element data + labels + padded lanes + α/β, a
      declared quadrature through the batch (area 2.6), different meshes
      panic.
14. ~~Fused `K^P = Bᵀ (M^P)⁻¹ B` per element~~: done (2026-11-20).
    - Module `lobattox::operators` (`src/operators/`): hand-written
      `ElementOperator`s for specific formulations, **written with the public
      API only** (so they can move to user code or another crate as is;
      `Space::same_mesh` added for that). Option 3 goes there too.
    - `operators::FusedKp<D, SU, SP, QB, QM>::new(&disp, &pres, mass_coef)`:
      `QB` the `b_h` rule (declared on either space; option 2 = `Gll<k+1>`),
      `QM` the pressure mass rule (declared on the pressure space, pressure
      counts, e.g. `Gl<k−1>`); per batch: gradients at the `QB` points,
      weighted divergence → pressure nodes, tensor `(M^P)⁻¹` per element,
      pressure → `QB` points, pulled-back `p·cof` → displacement nodes (the
      `QB` geometry is evaluated once for `B` and `Bᵀ`). Run with
      `mlt_add_element(&disp, &disp, &kp, λ, u, β, y)`.
    - Equal to the three global operators to 1e-12 (2D affine+Q1,
      affine+Q2, curved Q2, mapped, `b_h` with GLL or GL; 3D Q1/Q2);
      `⟨K^P u, u⟩ = ⟨B u, (M^P)⁻¹ B u⟩ ≥ 0`; wrong mass rule panics
      (`tests/fused.rs`, 3). A per-lane weight mutation fails 2/3.
    - Timing (14 threads, 8000 Q1 hexes, displacement P = 4, pressure P = 2,
      `λ K^P u` added to `y`): fused 1.14 ns per displacement dof vs 1.52
      for the three operators, and no global pressure vector.
15. ~~`b_h` option 3 (Green's formula) as a hard-coded `ElementOperator`~~: done
    (2026-11-20). **`GradDivQ4Q2` (lumped GLL 3 pressure mass) removed on
    2026-11-24** (decided: the Q4–Q2 operator uses the Gauss–Legendre mass
    only); its `B`/`Bᵀ` live in `FusedKpQ4Q2` (outputs bitwise unchanged),
    scratch `FusedKpQ4Q2Scratch`, and `tests/grad_div.rs` checks the
    equivalence of the options with the Gauss–Legendre mass (same meshes:
    Q1 to 1e-12, Q2 6e-16 2D / 8e-16 3D, Circle 1.3e-3, twisted cube
    4.3e-4). Record of the lumped operator:
    `operators::GradDivQ4Q2::<D>::new(&disp, mass_coef)`
    (D = 2, 3, Q4 displacement only): port of cornea `GradDiv.h` (decided:
    the hard-coded Q4–Q2 operator, no general face integrals). Precomputed
    `bndy` (`BndyMatrix`: `cof(J)·n̂ × 1D face weights` at the nodes of each
    reference face) and lumped GLL 3 (Simpson) pressure mass at the
    displacement nodes 0, 2, 4 (`InvMass`); per batch `B` = `Iᵀ(bndy·u) +
    Σ_d G_dᵀ(−w cof(J)ᵀ u)`, lumped `(M^P)⁻¹`, `Bᵀ` = `bndy·p − w cof(J) ∇ξ p`.
    cornea's hard-coded Q2 → Q4 interpolations are lobatto's rectangular
    kernels 3 → 5 (`Sweep` of the GLL 5 rule on `Iso<3>`).
    - **Equivalence of options 2 and 3** (`tests/grad_div.rs`, 3), against
      `FusedKp` with `b_h` on the GLL 5 rule and the same lumped GLL 3
      pressure mass: equal to 1e-12 on Q1 meshes (affine, bilinear, mixed,
      interpolated bending; 3D), as the article proves — and **also on Q2
      (curved) meshes** (1.7e-15 2D, 1.2e-15 3D): GLL summation by parts is
      exact, and so is the discrete Piola identity `Σ_b ∂_b cof(J)_cb = 0`
      since each `cof_cb` is a polynomial of degree ≤ 4 in `ξ_b` for a Q2
      map. On mapped elements with a non-separable map they differ (Circle
      arms 1.0e-3, twisted cube 3.0e-4). `K₃` symmetric, PSD. A flipped
      face normal fails 3/3.
    - Timing (14 threads, 8000 Q1 hexes, Q4/Q2, lumped GLL 3 mass, `λ K^P u`
      added): `GradDivQ4Q2` 0.90 ns per displacement dof vs `FusedKp`
      (option 2) 1.22.
    - Note: cornea's pressure mass is the lumped GLL 3 rule; the revised
      article uses Gauss–Legendre 3 points.
    - `operators::FusedKpQ4Q2::<D>::new(&disp, mass_coef)` (2026-11-20): `B`,
      `Bᵀ` of `GradDivQ4Q2` (its batch code takes the inverse pressure mass
      as a closure; `GradDivQ4Q2` passes the lumped one, results unchanged)
      with the article's Gauss–Legendre pressure mass (`PressureMass2/3` =
      `Gl<3>` rule, declared on the displacement space), tensor inverse
      `B_P W⁻¹ B_Pᵀ`. Equals `FusedKp` (option 2, same mass) to 1e-12 on
      affine, Q1, Q2 (2D, 3D); differs on the Circle (1.2e-3); symmetric
      PSD. Timing (as above): 1.04 ns per displacement dof vs `FusedKp`
      1.24 with the same mass.
16. ~~**Volume sources** `f(x)·g(t)`~~: done (2026-11-20).
    `Space::local_load::<C>(f)`: element-local `F_i = ∫ f φ_i` at the GLL
    nodes (`w_i |det J_i| f(x_i)`; with the lumped mass `M⁻¹ Σ_shared F` =
    nodal `f / ρ`); `local_load_with::<Q, C>(f)`: the same with a declared
    quadrature (`interpolate_t` of the weighted values; over-integrated or
    projection loads). The time factor `g(tⁿ)` scales `M⁻¹F` in the caller's
    loop. Tests `tests/sources.rs` (3): `f = 1` is the local mass; nodal
    `M⁻¹F`; `Σ F = ∫ f` and `⟨F, v⟩ = ∫ f v` exact with Gauss–Legendre, the
    node rule = `local_load`; forced leapfrog reproducing `cos πx cos πy · t²`
    (1e-5).
17. **I/O**: ~~tessellated ASCII VTK~~ (done with step 6); ~~Vizir HOSolAt~~
    (done 2026-11-20, `vizir.rs`, port of C++ `IO_Vizir.h`):
    `VizirWriter::new(&space, &mesh)` (header and GLL node positions once),
    `write(path, data, components)` (rows formatted in parallel, straight
    from the duplicated storage). Rows follow the element order of the
    `.mesh` section written by Anamesh `write_mesh` (mesh-element order, not
    storage order); `Dimension 3` to match Anamesh's `.mesh` (the C++ wrote
    the space dimension); `1 C` then `P NLOC`, as the C++. One element type
    and an isotropic shape (panics otherwise). Tests `tests/vizir.rs` (2):
    parsed back (header, positions = reference nodes, rows = values in mesh
    order, count = the `.mesh` section) on Q1 (storage ≠ mesh order), Q2,
    mapped, 3D Q2; anisotropic panics. Not checked in Vizir itself.
18. **Point sources** via `(element, uvw)`; `Mesh::locate` in Anamesh.

Examples ported from the C++ (2026-11-21), **one file per C++ example, no
shared code** (for newcomers: one `main` with numbered steps mesh → space →
physics → time step → source → output → leapfrog → timing; helpers such as
`ricker` and `bump` repeated in each file): `examples/acoustic_wave_square.rs`,
`acoustic_wave_cube.rs`, `acoustic_wave_circle.rs`, `acoustic_wave_2d_mesh.rs`
(same parameters as `core/Testing/FESpace/AcousticWave_{Square, Cube, Circle,
2D_Mesh}.cpp`: `CInfty0` bump × `Ricker`, or the square's initial velocity;
anisotropic `A` through the kernel `AnisotropicAcoustic`; source as
`M⁻¹ Σ local_load`). **Output by default** in `out/<example>/` under the
crate (`CARGO_MANIFEST_DIR`, so independent of the launch directory;
`/out/` is git-ignored): `<example>.mesh` (Anamesh), `<example>.<k>.vtk` and
`<example>.<k:05>.sol` every Δt_out, written in the loop. The disc is Anamesh
`Circle(n = 54)` (1620 elements, exact geometry) vs the C++ `Circle(6, 6)`
(1728). The cube uses 12³ hexahedra instead of the C++ 3³, which does not
resolve the source (bump of radius 0.05, Ricker ~14 Hz, speed 0.71 along z):
‖u(0.4)‖ is 5× too large on 3³, within 1 % on 12³, 0.3 % on 24³ (refinement
study 2026-11-21). The square writes every 0.1 (C++: 0.01). The default mesh
of `acoustic_wave_2d_mesh` is
`../ondomathx-mesh/assets/Potatoe.msh` (or the first argument).
Elastodynamics examples (2026-11-21), same style (one file each, no shared
code, output in `out/<example>/`), bump of radius 0.1 on the x component ×
`Ricker(t, 0.2, 500)`, T = 1, snapshots every 0.1, order 4, 40 × 40 squares and
12³ cubes:
- `elastic_wave_square.rs`, `elastic_wave_cube.rs`: isotropic `IsotropicElastic`
  (ρ = 1, λ = 2, μ = 1), lumped mass, leapfrog.
- `nearly_incompressible_wave_square.rs`, `…_cube.rs`: Ramiche–Imperiale
  formulation, `M Ü + R U + λ K^P U = F` with `R` = `IsotropicElastic`
  (λ = 0, μ = 1), `K^P` = `FusedKpQ4Q2` (Green's `B`, Gauss–Legendre 3 pressure
  mass), λ = 1000; CFL step by `power_iteration` on `M⁻¹(R + λK^P)` (dt ≈ 15.5×
  smaller than the elastic one, as `√((λ+2μ)/(2+2))`); the pressure
  `p = λ (M^P)⁻¹ B u` (`mlt_add_mixed` with `Divergence` on the displacement GLL
  rule, `inverse_mass_with` on the discontinuous Q2 pressure space) written at
  the snapshots (`<example>_pressure.<k>.vtk/.sol`). Plain leapfrog (the
  leapfrog-Chebyshev scheme of Merlini et al. is not implemented).
- Timings (14 threads, with output): elastic square 0.66 ms/step (724 steps),
  cube 2.3 ms/step (257); nearly incompressible square 1.34 ms/step (11 252
  steps, 15 s), cube 3.6 ms/step (4085 steps, 15 s). Amplitudes stay bounded
  over the snapshots (stable).
- Comparison (measured with the former combined example, same loops; 14
  threads, same machine; C++ `-O3` OpenMP, patched copies with a chrono
  around the loop, output on/off), ms per step without output,
  LobattoX vs C++: square 0.69 vs 1.14 (1.6×), cube 0.22 vs 2.70 (12×, tiny:
  the C++ loop applies stiffness + two masses), circle 0.57 vs 1.63 (2.9×;
  per DoF 9.7 vs 26.1 ns), potato mesh 0.38 vs 2.45 (6.4×). With output the
  square (101 snapshots) takes 1.30 s vs 41.9 s for the loop. ASCII output is
  ~2× larger than the C++ (17 significant digits vs 6).

Later, unordered:
- Boundary conditions: non-homogeneous Dirichlet and boundary sources
  (`get_parent_face` is there); paraxial ABC (needs Anamesh `get_parent_face`
  plus boundary kernels); PML.
- Physics: anisotropic `Cijkl`, nonlinear (Neo-Hookean) laws.
- Output: binary VTK (ASCII 3D snapshots are ~150 MB), receivers.
- Performance: ~~skip the identity sweeps~~ (done 2026-11-21: a `Sweep`
  pass whose 1D interpolation is the identity — Lobatto rule with `M = N` —
  is not run, bitwise the same results; 8000 Q1 hexes, 14 threads: mixed
  Laplace on the nodes 1.94 → 1.45 ns/dof (`mlt_add` 1.2), `B` on the GLL 5
  rule 0.85 → 0.65, `Bᵀ` 1.0 → 0.75 ns per displacement dof); ~~the hc
  derivative kernels on the remaining `D_b` pass of the node rule~~ and
  ~~one product path~~ (done 2026-11-24, below); W > 1 in
  `quadrature_values`.
  - **One product path** (2026-11-24): `Shape::Nodes` (the node rule as a
    quadrature type), hc kernels in `Sweep::gradient(_t)` on the node rule,
    `Space::mlt_add(_batched)` = `Pointwise::<K, S::Nodes, S, S>` run by
    `mlt_add_element_batched`; the engine's own driver and point loop
    (~250 lines) and `Shape::ref_derivative` removed. Outputs bitwise
    identical to before (hash of 14 cases: 2D/3D, affine/Q1/Q2/mapped,
    stored/recomputed, hc and generic counts, W = 1, 3, 4, 8, five
    kernels). What made `Pointwise` as fast as the engine (found in the
    assembly, not the source): `geo_at` / `tab_eval` forced inline (they were
    out-of-line calls per point and lane); every scratch buffer re-sliced to
    its constant length at the top of the per-kind loop (bounds checks
    dropped: the scratch is allocated in another function); driver sizes
    recomputed from the consts inside the rayon closure (captured, they were
    unknown runtime values in the gather/scatter loops); one gradient/flux
    buffer; sweep tables borrowed, not rebuilt per call (20 µs at P = 4);
    unused scratch buffers empty; the first transposed pass writes
    `0 + t` instead of zeroing `y`. Timings (scratch benchmark, ns/dof,
    before → after; 3D P = 4, 512 hexes, 1 thread): `mlt_add` Laplace
    affine 6.0–6.4 → 5.6, Q1 7.3–8.0 → 5.6–5.8; mixed on the nodes Laplace
    Q1 8.5–9.8 → 5.5–5.6, elastic 7.0–8.1 → 5.9–6.0; `div` on GL 3 2.1–2.7 →
    1.8–2.0. 2D P = 4 (40 × 40, 1 thread): Laplace 5.1 → 4.1–4.2, elastic
    5.17 → 5.26–5.35 (+2 %, the one case slower). 14 threads, 12³ P = 4
    cube: `mlt_add` Laplace 1.15–1.35 → 1.07–1.14, mixed on the nodes
    1.5–1.7 → 1.05–1.14 (Laplace), 1.46–1.53 → 0.87–0.92 (elastic). The
    examples `acoustic_wave_cube` and `elastic_wave_square` measured 0–20 %
    slower per step, but under a load average of 3–10 from other processes:
    **to re-measure on a quiet machine** before trusting either way.
  - Follow-up simplifications (2026-11-24, no public name changed):
    `local_load` = `local_load_with::<S::Nodes>` (bitwise); the hc helpers
    (`hc_2d`, `hc_3d`) live in `quadrature.rs`, their only user;
    `Space::weight` reads `node_weights`; `PointIn::zeros` /
    `PointOut::zeros` (crate-private); `ops::scale_nodes` shared by
    `scale_by` and the diagonal `InverseMass::apply`. Test
    `node_rule_is_implicit` (implicit = declared node rule, `mlt_add_mixed`
    on it = `mlt_add`, `local_load` = `local_load_with`, all bitwise).
  - Readability pass (2026-11-25), outputs bitwise unchanged (`mlt_add`
    hash of 14 cases, `FusedKpQ4Q2` hash of 5 meshes), same speed:
    - **Call sites**: kernels and fields are `impl Trait` arguments, and the
      mixed and element products are methods of the trial space, like
      `mlt_add`: `space.mlt_add::<C>(&k, …)`,
      `u_space.mlt_add_mixed::<Q, CU, CV, _>(&v_space, &k, …)` (the `_` is
      the test shape: Rust infers no generic argument once one is written,
      and `Q` must be written), `u_space.mlt_add_element(&v_space, &op, …)`
      (`CU`, `CV` inferred from the operator), `c_apply::<C>`,
      `c_max_eigenvalue::<C>`, `inverse_mass_with::<Q>`. The free functions
      `mlt_add_mixed` / `mlt_add_element` are gone.
    - **Named geometry groups**: `GroupRanges { affine, q1, q2, mapped }`
      replaces `ranges: [Range; 4]` + `GeoKind::index()`; `PointGeo { q1,
      q2, mapped }` with `GroupGeo::{Tabulated, Stored}` replaces
      `tabs[g − 1]` / `stored[g − 1]`; `geo_at` and `map_and_jacobian` are a
      `match` on the kind.
    - **Named steps**: `Space::construct` = `check_mesh`,
      `classify_elements` (`classify_mapped`, `classify_nodal`),
      `group_by_kind`, `storage_fingerprint`, `tensor_weights`;
      `Pointwise::apply_kind` = `to_points`, `point_loop`, `to_nodes`
      (forced inline, buffers as arguments).
    - **`space/` module** (was `space.rs`, 1264 lines; code moved, not
      changed): `mod.rs` (types, `Space` struct, queries, `sample`),
      `build.rs` (`SpaceBuilder`, construction, classification,
      `point_geo`, quadrature entries), `geo.rs` (`geo_at`,
      `map_and_jacobian`), `points.rs` (node set, quadratures, sweep tables,
      `quadrature_*`), `mass.rs` (`local_mass`, `local_load*`,
      `inverse_mass*`, `InverseMass`), `continuous.rs` (the numbering and
      `c_sum_shared`, `c_multiplicity`, `c_boundary_mask`). Older entries
      of this record still say `space.rs`. Element renumbering for locality; fusing the shared-DOF sum with
  the update; benchmarking nalgebra `SMatrix` against hand-rolled ≤3×3 algebra in
  the point kernel (geometry and push-forward), and switching if it pays.
  - Q1/Q2 geometry by sum factorization over the batch (interpolate the
    geometric nodes direction by direction, `D` derivative passes), to compare
    with the per-point full-table sums (step 8). Q1 is now about 2.2× the
    affine cost for Laplace, Q2 about 5.5×.
  - A criterion bench for `mlt_add` (affine/Q1/Q2, P = 1..8, D = 2, 3).
