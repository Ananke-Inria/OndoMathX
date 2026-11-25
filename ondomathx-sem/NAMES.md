# LobattoX — public names

Every public name of the crate (`src/`), grouped by module, with what it does.
Column **Tied** marks names that refer to a physics or to the article
(candidates for a more generic name): `phys` = physics, `art` = the
Ramiche–Imperiale article / cornea, blank = generic.

Notation: `D` dimension, `C` components, `NLOC` nodes per element, `n_dup =
n_elem·NLOC` duplicated nodes, `W` batch width (lanes), storage
`[elem][comp][node]`.

## Crate root re-exports (`lib.rs`)

`Batch, ElementOperator, PointSet` · `PointIn, PointKernel,
PointOut` · `GeoStorage, cofactor, det` · `Pointwise` · `dot, power_iteration, stable_dt, EigenEstimate` · `Gl, Gll,
Quad2, Quad3, Quadrature, Rule, Sweep` · `Gll2, Gll3, Iso2, Iso3, Shape` ·
`GeoKind, InverseMass, Numbering, Space, SpaceBuilder` · `VizirWriter` ·
`VtkField, VtkWriter`.

Modules: `element`, `engine`, `fields`, `geometry`, `kernels`, `mixed`,
`operators`, `ops`, `quadrature`, `shape`, `space`, `vizir`, `vtk`.

---

## `space` — the discrete space

### Types

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `Space<D, S>` | struct | | Discontinuous GLL space on a mesh (duplicated storage), optional continuous numbering, declared quadratures. Does not borrow the mesh. |
| `SpaceBuilder<'m, D, S, M>` | struct | | Builder of a `Space`: numbering, node geometry storage, quadratures. |
| `Numbering` | enum | | Numbering on top of the storage. |
| `Numbering::None` | variant | | Discontinuous space; `c_` functions panic. |
| `Numbering::Continuous` | variant | | Copies of a shared node are one global DoF. |
| `GeoKind` | enum | | Geometric type of an element; storage is grouped by kind. |
| `GeoKind::Affine` | variant | | `x = x0 + Jξ`, one Jacobian per element. |
| `GeoKind::Q1` | variant | | General bi/trilinear element (`2^D` nodes). |
| `GeoKind::Q2` | variant | | General bi/triquadratic element (`3^D` nodes). |
| `GeoKind::Mapped` | variant | | Map given by the mesh (analytic or attached deformation), not affine; geometry stored. |
| `GeoKind::ALL` | const | | Storage order of the groups. |
| `InverseMass<D, S>` | struct | | Inverse mass operator (diagonal or tensor form). |

### `Space` — construction and queries

| Name | Tied | What it does |
|---|---|---|
| `Space::new(mesh, numbering)` | | Shorthand for the builder: numbering, stored node geometry, no quadrature. |
| `Space::builder(mesh)` | | Starts a `SpaceBuilder`. |
| `Space::NLOC` | | Nodes per element and component, `N0·N1(·N2)`. |
| `same_mesh(other)` | | Whether `other` was built from the same mesh (condition of mixed/element products). |
| `has_numbering()` | | Whether a continuous numbering exists. |
| `multi_index(node)` | | `(i, j[, k])` of a local node (tensor order, `i` fastest). |
| `points_1d(d)` | | GLL points on `[0,1]` of direction `d`. |
| `weights_1d(d)` | | GLL weights of direction `d`. |
| `ref_point(node)` | | Reference point of a local node in `[0,1]^D`. |
| `weight(node)` | | Tensor GLL weight `w_i·w_j(·w_k)` of a local node. |
| `n_elem()` | | Number of volume elements. |
| `n_dup()` | | Number of duplicated nodes, `n_elem·NLOC`. |
| `c_n_global()` | | Number of global DoFs of the continuous numbering. |
| `elements(kind)` | | Storage range of the elements of a `GeoKind`. |
| `kind(e)` | | `GeoKind` of storage element `e`. |
| `mesh_element(e)` | | Mesh index of storage element `e`. |
| `label(e)` | | Mesh label of storage element `e`. |
| `map_and_jacobian(e, node)` | | `x` and `J = ∂x/∂ξ` at a node (setup, tests). |
| `node_points()` | | The nodes as a `PointSet` (for `ElementOperator`s). |
| `quadrature_points::<Q>()` | | Declared quadrature `Q` as a `PointSet`, if declared. |

### `Space` — vectors, assembly, masses, loads

| Name | Tied | What it does |
|---|---|---|
| `c_sum_shared::<C>(v)` | | Shared-DoF sum: each copy of a shared node gets the sum over its copies (assembly). |
| `c_multiplicity()` | | Number of copies of each duplicated node. |
| `sample::<C>(f)` | | Samples `f(x)` at every duplicated node. |
| `local_mass(coef)` | | Element-local lumped mass `c·w·|det J|`, one value per duplicated node. |
| `mass_inner::<C>(m, a, b)` | | `Σ m·a·b`: the mass product with `m` from `local_mass`. |
| `scale_by::<C>(diag, v)` | | `v[e][c][node] *= diag[e·NLOC+node]` for every component. |
| `local_load::<C>(f)` | | Element-local load `F_i = ∫ f φ_i` with the GLL rule. |
| `local_load_with::<Q, C>(f)` | | Same with the declared quadrature `Q`. |
| `inverse_mass(coef)` | | `InverseMass` of the element-local lumped mass (discontinuous). |
| `c_inverse_mass(coef)` | | `InverseMass` of the assembled lumped mass (continuous). |
| `inverse_mass_with::<Q>(coef)` | | `InverseMass` of the element-local mass integrated with `Q` (tensor form `B W⁻¹ Bᵀ`). |
| `c_boundary_mask(mesh, labels)` | | Dirichlet mask: `0` on the (labelled or whole) boundary, `1` elsewhere. |
| `quadrature_weights::<Q>()` | | `w_q·|det J|` at every point of `Q`, `[elem][q]`. |
| `quadrature_xyz::<Q>()` | | Physical points of `Q`, `[elem][q]`. |
| `quadrature_values::<Q, C>(u)` | | Values of `u` at the points of `Q`, `[elem][comp][q]`. |

### `Space` — operators (`engine.rs`, `ops.rs`)

| Name | Tied | What it does |
|---|---|---|
| `mlt_add::<C>(kernel, α, u, β, v)` | | `v ← βv + αAu` for a `PointKernel`, same space, element-local, `W = 4` (the mixed product on the node rule `S::Nodes`). |
| `mlt_add_batched::<C, W>(…)` | | Same with an explicit batch width `W`. |
| `c_apply::<C>(kernel, u, out)` | | `out = Σ_shared(K_loc u)`: `mlt_add` (β = 0) then `c_sum_shared`. |
| `c_start_vector::<C>()` | | Reproducible pseudo-random continuous vector (power-iteration start). |
| `c_max_eigenvalue::<C>(kernel, mass, …)` | | Largest eigenvalue of `M⁻¹K` by power iteration; returns `EigenEstimate`. |

### `SpaceBuilder`

| Name | Tied | What it does |
|---|---|---|
| `numbering(n)` | | Sets the numbering (default `None`). |
| `node_geometry(storage)` | | Stored or recomputed geometry at the nodes (default `Store`). |
| `quadrature::<Q>(storage)` | | Declares quadrature `Q` and its geometry storage; panics if declared twice. |
| `build()` | | Builds the `Space`. |

### `InverseMass`

| Name | Tied | What it does |
|---|---|---|
| `apply::<C>(v)` | | `v ← M⁻¹v`. |
| `diagonal()` | | The diagonal of `M⁻¹` when it is diagonal, else `None`. |

---

## `shape` — GLL point counts (type markers)

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `Shape<D>` | trait | | GLL counts per direction of a space; 1D tables and derivative sweeps. |
| `Shape::N` | const | | Points per direction (`[usize; 3]`, unused entries 1). |
| `Shape::NLOC` | const | | Nodes per element. |
| `Shape::ISOTROPIC` | const | | Same count in every direction. |
| `Shape::Tables` | assoc. type | | Per-direction 1D data (GLL points, weights). |
| `Shape::Nodes` | assoc. type | | The shape's own node rule as a quadrature type (`Quad2<Gll<N0>, Gll<N1>>`, …): the rule of the same-space products. |
| `Shape::tables()` | fn | | Builds them from `lobatto::tables`. |
| `Shape::points_1d(t, d)` / `weights_1d(t, d)` | fn | | 1D GLL points / weights of direction `d`. |
| `Gll1<N>` | struct | | 1D GLL data: `xi` (points), `w` (weights). |
| `Gll2<N0, N1>` | struct | | 2D shape, `N0 × N1` GLL points. |
| `Gll3<N0, N1, N2>` | struct | | 3D shape, `N0 × N1 × N2` GLL points. |
| `Iso2<N>` | type alias | | `Gll2<N, N>` (degree `N − 1`). |
| `Iso3<N>` | type alias | | `Gll3<N, N, N>`. |

---

## `quadrature` — quadrature rules (type markers)

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `Family` | trait | | Point family of a 1D rule (`GAUSS` const). |
| `Legendre` | struct | | Gauss–Legendre family (open points). |
| `Lobatto` | struct | | Gauss–Lobatto family (with the endpoints). |
| `Rule<F, M>` | struct | | 1D rule: `M` points of family `F` on `[0,1]`. |
| `Gl<M>` | type alias | | `Rule<Legendre, M>` (exact to degree `2M − 1`). |
| `Gll<M>` | type alias | | `Rule<Lobatto, M>` (exact to degree `2M − 3`). |
| `Quad2<A, B>` | struct | | 2D tensor rule from two 1D rules. |
| `Quad3<A, B, C>` | struct | | 3D tensor rule. |
| `Quadrature<D>` | trait | | A tensor rule: `M` (points per direction), `NQ` (points per element), `families()`. |
| `Sweep<D, S>` | trait | | Node ↔ point operators between shape `S` and the rule, by sum factorization. |
| `Sweep::Tables`, `TMP`, `tables()` | items | | 1D matrices, scratch size per lane, constructor. |
| `Sweep::interpolate` / `interpolate_t` | fn | | Nodes → points values `I` / transpose `Iᵀ`. |
| `Sweep::gradient` / `gradient_t` | fn | | Nodes → points reference derivative `G_b` / transpose. |
| `Sweep::back` / `back_t` | fn | | Points → nodes interpolation back `B` / transpose. |
| `Tables1<N, M>` | struct | | 1D matrices of one direction: `interp`, `deriv`, `back`. |

---

## `geometry`

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `GeoStorage` | enum | | How the geometry of a point set is evaluated (affine/Q1/Q2 groups). |
| `GeoStorage::Store` | variant | | `x`, `det J`, `cof J` stored at every point. |
| `GeoStorage::Recompute` | variant | | Recomputed at each use from the geometric nodes. |
| `cofactor(m)` | fn | | Cofactor matrix `det(J)·J⁻ᵀ` (2×2, 3×3). |
| `det(m)` | fn | | Determinant (2×2, 3×3). |

---

## `engine` — point kernels

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `PointKernel<D, CU, CV = CU>` | trait | | Pointwise map of an operator `a(u, v) = ∫ G(u):∇v + S(u)·v`, `CU` → `CV` components; `PointKernel<D, C>` for a same-space one. |
| `apply(&PointIn, &mut PointOut)` | fn | | The pointwise map. |
| `USES_VALUE`, `USES_GRAD`, `USES_XYZ`, `USES_LABEL`, `USES_DOF` | consts | | Compile-time switches of the inputs. |
| `OUTPUTS_FLUX`, `OUTPUTS_SOURCE` | consts | | Compile-time switches of the outputs. |
| `PointIn<D, C>` | struct | | Kernel input: `value`, `grad` (`[c][d] = ∂u_c/∂x_d`), `xyz`, `label`, `dof`. |
| `PointOut<D, C>` | struct | | Kernel output: `flux` (tested with `∇v`), `source` (tested with `v`). |

---

## `fields` — scalar coefficients

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `Field<D>` | trait | | Scalar coefficient evaluated at a point: `eval(&PointIn)`, flags `NEEDS_XYZ/LABEL/DOF`. |
| `Constant(f64)` | struct | | Same value everywhere. |
| `PerLabel(Vec<f64>)` | struct | | One value per mesh label; `from_pairs(&[(label, value)])`, missing labels `NaN`. |
| `Function { f, params }` | struct | | `f(params, xyz)` evaluated on the fly. |
| `PointData(Vec<f64>)` | struct | | One value per duplicated node; `PointData::sample(&space, f)`. |
| `Inverse<F>(F)` | struct | | `1 / F`. |

---

## `kernels` — ready-made point kernels

| Name | Fields | Tied | Form / what it does |
|---|---|---|---|
| `AcousticStiffness<R>` | `rho` | phys | `∫ (1/ρ) ∇u·∇v` (flux `∇u/ρ`), scalar. |
| `AnisotropicAcoustic<D>` | `a: [[f64; D]; D]` | phys | `∫ (A∇u)·∇v`, constant matrix `A`, scalar. |
| `IsotropicElastic<L, M>` | `lambda`, `mu` | phys | `∫ σ(u):ε(v)`, `σ = λ tr(ε) I + 2μ ε`, `C = D`. |
| `Mass<F>` | `coef` | | `∫ c u·v`, any `C` (no derivative). |
| `Divergence<F>` | `coef` | | Mixed `D → 1`: `∫ c q div u`. |
| `DivergenceTranspose<F>` | `coef` | | Mixed `1 → D`: `∫ c p div v`. |
| `L2Error<F>` | `exact` | | Mixed `C → 1`: source `|u − u_ex(x)|²` (sum of the output = squared L² error). |
| `H1Error<F>` | `exact_grad` | | Mixed `C → 1`: source `|∇u − ∇u_ex(x)|²` (H¹ semi-norm error). |

---

## `mixed` — products between two spaces

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `u_space.mlt_add_mixed::<Q, CU, CV, _>(v_space, kernel, α, u, β, v)` | method | | `v ← βv + αAu` for a `PointKernel<D, CU, CV>`, integrated with quadrature `Q` of either space (declared, or the node rule `S::Nodes`). |
| `Pointwise<'a, D, K, Q, SU, SV>` | struct | | The `ElementOperator` behind `mlt_add_mixed` and `Space::mlt_add`; `Pointwise::new(u_space, v_space, kernel)`. |
| `PointwiseScratch` | struct | | Per-task scratch of `Pointwise`. |

---

## `element` — hand-written element operators

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `ElementOperator<D, SU, SV, CU, CV>` | trait | | Local operator `v_e ← A_e u_e` on a batch of `W` elements. |
| `ElementOperator::Scratch`, `scratch::<W>()` | items | | Per-task scratch type and constructor. |
| `ElementOperator::READS_INPUT` | const | | Whether the input is gathered. |
| `ElementOperator::apply::<W>(batch, u, v, scratch)` | fn | | The local operator (`[c][node][lane]` in and out). |
| `u_space.mlt_add_element(v_space, op, α, u, β, v)` | method | | Driver: groups, batches, parallel chunks, gather/scatter. |
| `Batch<'a, D, SU, SV>` | struct | | One batch: `u_space`, `v_space`, `kind`, `first`, `n_valid`. |
| `Batch::element(lane)` / `label(lane)` | fn | | Storage element / mesh label of a lane. |
| `Batch::geometry(set, lane, p)` | fn | | `(x, det J, cof J)` at point `p` of a point set. |
| `Batch::quadrature::<Q>()` | fn | | Declared quadrature `Q` of either space as a `PointSet`. |
| `PointSet<'a, D>` | struct | | Reference points, weights and geometry of a point set. |
| `PointSet::len` / `is_empty` / `weight(p)` / `ref_point(p)` | fn | | Size, reference weight, reference point. |

---

## `operators` — hard-coded element operators

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `FusedKp<'a, D, SU, SP, QB, QM>` | struct | art | `K^P = Bᵀ (M^P)⁻¹ B` per element, `B` from `∫ q div u` on rule `QB`, pressure mass on rule `QM` (option 2). `FusedKp::new(&disp, &pres, mass_coef)`. |
| `FusedKpScratch<D>` | struct | art | Scratch of `FusedKp`. |
| `FusedKpQ4Q2<D>` | struct | art | Same `K^P` for Q4 displacement / Q2 pressure with Green's formula `B` (option 3, cornea) and the **Gauss–Legendre 3** pressure mass (the article). `new(&disp, mass_coef)`. |
| `FusedKpQ4Q2Scratch` | struct | art | Scratch of `FusedKpQ4Q2`. |
| `PressureMass2` | type alias | art | `Quad2<Gl<3>, Gl<3>>`. |
| `PressureMass3` | type alias | art | `Quad3<Gl<3>, Gl<3>, Gl<3>>`. |

---

## `ops` — free functions

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `power_iteration(x0, apply, inner, max_iter, rtol)` | fn | | Largest eigenvalue of `M⁻¹A` from closures (`apply` = `M⁻¹A`, `inner` = `M` product). |
| `EigenEstimate` | struct | | Result: `lambda`, `iterations`, `rel_change`. |
| `stable_dt(lambda_max, safety)` | fn | | Leapfrog step `safety · 2/√λ_max`. |
| `dot(a, b)` | fn | | Parallel dot product. |

---

## `vtk`, `vizir` — output

| Name | Kind | Tied | What it does |
|---|---|---|---|
| `VtkWriter` | struct | | Legacy ASCII VTK on tessellated elements; `new(&space)`, `write(path, &[VtkField])`. |
| `VtkField<'a>` | struct | | A named field: `name`, `components`, `data`. |
| `VizirWriter` | struct | | Vizir `.sol` (HOSolAt…); `new(&space, &mesh)`, `write(path, data, components)`, `tag()`. |

---

## Helpers repeated in the examples (not in the library)

| Name | What it does |
|---|---|
| `ricker(t, t0, s0)` | Ricker wavelet time signal (7 examples). |
| `gaussian(t, t0, s0)` | Gaussian time signal (1 example). |
| `bump(x, center, r0, s0)` | C∞ compactly supported bump of radius `r0` (all examples). |
