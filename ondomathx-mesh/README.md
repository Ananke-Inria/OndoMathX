# Anamesh

Anamesh is a Rust library for finite element mesh handling. It provides the building blocks needed to go from a mesh file or a parametric mesh description all the way to a global DOF numbering ready for finite element assembly: reference elements, geometric elements with shape functions, mesh I/O, mesh topology, deformation fields, and DOF management.

---

## Modules

### `reference_elements`

Defines the combinatorial and geometric structure of the reference cells that the library knows about:

| Constant     | Shape          | Dimension | Vertices | Edges | Faces |
|--------------|----------------|-----------|----------|-------|-------|
| `VERTEX`     | Point          | 0         | 1        | 0     | 0     |
| `EDGE`       | Segment [0,1]  | 1         | 2        | 1     | 0     |
| `TRIANGLE`   | Right triangle | 2         | 3        | 3     | 1     |
| `QUADRANGLE` | Unit square    | 2         | 4        | 4     | 1     |
| `TETRAHEDRON`| Right tet      | 3         | 4        | 6     | 4     |
| `HEXAHEDRON` | Unit cube      | 3         | 8        | 12    | 6     |

Each `ReferenceElement` records the vertex coordinates, edge/face connectivity, outward normals, and a compact topology summary. It provides spatial queries (`is_inside`, `is_on_edge`, `is_on_face`, `is_on_vertex`) and coordinate-change helpers (`uvw_edge_transform`, `uvw_face_transform`) that account for all possible orientations of a shared face or edge. These orientation maps are the foundation of the DOF numbering logic.

### `geometric_elements`

Pairs each reference cell with a set of scalar Lagrange shape functions and their gradients, producing a `GeometricElement`. Each element also stores the Gmsh element-type ID (`msh_id`) so it can be written back to `.msh` files.

Currently implemented elements:

| Constant       | Shape       | Order | Nodes | Gmsh ID |
|----------------|-------------|-------|-------|---------|
| `EDGE2`        | Segment     | P1    | 2     | 1       |
| `EDGE3`        | Segment     | P2    | 3     | 8       |
| `TRIANGLE3`    | Triangle    | P1    | 3     | 2       |
| `TRIANGLE6`    | Triangle    | P2    | 6     | 9       |
| `QUADRANGLE4`  | Quadrangle  | Q1    | 4     | 3       |
| `QUADRANGLE9`  | Quadrangle  | Q2    | 9     | 10      |
| `TETRAHEDRON4` | Tetrahedron | P1    | 4     | 4       |
| `TETRAHEDRON10`| Tetrahedron | P2    | 10    | 11      |
| `HEXAHEDRON8`  | Hexahedron  | Q1    | 8     | 5       |
| `HEXAHEDRON27` | Hexahedron  | Q2    | 27    | 12      |

`QUADRANGLE9` and `HEXAHEDRON27` use the full tensor-product Q2 basis built from the 1D quadratic Lagrange polynomials on [0,1] (vertex, midpoint, and interior node per direction). Their node ordering follows the topology tables of `reference_elements` (vertices, then edge midpoints in edge order, then face centers, then the interior node); for `HEXAHEDRON27` and `TETRAHEDRON10` this differs from Gmsh's node ordering, so a permutation will be needed when wiring them into the `.msh` reader/writer.

### `diffeomorphism`

A `Diffeomorphism` represents a smooth map f : R^`inner_dim` → R^`outer_dim` through two
function pointers, plus a `params: Vec<f64>` slice of auxiliary constants passed to them
(the pointers are plain `fn`s and cannot capture, so per-map data travels through `params`):

- `diffeo(params, x, f_x)` — evaluates f(x), writing the result into `f_x`.
- `grad_diffeo(params, x, grad)` — multiplies `grad` in-place by the Jacobian Df(x), i.e. `grad ← Df(x) · grad`.

Call them through the methods `eval(x, f_x)` and `grad_eval(x, grad)`, which forward the stored
`params` and apply any inner deformation. Maps can be composed: `field.compose(inner)` sets an
`inner_deformation` g so `field` evaluates f ∘ g, with the chain rule applied automatically in
`grad_eval`. The helper `test_grad` validates the Jacobian against centred finite differences,
which is useful when implementing new deformation maps.

The `Square` and `Circle` meshes and `UnstructuredMesh` all accept an optional `Diffeomorphism`.
When a deformation is attached, node coordinates are passed through it when writing to `.msh` and
when computing the Jacobian of the element map (`get_element_grad_def`). `UnstructuredMesh::apply_deformation`
permanently bakes the deformation into the node list.

### `mesh` — the `Mesh` trait

All mesh types implement the `Mesh` trait, which exposes:

- Node access: `get_nnodes`, `get_node`.
- Element access: `get_nelements`, `get_element`, `get_elements_by_dimension`, `get_elements_by_dimension_and_label`.
- Geometry: `get_geometric_dimension`, `get_topologic_dimension`.
- Element Jacobian: `get_element_grad_def(element_index, uvw, grad_def)` — fills `grad_def` with ∂x/∂u evaluated at reference coordinates `uvw`, taking any attached deformation into account.
- Coloring: `get_ncolors`, `get_element_color`, `get_nelements_by_color`, `get_element_index_by_color_to_element_index`.
- Conversion: `get_unstructured_mesh` (default implementation), `write_msh` (Gmsh 2.2), `write_mesh` (Medit ASCII, readable by Vizir4).

#### `Square`

A structured 2D quadrilateral mesh over the rectangle `[origin_x, origin_x + lx] × [origin_y, origin_y + ly]` divided into `nx × ny` cells. Boundary edges are exposed as four groups labelled 0 (bottom), 1 (right), 2 (top), 3 (left). The mesh supports an optional `Diffeomorphism`, which makes it easy to produce non-rectangular domains analytically without storing any extra nodes.

The constructor takes an optional `q2` flag (default `false`). When active the mesh carries Q2 geometric nodes — the Q1 vertices keep their numbering and the edge midpoints / cell centers are appended after them — and the elements become `QUADRANGLE9` / `EDGE3`, so `write_msh` and `get_unstructured_mesh` produce a Q2 mesh (useful to represent a deformed domain with second-order accuracy).

The coloring is hardcoded to 4 colours in a 2×2 checkerboard pattern, suitable for lock-free parallel assembly.

#### `Circle`

A structured quadrilateral mesh of a 2D disk of radius `R`. The reference domain
`[-r_ref, r_ref]²` (`r_ref = R/√2`) is split into an inner square and four arms, then mapped
to the disk by a built-in `Diffeomorphism` (radial blend on the arms, identity on the inner
square). The refinement parameters `n_x`, `n_rho`, and `alpha` are chosen automatically so that
elements are close to square.

Like `Square`, the constructor takes an optional `q2` flag (default `false`). When active the
mesh carries Q2 geometric nodes — the images under the circle map of the reference-domain edge
midpoints and cell centers — and the elements become `QUADRANGLE9` / `EDGE3`. In particular the
boundary midpoints lie exactly on the circle, so `write_msh` and `get_unstructured_mesh` produce
a curved, second-order accurate description of the disk.

#### `UnstructuredMesh`

A general mixed-element mesh. Once constructed it computes and stores:

- **Indexing** (`elements_by_dim`, `elements_by_dim_and_label`): fast lookup of elements by spatial dimension and physical label.
- **Topology** (`neighbour_faces`, `neighbour_edges`, `neighbour_vertexes`, `faces_neighbour`, `edges_neighbour`, `vertexes_neighbour`): for every element the list of its mesh faces/edges/vertices (with global indices), and for every face/edge/vertex the list of highest-dimension elements that share it.
- **Graph colouring** (`coloring`, `elements_by_color`): a greedy graph colouring of the dual element graph (two elements are adjacent if they share a vertex) so that elements of the same colour do not share any node — enabling race-free parallel computations.

Construction methods:

| Method | Source |
|---|---|
| `new_from_points_and_elements` | In-memory node/element lists |
| `new_from_msh(path)` | Gmsh format 2.2 (`.msh`) |
| `new_from_mesh(path)` | Medit format (`.mesh`) |

Writing: `write_msh(path)` writes Gmsh 2.2. If a deformation is attached, the deformed coordinates are written.

Supported element types when reading `.msh` files: Edge2, Triangle3, Quadrangle4, Tetrahedron4, Hexahedron8, plus the P2/Q2 elements Edge3, Triangle6, Quadrangle9, Tetrahedron10, Hexahedron27 (node ordering permuted from Gmsh's to the internal convention on read). `.mesh` files: the P1 sections plus `EdgesP2`, `TrianglesP2`, `QuadrilateralsQ2`, `TetrahedraP2` (Medit/libMeshb P2 node ordering, which coincides with the internal one) and `HexahedraQ2` (following the node ordering of the C++ OndoMathX `IO_MESH` writer). Writing: `write_msh` (Gmsh 2.2) applies the inverse permutation for Tetrahedron10/Hexahedron27, so read/write round-trips preserve the node ordering. `write_mesh` writes the ASCII Medit format (readable by Vizir4 and by `new_from_mesh`) with the same sections as the reader, including the P2/Q2 ones; elements are grouped by type into sections, so the global element ordering is not preserved on a round-trip (connectivity and labels are). The dimension is always written as 3 (z = 0 for 2D meshes) and vertex references are 0.

### `topology`

Provides `Edge` and `Face` types used internally by `UnstructuredMesh` to build the topology maps. Both implement a canonical sorted-index representation so that a shared edge or face (possibly listed with nodes in different order by adjacent elements) hashes to the same key in a `BTreeMap`. The `get_orientation` / `get_relative_orientation` methods return the rotation/reflection index needed by the DOF numbering step.

Faces use a 4-integer representation where repeated last indices signal a triangular face, allowing the same type to store both triangular and quadrilateral faces of mixed-element meshes.

### `dof_set`

`DoFSet` describes the layout of degrees of freedom on a single reference element. You construct it by providing a list of reference-coordinate positions `[u, v, w]` — one per local DOF — and the reference element they live on:

```rust
let q2_dofs = DoFSet::new(
    vec![
        [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0],
    ],
    &QUADRANGLE,
);
```

The constructor checks that every DOF lies inside the reference element and then classifies each one as belonging to a vertex, an edge (with both orientations pre-computed), a face (with all orientations), or the element interior. This classification is the input to `UnstructuredMesh::get_numbering`.

#### Global DOF numbering

`UnstructuredMesh::get_numbering(dof_set)` returns a `(numbering, n_dof)` pair where `numbering[element_index][local_dof]` is the global DOF index. The algorithm:

1. Iterates over elements of the highest topological dimension.
2. Assigns fresh global indices to vertex DOFs the first time a vertex is encountered; subsequent elements sharing the vertex reuse them.
3. Does the same for edge-interior DOFs, using the precomputed orientation to pick the correct local ordering.
4. Does the same for face-interior DOFs.
5. Always assigns fresh global indices for volume-interior DOFs.
6. Fills the numbering of lower-dimensional boundary elements (edges, faces) by reading the global indices already assigned to the higher-dimensional elements they border.

This produces a conforming global numbering that respects orientation, works for both P1 and Q2 DOF sets, and supports mixed-dimension meshes (volume elements + surface elements for boundary conditions).

---

## Quick start

```rust
use anamesh::mesh::square::*;
use anamesh::mesh::Mesh;

// Structured 3×3 quad mesh on [0,1]²
let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 3, None, None);

// Convert to an UnstructuredMesh (computes topology and coloring)
let mesh = square.get_unstructured_mesh();

// Q1 DOF set: one DOF per vertex of the reference quadrangle
let q1 = DoFSet::new(
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    &QUADRANGLE,
);

let (numbering, n_dof) = mesh.get_numbering(&q1);
println!("{n_dof} global DOFs");

// Read an unstructured mesh from a Gmsh file
let unstr = UnstructuredMesh::new_from_msh("./assets/L.msh").unwrap();
unstr.write_msh("./output/L_out.msh").unwrap();
```

---

## Dependencies

| Crate | Use |
|---|---|
| `serde` / `serde_json` | Serialization (available for future use) |
| `itertools` | Format element index lists when writing `.msh` files |

---

## Status (2025-05-22)

- P1 elements fully working end-to-end: mesh I/O, topology, coloring, DOF numbering.
- Q2 (`QUADRANGLE9`, `HEXAHEDRON27`) shape functions and DOF set classification implemented; P2/Q2 elements are read from `.msh` (and `HexahedraQ2` from `.mesh`) and written back to `.msh` with the correct Gmsh node ordering.
- `EDGE3` (P2 segment), `TRIANGLE6` (P2 triangle), and `TETRAHEDRON10` (P2 tetrahedron) shape functions available.
- Only Gmsh 2.2 and Medit format reading; writing is Gmsh 2.2 only.
- Parametric meshes: `Square` and `Circle` implemented. Cylinder, ball, cube, and other shapes are planned (see [TODO.md](TODO.md)).
