# Rouage

Rouage is a Rust library for finite element mesh handling. It provides the building blocks needed to go from a mesh file or a parametric mesh description all the way to a global DOF numbering ready for finite element assembly: reference elements, geometric elements with shape functions, mesh I/O, mesh topology, deformation fields, and DOF management.

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
| `QUADRANGLE4`  | Quadrangle  | Q1    | 4     | 3       |
| `QUADRANGLE9`  | Quadrangle  | Q2    | 9     | 10      |
| `TETRAHEDRON4` | Tetrahedron | P1    | 4     | 4       |
| `HEXAHEDRON8`  | Hexahedron  | Q1    | 8     | 5       |

`QUADRANGLE9` uses the full tensor-product Q2 basis built from the 1D quadratic Lagrange polynomials on [0,1] (vertex, midpoint, and interior node per direction).

### `deformation_field`

A `DeformationField` represents a smooth map f : R^`inner_dim` → R^`outer_dim` through two function pointers:

- `diffeo(x, f_x)` — evaluates f(x).
- `grad_diffeo(x, grad)` — multiplies `grad` in-place by the Jacobian Df(x), i.e. `grad ← Df(x) · grad`.

Fields can be composed: `field.compose(inner)` turns `field` into `f ∘ g`, with the chain rule applied automatically in `grad_eval`. The helper `test_grad` validates the Jacobian against centred finite differences, which is useful when implementing new deformation maps.

The `Square` mesh and `UnstructuredMesh` both accept an optional `DeformationField`. When a deformation is attached, node coordinates are passed through it when writing to `.msh` and when computing the Jacobian of the element map (`get_element_grad_def`). `apply_deformation` permanently bakes the deformation into the node list.

### `mesh` — the `Mesh` trait

All mesh types implement the `Mesh` trait, which exposes:

- Node access: `get_nnodes`, `get_node`.
- Element access: `get_nelements`, `get_element`, `get_elements_by_dimension`, `get_elements_by_dimension_and_label`.
- Geometry: `get_geometric_dimension`, `get_topologic_dimension`.
- Element Jacobian: `get_element_grad_def(element_index, uvw, grad_def)` — fills `grad_def` with ∂x/∂u evaluated at reference coordinates `uvw`, taking any attached deformation into account.
- Coloring: `get_ncolors`, `get_element_color`, `get_nelements_by_color`, `get_element_index_by_color_to_element_index`.
- Conversion: `get_unstructured_mesh` (default implementation), `write_msh`.

#### `Square`

A structured 2D quadrilateral mesh over the rectangle `[origin_x, origin_x + lx] × [origin_y, origin_y + ly]` divided into `nx × ny` cells. Boundary edges are exposed as four groups labelled 0 (bottom), 1 (right), 2 (top), 3 (left). The mesh supports an optional `DeformationField`, which makes it easy to produce non-rectangular domains analytically without storing any extra nodes.

The coloring is hardcoded to 4 colours in a 2×2 checkerboard pattern, suitable for lock-free parallel assembly.

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

Supported element types in `.msh` / `.mesh` files: Edge2, Triangle3, Quadrangle4, Tetrahedron4, Hexahedron8 (P1 elements only for now).

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
use rouage::mesh::square::*;
use rouage::mesh::Mesh;

// Structured 3×3 quad mesh on [0,1]²
let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 3, None);

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

## Deformation fields

Deformation fields let you describe curved or non-trivial domains analytically while keeping a simple parametric mesh as the computational backbone.

```rust
use rouage::deformation_field::DeformationField;
use rouage::mesh::square::*;
use rouage::mesh::Mesh;

// Map the unit square to an annulus: (u,v) -> ((1+u)*cos(2π*v), (1+u)*sin(2π*v))
let annulus = DeformationField {
    inner_dim: 2,
    outer_dim: 2,
    diffeo: |uv, xy| {
        let r = 1.0 + uv[0];
        let theta = 2.0 * std::f64::consts::PI * uv[1];
        xy[0] = r * theta.cos();
        xy[1] = r * theta.sin();
    },
    grad_diffeo: |uv, grad| {
        // Df applied to grad (chain rule)
        let theta = 2.0 * std::f64::consts::PI * uv[1];
        let r = 1.0 + uv[0];
        let (c, s) = (theta.cos(), theta.sin());
        let tau = 2.0 * std::f64::consts::PI;
        let g0 = grad[0];
        let g1 = grad[1];
        grad[0] = c * g0 - r * tau * s * g1;
        grad[1] = s * g0 + r * tau * c * g1;
    },
    inner_diffeo: None,
};

let square = Square::new([0.0, 0.0], 1.0, 1.0, 16, 32, Some(annulus));
square.write_msh("annulus.msh").unwrap();
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
- Q2 (`QUADRANGLE9`) shape functions and DOF set classification implemented; not yet wired into the `.msh` / `.mesh` parsers.
- `EDGE3` (P2 segment) shape functions available.
- Only Gmsh 2.2 and Medit format reading; writing is Gmsh 2.2 only.
- Parametric meshes: `Square` only. Circle, cylinder, ball, cube, and other shapes are planned (see [TODO.md](TODO.md)).
