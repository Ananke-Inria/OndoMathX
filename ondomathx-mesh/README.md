# Anamesh

Anamesh is a Rust library for finite element mesh handling. It provides the building
blocks needed to go from a mesh file or a parametric mesh description to a global DOF
numbering ready for finite element assembly:

- **Reference and geometric elements** — P1 and P2/Q2 Lagrange elements on segments,
  triangles, quadrangles, tetrahedra, and hexahedra.
- **Parametric meshes** — structured `Square` and `Circle` (disk) meshes, with an
  optional `q2` flag for second-order geometry.
- **Analytic deformations** — attach a `Diffeomorphism` to a mesh to represent
  curved domains without storing extra nodes; element Jacobians account for it.
- **Mesh I/O** — Gmsh 2.2 (`.msh`) and Medit ASCII (`.mesh`, readable by Vizir4).
- **Topology and coloring** — vertex/edge/face connectivity and a graph coloring for
  race-free parallel assembly.
- **Conforming DOF numbering** — orientation-aware global numbering from a
  user-defined DOF layout.

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

## Supported elements

| Constant        | Shape       | Order | Nodes | Gmsh ID |
|-----------------|-------------|-------|-------|---------|
| `EDGE2`         | Segment     | P1    | 2     | 1       |
| `EDGE3`         | Segment     | P2    | 3     | 8       |
| `TRIANGLE3`     | Triangle    | P1    | 3     | 2       |
| `TRIANGLE6`     | Triangle    | P2    | 6     | 9       |
| `QUADRANGLE4`   | Quadrangle  | Q1    | 4     | 3       |
| `QUADRANGLE9`   | Quadrangle  | Q2    | 9     | 10      |
| `TETRAHEDRON4`  | Tetrahedron | P1    | 4     | 4       |
| `TETRAHEDRON10` | Tetrahedron | P2    | 10    | 11      |
| `HEXAHEDRON8`   | Hexahedron  | Q1    | 8     | 5       |
| `HEXAHEDRON27`  | Hexahedron  | Q2    | 27    | 12      |

All element types are read from and written to both file formats (node orderings are
converted automatically where they differ from the internal convention).

## Mesh types

- **`Square`** — structured 2D quad mesh of `[origin_x, origin_x + lx] ×
  [origin_y, origin_y + ly]` with `nx × ny` cells. Boundary edges come in four
  labelled groups: 0 (bottom), 1 (right), 2 (top), 3 (left).
- **`Circle`** — structured quad mesh of a disk (an inner square plus four curved
  arms). Refinement parameters are chosen automatically for near-square elements;
  with `q2`, boundary midpoints lie exactly on the circle.
- **`UnstructuredMesh`** — general mixed-element mesh, built from in-memory data
  (`new_from_points_and_elements`) or from a file (`new_from_msh`, `new_from_mesh`).
  Construction computes element indexing, full topology, and the coloring.

All mesh types implement the `Mesh` trait: node/element access, element Jacobians
(`get_element_grad_def`), coloring, `write_msh`/`write_mesh`, and conversion to an
`UnstructuredMesh`.

## Deformations

A `Diffeomorphism` is a smooth map given by two functions (evaluation and Jacobian)
plus a vector of parameters; maps can be composed. Attach one to any mesh: written
node coordinates and element Jacobians then include the deformation.
`UnstructuredMesh::apply_deformation` bakes it permanently into the node list.

## DOF numbering

Describe a DOF layout on the reference element with `DoFSet::new(positions, &REF)` —
one reference-coordinate position per local DOF. `mesh.get_numbering(&dof_set)`
returns `(numbering, n_dof)` with `numbering[element][local_dof]` the global index:
a conforming, orientation-aware numbering that supports mixed-dimension meshes
(volume elements plus surface elements for boundary conditions).

---

Development documentation (conventions, contracts, design decisions, roadmap) lives
in [CLAUDE.md](CLAUDE.md).
