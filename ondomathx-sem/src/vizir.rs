//! Vizir high-order solutions (`HOSolAt…`, libMeshb `.sol`), port of the C++
//! `core/Sources/FEM/LobattoX/IO/IO_Vizir.h`.
//!
//! The mesh itself is written by Anamesh (`Mesh::write_mesh`, Medit `.mesh`);
//! the `.sol` file holds, per element, the solution at the GLL nodes, whose
//! reference positions are given once (`HOSolAt<tag>NodesPositions`).
//! Duplicated storage is exactly this per-element layout: no averaging.
//!
//! ```ignore
//! mesh.write_mesh("wave.mesh")?;
//! let vizir = VizirWriter::new(&space, &mesh);
//! vizir.write("wave.00010.sol", &u, 2)?; // 2-component field
//! ```
//!
//! Conventions (as the C++ writer, except `Dimension`):
//! - `Dimension 3`, to match Anamesh's `write_mesh`, which always writes a 3D
//!   mesh (`z = 0` in 2D); the node positions have `D` reference coordinates.
//! - Rows follow the order of the elements in the `.mesh` section (Anamesh
//!   writes each section in mesh-element order), not the storage order.
//! - `1 C` after the element count: one solution field with `C` values per
//!   node; then `P NLOC` (degree, nodes per element).
//! - One element type per mesh (QUADRANGLE4/9 or HEXAHEDRON8/27) and an
//!   isotropic shape (one degree): the writer panics otherwise.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use ondomathx_mesh::mesh::Mesh;
use rayon::prelude::*;

use crate::shape::Shape;
use crate::space::Space;

/// Writer of Vizir `.sol` files for one space.
pub struct VizirWriter {
    /// Header up to the element count (positions included).
    header: Vec<u8>,
    tag: &'static str,
    degree: usize,
    nloc: usize,
    /// Storage elements in the order of the `.mesh` section.
    order: Vec<usize>,
}

impl VizirWriter {
    /// Formats the header and the node positions of `space`, built on `mesh`.
    ///
    /// Panics for an anisotropic shape or a mesh with several (or
    /// unsupported) volume element types.
    pub fn new<const D: usize, S: Shape<D>, M: Mesh>(space: &Space<D, S>, mesh: &M) -> Self {
        assert!(
            S::ISOTROPIC,
            "Vizir HOSolAt needs one degree: anisotropic shape {:?}",
            &S::N[..D]
        );
        let msh_id = |e: usize| {
            mesh.get_element(space.mesh_element(e))
                .geometric_element
                .msh_id
        };
        let id = msh_id(0);
        assert!(
            (0..space.n_elem()).all(|e| msh_id(e) == id),
            "Vizir HOSolAt needs a single volume element type"
        );
        let tag = match (D, id) {
            (2, 3) => "QuadrilateralsQ1",
            (2, 10) => "QuadrilateralsQ2",
            (3, 5) => "HexahedraQ1",
            (3, 12) => "HexahedraQ2",
            _ => panic!("Vizir HOSolAt: unsupported element type (gmsh {id}) in {D}D"),
        };
        let nloc = S::NLOC;
        let mut header = Vec::new();
        write!(
            header,
            "MeshVersionFormatted 2\nDimension\n3\nHOSolAt{tag}NodesPositions\n{nloc}\n"
        )
        .unwrap();
        for node in 0..nloc {
            for x in space.ref_point(node) {
                write!(header, "{x} ").unwrap();
            }
            header.push(b'\n');
        }
        write!(header, "HOSolAt{tag}\n{}\n", space.n_elem()).unwrap();
        let mut order: Vec<usize> = (0..space.n_elem()).collect();
        order.sort_by_key(|&e| space.mesh_element(e));
        VizirWriter {
            header,
            tag,
            degree: S::N[0] - 1,
            nloc,
            order,
        }
    }

    /// Geometric tag (`QuadrilateralsQ1`, …, `HexahedraQ2`).
    pub fn tag(&self) -> &'static str {
        self.tag
    }

    /// Writes the `components`-component field `data` (`[elem][comp][node]`,
    /// the space's layout) to `path`.
    pub fn write(&self, path: impl AsRef<Path>, data: &[f64], components: usize) -> io::Result<()> {
        let (nloc, c) = (self.nloc, components);
        assert_eq!(data.len(), self.order.len() * c * nloc, "data length");
        let rows: Vec<Vec<u8>> = self
            .order
            .par_iter()
            .map(|&e| {
                let block = &data[e * c * nloc..(e + 1) * c * nloc];
                let mut b = Vec::with_capacity(nloc * c * 24);
                for node in 0..nloc {
                    for k in 0..c {
                        write!(b, "{} ", block[k * nloc + node]).unwrap();
                    }
                }
                b.push(b'\n');
                b
            })
            .collect();
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(&self.header)?;
        writeln!(out, "1 {c}")?;
        writeln!(out, "{} {nloc}", self.degree)?;
        for r in &rows {
            out.write_all(r)?;
        }
        out.write_all(b"End\n")?;
        out.flush()
    }
}
