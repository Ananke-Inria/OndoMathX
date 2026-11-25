//! Vizir high-order solutions (`HOSolAt…`, libMeshb `.sol`), port of the C++
//! `core/Sources/FEM/LobattoX/IO/IO_Vizir.h`.
//!
//! The mesh itself is written by `ondomathx-mesh` (`Mesh::write_mesh`, Medit `.mesh`);
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
//! - `Dimension 3`, to match `ondomathx-mesh`'s `write_mesh`, which always writes a 3D
//!   mesh (`z = 0` in 2D); the node positions have `D` reference coordinates.
//! - Rows follow the order of the elements in the `.mesh` section (`ondomathx-mesh`
//!   writes each section in mesh-element order), not the storage order.
//! - `1 C` after the element count: one solution field with `C` values per
//!   node; then `P NLOC` (degree, nodes per element).
//! - One element type per mesh (QUADRANGLE4/9 or HEXAHEDRON8/27) and an
//!   isotropic shape (one degree): the writer panics otherwise.

use std::io;
use std::path::Path;

use ondomathx_mesh::mesh::Mesh;
use ondomathx_toolbox::io_vizir::HoSolution;

use crate::shape::Shape;
use crate::space::Space;

/// Writer of Vizir `.sol` files for one space (the format:
/// `ondomathx_toolbox::io_vizir`).
pub struct VizirWriter {
    sol: HoSolution,
    tag: &'static str,
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
            (0..space.num_elements()).all(|e| msh_id(e) == id),
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
        let positions: Vec<Vec<f64>> = (0..nloc).map(|node| space.loc_dof_coordinates(node).to_vec()).collect();
        let sol = HoSolution::new(tag, S::N[0] - 1, &positions, space.num_elements());
        let mut order: Vec<usize> = (0..space.num_elements()).collect();
        order.sort_by_key(|&e| space.mesh_element(e));
        VizirWriter { sol, tag, nloc, order }
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
        self.sol
            .write(path, c, |row, node, k| data[self.order[row] * c * nloc + k * nloc + node])
    }
}
