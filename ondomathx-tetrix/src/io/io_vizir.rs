//! Vizir high-order solutions (`HOSolAtTetrahedraP1`, `.sol` files) on the
//! P2 sub-space. The format is `ondomathx_toolbox::io_vizir`; [`VizirWriter`]
//! formats the header once, as `ondomathx-sem`'s writer.

use std::io;
use std::path::Path;

use ondomathx_mesh::mesh::UnstructuredMesh;
use ondomathx_toolbox::io_vizir::HoSolution;

use crate::space::p2_plus::{P2PlusSpace, ORDER};

/// Writer of Vizir `.sol` files for one [`P2PlusSpace`].
pub struct VizirWriter {
    sol: HoSolution,
    /// Global DoF of each P2 node of each element, `[elem][node]`.
    glob: Vec<usize>,
    nloc: usize,
    n_dof: usize,
}

impl VizirWriter {
    /// Formats the header and the P2 node positions (barycentric) of
    /// `fe_space`, built on `mesh`.
    pub fn new(fe_space: &P2PlusSpace, _mesh: &UnstructuredMesh) -> Self {
        let nloc = fe_space.num_loc_p2_dofs();
        let positions: Vec<Vec<f64>> = (0..nloc)
            .map(|i| fe_space.get_barycentric_coordinates(fe_space.loc_dof_coordinates(i)).to_vec())
            .collect();
        let n_elem = fe_space.num_elements();
        let glob = (0..n_elem)
            .flat_map(|e| (0..nloc).map(move |i| (e, i)))
            .map(|(e, i)| fe_space.p2_glob2glob(fe_space.p2_loc2glob(e, i)))
            .collect();
        VizirWriter {
            sol: HoSolution::new("TetrahedraP1", ORDER, &positions, n_elem),
            glob,
            nloc,
            n_dof: fe_space.num_dofs(),
        }
    }

    /// Geometric tag (`TetrahedraP1`).
    pub fn tag(&self) -> &str {
        self.sol.tag()
    }

    /// Writes the `components`-component field `data` (a global vector,
    /// `[dof][comp]`) to `path`.
    pub fn write(&self, path: impl AsRef<Path>, data: &[f64], components: usize) -> io::Result<()> {
        assert_eq!(data.len(), self.n_dof * components, "data length");
        let (nloc, glob) = (self.nloc, &self.glob);
        self.sol.write(path, components, |row, node, k| data[components * glob[row * nloc + node] + k])
    }
}

/// Write a Vizir `.sol` file for a scalar or vector solution, to
/// `{file_name}.{n:05}.sol` (a [`VizirWriter`] built for this one file).
pub fn write_solution(file_name: &str, fe_space: &P2PlusSpace, mesh: &UnstructuredMesh, solution: &[f64], n: usize) {
    let dim_sol = solution.len() / fe_space.num_dofs();
    let file_path = format!("{}.{:05}.sol", file_name, n);
    VizirWriter::new(fe_space, mesh)
        .write(&file_path, solution, dim_sol)
        .unwrap_or_else(|e| panic!("Cannot write {}: {}", file_path, e));
}
