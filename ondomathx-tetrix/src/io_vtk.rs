//! VTK output (ParaView): the P2 sub-space as quadratic tetrahedra (VTK cell
//! type 24, 10 nodes per cell). The format is `ondomathx_toolbox::io_vtk`;
//! [`VtkWriter`] formats the geometry once and writes several fields per
//! snapshot, as `ondomathx-sem`'s writer.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use ondomathx_toolbox::io_vtk::{Field, VtkGrid};

use crate::tetrix_p2_space::TetriXP2Space;

/// A named field of `components` components, a global (CG) vector in the
/// layout `[dof][comp]` (`num_dofs · components` entries).
#[derive(Clone, Copy, Debug)]
pub struct VtkField<'a> {
    pub name: &'a str,
    pub components: usize,
    pub data: &'a [f64],
}

/// Writes snapshots of fields defined on one [`TetriXP2Space`].
pub struct VtkWriter {
    grid: VtkGrid,
    /// Global DoF of each P2 point.
    glob: Vec<usize>,
    n_dof: usize,
}

impl VtkWriter {
    /// Formats the P2 points and the quadratic tetrahedra of `fe_space`.
    pub fn new(fe_space: &TetriXP2Space) -> Self {
        let n_p2 = fe_space.p2_num_dofs();
        let grid = VtkGrid::new(
            "ondomathx-tetrix",
            3,
            n_p2,
            |i| *fe_space.p2_dof_coordinate(i),
            24,
            10,
            fe_space.num_elements(),
            |e, k| fe_space.p2_loc2glob(e, k),
        );
        VtkWriter {
            grid,
            glob: (0..n_p2).map(|i| fe_space.p2_glob2glob(i)).collect(),
            n_dof: fe_space.num_dofs(),
        }
    }

    /// Writes one snapshot to `path`: a field with one component as
    /// `SCALARS`, with 3 as `VECTORS`, any other count as one `SCALARS` per
    /// component.
    pub fn write(&self, path: impl AsRef<Path>, fields: &[VtkField<'_>]) -> io::Result<()> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        self.write_to(&mut w, fields)?;
        w.flush()
    }

    /// Writes one snapshot to `w` (see [`VtkWriter::write`]).
    pub fn write_to(&self, w: &mut impl Write, fields: &[VtkField<'_>]) -> io::Result<()> {
        let values: Vec<_> = fields
            .iter()
            .map(|f| {
                assert_eq!(f.data.len(), self.n_dof * f.components, "field {}: length", f.name);
                let (data, nc, glob) = (f.data, f.components, &self.glob);
                move |p: usize, c: usize| data[nc * glob[p] + c]
            })
            .collect();
        let toolbox: Vec<Field<'_>> = fields
            .iter()
            .zip(&values)
            .map(|(f, v)| Field { name: f.name, components: f.components, value: v })
            .collect();
        self.grid.write_to(w, &toolbox)
    }

    /// The scalar `SOLUTION`: component 0 of `solution` (`num_dofs · sys_dim`
    /// entries, `[dof][comp]`), as the former `write`.
    fn write_solution_to(&self, w: &mut impl Write, solution: &[f64]) -> io::Result<()> {
        let sys_dim = solution.len() / self.n_dof;
        let glob = &self.glob;
        let value = |p: usize, _c: usize| solution[sys_dim * glob[p]];
        self.grid.write_to(w, &[Field { name: "SOLUTION", components: 1, value: &value }])
    }
}

/// Write a VTK unstructured-grid file for a scalar solution: component 0 of
/// `solution` as `SOLUTION` (a [`VtkWriter`] built for this one file).
pub fn write<W: Write>(writer: &mut W, fe_space: &TetriXP2Space, solution: &[f64]) -> io::Result<()> {
    VtkWriter::new(fe_space).write_solution_to(writer, solution)
}

/// Convenience wrapper that opens a file and calls `write`.
pub fn write_vtk(file_path: &str, fe_space: &TetriXP2Space, solution: &[f64]) {
    let file = std::fs::File::create(file_path)
        .unwrap_or_else(|e| panic!("Cannot create {}: {}", file_path, e));
    let mut buf = std::io::BufWriter::new(file);
    write(&mut buf, fe_space, solution).expect("VTK write failed");
}
