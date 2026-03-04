use std::io::{self, Write};

use crate::tetrix_p2_space::TetriXP2Space;

/// Write a VTK unstructured-grid file for a scalar solution.
///
/// The mesh is exported as quadratic tetrahedra (VTK cell type 24, 10 nodes
/// per cell) using the P2 sub-space of the FE space.
pub fn write<W: Write>(
    writer: &mut W,
    fe_space: &TetriXP2Space,
    solution: &[f64],
) -> io::Result<()> {
    let n_dof = fe_space.p2_num_dofs();
    let n_dof_glob = fe_space.num_dofs();
    let sys_dim = solution.len() / n_dof_glob;
    let n_tet = fe_space.num_elements();

    // Header
    writeln!(writer, "# vtk DataFile Version 2.0")?;
    writeln!(writer, "OndoMathX TetriX VTK IO")?;
    writeln!(writer, "ASCII")?;
    writeln!(writer, "DATASET UNSTRUCTURED_GRID")?;

    // Points (P2 nodes)
    writeln!(writer, "POINTS {} double", n_dof)?;
    for i_dof in 0..n_dof {
        let xyz = fe_space.p2_dof_coordinate(i_dof);
        writeln!(writer, "{:.8e} {:.8e} {:.8e}", xyz[0], xyz[1], xyz[2])?;
    }

    // Cells: 10-node quadratic tetrahedra (type 24)
    writeln!(writer, "CELLS {} {}", n_tet, 11 * n_tet)?;
    for elem in 0..n_tet {
        write!(writer, "10")?;
        for i_loc in 0..10 {
            write!(writer, " {}", fe_space.p2_loc2glob(elem, i_loc))?;
        }
        writeln!(writer)?;
    }

    writeln!(writer, "CELL_TYPES {}", n_tet)?;
    for _ in 0..n_tet {
        writeln!(writer, "24")?;
    }

    // Scalar solution at P2 nodes
    writeln!(writer, "POINT_DATA {}", n_dof)?;
    writeln!(writer, "SCALARS SOLUTION double 1")?;
    writeln!(writer, "LOOKUP_TABLE default")?;
    for i_dof in 0..n_dof {
        let i_glob = fe_space.p2_glob2glob(i_dof);
        let mut v = solution[sys_dim * i_glob];
        if v.abs() < 1e-40 {
            v = 0.0;
        }
        writeln!(writer, "{}", v)?;
    }

    Ok(())
}

/// Convenience wrapper that opens a file and calls `write`.
pub fn write_vtk(file_path: &str, fe_space: &TetriXP2Space, solution: &[f64]) {
    let file = std::fs::File::create(file_path)
        .unwrap_or_else(|e| panic!("Cannot create {}: {}", file_path, e));
    let mut buf = std::io::BufWriter::new(file);
    write(&mut buf, fe_space, solution).expect("VTK write failed");
}
