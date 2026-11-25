//! Legacy ASCII VTK (ParaView) unstructured grids: the file format shared by
//! the writers of `ondomathx-tetrix` and `ondomathx-sem`, which supply the
//! points, the cells and the values through functions (no copy of their
//! data, each crate keeping its own layout).
//!
//! The header, points and cells are formatted once in [`VtkGrid::new`];
//! each [`VtkGrid::write`] formats only the point data, in parallel. Numbers
//! are written in their shortest exact form (`{}`); point-data values with
//! `|v| < 1e-40` are written as `0` (ParaView misreads denormal numbers).

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use rayon::prelude::*;

/// Points (or cells) formatted per parallel task.
const BLOCK: usize = 4096;

/// Values below this magnitude are written as `0`.
pub const FLUSH_BELOW: f64 = 1e-40;

/// A named point field: `components` values per point, `value(point, comp)`.
pub struct Field<'a> {
    pub name: &'a str,
    pub components: usize,
    pub value: &'a (dyn Fn(usize, usize) -> f64 + Sync),
}

/// An unstructured grid of one cell type, its geometry formatted once.
pub struct VtkGrid {
    geometry: Vec<u8>,
    dim: usize,
    n_points: usize,
}

/// Formats `f(i)` for `i in 0..n` into one buffer, in parallel blocks.
fn format_blocks(n: usize, f: impl Fn(usize, &mut Vec<u8>) + Sync) -> Vec<u8> {
    let blocks: Vec<Vec<u8>> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|k| {
            let mut b = Vec::new();
            for i in k * BLOCK..((k + 1) * BLOCK).min(n) {
                f(i, &mut b);
            }
            b
        })
        .collect();
    blocks.concat()
}

fn flushed(v: f64) -> f64 {
    if v.abs() < FLUSH_BELOW {
        0.0
    } else {
        v
    }
}

impl VtkGrid {
    /// The grid of `n_points` points `point(i)` (3 coordinates; `z = 0` in
    /// 2D) and `n_cells` cells of VTK type `cell_type` (e.g. 9 quadrangle, 12
    /// hexahedron, 24 quadratic tetrahedron), each of `nodes_per_cell`
    /// points `cell_node(cell, k)`. `dim` (2 or 3) is the space dimension: a
    /// field with `dim` components is written as `VECTORS`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        title: &str,
        dim: usize,
        n_points: usize,
        point: impl Fn(usize) -> [f64; 3] + Sync,
        cell_type: u8,
        nodes_per_cell: usize,
        n_cells: usize,
        cell_node: impl Fn(usize, usize) -> usize + Sync,
    ) -> Self {
        assert!(dim == 2 || dim == 3, "dimension {dim}");
        let mut out = Vec::new();
        write!(
            out,
            "# vtk DataFile Version 3.0\n{title}\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS {n_points} double\n"
        )
        .unwrap();
        out.extend(format_blocks(n_points, |i, b| {
            let x = point(i);
            writeln!(b, "{} {} {}", x[0], x[1], x[2]).unwrap();
        }));
        writeln!(out, "CELLS {n_cells} {}", n_cells * (nodes_per_cell + 1)).unwrap();
        out.extend(format_blocks(n_cells, |c, b| {
            write!(b, "{nodes_per_cell}").unwrap();
            for k in 0..nodes_per_cell {
                write!(b, " {}", cell_node(c, k)).unwrap();
            }
            b.push(b'\n');
        }));
        writeln!(out, "CELL_TYPES {n_cells}").unwrap();
        out.extend_from_slice(format!("{cell_type}\n").repeat(n_cells).as_bytes());
        writeln!(out, "POINT_DATA {n_points}").unwrap();
        VtkGrid {
            geometry: out,
            dim,
            n_points,
        }
    }

    /// Number of points.
    pub fn num_points(&self) -> usize {
        self.n_points
    }

    /// Writes one snapshot to the file `path`.
    pub fn write(&self, path: impl AsRef<Path>, fields: &[Field<'_>]) -> io::Result<()> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        self.write_to(&mut w, fields)?;
        w.flush()
    }

    /// Writes one snapshot to `w`: a field of one component as `SCALARS`, one
    /// of `dim` components as `VECTORS` (padded to 3 in 2D), any other count
    /// as one `SCALARS` per component (`name_0`, `name_1`, …).
    pub fn write_to(&self, w: &mut impl Write, fields: &[Field<'_>]) -> io::Result<()> {
        w.write_all(&self.geometry)?;
        for f in fields {
            if f.components == self.dim {
                writeln!(w, "VECTORS {} double", f.name)?;
                w.write_all(&format_blocks(self.n_points, |p, b| {
                    for c in 0..3 {
                        let v = if c < self.dim { flushed((f.value)(p, c)) } else { 0.0 };
                        let sep = if c == 2 { '\n' } else { ' ' };
                        write!(b, "{v}{sep}").unwrap();
                    }
                }))?;
            } else if f.components == 1 {
                writeln!(w, "SCALARS {} double 1\nLOOKUP_TABLE default", f.name)?;
                w.write_all(&format_blocks(self.n_points, |p, b| writeln!(b, "{}", flushed((f.value)(p, 0))).unwrap()))?;
            } else {
                for c in 0..f.components {
                    writeln!(w, "SCALARS {}_{c} double 1\nLOOKUP_TABLE default", f.name)?;
                    w.write_all(&format_blocks(self.n_points, |p, b| writeln!(b, "{}", flushed((f.value)(p, c))).unwrap()))?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles of a unit square, a scalar and a 2D vector field.
    #[test]
    fn a_small_grid() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let tri = [[0, 1, 2], [0, 2, 3]];
        let grid = VtkGrid::new("test", 2, 4, |i| pts[i], 5, 3, 2, |c, k| tri[c][k]);
        let s = |p: usize, _c: usize| if p == 3 { 1e-45 } else { p as f64 * 0.5 };
        let v = |p: usize, c: usize| (p * 10 + c) as f64;
        let mut out = Vec::new();
        grid.write_to(&mut out, &[
            Field { name: "s", components: 1, value: &s },
            Field { name: "v", components: 2, value: &v },
        ])
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        let expected = "# vtk DataFile Version 3.0\ntest\nASCII\nDATASET UNSTRUCTURED_GRID\n\
            POINTS 4 double\n0 0 0\n1 0 0\n1 1 0\n0 1 0\nCELLS 2 8\n3 0 1 2\n3 0 2 3\nCELL_TYPES 2\n5\n5\n\
            POINT_DATA 4\nSCALARS s double 1\nLOOKUP_TABLE default\n0\n0.5\n1\n0\n\
            VECTORS v double\n0 1 0\n10 11 0\n20 21 0\n30 31 0\n";
        assert_eq!(text, expected);
    }
}
