//! Legacy ASCII VTK output (ParaView) on tessellated elements.
//!
//! Every element is split into `P^D` linear sub-cells (VTK_QUAD 9 /
//! VTK_HEXAHEDRON 12) on its GLL points. Points are duplicated per element in
//! storage order (`e·NLOC + node`), so a vector's values are written straight
//! from storage. The points and cells are formatted once in
//! [`VtkWriter::new`]; each [`VtkWriter::write`] only formats the data.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use rayon::prelude::*;

use crate::shape::Shape;
use crate::space::Space;

/// A named point field of `components` components, in the space's
/// `[elem][comp][node]` storage.
#[derive(Clone, Copy, Debug)]
pub struct VtkField<'a> {
    pub name: &'a str,
    pub components: usize,
    pub data: &'a [f64],
}

/// Writes snapshots of fields defined on one [`Space`].
pub struct VtkWriter {
    /// Header, points and cells, formatted once.
    geometry: Vec<u8>,
    dim: usize,
    nloc: usize,
    n_elem: usize,
}

impl VtkWriter {
    /// Formats the tessellated geometry of `space`.
    pub fn new<const D: usize, S: Shape<D>>(space: &Space<D, S>) -> Self {
        let nloc = S::NLOC;
        let n_elem = space.n_elem();
        let n_points = n_elem * nloc;
        // Sub-cells per direction (1 for the unused directions).
        let p: [usize; 3] = core::array::from_fn(|d| if d < D { S::N[d] - 1 } else { 1 });
        let sub_per_elem: usize = p[..D].iter().product();
        let corners = 1usize << D;

        let mut out = Vec::new();
        write!(
            out,
            "# vtk DataFile Version 3.0\nlobattox\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS {n_points} double\n"
        )
        .unwrap();
        let blocks: Vec<Vec<u8>> = (0..n_elem)
            .into_par_iter()
            .map(|e| {
                let mut b = Vec::with_capacity(nloc * 3 * 24);
                for node in 0..nloc {
                    let x = space.map_and_jacobian(e, node).0;
                    let z = if D == 3 { x[2] } else { 0.0 };
                    writeln!(b, "{} {} {}", x[0], x[1], z).unwrap();
                }
                b
            })
            .collect();
        blocks.iter().for_each(|b| out.extend_from_slice(b));

        // Sub-cell corners in VTK order: (0,0,0) (1,0,0) (1,1,0) (0,1,0), then
        // the same at z + 1.
        const QUAD: [[usize; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ];
        let n_cells = n_elem * sub_per_elem;
        writeln!(out, "CELLS {n_cells} {}", n_cells * (corners + 1)).unwrap();
        let blocks: Vec<Vec<u8>> = (0..n_elem)
            .into_par_iter()
            .map(|e| {
                let mut b = Vec::new();
                for s in 0..sub_per_elem {
                    let base = [s % p[0], (s / p[0]) % p[1], s / (p[0] * p[1])];
                    write!(b, "{corners}").unwrap();
                    for c in &QUAD[..corners] {
                        let mut node = 0;
                        let mut stride = 1;
                        for d in 0..D {
                            node += (base[d] + c[d]) * stride;
                            stride *= S::N[d];
                        }
                        write!(b, " {}", e * nloc + node).unwrap();
                    }
                    b.push(b'\n');
                }
                b
            })
            .collect();
        blocks.iter().for_each(|b| out.extend_from_slice(b));
        writeln!(out, "CELL_TYPES {n_cells}").unwrap();
        let ty = if D == 2 { "9\n" } else { "12\n" };
        out.extend_from_slice(ty.repeat(n_cells).as_bytes());
        writeln!(out, "POINT_DATA {n_points}").unwrap();

        VtkWriter {
            geometry: out,
            dim: D,
            nloc,
            n_elem,
        }
    }

    /// Writes one snapshot to `path`.
    ///
    /// A field with one component is written as `SCALARS`, one with `D`
    /// components as `VECTORS` (padded to 3 in 2D), any other count as one
    /// `SCALARS` per component (`name_0`, `name_1`, …).
    pub fn write(&self, path: impl AsRef<Path>, fields: &[VtkField<'_>]) -> io::Result<()> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        w.write_all(&self.geometry)?;
        for f in fields {
            assert_eq!(
                f.data.len(),
                self.n_elem * self.nloc * f.components,
                "field {}: length",
                f.name
            );
            if f.components == self.dim {
                writeln!(w, "VECTORS {} double", f.name)?;
                w.write_all(&self.format(f, None))?;
            } else if f.components == 1 {
                writeln!(w, "SCALARS {} double 1\nLOOKUP_TABLE default", f.name)?;
                w.write_all(&self.format(f, Some(0)))?;
            } else {
                for c in 0..f.components {
                    writeln!(w, "SCALARS {}_{c} double 1\nLOOKUP_TABLE default", f.name)?;
                    w.write_all(&self.format(f, Some(c)))?;
                }
            }
        }
        w.flush()
    }

    /// Formats component `comp` of `f` (or all `D` components as 3-vectors if
    /// `None`), one point per line, in parallel over elements.
    fn format(&self, f: &VtkField<'_>, comp: Option<usize>) -> Vec<u8> {
        let (nloc, nc, dim) = (self.nloc, f.components, self.dim);
        let blocks: Vec<Vec<u8>> = f
            .data
            .par_chunks(nc * nloc)
            .map(|block| {
                let mut b = Vec::with_capacity(nloc * 24 * if comp.is_some() { 1 } else { 3 });
                for node in 0..nloc {
                    match comp {
                        Some(c) => writeln!(b, "{}", block[c * nloc + node]).unwrap(),
                        None => {
                            for c in 0..3 {
                                let v = if c < dim { block[c * nloc + node] } else { 0.0 };
                                let sep = if c == 2 { '\n' } else { ' ' };
                                write!(b, "{v}{sep}").unwrap();
                            }
                        }
                    }
                }
                b
            })
            .collect();
        blocks.concat()
    }
}
