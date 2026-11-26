//! Legacy ASCII VTK output (ParaView) on tessellated elements.
//!
//! Every element is split into `P^D` linear sub-cells (VTK_QUAD 9 /
//! VTK_HEXAHEDRON 12) on its GLL points. Points are duplicated per element in
//! storage order (`e·NLOC + node`), so a vector's values are written straight
//! from storage. The points and cells are formatted once in
//! [`VtkWriter::new`]; each [`VtkWriter::write`] only formats the data. The
//! format is `ondomathx_toolbox::io_vtk` (values with `|v| < 1e-40` written
//! as `0`).

use std::io;
use std::path::Path;

use ondomathx_toolbox::io_vtk::{Field, VtkGrid};

use crate::element::shape::Shape;
use crate::space::Space;

/// A named point field of `components` components, in the space's
/// `[elem][comp][node]` storage.
#[derive(Clone, Copy, Debug)]
pub struct VtkField<'a> {
    pub name: &'a str,
    pub components: usize,
    pub data: &'a [f64],
}

/// Writes snapshots of fields defined on one [`Space`] (the format:
/// `ondomathx_toolbox::io_vtk`).
pub struct VtkWriter {
    grid: VtkGrid,
    nloc: usize,
    n_elem: usize,
}

impl VtkWriter {
    /// Formats the tessellated geometry of `space`.
    pub fn new<const D: usize, S: Shape<D>>(space: &Space<D, S>) -> Self {
        let nloc = S::NLOC;
        let n_elem = space.num_elements();
        // Sub-cells per direction (1 for the unused directions).
        let p: [usize; 3] = core::array::from_fn(|d| if d < D { S::N[d] - 1 } else { 1 });
        let sub_per_elem: usize = p[..D].iter().product();
        let corners = 1usize << D;
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
        let grid = VtkGrid::new(
            "ondomathx-sem",
            D,
            n_elem * nloc,
            |i| {
                let x = space.map_and_jacobian(i / nloc, i % nloc).0;
                [x[0], x[1], if D == 3 { x[2] } else { 0.0 }]
            },
            if D == 2 { 9 } else { 12 },
            corners,
            n_elem * sub_per_elem,
            |cell, k| {
                let (e, s) = (cell / sub_per_elem, cell % sub_per_elem);
                let base = [s % p[0], (s / p[0]) % p[1], s / (p[0] * p[1])];
                let c = QUAD[k];
                let mut node = 0;
                let mut stride = 1;
                for d in 0..D {
                    node += (base[d] + c[d]) * stride;
                    stride *= S::N[d];
                }
                e * nloc + node
            },
        );
        VtkWriter { grid, nloc, n_elem }
    }

    /// Writes one snapshot to `path`.
    ///
    /// A field with one component is written as `SCALARS`, one with `D`
    /// components as `VECTORS` (padded to 3 in 2D), any other count as one
    /// `SCALARS` per component (`name_0`, `name_1`, …).
    pub fn write(&self, path: impl AsRef<Path>, fields: &[VtkField<'_>]) -> io::Result<()> {
        let nloc = self.nloc;
        let values: Vec<_> = fields
            .iter()
            .map(|f| {
                assert_eq!(f.data.len(), self.n_elem * nloc * f.components, "field {}: length", f.name);
                let (data, nc) = (f.data, f.components);
                move |p: usize, c: usize| data[(p / nloc) * nc * nloc + c * nloc + p % nloc]
            })
            .collect();
        let toolbox: Vec<Field<'_>> = fields
            .iter()
            .zip(&values)
            .map(|(f, v)| Field {
                name: f.name,
                components: f.components,
                value: v,
            })
            .collect();
        self.grid.write(path, &toolbox)
    }
}
