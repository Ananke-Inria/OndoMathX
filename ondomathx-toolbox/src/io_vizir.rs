//! Vizir high-order solutions (`HOSolAt…`, libMeshb `.sol` files): the file
//! format shared by the writers of `ondomathx-tetrix` and `ondomathx-sem`.
//!
//! A `.sol` file holds, per element of the matching `.mesh` section, the
//! solution at the element's nodes, whose reference positions are given
//! once (`HOSolAt<tag>NodesPositions`). Format (`Dimension 3`, the
//! dimension of the meshes `ondomathx-mesh` writes):
//!
//! ```text
//! MeshVersionFormatted 2
//! Dimension
//! 3
//! HOSolAt<tag>NodesPositions
//! <nloc>
//! <position of node 0> …                (one line per node)
//! HOSolAt<tag>
//! <n_elem>
//! 1 <components>
//! <degree> <nloc>
//! <values of element 0>                 (node by node, components inner)
//! …
//! End
//! ```
//!
//! The header is formatted once in [`HoSolution::new`]; each
//! [`HoSolution::write`] formats the rows, in parallel. Numbers are written
//! in their shortest exact form (`{}`).

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use rayon::prelude::*;

/// A `HOSolAt<tag>` layout: element type, degree, node positions, number of
/// elements.
pub struct HoSolution {
    header: Vec<u8>,
    tag: String,
    degree: usize,
    nloc: usize,
    n_elem: usize,
}

impl HoSolution {
    /// The layout of tag `tag` (e.g. `TetrahedraP1`, `QuadrilateralsQ2`,
    /// `HexahedraQ1`), polynomial degree `degree`, `positions[node]` the
    /// reference (or barycentric) coordinates of each node, `n_elem` rows.
    pub fn new(tag: &str, degree: usize, positions: &[Vec<f64>], n_elem: usize) -> Self {
        let nloc = positions.len();
        let mut header = Vec::new();
        write!(header, "MeshVersionFormatted 2\nDimension\n3\nHOSolAt{tag}NodesPositions\n{nloc}\n").unwrap();
        for p in positions {
            for x in p {
                write!(header, "{x} ").unwrap();
            }
            header.push(b'\n');
        }
        write!(header, "HOSolAt{tag}\n{n_elem}\n").unwrap();
        HoSolution {
            header,
            tag: tag.to_string(),
            degree,
            nloc,
            n_elem,
        }
    }

    /// The element tag.
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Writes a `components`-component solution to the file `path`:
    /// `value(row, node, comp)`, `row` in the order of the `.mesh` section.
    pub fn write(
        &self,
        path: impl AsRef<Path>,
        components: usize,
        value: impl Fn(usize, usize, usize) -> f64 + Sync,
    ) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        self.write_to(&mut w, components, value)?;
        w.flush()
    }

    /// Writes a `components`-component solution to `w` (see [`HoSolution::write`]).
    pub fn write_to(
        &self,
        w: &mut impl Write,
        components: usize,
        value: impl Fn(usize, usize, usize) -> f64 + Sync,
    ) -> io::Result<()> {
        let rows: Vec<Vec<u8>> = (0..self.n_elem)
            .into_par_iter()
            .map(|r| {
                let mut b = Vec::with_capacity(self.nloc * components * 24);
                for node in 0..self.nloc {
                    for c in 0..components {
                        write!(b, "{} ", value(r, node, c)).unwrap();
                    }
                }
                b.push(b'\n');
                b
            })
            .collect();
        w.write_all(&self.header)?;
        writeln!(w, "1 {components}")?;
        writeln!(w, "{} {}", self.degree, self.nloc)?;
        for r in &rows {
            w.write_all(r)?;
        }
        w.write_all(b"End\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_solution() {
        let pos = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let sol = HoSolution::new("EdgesP1", 1, &pos, 2);
        let mut out = Vec::new();
        sol.write_to(&mut out, 2, |r, n, c| (100 * r + 10 * n + c) as f64).unwrap();
        let expected = "MeshVersionFormatted 2\nDimension\n3\nHOSolAtEdgesP1NodesPositions\n2\n0 0 \n1 0 \n\
            HOSolAtEdgesP1\n2\n1 2\n1 2\n0 1 10 11 \n100 101 110 111 \nEnd\n";
        assert_eq!(String::from_utf8(out).unwrap(), expected);
        assert_eq!(sol.tag(), "EdgesP1");
    }
}
