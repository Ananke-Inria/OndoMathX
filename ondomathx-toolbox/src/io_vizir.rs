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
//! <number of fields> <type of each field>
//! <degree> <nloc>
//! <values of element 0>                 (node by node, components inner)
//! …
//! End
//! ```
//!
//! The field types are libMeshb's: `1` scalar, `2` vector (3 values in
//! dimension 3), `3` symmetric matrix, `4` matrix. A solution of
//! `components` values per node is written as one scalar (`1 1`) for one
//! component, one vector (`1 2`, 2D vectors padded with `z = 0`) for two or
//! three, and `components` scalars (`C 1 1 … 1`) otherwise. Writing the
//! component count as the type (`1 3` for a 3D vector) made Vizir read a
//! symmetric matrix: 6 values per node, the data misaligned.
//!
//! [`write_sols`] writes the list of a time series (`.sols`), opened from
//! the directory of the files with `vizir4 -in <mesh> -sol <first> -sols
//! <name>.sols` (Utilities, Non Stationary Solutions; F9 / F10 step).
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
        assert!(components > 0, "no component");
        // A vector (2 or 3 components) is written with 3 values, the 2D ones
        // padded with 0 (the files are in dimension 3).
        let (written, types) = match components {
            1 => (1, "1 1".to_string()),
            2 | 3 => (3, "1 2".to_string()),
            c => (c, format!("{c}{}", " 1".repeat(c))),
        };
        let rows: Vec<Vec<u8>> = (0..self.n_elem)
            .into_par_iter()
            .map(|r| {
                let mut b = Vec::with_capacity(self.nloc * written * 24);
                for node in 0..self.nloc {
                    for c in 0..written {
                        let v = if c < components { value(r, node, c) } else { 0.0 };
                        write!(b, "{v} ").unwrap();
                    }
                }
                b.push(b'\n');
                b
            })
            .collect();
        w.write_all(&self.header)?;
        writeln!(w, "{types}")?;
        writeln!(w, "{} {}", self.degree, self.nloc)?;
        for r in &rows {
            w.write_all(r)?;
        }
        w.write_all(b"End\n")
    }
}

/// Writes the list of the solution files of a time series (one path per
/// line, written as given), to open with `vizir4 -in <mesh> -sols <path>`.
/// Vizir resolves a relative path against its working directory, not the
/// directory of the `.sols` file: with bare file names, launch it from the
/// directory of the files. `-sols` does not open the first file; give it
/// with `-sol`.
pub fn write_sols(path: impl AsRef<Path>, files: &[impl AsRef<str>]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for f in files {
        writeln!(w, "{}", f.as_ref())?;
    }
    w.flush()
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
            HOSolAtEdgesP1\n2\n1 2\n1 2\n0 1 0 10 11 0 \n100 101 0 110 111 0 \nEnd\n";
        assert_eq!(String::from_utf8(out).unwrap(), expected);
        assert_eq!(sol.tag(), "EdgesP1");
    }

    /// The libMeshb field types: a scalar, a vector (3 values), and any
    /// other count as that many scalars.
    #[test]
    fn field_types_by_component_count() {
        let sol = HoSolution::new("EdgesP1", 1, &[vec![0.0], vec![1.0]], 1);
        let types = |c: usize| {
            let mut out = Vec::new();
            sol.write_to(&mut out, c, |_, n, k| (10 * n + k) as f64).unwrap();
            let text = String::from_utf8(out).unwrap();
            let lines: Vec<String> = text.lines().map(String::from).collect();
            let i = lines.iter().position(|l| l == "HOSolAtEdgesP1").unwrap();
            (lines[i + 2].clone(), lines[i + 4].clone())
        };
        assert_eq!(types(1), ("1 1".into(), "0 10 ".into()));
        assert_eq!(types(3), ("1 2".into(), "0 1 2 10 11 12 ".into()));
        assert_eq!(types(4), ("4 1 1 1 1".into(), "0 1 2 3 10 11 12 13 ".into()));
    }

    #[test]
    fn a_sols_list() {
        let path = std::env::temp_dir().join(format!("toolbox-sols-{}.sols", std::process::id()));
        write_sols(&path, &["a.00000.sol", "a.00001.sol"]).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a.00000.sol\na.00001.sol\n");
        std::fs::remove_file(&path).unwrap();
    }
}
