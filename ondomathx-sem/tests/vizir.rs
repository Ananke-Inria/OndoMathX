//! Vizir `.sol` output (`VizirWriter`): header, node positions, rows in the
//! order of Anamesh's `.mesh` section, values.

mod common;

use anamesh::mesh::square::Square;
use anamesh::mesh::Mesh;
use common::*;
use lobattox::{Gll2, Iso2, Iso3, Numbering, Space, VizirWriter};

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("lobattox-vizir-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// A parsed `.sol`: (tag, positions, n_elem, components, degree, rows).
type Sol = (String, Vec<Vec<f64>>, usize, usize, usize, Vec<Vec<f64>>);

/// Parses a `.sol`.
fn parse(path: &std::path::Path) -> Sol {
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(&lines[..3], &["MeshVersionFormatted 2", "Dimension", "3"]);
    let tag = lines[3].strip_prefix("HOSolAt").unwrap().strip_suffix("NodesPositions").unwrap().to_string();
    let nloc: usize = lines[4].parse().unwrap();
    let pos = lines[5..5 + nloc]
        .iter()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect();
    let mut i = 5 + nloc;
    assert_eq!(lines[i], format!("HOSolAt{tag}"));
    let n_elem: usize = lines[i + 1].parse().unwrap();
    let c: usize = lines[i + 2].split_whitespace().nth(1).unwrap().parse().unwrap();
    let dl: Vec<usize> = lines[i + 3].split_whitespace().map(|v| v.parse().unwrap()).collect();
    assert_eq!(dl[1], nloc);
    i += 4;
    let rows = lines[i..i + n_elem]
        .iter()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(lines[i + n_elem], "End");
    (tag, pos, n_elem, c, dl[0], rows)
}

/// Number of entries of a section of a Medit `.mesh` written by Anamesh.
fn section_count(path: &std::path::Path, section: &str) -> usize {
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let k = lines.iter().position(|l| l.trim() == section).unwrap();
    lines[k + 1].trim().parse().unwrap()
}

fn check<const D: usize, S: lobattox::Shape<D>>(space: &Space<D, S>, mesh: &impl Mesh, tag: &str, name: &str) {
    let c = D;
    let data = space.sample::<D>(|x| core::array::from_fn(|a| x[a] + 10.0 * a as f64));
    let w = VizirWriter::new(space, mesh);
    assert_eq!(w.tag(), tag);
    let sol = temp(&format!("{name}.sol"));
    w.write(&sol, &data, c).unwrap();
    let (t, pos, n_elem, cc, degree, rows) = parse(&sol);
    assert_eq!((t.as_str(), n_elem, cc, degree), (tag, space.n_elem(), c, S::N[0] - 1));
    for (node, p) in pos.iter().enumerate() {
        assert_eq!(p.as_slice(), space.ref_point(node).as_slice());
    }
    // Row k: the k-th volume element of the mesh (mesh order), node-major.
    let mut by_mesh: Vec<usize> = (0..space.n_elem()).collect();
    by_mesh.sort_by_key(|&e| space.mesh_element(e));
    let nloc = S::NLOC;
    for (row, &e) in rows.iter().zip(&by_mesh) {
        assert_eq!(row.len(), nloc * c);
        for node in 0..nloc {
            for k in 0..c {
                assert_eq!(row[node * c + k], data[(e * c + k) * nloc + node]);
            }
        }
    }
    // The same count in Anamesh's .mesh section.
    let mesh_path = temp(&format!("{name}.mesh"));
    mesh.write_mesh(mesh_path.to_str().unwrap()).unwrap();
    let section = tag.strip_suffix("Q1").unwrap_or(tag);
    assert_eq!(section_count(&mesh_path, section), n_elem);
}

#[test]
fn vizir_2d_and_3d() {
    // Mixed affine/Q1 storage order differs from the mesh order.
    let mesh = square(5, 3, false, Some(partial_2d()));
    check(&Space::<2, Iso2<4>>::new(&mesh, Numbering::None), &mesh, "QuadrilateralsQ1", "q1");
    let mesh = square(3, 2, true, Some(bending_2d()));
    check(&Space::<2, Iso2<5>>::new(&mesh, Numbering::None), &mesh, "QuadrilateralsQ2", "q2");
    let sq = Square::new([0.0, 0.0], 2.0, 1.0, 3, 2, Some(bending_2d()), None);
    check(&Space::<2, Iso2<3>>::new(&sq, Numbering::None), &sq, "QuadrilateralsQ1", "mapped");
    let mesh = cube([2, 1, 2], true, Some(bending_3d()));
    check(&Space::<3, Iso3<3>>::new(&mesh, Numbering::None), &mesh, "HexahedraQ2", "hex");
}

#[test]
#[should_panic(expected = "anisotropic")]
fn vizir_anisotropic_panics() {
    let mesh = square(2, 2, false, None);
    let _ = VizirWriter::new(&Space::<2, Gll2<3, 4>>::new(&mesh, Numbering::None), &mesh);
}
