//! The P2 space's DG operations on the 6-tetrahedron unit cube
//! (`ondomathx-mesh/assets/Cube.mesh`).

use ondomathx_mesh::mesh::UnstructuredMesh;
use ondomathx_tetrix::tetrix_p2_space::TetriXP2Space;

fn cube() -> UnstructuredMesh {
    UnstructuredMesh::new_from_mesh(concat!(env!("CARGO_MANIFEST_DIR"), "/../ondomathx-mesh/assets/Cube.mesh"))
        .expect("Cube.mesh")
}

/// `mlt_add_stiffness_dg` adds exactly what `mlt_stiffness_dg` writes.
#[test]
fn mlt_add_stiffness_dg_accumulates() {
    let mesh = cube();
    let space = TetriXP2Space::new(&mesh);
    let n = space.num_dg_dofs();
    let u: Vec<f64> = (0..n).map(|k| ((k as f64) * 0.31).sin()).collect();
    let v0: Vec<f64> = (0..n).map(|k| ((k as f64) * 0.17).cos()).collect();
    let mut ku = vec![f64::NAN; n];
    space.mlt_stiffness_dg(&u, &mut ku, 2.0);
    let mut v = v0.clone();
    space.mlt_add_stiffness_dg(&u, &mut v, 2.0);
    let expected: Vec<f64> = v0.iter().zip(&ku).map(|(a, b)| a + b).collect();
    assert_eq!(v, expected);
}

/// `sample` is the scatter of the function at the global DoFs, component by
/// component in the layout `[elem][comp][node]`.
#[test]
fn sample_is_the_scatter_of_the_dof_values() {
    let mesh = cube();
    let space = TetriXP2Space::new(&mesh);
    let (n_dof, n_loc, n_elem) = (space.num_dofs(), space.num_loc_dofs(), space.num_elements());
    let f = |x: [f64; 3]| [x[0] + 2.0 * x[1], x[2] * x[2]];
    let v = space.sample::<2>(f);
    assert_eq!(v.len(), 2 * space.num_dg_dofs());
    for c in 0..2 {
        let cg: Vec<f64> = (0..n_dof).map(|g| f(*space.dof_coordinate(g))[c]).collect();
        let mut dg = vec![0.0; space.num_dg_dofs()];
        space.scatter_cg(&cg, &mut dg);
        for el in 0..n_elem {
            for i in 0..n_loc {
                assert_eq!(v[el * 2 * n_loc + c * n_loc + i], dg[el * n_loc + i], "el {el}, node {i}, comp {c}");
            }
        }
    }
}

/// The VTK writer: the P2 grid once, several fields per snapshot (a scalar
/// and a 3-component vector), and the former `io_vtk::write` as the scalar
/// `SOLUTION` of component 0.
#[test]
fn vtk_writer_fields_and_the_solution_wrapper() {
    use ondomathx_tetrix::io_vtk::{self, VtkField, VtkWriter};
    let mesh = cube();
    let space = TetriXP2Space::new(&mesh);
    let n_dof = space.num_dofs();
    let s: Vec<f64> = (0..n_dof).map(|g| g as f64).collect();
    let v: Vec<f64> = (0..3 * n_dof).map(|i| i as f64 * 0.5).collect();
    let writer = VtkWriter::new(&space);
    let mut out = Vec::new();
    writer
        .write_to(&mut out, &[
            VtkField { name: "s", components: 1, data: &s },
            VtkField { name: "v", components: 3, data: &v },
        ])
        .unwrap();
    let text = String::from_utf8(out).unwrap();
    let (n_p2, n_elem) = (space.p2_num_dofs(), space.num_elements());
    assert!(text.starts_with("# vtk DataFile Version 3.0\nondomathx-tetrix\n"));
    assert!(text.contains(&format!("POINTS {n_p2} double\n")));
    assert!(text.contains(&format!("CELLS {n_elem} {}\n", 11 * n_elem)));
    assert!(text.contains("SCALARS s double 1\n") && text.contains("VECTORS v double\n"));
    // The P2 point i carries the value of its global DoF.
    let after = text.split("VECTORS v double\n").nth(1).unwrap();
    let g = space.p2_glob2glob(1);
    let line: Vec<f64> = after.lines().nth(1).unwrap().split(' ').map(|t| t.parse().unwrap()).collect();
    assert_eq!(line, [v[3 * g], v[3 * g + 1], v[3 * g + 2]]);

    let mut wrapped = Vec::new();
    io_vtk::write(&mut wrapped, &space, &v).unwrap();
    let mut direct = Vec::new();
    let first: Vec<f64> = (0..n_dof).map(|g| v[3 * g]).collect();
    writer.write_to(&mut direct, &[VtkField { name: "SOLUTION", components: 1, data: &first }]).unwrap();
    assert_eq!(wrapped, direct);
}

/// The Vizir writer: `HOSolAtTetrahedraP1`, one row of 10 nodes × C values
/// per element.
#[test]
fn vizir_writer_rows() {
    use ondomathx_tetrix::io_vizir::VizirWriter;
    let mesh = cube();
    let space = TetriXP2Space::new(&mesh);
    let data: Vec<f64> = (0..2 * space.num_dofs()).map(|i| i as f64).collect();
    let path = std::env::temp_dir().join(format!("tetrix-vizir-{}.sol", std::process::id()));
    let writer = VizirWriter::new(&space, &mesh);
    assert_eq!(writer.tag(), "TetrahedraP1");
    writer.write(&path, &data, 2).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let i = lines.iter().position(|l| *l == "HOSolAtTetrahedraP1").unwrap();
    assert_eq!(lines[i + 1].parse::<usize>().unwrap(), space.num_elements());
    assert_eq!((lines[i + 2], lines[i + 3]), ("1 2", "2 10"));
    let row: Vec<f64> = lines[i + 4].split_whitespace().map(|t| t.parse().unwrap()).collect();
    assert_eq!(row.len(), 20);
    let g = space.p2_glob2glob(space.p2_loc2glob(0, 0));
    assert_eq!((row[0], row[1]), (data[2 * g], data[2 * g + 1]));
    assert_eq!(*lines.last().unwrap(), "End");
}
