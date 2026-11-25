use ondomathx_mesh::mesh::UnstructuredMesh;

use crate::tetrix_p2_space::{TetriXP2Space, ORDER};
/// Write a Vizir `.sol` file for a scalar or vector solution.
///
/// The file follows the `MeshVersionFormatted 2` / `HOSolAt` convention used
/// by the Vizir visualiser.
pub fn write_solution(
    file_name: &str,
    fe_space: &TetriXP2Space,
    _mesh: &UnstructuredMesh,
    solution: &[f64],
    n: usize,
) {
    let dim_sol = solution.len() / fe_space.num_dofs();
    let file_path = format!("{}.{:05}.sol", file_name, n);

    let file = std::fs::File::create(&file_path)
        .unwrap_or_else(|e| panic!("Cannot create {}: {}", file_path, e));
    let mut ofs = std::io::BufWriter::new(file);

    use std::io::Write;

    let dimension: usize = 3;
    let geo_element_tag = "TetrahedraP1";
    let n_loc_p2 = fe_space.num_loc_p2_dofs();

    writeln!(ofs, "MeshVersionFormatted 2").unwrap();
    writeln!(ofs, "Dimension").unwrap();
    writeln!(ofs, "{}", dimension).unwrap();
    write!(ofs, "HOSolAt{}NodesPositions", geo_element_tag).unwrap();
    writeln!(ofs).unwrap();
    writeln!(ofs, "{}", n_loc_p2).unwrap();

    for i_loc in 0..n_loc_p2 {
        let uvw = fe_space.loc_dof_coordinates(i_loc);
        let bc = fe_space.get_barycentric_coordinates(uvw);
        for coord in &bc {
            write!(ofs, "{} ", coord).unwrap();
        }
        writeln!(ofs).unwrap();
    }

    writeln!(ofs, "HOSolAt{}", geo_element_tag).unwrap();
    writeln!(ofs, "{}", fe_space.num_elements()).unwrap();
    writeln!(ofs, "1 {}", dim_sol).unwrap();
    writeln!(ofs, "{} {}", ORDER, n_loc_p2).unwrap();

    for i_elt in 0..fe_space.num_elements() {
        for i_loc in 0..n_loc_p2 {
            for k in 0..dim_sol {
                let i_glob_p2 = fe_space.p2_loc2glob(i_elt, i_loc);
                let i_glob = fe_space.p2_glob2glob(i_glob_p2);
                write!(ofs, "{} ", solution[dim_sol * i_glob + k]).unwrap();
            }
        }
        writeln!(ofs).unwrap();
    }

    writeln!(ofs, "End").unwrap();
}
