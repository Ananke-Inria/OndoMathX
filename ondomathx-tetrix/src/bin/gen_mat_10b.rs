/// Code generator for the K_BB basis stiffness matrices used in `finite_element/tet_10b.rs`.
///
/// Computes 6 symmetric 5×5 matrices:
///   KBB_AB[p][q] = ∫_T A_AB ∇φ_{10+p} · ∇φ_{10+q} dV
/// for each basis element A of the space of symmetric 3×3 matrices.
///
/// The K_BB block for a diffusion tensor A is then:
///   K_BB = a00·KBB_00 + a01·KBB_01 + a02·KBB_02 + a11·KBB_11 + a12·KBB_12 + a22·KBB_22
///
/// Runs with:
///   cargo run --bin gen_mat_10b

use ondomathx_tetrix::finite_element::{tet_10b, tet_14};

const ZERO_THRESHOLD: f64 = 1e-15;

fn clean(v: f64) -> f64 {
    if v.abs() < ZERO_THRESHOLD { 0.0 } else { v }
}

fn main() {
    // Accumulate 6 5×5 matrices by numerical quadrature over the reference tetrahedron.
    // Ordering: [M00, M01, M02, M11, M12, M22]
    let mut mats = [[[0.0f64; 5]; 5]; 6];

    for i_quad in 0..tet_14::WEIGHTS.len() {
        let w = tet_14::WEIGHTS[i_quad];
        let pt = &tet_14::POINTS[i_quad];

        let du: [f64; 5] = std::array::from_fn(|k| tet_10b::d_shape_function(10 + k, 0, pt));
        let dv: [f64; 5] = std::array::from_fn(|k| tet_10b::d_shape_function(10 + k, 1, pt));
        let dw: [f64; 5] = std::array::from_fn(|k| tet_10b::d_shape_function(10 + k, 2, pt));

        for p in 0..5 {
            for q in 0..5 {
                mats[0][p][q] += w * du[p] * du[q];
                mats[1][p][q] += w * (dv[p] * du[q] + du[p] * dv[q]);
                mats[2][p][q] += w * (dw[p] * du[q] + du[p] * dw[q]);
                mats[3][p][q] += w * dv[p] * dv[q];
                mats[4][p][q] += w * (dw[p] * dv[q] + dv[p] * dw[q]);
                mats[5][p][q] += w * dw[p] * dw[q];
            }
        }
    }

    let names = ["KBB_00", "KBB_01", "KBB_02", "KBB_11", "KBB_12", "KBB_22"];
    let comments = [
        "A = [[1,0,0],[0,0,0],[0,0,0]]",
        "A = [[0,1,0],[1,0,0],[0,0,0]]",
        "A = [[0,0,1],[0,0,0],[1,0,0]]",
        "A = [[0,0,0],[0,1,0],[0,0,0]]",
        "A = [[0,0,0],[0,0,1],[0,1,0]]",
        "A = [[0,0,0],[0,0,0],[0,0,1]]",
    ];

    println!("/// K_BB basis stiffness matrices: KBB_AB[p][q] = ∫_T A_AB ∇φ_{{10+p}} · ∇φ_{{10+q}} dV");
    println!("/// K_BB = a00·KBB_00 + a01·KBB_01 + a02·KBB_02 + a11·KBB_11 + a12·KBB_12 + a22·KBB_22");
    println!();
    for (m, (name, comment)) in mats.iter().zip(names.iter().zip(comments.iter())) {
        println!("/// {}", comment);
        println!("pub const {}: [[f64; 5]; 5] = [", name);
        for (p, row) in m.iter().enumerate() {
            let entries: Vec<String> = row.iter().map(|&v| format!("{:.16e}", clean(v))).collect();
            println!("    [{}], // p={}", entries.join(", "), p);
        }
        println!("];");
        println!();
    }
}
