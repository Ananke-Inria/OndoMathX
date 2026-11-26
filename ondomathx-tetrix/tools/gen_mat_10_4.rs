/// Code generator for `GRADIENT_MAT` used by `gradient_interpolation` and
/// `transpose_gradient_interpolation` in `element/tet_10_4.rs`.
///
/// Runs with:
///   cargo run --example gen_mat_10_4
///
/// GRADIENT_MAT[k][i] = ∂φ_i/∂x_d(q_j)   where k = j*3+d  (12 rows × 10 cols)

use ondomathx_tetrix::element::{tet_4, tet_10};

const ZERO_THRESHOLD: f64 = 1e-15;

fn clean(v: f64) -> f64 {
    if v.abs() < ZERO_THRESHOLD { 0.0 } else { v }
}

fn main() {
    println!("/// 12×10 matrix: GRADIENT_MAT[k][i] = ∂φ_i/∂x_d(q_j), k = j*3+d");
    println!("const GRADIENT_MAT: [[f64; 10]; 12] = [");
    for k in 0..12_usize {
        let j = k / 3;
        let d = k % 3;
        let row: Vec<String> = (0..10)
            .map(|i| format!("{:.16e}", clean(tet_10::d_shape_function(i, d, &tet_4::POINTS[j]))))
            .collect();
        println!("    [{}], // j={j}, d={d}", row.join(", "));
    }
    println!("];");
}
