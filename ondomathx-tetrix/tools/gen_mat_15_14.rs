/// Code generator for `GRADIENT_MAT` used by `gradient_interpolation` and
/// `transpose_gradient_interpolation` in `element/tet_15_14.rs`.
///
/// Runs with:
///   cargo run --example gen_mat_15_14
///
/// GRADIENT_MAT[k][i] = ∂φ_i/∂x_d(q_j)   where k = j*3+d  (42 rows × 15 cols)
///
/// Both functions share this single matrix:
///   gradient_interpolation:           output[k] = Σ_i GRADIENT_MAT[k][i] * input[i]
///   transpose_gradient_interpolation: output[i] += GRADIENT_MAT[k][i] * input[k]  (axpy)

use ondomathx_tetrix::element::{tet_14, tet_15};

/// Snap values below this threshold to exactly 0.0 to eliminate floating-point
/// noise from entries that are mathematically zero.
const ZERO_THRESHOLD: f64 = 1e-15;

fn clean(v: f64) -> f64 {
    if v.abs() < ZERO_THRESHOLD { 0.0 } else { v }
}

fn main() {
    println!("/// 42×15 matrix: GRADIENT_MAT[k][i] = ∂φ_i/∂x_d(q_j), k = j*3+d");
    println!("const GRADIENT_MAT: [[f64; 15]; 42] = [");
    for k in 0..42_usize {
        let j = k / 3;
        let d = k % 3;
        let row: Vec<String> = (0..15)
            .map(|i| format!("{:.16e}", clean(tet_15::d_shape_function(i, d, &tet_14::POINTS[j]))))
            .collect();
        println!("    [{}], // j={j}, d={d}", row.join(", "));
    }
    println!("];");
}
