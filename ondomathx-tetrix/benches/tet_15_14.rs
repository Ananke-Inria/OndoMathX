use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector};
use tetrix::finite_element::{tet_14, tet_15, tet_15_14};

/// Build the 42×15 gradient matrix M where M[j*3+d, i] = ∂φ_i/∂x_d(q_j).
///
/// `gradient_interpolation`           computes M   · v  (15→42)
/// `transpose_gradient_interpolation` computes M^T · z  (42→15)
fn build_matrix() -> DMatrix<f64> {
    DMatrix::from_fn(42, 15, |row, col| {
        let j = row / 3;
        let d = row % 3;
        tet_15::d_shape_function(col, d, &tet_14::POINTS[j])
    })
}

fn bench_gradient_interpolation(c: &mut Criterion) {
    let mat = build_matrix(); // 42×15

    let input_arr = [1.0_f64; 15];
    let mut output_arr = [0.0_f64; 42];

    let vec_in = DVector::from_fn(15, |i, _| input_arr[i]);
    let mut vec_out = DVector::zeros(42);

    let mut group = c.benchmark_group("gradient_interpolation  (15→42)");

    group.bench_function("precomputed coefficients", |b| {
        b.iter(|| {
            tet_15_14::gradient_interpolation(
                std::hint::black_box(&input_arr),
                std::hint::black_box(&mut output_arr),
            )
        })
    });

    group.bench_function("nalgebra  M·v", |b| {
        b.iter(|| {
            mat.mul_to(
                std::hint::black_box(&vec_in),
                std::hint::black_box(&mut vec_out),
            )
        })
    });

    group.finish();
}

fn bench_transpose_gradient_interpolation(c: &mut Criterion) {
    let mat = build_matrix();   // 42×15
    let mat_t = mat.transpose(); // 15×42

    let input_arr = [1.0_f64; 42];
    let mut output_arr = [0.0_f64; 15];

    let vec_in = DVector::from_fn(42, |i, _| input_arr[i]);
    let mut vec_out = DVector::zeros(15);

    let mut group = c.benchmark_group("transpose_gradient_interpolation  (42→15)");

    group.bench_function("precomputed coefficients", |b| {
        b.iter(|| {
            tet_15_14::transpose_gradient_interpolation(
                std::hint::black_box(&input_arr),
                std::hint::black_box(&mut output_arr),
            )
        })
    });

    group.bench_function("nalgebra  M^T·v  (tr_mul_to)", |b| {
        b.iter(|| {
            // computes M^T · v without building the transpose matrix
            mat.tr_mul_to(
                std::hint::black_box(&vec_in),
                std::hint::black_box(&mut vec_out),
            )
        })
    });

    group.bench_function("nalgebra  M^T·v  (prebuilt transpose)", |b| {
        b.iter(|| {
            // prebuilt 15×42 matrix multiplied by a 42-vector
            mat_t.mul_to(
                std::hint::black_box(&vec_in),
                std::hint::black_box(&mut vec_out),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gradient_interpolation,
    bench_transpose_gradient_interpolation
);
criterion_main!(benches);
