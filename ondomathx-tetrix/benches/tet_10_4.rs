use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector};
use ondomathx_tetrix::element::{tet_4, tet_10, tet_10_4};

fn build_matrix() -> DMatrix<f64> {
    DMatrix::from_fn(12, 10, |row, col| {
        let j = row / 3;
        let d = row % 3;
        tet_10::d_shape_function(col, d, &tet_4::POINTS[j])
    })
}

fn bench_gradient_interpolation(c: &mut Criterion) {
    let mat = build_matrix(); // 12×10

    let input_arr = [1.0_f64; 10];
    let mut output_arr = [0.0_f64; 12];

    let vec_in = DVector::from_fn(10, |i, _| input_arr[i]);
    let mut vec_out = DVector::zeros(12);

    let mut group = c.benchmark_group("gradient_interpolation  (10→12)");

    group.bench_function("precomputed coefficients", |b| {
        b.iter(|| {
            tet_10_4::gradient_interpolation(
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
    let mat = build_matrix();    // 12×10
    let mat_t = mat.transpose(); // 10×12

    let input_arr = [1.0_f64; 12];
    let mut output_arr = [0.0_f64; 10];

    let vec_in = DVector::from_fn(12, |i, _| input_arr[i]);
    let mut vec_out = DVector::zeros(10);

    let mut group = c.benchmark_group("transpose_gradient_interpolation  (12→10)");

    group.bench_function("precomputed coefficients", |b| {
        b.iter(|| {
            tet_10_4::transpose_gradient_interpolation(
                std::hint::black_box(&input_arr),
                std::hint::black_box(&mut output_arr),
            )
        })
    });

    group.bench_function("nalgebra  M^T·v  (tr_mul_to)", |b| {
        b.iter(|| {
            mat.tr_mul_to(
                std::hint::black_box(&vec_in),
                std::hint::black_box(&mut vec_out),
            )
        })
    });

    group.bench_function("nalgebra  M^T·v  (prebuilt transpose)", |b| {
        b.iter(|| {
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
