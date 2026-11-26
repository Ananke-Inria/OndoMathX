use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{Matrix3, SVector};
use ondomathx_tetrix::element::tet_10::{build_local_stiffness, mlt_stiffness};

fn test_matrix() -> Matrix3<f64> {
    Matrix3::new(2.0, 0.5, 0.1, 0.5, 3.0, 0.2, 0.1, 0.2, 1.5)
}

fn bench_mlt_stiffness(c: &mut Criterion) {
    let sym_grad = test_matrix();

    let k = build_local_stiffness(&sym_grad);

    let input: [f64; 10] = [1.0, -2.0, 0.5, 3.0, -1.5, 0.7, -0.3, 2.1, -0.9, 1.3];
    let mut output = [0.0_f64; 10];

    let vec_in = SVector::<f64, 10>::from_fn(|i, _| input[i]);
    let mut vec_out = SVector::<f64, 10>::zeros();

    let mut group = c.benchmark_group("mlt_stiffness  (10→10)");

    group.bench_function("matrix-free  (mlt_stiffness)", |b| {
        b.iter(|| {
            mlt_stiffness(
                std::hint::black_box(&input),
                std::hint::black_box(&sym_grad),
                std::hint::black_box(&mut output),
            )
        })
    });

    group.bench_function("nalgebra  K·v  (prebuilt 10×10)", |b| {
        b.iter(|| {
            k.mul_to(
                std::hint::black_box(&vec_in),
                std::hint::black_box(&mut vec_out),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_mlt_stiffness);
criterion_main!(benches);
