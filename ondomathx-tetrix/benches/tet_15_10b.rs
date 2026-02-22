use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector};
use tetrix::finite_element::{tet_10b, tet_15, tet_15_10b};

/// Build the 15×15 change-of-basis matrix M where M[i][j] = φ_j^{10b}(POINTS_15[i]).
///
/// `tet_10b_to_tet_15` computes  a = M  · b  (tet_10b → tet_15)
/// `tet_15_to_tet_10b` computes  b = M⁻¹· a  (tet_15 → tet_10b, via forward substitution)
fn build_matrix() -> DMatrix<f64> {
    DMatrix::from_fn(15, 15, |i, j| {
        tet_10b::shape_function(j, &tet_15::POINTS[i])
    })
}

/// Naive tet_10b → tet_15: on-the-fly shape evaluation, no precomputed data.
fn naive_10b_to_15(b: &[f64; 15], a: &mut [f64; 15]) {
    for i in 0..15 {
        let pt = &tet_15::POINTS[i];
        a[i] = (0..15).map(|j| b[j] * tet_10b::shape_function(j, pt)).sum();
    }
}

/// Naive tet_15 → tet_10b: on-the-fly shape evaluation + forward substitution.
fn naive_15_to_10b(a: &[f64; 15], b: &mut [f64; 15]) {
    let mut val = [0.0f64; 15];
    for j in 0..15 {
        let pt = &tet_10b::POINTS[j];
        val[j] = (0..15).map(|i| a[i] * tet_15::shape_function(i, pt)).sum();
    }
    for j in 0..15 {
        let pt = &tet_10b::POINTS[j];
        let sub: f64 = (0..j).map(|k| b[k] * tet_10b::shape_function(k, pt)).sum();
        b[j] = val[j] - sub;
    }
}

fn bench_10b_to_15(c: &mut Criterion) {
    let mat = build_matrix(); // 15×15

    let b_arr = [
        1.0, -2.0, 0.5, 3.0, -1.5, 0.7, -0.3, 2.1, -0.9, 1.3, -0.4, 0.8, -1.1, 0.6, -0.2_f64,
    ];
    let mut a_arr = [0.0_f64; 15];

    let b_vec = DVector::from_fn(15, |i, _| b_arr[i]);
    let mut a_vec = DVector::zeros(15);

    let mut group = c.benchmark_group("tet_10b_to_tet_15  (15→15)");

    group.bench_function("precomputed BUBBLE_MAT", |b| {
        b.iter(|| {
            tet_15_10b::tet_10b_to_tet_15(
                std::hint::black_box(&b_arr),
                std::hint::black_box(&mut a_arr),
            )
        })
    });

    group.bench_function("nalgebra  M·v", |b| {
        b.iter(|| {
            mat.mul_to(
                std::hint::black_box(&b_vec),
                std::hint::black_box(&mut a_vec),
            )
        })
    });

    group.bench_function("naive on-the-fly", |b| {
        b.iter(|| {
            naive_10b_to_15(
                std::hint::black_box(&b_arr),
                std::hint::black_box(&mut a_arr),
            )
        })
    });

    group.finish();
}

fn bench_15_to_10b(c: &mut Criterion) {
    let a_arr = [
        0.5, 1.2, -0.8, 2.0, -1.0, 0.3, 0.9, -0.4, 1.7, -0.6, 0.1, -1.3, 0.4, -0.7, 0.2_f64,
    ];
    let mut b_arr = [0.0_f64; 15];

    let mut group = c.benchmark_group("tet_15_to_tet_10b  (15→15)");

    group.bench_function("precomputed BUBBLE_MAT", |b| {
        b.iter(|| {
            tet_15_10b::tet_15_to_tet_10b(
                std::hint::black_box(&a_arr),
                std::hint::black_box(&mut b_arr),
            )
        })
    });

    group.bench_function("naive on-the-fly", |b| {
        b.iter(|| {
            naive_15_to_10b(
                std::hint::black_box(&a_arr),
                std::hint::black_box(&mut b_arr),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_10b_to_15, bench_15_to_10b);
criterion_main!(benches);
