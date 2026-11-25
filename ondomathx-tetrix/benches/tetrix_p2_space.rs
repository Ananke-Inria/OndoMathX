use ondomathx_mesh::mesh::UnstructuredMesh;
use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use sprs::prod::mul_acc_mat_vec_csr;
use ondomathx_tetrix::tetrix_p2_space::TetriXP2Space;

fn setup() -> (UnstructuredMesh,) {
    let mesh = UnstructuredMesh::new_from_mesh(concat!(env!("CARGO_MANIFEST_DIR"), "/../ondomathx-mesh/assets/Heart.mesh"))
        .expect("Failed to load Heart.mesh");
    (mesh,)
}

fn bench_mlt_stiffness(c: &mut Criterion) {
    let (mesh,) = setup();
    let fe_space = TetriXP2Space::new(&mesh);
    let n_dof = fe_space.num_dofs();

    let u: Vec<f64> = (0..n_dof).map(|i| (i as f64).sin()).collect();
    let mut v = vec![0.0_f64; n_dof];

    c.bench_function("mlt_stiffness", |b| {
        b.iter(|| {
            fe_space.mlt_stiffness(std::hint::black_box(&u), std::hint::black_box(&mut v), 1.0)
        })
    });
}

fn bench_mlt_stiffness_dg(c: &mut Criterion) {
    let (mesh,) = setup();
    let fe_space = TetriXP2Space::new(&mesh);
    let n_elem = fe_space.num_elements();
    let n_loc = fe_space.num_loc_dofs();
    let n_dg = n_elem * n_loc;

    let u: Vec<f64> = (0..n_dg).map(|i| (i as f64).sin()).collect();
    let mut v = vec![0.0_f64; n_dg];

    c.bench_function("mlt_stiffness_dg", |b| {
        b.iter(|| {
            fe_space.mlt_stiffness_dg(std::hint::black_box(&u), std::hint::black_box(&mut v), 1.0)
        })
    });
}

fn bench_mlt_stiffness_dg_seq(c: &mut Criterion) {
    let (mesh,) = setup();
    let fe_space = TetriXP2Space::new(&mesh);
    let n_elem = fe_space.num_elements();
    let n_loc = fe_space.num_loc_dofs();
    let n_dg = n_elem * n_loc;

    let u: Vec<f64> = (0..n_dg).map(|i| (i as f64).sin()).collect();
    let mut v = vec![0.0_f64; n_dg];

    c.bench_function("mlt_stiffness_dg_seq", |b| {
        b.iter(|| {
            fe_space.mlt_stiffness_dg_seq(
                std::hint::black_box(&u),
                std::hint::black_box(&mut v),
                1.0,
            )
        })
    });
}

fn bench_mlt_inverse_mass(c: &mut Criterion) {
    let (mesh,) = setup();
    let fe_space = TetriXP2Space::new(&mesh);
    let n_dof = fe_space.num_dofs();
    let n_elem = fe_space.num_elements();
    let n_loc = fe_space.num_loc_dofs();
    let n_dg = n_elem * n_loc;

    let mut mass = vec![0.0_f64; n_dof];
    fe_space.assemble_mass(&mut mass, 1.0);

    let mut dg: Vec<f64> = (0..n_dg).map(|i| (i as f64).sin()).collect();

    c.bench_function("mlt_inverse_mass", |b| {
        b.iter(|| {
            fe_space.mlt_inverse_mass(std::hint::black_box(&mut dg), std::hint::black_box(&mass))
        })
    });
}

/// Compare: sparse K·u (sprs SpMV) + lumped M⁻¹·v  vs  matrix-free DG + mlt_inverse_mass.
///
/// Both represent one application of M⁻¹K on a vector.
///   - sparse CG:   K_csr · u_cg  →  v_cg[i] /= mass[i]
///   - matfree DG:  mlt_stiffness_dg(u_dg, v_dg)  →  mlt_inverse_mass(v_dg, mass)
fn bench_stiffness_inverse_mass(c: &mut Criterion) {
    let (mesh,) = setup();
    let fe_space = TetriXP2Space::new(&mesh);
    let n_dof = fe_space.num_dofs();
    let n_dg = fe_space.num_elements() * fe_space.num_loc_dofs();

    // --- shared setup ---
    let k_csr = fe_space.assemble_stiffness(1.0);
    let mut mass = vec![0.0_f64; n_dof];
    fe_space.assemble_mass(&mut mass, 1.0);

    let u_cg: Vec<f64> = (0..n_dof).map(|i| (i as f64).sin()).collect();
    let mut v_cg = vec![0.0_f64; n_dof];

    let mut u_dg = vec![0.0_f64; n_dg];
    fe_space.scatter_cg(&u_cg, &mut u_dg);
    let mut v_dg = vec![0.0_f64; n_dg];

    // nalgebra-sparse: convert sprs CsMat → CsrMatrix
    let k_na = CsrMatrix::try_from_csr_data(
        k_csr.rows(),
        k_csr.cols(),
        k_csr.indptr().as_slice().unwrap().to_vec(),
        k_csr.indices().to_vec(),
        k_csr.data().to_vec(),
    )
    .unwrap();
    let u_na = DVector::from_vec(u_cg.clone());

    let mut group = c.benchmark_group("M⁻¹ K·u");

    group.bench_function("sparse CSR + lumped CG M⁻¹", |b| {
        b.iter(|| {
            v_cg.fill(0.0);
            mul_acc_mat_vec_csr(
                k_csr.view(),
                std::hint::black_box(u_cg.as_slice()),
                std::hint::black_box(v_cg.as_mut_slice()),
            );
            for i in 0..n_dof {
                v_cg[i] /= mass[i];
            }
        })
    });

    group.bench_function("nalgebra-sparse CSR + lumped CG M⁻¹", |b| {
        b.iter(|| {
            let mut v_na = &k_na * std::hint::black_box(&u_na);
            for i in 0..n_dof {
                v_na[i] /= mass[i];
            }
        })
    });

    group.bench_function("matfree DG + mlt_inverse_mass", |b| {
        b.iter(|| {
            fe_space.mlt_stiffness_dg(
                std::hint::black_box(&u_dg),
                std::hint::black_box(&mut v_dg),
                1.0,
            );
            fe_space.mlt_inverse_mass(std::hint::black_box(&mut v_dg), std::hint::black_box(&mass));
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mlt_stiffness,
    bench_mlt_stiffness_dg,
    bench_mlt_stiffness_dg_seq,
    bench_mlt_inverse_mass,
    bench_stiffness_inverse_mass,
);
criterion_main!(benches);
