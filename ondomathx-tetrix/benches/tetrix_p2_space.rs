use criterion::{criterion_group, criterion_main, Criterion};
use rouage::mesh::UnstructuredMesh;
use tetrix::tetrix_p2_space::TetriXP2Space;

fn setup() -> (UnstructuredMesh,) {
    let mesh = UnstructuredMesh::new_from_mesh("../Rouage/assets/Heart.mesh")
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

criterion_group!(
    benches,
    bench_mlt_stiffness,
    bench_mlt_stiffness_dg,
    bench_mlt_stiffness_dg_seq,
    bench_mlt_inverse_mass
);
criterion_main!(benches);
