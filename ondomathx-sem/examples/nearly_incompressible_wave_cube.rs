//! Nearly incompressible elastic wave in a cube, excited by a source.
//!
//! ```text
//!   ρ ∂²u/∂t² − div(2μ ε(u)) − ∇p = f(x) g(t),   div u = p / λ   in [0, 1]³,
//!   free boundary,   ρ = 1, μ = 1, λ = 1000 (Poisson ratio ≈ 0.4995),
//!   f = (bump, 0, 0) at the centre,   g = Ricker wavelet,
//! ```
//! discretized as in Ramiche & Imperiale (inf-sup article; see also Merlini
//! et al.): displacement in continuous Q4 (Gauss–Lobatto, mass lumping),
//! pressure in discontinuous Q2, eliminated element by element (12 × 12 × 12
//! hexahedra here):
//! ```text
//!   M d²U/dt² + R U + λ K^P U = F,   K^P = Bᵀ (M^P)⁻¹ B,
//! ```
//! `R` the shear stiffness `∫ 2μ ε(u) : ε(v)`, `B` the divergence
//! `b(u, q) = ∫ q div u`, `M^P` the pressure mass with 3 Gauss–Legendre points
//! per direction. Leapfrog in time with `dt` from the spectral radius of
//! `M⁻¹(R + λ K^P)` (it shrinks like `1/√λ`).
//!
//! ```text
//! cargo run --release --example nearly_incompressible_wave_cube
//! ```
//! Output in `out/nearly_incompressible_wave_cube/` (git-ignored): the mesh
//! for Vizir (`.mesh`), and every 0.1 time unit the displacement and the
//! pressure `p = λ (M^P)⁻¹ B u` for ParaView (`.vtk`) and for Vizir (`.sol`).

use std::time::Instant;

use ondomathx_mesh::mesh::cube::Cube;
use ondomathx_mesh::mesh::Mesh;
use ondomathx_sem::fields::Constant;
use ondomathx_sem::kernels::{Divergence, IsotropicElastic};
use ondomathx_sem::operators::{FusedKpQ4Q2, PressureMass3};
use ondomathx_sem::{
    GeoStorage, Gll, Iso3, Numbering, Quad3, Space, VizirWriter,
    VtkField, VtkWriter,
};
use rayon::prelude::*;

use ondomathx_toolbox::lal::power_iteration;
use ondomathx_toolbox::signals::{bump, ricker};

/// The Gauss–Lobatto rule at the displacement nodes, used for `b(u, q)` when
/// the pressure is written.
type DisplacementNodes = Quad3<Gll<5>, Gll<5>, Gll<5>>;

fn main() {
    let lambda = 1000.0;

    // 1. The mesh: [0, 1]³ cut into 12 × 12 × 12 hexahedra (`ondomathx-mesh`).
    let mesh = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 12, 12, 12, None, None);

    // 2. The spaces, on the same mesh:
    //    - displacement: continuous Q4, with the two quadratures it is used with
    //      (the pressure mass rule, needed by K^P; its own nodes, for b(u, q));
    //    - pressure: discontinuous Q2 (no numbering), with the pressure mass rule.
    let disp = Space::<3, Iso3<5>>::builder(&mesh)
        .numbering(Numbering::Continuous)
        .quadrature::<PressureMass3>(GeoStorage::Store)
        .quadrature::<DisplacementNodes>(GeoStorage::Store)
        .build();
    let pres = Space::<3, Iso3<3>>::builder(&mesh)
        .quadrature::<PressureMass3>(GeoStorage::Store)
        .build();
    println!(
        "{} elements, {} displacement nodes ({} dofs), {} pressure dofs",
        disp.num_elements(),
        disp.num_dofs(),
        3 * disp.num_dofs(),
        pres.num_dg_dofs()
    );

    // 3. The physics:
    //    - R: the shear stiffness ∫ 2μ ε(u) : ε(v) (isotropic kernel with λ = 0);
    //    - K^P = Bᵀ (M^P)⁻¹ B, applied element by element in one pass (the
    //      pressure is discontinuous, so no global pressure vector is needed);
    //    - the lumped mass ∫ ρ u·v.
    let shear = IsotropicElastic {
        lambda: Constant(0.0),
        mu: Constant(1.0),
    };
    let kp = FusedKpQ4Q2::<3>::new(&disp, 1.0);
    let density = Constant(1.0);
    let inv_mass = disp.inverse_mass(&density);
    // y = (R + λ K^P) u, assembled.
    let apply = |u: &[f64], y: &mut [f64]| {
        disp.mlt_dg::<3>(&shear, u, y, 1.0);
        disp.mlt_add_element_dg(&disp, &kp, u, y, lambda);
        disp.sum_shared::<3>(y);
    };

    // 4. The time step: 95 % of 2 / √ρ(M⁻¹(R + λ K^P)), the spectral radius by
    //    power iteration on that composite operator.
    let m = disp.local_mass(&density);
    let eig = power_iteration(
        |y| inv_mass.apply::<3>(y),
        |x, y| apply(x, y),
        |a, b| disp.mass_inner::<3>(&m, a, b),
        disp.start_vector::<3>(),
        1e-3,
        5000,
    );
    let dt = 0.95 * 2.0 / eig.lambda.sqrt(); // 95 % of the leapfrog limit 2 / √λ_max
    let t_end = 1.0;
    let steps = (t_end / dt).ceil() as usize;
    println!(
        "dt = {dt:.4e} ({} power iterations), {steps} steps",
        eig.n_iter
    );

    // 5. The source f(x) g(t), along x: `source` = M⁻¹F, F = ∫ f·φ_i.
    let mut source = disp.local_load::<3>(|x| [bump(x, [0.5, 0.5, 0.5], 0.1, 1.0), 0.0, 0.0]);
    disp.sum_shared::<3>(&mut source);
    inv_mass.apply::<3>(&mut source);

    // 6. The output: the mesh once; every 0.1, the displacement and the
    //    pressure p = λ (M^P)⁻¹ B u (B: the divergence kernel between the two
    //    spaces, (M^P)⁻¹: the inverse of the Gauss–Legendre pressure mass).
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("out/nearly_incompressible_wave_cube");
    std::fs::create_dir_all(&dir).unwrap();
    let name = "nearly_incompressible_wave_cube";
    mesh.write_mesh(dir.join(format!("{name}.mesh")).to_str().unwrap())
        .unwrap();
    let (vtk_u, vizir_u) = (VtkWriter::new(&disp), VizirWriter::new(&disp, &mesh));
    let (vtk_p, vizir_p) = (VtkWriter::new(&pres), VizirWriter::new(&pres, &mesh));
    let inv_mass_p = pres.inverse_mass_with_dg::<PressureMass3>(&Constant(1.0));
    let divergence = Divergence {
        coef: Constant(1.0),
    };
    let mut p = vec![0.0; pres.num_dg_dofs()];
    let output_every = (0.1 / dt).floor() as usize;

    // 7. The leapfrog scheme, u(0) = u'(0) = 0:
    //    uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹(R + λ K^P) uⁿ + dt² g(tⁿ) M⁻¹F.
    let n = 3 * disp.num_dg_dofs();
    let mut u_prev = vec![0.0; n]; // uⁿ⁻¹
    let mut u_now = vec![0.0; n]; // uⁿ
    let mut au = vec![0.0; n]; // M⁻¹(R + λ K^P) uⁿ
    let mut n_out = 0;
    let start = Instant::now();
    for step in 0..steps {
        let t = step as f64 * dt;
        apply(&u_now, &mut au);
        inv_mass.apply::<3>(&mut au);
        let g = ricker(t, 7.0, 0.2);
        u_prev
            .par_iter_mut()
            .zip(&u_now)
            .zip(&au)
            .zip(&source)
            .for_each(|(((up, &u), &a), &s)| *up = 2.0 * u - *up - dt * dt * a + dt * dt * g * s);
        std::mem::swap(&mut u_prev, &mut u_now);

        if step > 0 && step % output_every == 0 {
            disp.mlt_mixed_dg::<DisplacementNodes, 3, 1, _>(
                &pres,
                &divergence,
                &u_now,
                &mut p,
                lambda,
            );
            inv_mass_p.apply::<1>(&mut p); // p = λ (M^P)⁻¹ B u
            let u_field = VtkField {
                name: "u",
                components: 3,
                data: &u_now,
            };
            let p_field = VtkField {
                name: "p",
                components: 1,
                data: &p,
            };
            vtk_u
                .write(dir.join(format!("{name}.{n_out}.vtk")), &[u_field])
                .unwrap();
            vtk_p
                .write(dir.join(format!("{name}_pressure.{n_out}.vtk")), &[p_field])
                .unwrap();
            vizir_u
                .write(dir.join(format!("{name}.{n_out:05}.sol")), &u_now, 3)
                .unwrap();
            vizir_p
                .write(dir.join(format!("{name}_pressure.{n_out:05}.sol")), &p, 1)
                .unwrap();
            n_out += 1;
        }
    }

    // 8. Timing.
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{steps} steps in {elapsed:.3} s ({:.3} ms/step), {n_out} snapshots in {}",
        1e3 * elapsed / steps as f64,
        dir.display()
    );
}
