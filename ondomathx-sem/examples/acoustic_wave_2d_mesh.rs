//! Acoustic wave in a 2D domain read from a Gmsh file, excited by a source.
//!
//! Port of the C++ example `core/Testing/FESpace/AcousticWave_2D_Mesh.cpp`:
//! ```text
//!   ∂²u/∂t² − div(A ∇u) = f(x) g(t)   in the "potatoe" domain,   A ∇u · n = 0 on the boundary,
//!   A = diag(2, 1),   f = smooth bump at (−1, 1.5),   g = Ricker wavelet,
//! ```
//! with spectral elements of order 4 on the quadrangles of the mesh and the
//! leapfrog scheme in time.
//!
//! ```text
//! cargo run --release --example acoustic_wave_2d_mesh [-- path/to/mesh.msh]
//! ```
//! The default mesh is ondomathx-mesh's asset `Potatoe.msh`. Output in
//! `out/acoustic_wave_2d_mesh/` (git-ignored): the mesh for Vizir (`.mesh`),
//! and every 0.1 time unit a snapshot for ParaView (`.vtk`) and for Vizir
//! (`.sol`).

use std::time::Instant;

use ondomathx_mesh::mesh::{Mesh, UnstructuredMesh};
use ondomathx_sem::fields::Constant;
use ondomathx_sem::kernels::AnisotropicAcoustic;
use ondomathx_sem::{stable_dt, Iso2, Numbering, Space, VizirWriter, VtkField, VtkWriter};

/// Ricker wavelet in time (C++ `Ricker(t, t0, s0)`).
fn ricker(t: f64, t0: f64, s0: f64) -> f64 {
    let t = t - t0;
    (4.0 * s0 * s0 * t * t - 2.0 * s0) * (-t * t * s0).exp()
}

/// Smooth bump with compact support, 1 at `center`, 0 beyond `r0`
/// (C++ `CInfty0(x, center, r0, s0)`).
fn bump(x: [f64; 2], center: [f64; 2], r0: f64, s0: f64) -> f64 {
    let r2 = (x[0] - center[0]).powi(2) + (x[1] - center[1]).powi(2);
    if r2 >= r0 * r0 {
        0.0
    } else {
        (s0 * r0 * r0 / (r2 - r0 * r0) + s0).exp()
    }
}

fn main() {
    // 1. The mesh: read from a Gmsh file (`ondomathx-mesh`). The space uses its
    //    quadrangles; the labelled boundary edges are not needed here.
    let file = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../ondomathx-mesh/assets/Potatoe.msh").to_string()
    });
    let mesh = UnstructuredMesh::new_from_msh(&file).unwrap_or_else(|e| panic!("{file}: {e}"));

    // 2. The space: Gauss–Lobatto elements of order 4 (5 points per direction),
    //    continuous across the elements.
    let space = Space::<2, Iso2<5>>::new(&mesh, Numbering::Continuous);
    println!(
        "{} elements, {} degrees of freedom",
        space.n_elem(),
        space.c_n_global()
    );

    // 3. The physics: the stiffness ∫ (A ∇u)·∇v and the mass ∫ u v (density 1).
    //    The mass is diagonal (mass lumping): its inverse is a vector.
    let stiffness = AnisotropicAcoustic {
        a: [[2.0, 0.0], [0.0, 1.0]],
    };
    let density = Constant(1.0);
    let inv_mass = space.c_inverse_mass(&density);

    // 4. The time step: 95 % of the stability limit 2 / √λ_max(M⁻¹K), with
    //    λ_max estimated by power iteration.
    let eig = space.c_max_eigenvalue::<1>(&stiffness, &density, 5000, 1e-3);
    let dt = stable_dt(eig.lambda, 0.95);
    let t_end = 10.0;
    let steps = (t_end / dt).ceil() as usize;
    println!(
        "dt = {dt:.4e} ({} power iterations), {steps} steps",
        eig.iterations
    );

    // 5. The source f(x) g(t). The load vector F = ∫ f φ_i is computed once;
    //    `source` = M⁻¹F, so that each step adds dt² g(t) M⁻¹F.
    let mut source = space.local_load::<1>(|x| [bump(x, [-1.0, 1.5], 0.2, 4.0)]);
    space.c_sum_shared::<1>(&mut source); // assemble: sum the copies of shared nodes
    inv_mass.apply::<1>(&mut source);

    // 6. The output: the mesh once, snapshots every 0.1.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("out/acoustic_wave_2d_mesh");
    std::fs::create_dir_all(&dir).unwrap();
    mesh.write_mesh(dir.join("acoustic_wave_2d_mesh.mesh").to_str().unwrap())
        .unwrap();
    let vtk = VtkWriter::new(&space);
    let vizir = VizirWriter::new(&space, &mesh);
    let output_every = (0.1 / dt).floor() as usize;

    // 7. The leapfrog scheme, u(0) = u'(0) = 0:
    //    uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹K uⁿ + dt² g(tⁿ) M⁻¹F.
    //    Vectors hold the values at the nodes of every element (a node shared
    //    by several elements has one copy per element).
    let n = space.n_dup();
    let mut u_prev = vec![0.0; n]; // uⁿ⁻¹
    let mut u_now = vec![0.0; n]; // uⁿ
    let mut ku = vec![0.0; n]; // M⁻¹K uⁿ
    // M⁻¹ as one value per node, fused into the update below.
    let inv_diag = inv_mass.diagonal().unwrap();
    let mut n_out = 0;
    let start = Instant::now();
    for step in 0..steps {
        let t = step as f64 * dt;
        space.c_apply::<1>(&stiffness, &u_now, &mut ku); // K uⁿ, assembled
        let g = ricker(t, 0.2, 500.0);
        // uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹K uⁿ + dt² g M⁻¹f in one pass, written into
        // u_prev (uⁿ⁻¹ is no longer needed), then swapped.
        space.axpby_diag::<1>(2.0, &u_now, -1.0, &mut u_prev, -dt * dt, inv_diag, &ku, Some((dt * dt * g, &source)));
        std::mem::swap(&mut u_prev, &mut u_now);

        if step > 0 && step % output_every == 0 {
            let field = VtkField {
                name: "u",
                components: 1,
                data: &u_now,
            };
            vtk.write(
                dir.join(format!("acoustic_wave_2d_mesh.{n_out}.vtk")),
                &[field],
            )
            .unwrap();
            vizir
                .write(
                    dir.join(format!("acoustic_wave_2d_mesh.{n_out:05}.sol")),
                    &u_now,
                    1,
                )
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
