//! Acoustic wave in a cube, excited by a source.
//!
//! Port of the C++ example `core/Testing/FESpace/AcousticWave_Cube.cpp`:
//! ```text
//!   ∂²u/∂t² − div(A ∇u) = f(x) g(t)   in [0, 1]³,   A ∇u · n = 0 on the boundary,
//!   A = diag(4, 2, 0.5),   f = smooth bump at the centre,   g = Ricker wavelet,
//! ```
//! with spectral elements of order 4 on 12 × 12 × 12 hexahedra and the
//! leapfrog scheme in time.
//!
//! The C++ example uses 3 × 3 × 3 hexahedra, far too coarse for this source
//! (a bump of radius 0.05, a Ricker wavelet around 14 Hz, wave speed 0.71
//! along z): the L² norm of u(0.4) is 5 times too large there. With 12 × 12 × 12
//! it is within 1 % of the converged value.
//!
//! ```text
//! cargo run --release --example acoustic_wave_cube
//! ```
//! Output in `out/acoustic_wave_cube/` (git-ignored): the mesh for Vizir
//! (`.mesh`), and every 0.1 time unit a snapshot for ParaView (`.vtk`) and for
//! Vizir (`.sol`).

use std::time::Instant;

use ondomathx_mesh::mesh::cube::Cube;
use ondomathx_mesh::mesh::Mesh;
use ondomathx_toolbox::field::Constant;
use ondomathx_sem::kernels::AnisotropicAcoustic;
use ondomathx_sem::{Iso3, Numbering, Space, VizirWriter, VtkField, VtkWriter};

use ondomathx_toolbox::io_vizir::write_sols;
use ondomathx_toolbox::signals::{bump, ricker};

fn main() {
    // 1. The mesh: [0, 1]³ cut into 12 × 12 × 12 hexahedra (`ondomathx-mesh`).
    let mesh = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 12, 12, 12, None, None);

    // 2. The space: Gauss–Lobatto elements of order 4 (5 points per direction),
    //    continuous across the elements.
    let space = Space::<3, Iso3<5>>::new(&mesh, Numbering::Continuous);
    println!(
        "{} elements, {} degrees of freedom",
        space.num_elements(),
        space.num_dofs()
    );

    // 3. The physics: the stiffness ∫ (A ∇u)·∇v and the mass ∫ u v (density 1).
    //    The mass is diagonal (mass lumping): its inverse is a vector.
    let stiffness = AnisotropicAcoustic {
        a: [[4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]],
    };
    let density = Constant(1.0);
    let inv_mass = space.inverse_mass(&density);

    // 4. The time step: 95 % of the stability limit 2 / √λ_max(M⁻¹K), with
    //    λ_max estimated by power iteration.
    let eig = space.max_eigenvalue::<1>(&stiffness, &density, 5000, 1e-3);
    let dt = 0.95 * 2.0 / eig.lambda.sqrt(); // 95 % of the leapfrog limit 2 / √λ_max
    let t_end = 1.0;
    let steps = (t_end / dt).ceil() as usize;
    println!(
        "dt = {dt:.4e} ({} power iterations), {steps} steps",
        eig.n_iter
    );

    // 5. The source f(x) g(t). The load vector F = ∫ f φ_i is computed once;
    //    `source` = M⁻¹F, so that each step adds dt² g(t) M⁻¹F.
    let mut source = space.local_load::<1>(|x| [bump(x, [0.5, 0.5, 0.5], 0.05, 1.0)]);
    space.sum_shared::<1>(&mut source); // assemble: sum the copies of shared nodes
    inv_mass.apply::<1>(&mut source);

    // 6. The output: the mesh once, snapshots every 0.1.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("out/acoustic_wave_cube");
    std::fs::create_dir_all(&dir).unwrap();
    mesh.write_mesh(dir.join("acoustic_wave_cube.mesh").to_str().unwrap())
        .unwrap();
    let vtk = VtkWriter::new(&space);
    let vizir = VizirWriter::new(&space, &mesh);
    let output_every = (0.1 / dt).floor() as usize;

    // 7. The leapfrog scheme, u(0) = u'(0) = 0:
    //    uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹K uⁿ + dt² g(tⁿ) M⁻¹F.
    //    Vectors hold the values at the nodes of every element (a node shared
    //    by several elements has one copy per element).
    let n = space.num_dg_dofs();
    let mut u_prev = vec![0.0; n]; // uⁿ⁻¹
    let mut u_now = vec![0.0; n]; // uⁿ
    let mut ku = vec![0.0; n]; // M⁻¹K uⁿ
    // M⁻¹ as one value per node, fused into the update below.
    let inv_diag = inv_mass.diagonal().unwrap();
    let mut n_out = 0;
    let start = Instant::now();
    for step in 0..steps {
        let t = step as f64 * dt;
        space.mlt_assembled::<1>(&stiffness, &u_now, &mut ku); // K uⁿ, assembled
        let g = ricker(t, 14.0, 0.2);
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
                dir.join(format!("acoustic_wave_cube.{n_out}.vtk")),
                &[field],
            )
            .unwrap();
            vizir
                .write(
                    dir.join(format!("acoustic_wave_cube.{n_out:05}.sol")),
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

    // 9. Vizir: the time series (`.sols`) and the command that opens it.
    let sols: Vec<String> = (0..n_out).map(|k| format!("acoustic_wave_cube.{k:05}.sol")).collect();
    write_sols(dir.join("acoustic_wave_cube.sols"), &sols).unwrap();
    // Vizir adds the solution on the boundary faces (the .sol files hold the
    // volume elements only; `Checksurf`, user guide 7.10).
    std::fs::write(dir.join("default.vizir"), "Checksurf\n1\n").unwrap();
    println!("cd {} && vizir4 -in acoustic_wave_cube.mesh -sol acoustic_wave_cube.00000.sol -sols acoustic_wave_cube.sols", dir.display());
}
