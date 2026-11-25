//! Acoustic wave in a square, started by an initial velocity.
//!
//! Port of the C++ example `core/Testing/FESpace/AcousticWave_Square.cpp`:
//! ```text
//!   ∂²u/∂t² − Δu = 0   in [0, 1]²,   ∇u · n = 0 on the boundary,
//!   u(0) = 0,   ∂u/∂t(0) = v0(x, y) = G(x) G(y) cos(2π·20 (x + y)),
//! ```
//! (`G` Gaussians centred at x = 0.9 and y = 0.5), with spectral elements of
//! order 4 on 100 × 100 quadrangles and the leapfrog scheme in time.
//!
//! ```text
//! cargo run --release --example acoustic_wave_square
//! ```
//! Output in `out/acoustic_wave_square/` (git-ignored): the mesh for Vizir
//! (`.mesh`), and every 0.1 time unit a snapshot for ParaView (`.vtk`) and
//! for Vizir (`.sol`): 11 snapshots, about 0.3 GB (ASCII files). The C++
//! example writes one every 0.01 (101 snapshots).

use std::f64::consts::PI;
use std::time::Instant;

use ondomathx_mesh::mesh::square::Square;
use ondomathx_mesh::mesh::Mesh;
use ondomathx_sem::fields::Constant;
use ondomathx_sem::kernels::AnisotropicAcoustic;
use ondomathx_sem::{stable_dt, Iso2, Numbering, Space, VizirWriter, VtkField, VtkWriter};

/// Gaussian (C++ `Gaussian(t, t0, s0)`).
fn gaussian(t: f64, t0: f64, s0: f64) -> f64 {
    (-(t - t0) * (t - t0) * s0).exp()
}

fn main() {
    // 1. The mesh: [0, 1]² cut into 100 × 100 quadrangles (`ondomathx-mesh`).
    let mesh = Square::new([0.0, 0.0], 1.0, 1.0, 100, 100, None, None);

    // 2. The space: Gauss–Lobatto elements of order 4 (5 points per direction),
    //    continuous across the elements.
    let space = Space::<2, Iso2<5>>::new(&mesh, Numbering::Continuous);
    println!(
        "{} elements, {} degrees of freedom",
        space.n_elem(),
        space.c_n_global()
    );

    // 3. The physics: the stiffness ∫ ∇u·∇v (A = identity) and the mass ∫ u v
    //    (density 1). The mass is diagonal (mass lumping): its inverse is a vector.
    let stiffness = AnisotropicAcoustic {
        a: [[1.0, 0.0], [0.0, 1.0]],
    };
    let density = Constant(1.0);
    let inv_mass = space.c_inverse_mass(&density);

    // 4. The time step: 95 % of the stability limit 2 / √λ_max(M⁻¹K), with
    //    λ_max estimated by power iteration.
    let eig = space.c_max_eigenvalue::<1>(&stiffness, &density, 5000, 1e-3);
    let dt = stable_dt(eig.lambda, 0.95);
    let t_end = 1.0;
    let steps = (t_end / dt).ceil() as usize;
    println!(
        "dt = {dt:.4e} ({} power iterations), {steps} steps",
        eig.iterations
    );

    // 5. The initial state: u⁰ = 0 and, by a Taylor expansion,
    //    u¹ = dt v0 − dt³/6 M⁻¹K v0 (as the C++).
    let n = space.n_dup();
    let v0 = space.sample::<1>(|x| {
        [gaussian(x[0], 0.9, 500.0)
            * gaussian(x[1], 0.5, 500.0)
            * (2.0 * PI * 20.0 * (x[0] + x[1])).cos()]
    });
    let mut ku = vec![0.0; n]; // M⁻¹K u
    space.c_apply::<1>(&stiffness, &v0, &mut ku);
    inv_mass.apply::<1>(&mut ku);
    let mut u_prev = vec![0.0; n]; // uⁿ⁻¹
    let mut u_now: Vec<f64> = (0..n)
        .map(|i| dt * v0[i] - dt * dt * dt / 6.0 * ku[i])
        .collect(); // uⁿ

    // 6. The output: the mesh once, snapshots every 0.1.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("out/acoustic_wave_square");
    std::fs::create_dir_all(&dir).unwrap();
    mesh.write_mesh(dir.join("acoustic_wave_square.mesh").to_str().unwrap())
        .unwrap();
    let vtk = VtkWriter::new(&space);
    let vizir = VizirWriter::new(&space, &mesh);
    let output_every = (0.1 / dt).floor() as usize;

    // 7. The leapfrog scheme: uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹K uⁿ.
    //    Vectors hold the values at the nodes of every element (a node shared
    //    by several elements has one copy per element).
    // M⁻¹ as one value per node, fused into the update below.
    let inv_diag = inv_mass.diagonal().unwrap();
    let mut n_out = 0;
    let start = Instant::now();
    for step in 0..steps {
        space.c_apply::<1>(&stiffness, &u_now, &mut ku); // K uⁿ, assembled

        // uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹K uⁿ in one pass, written into u_prev (uⁿ⁻¹
        // is no longer needed), then swapped.
        space.axpby_diag::<1>(2.0, &u_now, -1.0, &mut u_prev, -dt * dt, inv_diag, &ku, None);
        std::mem::swap(&mut u_prev, &mut u_now);

        if step % output_every == 0 {
            let field = VtkField {
                name: "u",
                components: 1,
                data: &u_now,
            };
            vtk.write(
                dir.join(format!("acoustic_wave_square.{n_out}.vtk")),
                &[field],
            )
            .unwrap();
            vizir
                .write(
                    dir.join(format!("acoustic_wave_square.{n_out:05}.sol")),
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
