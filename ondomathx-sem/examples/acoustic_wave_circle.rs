//! Acoustic wave in a disc, excited by a source.
//!
//! Port of the C++ example `core/Testing/FESpace/AcousticWave_Circle.cpp`:
//! ```text
//!   ∂²u/∂t² − div(A ∇u) = f(x) g(t)   in the unit disc,   A ∇u · n = 0 on the circle,
//!   A = diag(2, 1),   f = smooth bump at the centre,   g = Ricker wavelet,
//! ```
//! with spectral elements of order 6 and the leapfrog scheme in time.
//!
//! The disc is `ondomathx-mesh`'s `Circle` (a square core and four curved arms). Its
//! arms follow the exact circle map, which the space stores at every node
//! ("mapped" elements), so the circle is exact. `ondomathx-mesh` splits the disc in
//! 1620 elements here, the C++ `Circle(6, 6)` in 1728.
//!
//! ```text
//! cargo run --release --example acoustic_wave_circle
//! ```
//! Output in `out/acoustic_wave_circle/` (git-ignored): the mesh for Vizir
//! (`.mesh`, curved Q2 elements), and every 0.1 time unit a snapshot for
//! ParaView (`.vtk`) and for Vizir (`.sol`).

use std::time::Instant;

use ondomathx_mesh::mesh::circle::Circle;
use ondomathx_mesh::mesh::Mesh;
use ondomathx_sem::fields::Constant;
use ondomathx_sem::kernels::AnisotropicAcoustic;
use ondomathx_sem::{Iso2, Numbering, Space, VizirWriter, VtkField, VtkWriter};

use ondomathx_toolbox::signals::{bump, ricker};

fn main() {
    // 1. The mesh: the unit disc (`ondomathx-mesh`), Q2 elements for the Vizir output.
    let mesh = Circle::new([0.0, 0.0], 1.0, 54, None, Some(true));

    // 2. The space: Gauss–Lobatto elements of order 6 (7 points per direction),
    //    continuous across the elements.
    let space = Space::<2, Iso2<7>>::new(&mesh, Numbering::Continuous);
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
    let dt = 0.95 * 2.0 / eig.lambda.sqrt(); // 95 % of the leapfrog limit 2 / √λ_max
    let t_end = 2.0;
    let steps = (t_end / dt).ceil() as usize;
    println!(
        "dt = {dt:.4e} ({} power iterations), {steps} steps",
        eig.n_iter
    );

    // 5. The source f(x) g(t). The load vector F = ∫ f φ_i is computed once;
    //    `source` = M⁻¹F, so that each step adds dt² g(t) M⁻¹F.
    let mut source = space.local_load::<1>(|x| [bump(x, [0.0, 0.0], 0.2, 4.0)]);
    space.c_sum_shared::<1>(&mut source); // assemble: sum the copies of shared nodes
    inv_mass.apply::<1>(&mut source);

    // 6. The output: the mesh once, snapshots every 0.1.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("out/acoustic_wave_circle");
    std::fs::create_dir_all(&dir).unwrap();
    mesh.write_mesh(dir.join("acoustic_wave_circle.mesh").to_str().unwrap())
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
        let g = ricker(t, 7.0, 0.2);
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
                dir.join(format!("acoustic_wave_circle.{n_out}.vtk")),
                &[field],
            )
            .unwrap();
            vizir
                .write(
                    dir.join(format!("acoustic_wave_circle.{n_out:05}.sol")),
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
