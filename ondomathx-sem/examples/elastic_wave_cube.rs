//! Elastic wave in a cube, excited by a source.
//!
//! ```text
//!   ρ ∂²u/∂t² − div σ(u) = f(x) g(t)   in [0, 1]³,   σ(u)·n = 0 on the boundary,
//!   σ(u) = λ tr(ε(u)) I + 2μ ε(u),   ρ = 1, λ = 2, μ = 1,
//!   f = (bump, 0, 0) at the centre,   g = Ricker wavelet,
//! ```
//! with spectral elements of order 4 on 12 × 12 × 12 hexahedra and the
//! leapfrog scheme in time.
//!
//! ```text
//! cargo run --release --example elastic_wave_cube
//! ```
//! Output in `out/elastic_wave_cube/` (git-ignored): the mesh for Vizir
//! (`.mesh`), and every 0.1 time unit the displacement for ParaView (`.vtk`)
//! and for Vizir (`.sol`).

use std::time::Instant;

use anamesh::mesh::cube::Cube;
use anamesh::mesh::Mesh;
use lobattox::fields::Constant;
use lobattox::kernels::IsotropicElastic;
use lobattox::{stable_dt, Iso3, Numbering, Space, VizirWriter, VtkField, VtkWriter};
use rayon::prelude::*;

/// Ricker wavelet in time (C++ `Ricker(t, t0, s0)`).
fn ricker(t: f64, t0: f64, s0: f64) -> f64 {
    let t = t - t0;
    (4.0 * s0 * s0 * t * t - 2.0 * s0) * (-t * t * s0).exp()
}

/// Smooth bump with compact support, 1 at `center`, 0 beyond `r0`
/// (C++ `CInfty0(x, center, r0, s0)`).
fn bump(x: [f64; 3], center: [f64; 3], r0: f64, s0: f64) -> f64 {
    let r2: f64 = (0..3).map(|d| (x[d] - center[d]).powi(2)).sum();
    if r2 >= r0 * r0 {
        0.0
    } else {
        (s0 * r0 * r0 / (r2 - r0 * r0) + s0).exp()
    }
}

fn main() {
    // 1. The mesh: [0, 1]³ cut into 12 × 12 × 12 hexahedra (Anamesh).
    let mesh = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 12, 12, 12, None, None);

    // 2. The space: Gauss–Lobatto elements of order 4 (5 points per direction),
    //    continuous across the elements. The displacement has 3 components.
    let space = Space::<3, Iso3<5>>::new(&mesh, Numbering::Continuous);
    println!(
        "{} elements, {} nodes, {} degrees of freedom",
        space.n_elem(),
        space.c_n_global(),
        3 * space.c_n_global()
    );

    // 3. The physics: the stiffness ∫ σ(u) : ε(v) and the mass ∫ ρ u·v. The
    //    mass is diagonal (mass lumping): its inverse is a vector.
    let stiffness = IsotropicElastic {
        lambda: Constant(2.0),
        mu: Constant(1.0),
    };
    let density = Constant(1.0);
    let inv_mass = space.c_inverse_mass(&density);

    // 4. The time step: 95 % of the stability limit 2 / √λ_max(M⁻¹K), with
    //    λ_max estimated by power iteration.
    let eig = space.c_max_eigenvalue::<3>(&stiffness, &density, 5000, 1e-3);
    let dt = stable_dt(eig.lambda, 0.95);
    let t_end = 1.0;
    let steps = (t_end / dt).ceil() as usize;
    println!(
        "dt = {dt:.4e} ({} power iterations), {steps} steps",
        eig.iterations
    );

    // 5. The source f(x) g(t), along x. The load vector F = ∫ f·φ_i is computed
    //    once; `source` = M⁻¹F, so that each step adds dt² g(t) M⁻¹F.
    let mut source = space.local_load::<3>(|x| [bump(x, [0.5, 0.5, 0.5], 0.1, 1.0), 0.0, 0.0]);
    space.c_sum_shared::<3>(&mut source); // assemble: sum the copies of shared nodes
    inv_mass.apply::<3>(&mut source);

    // 6. The output: the mesh once, snapshots every 0.1.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("out/elastic_wave_cube");
    std::fs::create_dir_all(&dir).unwrap();
    mesh.write_mesh(dir.join("elastic_wave_cube.mesh").to_str().unwrap())
        .unwrap();
    let vtk = VtkWriter::new(&space);
    let vizir = VizirWriter::new(&space, &mesh);
    let output_every = (0.1 / dt).floor() as usize;

    // 7. The leapfrog scheme, u(0) = u'(0) = 0:
    //    uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ − dt² M⁻¹K uⁿ + dt² g(tⁿ) M⁻¹F.
    //    Vectors hold, element by element, the x, y, z components at the nodes
    //    (a node shared by several elements has one copy per element).
    let n = 3 * space.n_dup();
    let mut u_prev = vec![0.0; n]; // uⁿ⁻¹
    let mut u_now = vec![0.0; n]; // uⁿ
    let mut ku = vec![0.0; n]; // M⁻¹K uⁿ
    let mut n_out = 0;
    let start = Instant::now();
    for step in 0..steps {
        let t = step as f64 * dt;
        space.c_apply::<3>(&stiffness, &u_now, &mut ku); // K uⁿ, assembled
        inv_mass.apply::<3>(&mut ku); // M⁻¹K uⁿ
        let g = ricker(t, 0.2, 500.0);
        // uⁿ⁺¹ is written into u_prev (uⁿ⁻¹ is no longer needed), then swapped.
        u_prev
            .par_iter_mut()
            .zip(&u_now)
            .zip(&ku)
            .zip(&source)
            .for_each(|(((p, &u), &a), &s)| *p = 2.0 * u - *p - dt * dt * a + dt * dt * g * s);
        std::mem::swap(&mut u_prev, &mut u_now);

        if step > 0 && step % output_every == 0 {
            let field = VtkField {
                name: "u",
                components: 3,
                data: &u_now,
            };
            vtk.write(dir.join(format!("elastic_wave_cube.{n_out}.vtk")), &[field])
                .unwrap();
            vizir
                .write(
                    dir.join(format!("elastic_wave_cube.{n_out:05}.sol")),
                    &u_now,
                    3,
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
