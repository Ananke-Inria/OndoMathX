//! Scalar acoustic wave on the heart mesh (`ondomathx-mesh/assets/Heart.mesh`),
//! P2 + bubble elements with DG storage, leapfrog in time, a Gaussian source
//! with a Ricker wavelet in time.
//!
//! ```text
//! cargo run --release --example acoustic_wave_heart
//! ```
//! Output in `out/acoustic_wave_heart/` (git-ignored): the mesh, the source
//! and every time unit a snapshot for ParaView (`.vtk`) and for Vizir
//! (`.sol`, listed in `wave.sols`); the run ends with the Vizir command.

use ondomathx_toolbox::field::Constant;
use ondomathx_tetrix::io::io_vizir;
use ondomathx_tetrix::io::io_vtk;
use ondomathx_tetrix::space::p2_plus::P2PlusSpace;

use ondomathx_mesh::mesh::{Mesh, UnstructuredMesh};
use ondomathx_toolbox::io_vizir::write_sols;
use ondomathx_toolbox::lal;
use ondomathx_toolbox::signals::{gaussian, ricker, ricker_delay};


use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("TetriX – scalar acoustic wave, DG storage, leap-frog scheme");

    // -----------------------------------------------------------------------
    // Mesh
    // -----------------------------------------------------------------------
    let mesh = UnstructuredMesh::new_from_mesh(concat!(env!("CARGO_MANIFEST_DIR"), "/../ondomathx-mesh/assets/Heart.mesh"))
        .expect("Failed to load Heart.mesh");
    println!(
        "  nodes={}, elements={}",
        mesh.get_nnodes(),
        mesh.get_nelements()
    );

    // -----------------------------------------------------------------------
    // FE space
    // -----------------------------------------------------------------------
    let fe_space = P2PlusSpace::new(&mesh);
    let n_dof = fe_space.num_dofs();
    let n_elem = fe_space.num_elements();
    let n_loc = fe_space.num_loc_dofs(); // 15
    let n_dg = n_elem * n_loc;
    println!(
        "  CG DoFs={}, DG size={}, colors={}",
        n_dof,
        n_dg,
        fe_space.num_colors()
    );

    // -----------------------------------------------------------------------
    // Diagonal mass matrix (assembled in CG, size n_dof)
    // -----------------------------------------------------------------------
    let mut mass = vec![0.0_f64; n_dof];
    fe_space.assemble_mass(&Constant(1.0), &mut mass, 1.0);

    // -----------------------------------------------------------------------
    // Source: Gaussian centred at (20, 70, 70) in CG, then scattered to DG
    // -----------------------------------------------------------------------
    let mut source_cg: Vec<f64> = vec![0.0_f64; n_dof];
    for i in 0..n_dof {
        let xyz = fe_space.dof_coordinate(i);
        let r =
            ((xyz[0] - 20.0).powi(2) + (xyz[1] - 70.0).powi(2) + (xyz[2] - 70.0).powi(2)).sqrt();
        source_cg[i] = gaussian(r, 400.0);
    }
    let mut source_dg: Vec<f64> = vec![0.0_f64; n_dg];
    fe_space.scatter_cg(&source_cg, &mut source_dg);

    let source_norm = lal::mass_norm(&source_cg, &mass);

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("out/acoustic_wave_heart");
    fs::create_dir_all(&dir).unwrap();
    let out = |name: &str| dir.join(name).to_str().unwrap().to_string();
    // The mesh for Vizir (the .sol files refer to its tetrahedra).
    mesh.write_mesh(&out("heart.mesh")).unwrap();
    io_vtk::write_vtk(&out("source.vtk"), &fe_space, &source_cg);
    io_vizir::write_solution(&out("source"), &fe_space, &mesh, &source_cg, 0);
    println!(
        "  Source written to {}  ||source||_M = {:.6e}",
        out("source.vtk"),
        source_norm
    );

    // -----------------------------------------------------------------------
    // Power iteration: spectral radius ρ of M⁻¹K  (eigenvectors in DG format)
    //
    // Operator in DG:
    //   MltAdd : u_dg  →  K u_dg            (mlt_stiffness_dg)
    //   Solve  : v_dg  →  M⁻¹ v_dg          (collect → /mass → scatter)
    // -----------------------------------------------------------------------
    println!("Power iteration…");
    let t0 = Instant::now();

    // M inner product of two DG vectors: the copies of each global DoF
    // averaged (exact for continuous vectors), weighted by the global mass.
    let inner = |a: &[f64], b: &[f64]| {
        let (mut a_cg, mut b_cg) = (vec![0.0_f64; n_dof], vec![0.0_f64; n_dof]);
        fe_space.collect_avg_dg(a, &mut a_cg);
        fe_space.collect_avg_dg(b, &mut b_cg);
        a_cg.iter().zip(&b_cg).zip(&mass).map(|((x, y), m)| m * x * y).sum::<f64>()
    };
    let eig = lal::power_iteration(
        // Solve: M⁻¹ in-place on a DG vector
        |v| fe_space.mlt_inverse_mass(v, &mass),
        // MltAdd: K on a DG vector
        |u, v| fe_space.mlt_stiffness_dg(&Constant(1.0), u, v, 1.0),
        inner,
        lal::start_vector(n_dg),
        1e-7,
        200,
    );
    let (rho, pi_iters, pi_res) = (eig.lambda, eig.n_iter, eig.residue);

    println!(
        "  ρ = {:.6e},  iters = {},  residue = {:.2e}  ({:.2?})",
        rho,
        pi_iters,
        pi_res,
        t0.elapsed()
    );

    let dt = 0.95 * 2.0 / rho.sqrt();
    println!("  dt = {:.6e}", dt);

    // -----------------------------------------------------------------------
    // Leap-frog time loop — all solution vectors live in DG format
    //
    //   u_{n+1} = 2 u_n − u_{n-1} − dt² M⁻¹K u_n  +  dt² R(t_n) · src
    //
    // Storage trick: we write u_{n+1} directly into u_prev (which is no
    // longer needed after the update), then swap u_prev ↔ u_curr.
    // After the swap: u_curr = u_{n+1}, u_prev = u_n.
    // -----------------------------------------------------------------------
    let t_end = 30.0_f64;
    let output_delta = 1.0_f64;
    let output_freq = (output_delta / dt).ceil() as usize;
    let n_steps = (t_end / dt).ceil() as usize;
    println!(
        "Leap-frog: {} steps, output every {} steps",
        n_steps, output_freq
    );

    let mut u_prev = vec![0.0_f64; n_dg]; // u_{n-1}, initialised to zero
    let mut u_curr = vec![0.0_f64; n_dg]; // u_n,     initialised to zero
    let mut ku_dg = vec![0.0_f64; n_dg]; // K u_n, assembled (DG)

    // M⁻¹ as one value per DG copy (the inverse mass of its global DoF),
    // fused into the update below.
    let inv_mass_cg: Vec<f64> = mass.iter().map(|m| 1.0 / m).collect();
    let mut inv_mass_dg = vec![0.0_f64; n_dg];
    fe_space.scatter_cg(&inv_mass_cg, &mut inv_mass_dg);

    let mut n_out = 0_usize;
    let tloop = Instant::now();

    for i_step in 0..n_steps {
        let t = i_step as f64 * dt;
        let dt2 = dt * dt;

        // 1. K u_curr  (element-local, parallel, result in DG)
        fe_space.mlt_stiffness_dg(&Constant(1.0), &u_curr, &mut ku_dg, 1.0);

        // 2. Assembly: every copy receives the sum of its DoF's copies
        fe_space.sum_shared(&mut ku_dg);

        // 3. u_next = 2 u_n − u_{n-1} − dt² M⁻¹K u_n + dt² R(t^n) src in one
        //    pass (M⁻¹ fused), written into u_prev (no longer needed as u_{n-1}).
        let rk = ricker(t, 0.20, ricker_delay(0.20));
        let dt2_rk = dt2 * rk;
        lal::axpby_diag::<1>(
            n_loc,
            2.0,
            &u_curr,
            -1.0,
            &mut u_prev,
            -dt2,
            &inv_mass_dg,
            &ku_dg,
            Some((dt2_rk, &source_dg)),
        );

        // 4. Swap: u_curr = u_{n+1},  u_prev = u_n
        std::mem::swap(&mut u_prev, &mut u_curr);

        // 5. Output u_curr = u_{n+1}: average DG contributions per global DoF.
        if i_step % output_freq == 0 && i_step > 0 {
            let mut u_cg = vec![0.0_f64; n_dof];
            fe_space.collect_avg_dg(&u_curr, &mut u_cg);
            let u_norm = lal::mass_norm(&u_cg, &mass);
            let path = out(&format!("wave.{:04}.vtk", n_out));
            io_vtk::write_vtk(&path, &fe_space, &u_cg);
            io_vizir::write_solution(&out("wave"), &fe_space, &mesh, &u_cg, n_out);
            println!(
                "  step {:6}  t = {:8.3}  ||u||_M = {:.6e}  ||src||_M = {:.6e}  → {}",
                i_step,
                t,
                u_norm,
                rk.abs() * source_norm,
                path
            );
            n_out += 1;
        }
    }

    println!(
        "Done. {} VTK files written in {:.2?}",
        n_out,
        tloop.elapsed()
    );

    // Vizir: the time series (`.sols`) and the command that opens it.
    let sols: Vec<String> = (0..n_out).map(|k| format!("wave.{k:05}.sol")).collect();
    write_sols(dir.join("wave.sols"), &sols).unwrap();
    // Vizir adds the solution on the boundary faces (the .sol files hold the
    // volume elements only; `Checksurf`, user guide 7.10).
    fs::write(dir.join("default.vizir"), "Checksurf\n1\n").unwrap();
    println!("cd {} && vizir4 -in heart.mesh -sol wave.00000.sol -sols wave.sols", dir.display());
}
