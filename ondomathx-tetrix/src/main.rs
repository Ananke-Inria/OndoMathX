use ondomathx_tetrix::io_vizir;
use ondomathx_tetrix::io_vtk;
use ondomathx_tetrix::tetrix_p2_space::TetriXP2Space;

use ondomathx_mesh::mesh::{Mesh, UnstructuredMesh};
use ondomathx_toolbox::lal;
use ondomathx_toolbox::signals::{gaussian, ricker, ricker_delay};

use rayon::prelude::*;

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
    let fe_space = TetriXP2Space::new(&mesh);
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
    fe_space.assemble_mass(&mut mass, 1.0);

    // -----------------------------------------------------------------------
    // Source: Gaussian centred at (20, 70, 70) in CG, then scattered to DG
    // -----------------------------------------------------------------------
    let mut source_cg: Vec<f64> = vec![0.0_f64; n_dof];
    for i in 0..n_dof {
        let xyz = fe_space.dof_coordinate(i);
        let r =
            ((xyz[0] - 20.0).powi(2) + (xyz[1] - 70.0).powi(2) + (xyz[2] - 70.0).powi(2)).sqrt();
        source_cg[i] = gaussian(r, 100.0);
    }
    let mut source_dg: Vec<f64> = vec![0.0_f64; n_dg];
    fe_space.scatter_cg(&source_cg, &mut source_dg);

    let source_norm = lal::mass_norm(&source_cg, &mass);

    fs::create_dir_all("output").unwrap();
    io_vtk::write_vtk("output/source.vtk", &fe_space, &source_cg);
    io_vizir::write_solution("output/source", &fe_space, &mesh, &source_cg, 0);
    println!(
        "  Source written to output/source.vtk  ||source||_M = {:.6e}",
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
        |u, v| fe_space.mlt_stiffness_dg(u, v, 1.0),
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
    let mut ku_dg = vec![0.0_f64; n_dg]; // M⁻¹ K u_n  (DG)

    let mut n_out = 0_usize;
    let tloop = Instant::now();

    for i_step in 0..n_steps {
        let t = i_step as f64 * dt;
        let dt2 = dt * dt;

        // 1. K u_curr  (element-local, parallel, result in DG)
        fe_space.mlt_stiffness_dg(&u_curr, &mut ku_dg, 1.0);

        // 2. M⁻¹: collect DG→CG, divide by diagonal mass, scatter CG→DG
        fe_space.mlt_inverse_mass(&mut ku_dg, &mass);

        // 3. u_next = 2 u_n − u_{n-1} − dt² M⁻¹K u_n + dt² R(t^n) src
        //    Written into u_prev (no longer needed as u_{n-1}).
        let rk = ricker(t, 0.20, ricker_delay(0.20));
        let dt2_rk = dt2 * rk;

        u_prev.par_iter_mut().enumerate().for_each(|(k, prev)| {
            *prev = 2.0 * u_curr[k] - *prev - dt2 * ku_dg[k] + dt2_rk * source_dg[k];
        });

        // 4. Swap: u_curr = u_{n+1},  u_prev = u_n
        std::mem::swap(&mut u_prev, &mut u_curr);

        // 5. Output u_curr = u_{n+1}: average DG contributions per global DoF.
        if i_step % output_freq == 0 && i_step > 0 {
            let mut u_cg = vec![0.0_f64; n_dof];
            fe_space.collect_avg_dg(&u_curr, &mut u_cg);
            let u_norm = lal::mass_norm(&u_cg, &mass);
            let path = format!("output/wave.{:04}.vtk", n_out);
            io_vtk::write_vtk(&path, &fe_space, &u_cg);
            io_vizir::write_solution("output/wave", &fe_space, &mesh, &u_cg, n_out);
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
}
