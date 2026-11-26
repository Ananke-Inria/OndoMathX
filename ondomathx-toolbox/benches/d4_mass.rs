//! D4 benchmark: three ways to finish a leapfrog step after the
//! element-local K u, on sem's big cubes (C = 1 and C = 3) and TetriX's
//! Heart mesh.
//!
//!   (a)  shared-DoF sum, then axpby_diag with the inverse mass per copy
//!   (b)  sum and divide over every DoF (mlt_inverse_mass), then the update
//!   (c)  one pass over glob2dg: sum, divide, write u^{n+1} to every copy
//!
//! Every variant writes u_prev ← 2u − u_prev − dt² M⁻¹(Σ K_loc u) + d·src.
//! Decision D4 (workspace CLAUDE.md, "API convergence"): (a), the fastest
//! or tied in nearly every case; (c) is 1.6–2.7× slower on the large cases.
//!
//! ```text
//! RAYON_NUM_THREADS=10 cargo bench -p ondomathx-toolbox --bench d4_mass
//! ```
use ondomathx_toolbox::field::Constant;
use std::time::Instant;

use ondomathx_mesh::mesh::cube::Cube;
use ondomathx_mesh::mesh::{Mesh, UnstructuredMesh};
use ondomathx_sem::kernels::{AnisotropicAcoustic, IsotropicElastic};
use ondomathx_sem::{Iso3, Numbering, PointKernel, Space};
use ondomathx_tetrix::space::p2_plus::P2PlusSpace;
use ondomathx_toolbox::lal;
use ondomathx_toolbox::numbering::DgMap;
use rayon::prelude::*;

const DT: f64 = 1e-3;
const D: f64 = 2e-6;
const REPS: usize = 10;

#[derive(Clone, Copy)]
struct P(*mut f64);
unsafe impl Send for P {}
unsafe impl Sync for P {}
impl P {
    fn get(self) -> *mut f64 {
        self.0
    }
}

/// Best wall time (ms) of `REPS` runs of `f`.
fn best(mut f: impl FnMut()) -> f64 {
    let mut t = f64::MAX;
    for _ in 0..REPS + 1 {
        let s = Instant::now();
        f();
        t = t.min(s.elapsed().as_secs_f64());
    }
    t * 1e3
}

/// Best wall time (ms) of `REPS` runs of `f(ku, prev)`, the buffers reset
/// from `ku0`, `prev0` before each run (not timed).
fn best_on(ku: &mut [f64], prev: &mut [f64], ku0: &[f64], prev0: &[f64], mut f: impl FnMut(&mut [f64], &mut [f64])) -> f64 {
    let mut t = f64::MAX;
    for _ in 0..REPS + 1 {
        ku.copy_from_slice(ku0);
        prev.copy_from_slice(prev0);
        let s = Instant::now();
        f(ku, prev);
        t = t.min(s.elapsed().as_secs_f64());
    }
    t * 1e3
}

fn update(prev: &mut [f64], u: &[f64], a: &[f64], src: &[f64]) {
    prev.par_iter_mut()
        .zip(u)
        .zip(a)
        .zip(src)
        .for_each(|(((p, &u), &a), &s)| *p = 2.0 * u - *p - DT * DT * a + D * s);
}

/// (c): one pass over the DoFs.
fn fused<const C: usize>(m: &DgMap, mass: &[f64], ku: &[f64], u: &[f64], src: &[f64], prev: &mut [f64]) {
    let nloc = m.num_loc_dofs();
    let at = |k: usize, c: usize| if C == 1 { k } else { (k / nloc) * C * nloc + c * nloc + k % nloc };
    let ptr = P(prev.as_mut_ptr());
    (0..m.num_dofs()).into_par_iter().for_each(|g| {
        let ptr = ptr.get();
        let copies = m.glob2dg(g);
        for c in 0..C {
            let s: f64 = copies.iter().map(|&k| ku[at(k as usize, c)]).sum::<f64>() / mass[g];
            for &k in copies {
                let i = at(k as usize, c);
                // SAFETY: copies of different DoFs are disjoint.
                unsafe { *ptr.add(i) = 2.0 * u[i] - *ptr.add(i) - DT * DT * s + D * src[i] };
            }
        }
    });
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

fn report(name: &str, n_dof: usize, n_dg_values: usize, rows: &[(&str, f64)], k_ms: f64, map_mb: f64) {
    println!("{name}: {n_dof} DoFs, {n_dg_values} DG values, K u {k_ms:.3} ms, DgMap {map_mb:.0} MB (one DG vector {:.0} MB)", n_dg_values as f64 * 8e-6);
    let best = rows.iter().map(|r| r.1).fold(f64::MAX, f64::min);
    for (label, t) in rows {
        println!("  {label:44} {t:8.3} ms   ×{:.2}", t / best);
    }
}

fn sem_case<const C: usize>(name: &str, n_el: usize, kernel: &impl PointKernel<3, C>) {
    let mesh = Cube::new([0.0; 3], 1.0, 1.0, 1.0, n_el, n_el, n_el, None, None);
    let space = Space::<3, Iso3<5>>::new(&mesh, Numbering::Continuous);
    let (numbering, n_num) = mesh.get_tensor_numbering([5, 5, 5]);
    let elements: Vec<usize> = (0..space.num_elements()).map(|e| space.mesh_element(e)).collect();
    let map = DgMap::from_numbering(&numbering, n_num, &elements);
    let n = space.num_dg_dofs() * C;
    let density = Constant(1.0);
    let inv_diag = space.inverse_mass(&density).diagonal().unwrap().to_vec();
    let mut mass = vec![0.0; map.num_dofs()];
    map.collect_dg::<1>(&space.local_mass(&density), &mut mass);

    // Inputs: a continuous u, K_loc u (unassembled), a source, u_prev.
    let mut u: Vec<f64> = (0..n).map(|i| ((i as f64) * 1e-3).sin()).collect();
    space.sum_shared::<C>(&mut u);
    let mut ku0 = vec![0.0; n];
    let k_ms = best(|| space.mlt_dg::<C>(kernel, &u, &mut ku0, 1.0));
    let src: Vec<f64> = (0..n).map(|i| ((i as f64) * 7e-4).cos()).collect();
    let prev0: Vec<f64> = (0..n).map(|i| ((i as f64) * 3e-4).sin()).collect();

    // The same shared sum in sem and in DgMap.
    let (mut s1, mut s2) = (ku0.clone(), ku0.clone());
    space.sum_shared::<C>(&mut s1);
    map.sum_shared::<C>(&mut s2);
    assert_eq!(s1, s2, "{name}: DgMap::sum_shared differs from sem's");

    let (mut ku, mut prev) = (ku0.clone(), prev0.clone());
    let mut results = Vec::new();
    let ta = best_on(&mut ku, &mut prev, &ku0, &prev0, |ku, prev| {
        space.sum_shared::<C>(ku);
        space.axpby_diag::<C>(2.0, &u, -1.0, prev, -DT * DT, &inv_diag, ku, Some((D, &src)));
    });
    results.push(prev.clone());
    let ta2 = best_on(&mut ku, &mut prev, &ku0, &prev0, |ku, prev| {
        map.sum_shared::<C>(ku);
        space.axpby_diag::<C>(2.0, &u, -1.0, prev, -DT * DT, &inv_diag, ku, Some((D, &src)));
    });
    let tb = best_on(&mut ku, &mut prev, &ku0, &prev0, |ku, prev| {
        map.mlt_inverse_mass::<C>(ku, &mass);
        update(prev, &u, ku, &src);
    });
    results.push(prev.clone());
    let tc = best_on(&mut ku, &mut prev, &ku0, &prev0, |_ku, prev| fused::<C>(&map, &mass, &ku0, &u, &src, prev));
    results.push(prev.clone());
    report(name, map.num_dofs() * C, n, &[
        ("(a) sem sum_shared + axpby_diag", ta),
        ("(a) DgMap sum_shared + axpby_diag", ta2),
        ("(b) DgMap mlt_inverse_mass + update", tb),
        ("(c) fused sum/divide/update", tc),
    ], k_ms, map.memory_bytes() as f64 * 1e-6);
    println!("  max relative difference: b vs a {:.1e}, c vs a {:.1e}, c vs b {:.1e}",
        max_rel(&results[1], &results[0]), max_rel(&results[2], &results[0]), max_rel(&results[2], &results[1]));
}

fn tetrix_case() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../ondomathx-mesh/assets/Heart.mesh");
    let mesh = UnstructuredMesh::new_from_mesh(path).expect("Heart.mesh");
    let space = P2PlusSpace::new(&mesh);
    let (n_elem, nloc, n_dof) = (space.num_elements(), space.num_loc_dofs(), space.num_dofs());
    let flat: Vec<usize> = (0..n_elem).flat_map(|e| (0..nloc).map(move |i| (e, i))).map(|(e, i)| space.loc2glob(e, i)).collect();
    let map = DgMap::new(&flat, nloc, n_dof);
    let n = n_elem * nloc;
    let mut mass = vec![0.0; n_dof];
    space.assemble_mass(&Constant(1.0), &mut mass, 1.0);
    let inv_diag: Vec<f64> = (0..n).map(|k| 1.0 / mass[map.dg2glob(k)]).collect();
    let u_cg: Vec<f64> = (0..n_dof).map(|g| ((g as f64) * 1e-3).sin()).collect();
    let mut u = vec![0.0; n];
    space.scatter_cg(&u_cg, &mut u);
    let mut ku0 = vec![0.0; n];
    let k_ms = best(|| space.mlt_stiffness_dg(&Constant(1.0), &u, &mut ku0, 1.0));
    let src: Vec<f64> = (0..n).map(|i| ((i as f64) * 7e-4).cos()).collect();
    let prev0: Vec<f64> = (0..n).map(|i| ((i as f64) * 3e-4).sin()).collect();
    let (mut ku, mut prev) = (ku0.clone(), prev0.clone());
    let mut results = Vec::new();
    let tb0 = best_on(&mut ku, &mut prev, &ku0, &prev0, |ku, prev| {
        space.mlt_inverse_mass(ku, &mass);
        update(prev, &u, ku, &src);
    });
    results.push(prev.clone());
    let tb = best_on(&mut ku, &mut prev, &ku0, &prev0, |ku, prev| {
        map.mlt_inverse_mass::<1>(ku, &mass);
        update(prev, &u, ku, &src);
    });
    results.push(prev.clone());
    let ta = best_on(&mut ku, &mut prev, &ku0, &prev0, |ku, prev| {
        map.sum_shared::<1>(ku);
        lal::axpby_diag::<1>(nloc, 2.0, &u, -1.0, prev, -DT * DT, &inv_diag, ku, Some((D, &src)));
    });
    results.push(prev.clone());
    let tc = best_on(&mut ku, &mut prev, &ku0, &prev0, |_ku, prev| fused::<1>(&map, &mass, &ku0, &u, &src, prev));
    results.push(prev.clone());
    report("TetriX Heart (P2+bubbles, 15 nodes)", n_dof, n, &[
        ("(b) TetriX mlt_inverse_mass + update", tb0),
        ("(b) DgMap mlt_inverse_mass + update", tb),
        ("(a) DgMap sum_shared + axpby_diag", ta),
        ("(c) fused sum/divide/update", tc),
    ], k_ms, map.memory_bytes() as f64 * 1e-6);
    println!("  bit-identical: DgMap (b) vs TetriX (b): {}; max relative difference: a vs b {:.1e}, c vs b {:.1e}",
        results[1] == results[0], max_rel(&results[2], &results[1]), max_rel(&results[3], &results[1]));
}

fn main() {
    println!("threads: {}", rayon::current_num_threads());
    let acoustic = AnisotropicAcoustic { a: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
    sem_case::<1>("sem acoustic cube 48³ Q4", 48, &acoustic);
    let elastic = IsotropicElastic { lambda: Constant(1.0), mu: Constant(1.0) };
    sem_case::<3>("sem elastic cube 32³ Q4", 32, &elastic);
    tetrix_case();
}
