use anamesh::mesh::{DoFSet, Mesh, UnstructuredMesh, TETRAHEDRON};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use sprs::{CsMat, TriMat};

use crate::finite_element::{tet_14, tet_15, tet_15_14};

pub const DIM: usize = 3;
pub const ORDER: usize = 2;
const N_LOC_DOF: usize = 15;
const P2_N_LOC_DOF: usize = 10;

pub struct TetriXP2Space<'a> {
    mesh: &'a UnstructuredMesh,
    elements: Vec<usize>,
    n_color: usize,
    elem_by_color: Vec<Vec<usize>>,
    n_elem: usize,
    n_dof: usize,
    barycenter: Vec<[f64; 3]>,
    dof_coordinates: Vec<[f64; 3]>,
    grad_def_origin: Vec<Matrix3<f64>>,
    sym_grad_def_origin: Vec<Matrix3<f64>>,
    jacobian_origin: Vec<f64>,
    affine: Vec<bool>,
    loc2glob: Vec<Vec<usize>>,
    glob2dg: Vec<Vec<usize>>,
    dg2glob: Vec<usize>,
    p2_n_dof: usize,
    p2_loc2glob: Vec<Vec<usize>>,
    p2_glob2glob: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Isoparametric mapping: reference coords uvw → physical xyz, using the
/// P1 shape functions of the element.
fn uvw2xyz(mesh: &UnstructuredMesh, elem: usize, uvw: &[f64; 3]) -> [f64; 3] {
    let element = mesh.get_element(elem);
    let geo = element.geometric_element;
    let mut xyz = [0.0f64; 3];
    for i in 0..geo.nshape_functions {
        let p = (geo.shape_functions[i])(uvw);
        let node = mesh.get_node(element.indexes[i]);
        xyz[0] += p * node[0];
        xyz[1] += p * node[1];
        xyz[2] += p * node[2];
    }
    xyz
}

/// Barycenter of an element (average of its vertices).
fn barycenter(mesh: &UnstructuredMesh, elem: usize) -> [f64; 3] {
    let element = mesh.get_element(elem);
    let n = element.geometric_element.nshape_functions;
    let mut b = [0.0f64; 3];
    for i in 0..n {
        let node = mesh.get_node(element.indexes[i]);
        b[0] += node[0];
        b[1] += node[1];
        b[2] += node[2];
    }
    let nf = n as f64;
    [b[0] / nf, b[1] / nf, b[2] / nf]
}

/// Jacobian matrix of the element mapping at uvw (returned as nalgebra Matrix3).
/// The matrix entry (i,j) = d(xyz_i)/d(uvw_j).
fn jacobian_matrix(mesh: &UnstructuredMesh, elem: usize, uvw: &[f64; 3]) -> Matrix3<f64> {
    let mut flat = [0.0f64; 9];
    mesh.get_element_grad_def(elem, uvw, &mut flat);
    Matrix3::from_row_slice(&flat)
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<'a> TetriXP2Space<'a> {
    pub fn new(mesh: &'a UnstructuredMesh) -> Self {
        assert_eq!(
            mesh.get_topologic_dimension(),
            3,
            "TetriXP2Space requires a 3-D mesh"
        );

        // 3-D element list (local index el → global element index elem)
        let elements = mesh.get_elements_by_dimension(3);
        let n_elem = elements.len();

        // Build DoFSet from the 15 Tet_15 reference points
        let pos_list: Vec<[f64; 3]> = tet_15::POINTS.to_vec();
        let dof_set = DoFSet::new(pos_list, &TETRAHEDRON);
        let (all_loc2global, n_dof) = mesh.get_numbering(&dof_set);

        // Per-element geometric data
        let mut barycenter_arr = vec![[0.0f64; 3]; n_elem];
        let mut loc2glob = vec![vec![]; n_elem];
        let mut grad_def_origin = Vec::with_capacity(n_elem);
        let mut sym_grad_def_origin = Vec::with_capacity(n_elem);
        let mut jacobian_origin = Vec::with_capacity(n_elem);
        let affine = vec![true; n_elem];
        let mut dof_coordinates = vec![[0.0f64; 3]; n_dof];

        let mut glob2glob_p2 = vec![0usize; n_dof];
        let mut p2_n_dof: usize = 0;

        for (el, &elem) in elements.iter().enumerate() {
            loc2glob[el] = all_loc2global[elem].clone();

            barycenter_arr[el] = barycenter(mesh, elem);

            let g = jacobian_matrix(mesh, elem, &[0.0, 0.0, 0.0]);
            let j = g.determinant();
            let g_inv = g.try_inverse().expect("singular element Jacobian");

            grad_def_origin.push(g);
            jacobian_origin.push(j);
            // sym_grad = J * G^{-T} * G^{-1}
            sym_grad_def_origin.push(j * g_inv.transpose() * g_inv);

            // P2 sub-numbering (first P2_N_LOC_DOF local DoFs)
            for i in 0..P2_N_LOC_DOF {
                let global_dof = loc2glob[el][i];
                if glob2glob_p2[global_dof] == 0 {
                    p2_n_dof += 1;
                    glob2glob_p2[global_dof] = p2_n_dof;
                }
            }

            // Physical coordinates of all 15 local DoFs
            for i in 0..N_LOC_DOF {
                let global_dof = loc2glob[el][i];
                let uvw = tet_15::POINTS[i];
                dof_coordinates[global_dof] = uvw2xyz(mesh, elem, &uvw);
            }
        }

        // Build p2_glob2glob (P2-local → global)
        let mut p2_glob2glob = vec![0usize; p2_n_dof];
        for i in 0..n_dof {
            if glob2glob_p2[i] != 0 {
                p2_glob2glob[glob2glob_p2[i] - 1] = i;
            }
        }

        // Build p2_loc2glob
        let mut p2_loc2glob = vec![vec![0usize; P2_N_LOC_DOF]; n_elem];
        for el in 0..n_elem {
            for i in 0..P2_N_LOC_DOF {
                p2_loc2glob[el][i] = glob2glob_p2[loc2glob[el][i]] - 1;
            }
        }

        // Build glob2dg and dg2glob: forward and inverse DG ↔ CG index maps
        let mut glob2dg = vec![vec![]; n_dof];
        let mut dg2glob = vec![0usize; n_elem * N_LOC_DOF];
        for el in 0..n_elem {
            for i_loc in 0..N_LOC_DOF {
                let g = loc2glob[el][i_loc];
                let dg = el * N_LOC_DOF + i_loc;
                glob2dg[g].push(dg);
                dg2glob[dg] = g;
            }
        }

        // Element coloring (mesh.coloring is indexed by local 3-D element index)
        let n_color = mesh.get_ncolors();
        let mut elem_by_color: Vec<Vec<usize>> = vec![vec![]; n_color];
        for el in 0..n_elem {
            let color = mesh.get_element_color(el);
            elem_by_color[color].push(el);
        }

        TetriXP2Space {
            mesh,
            elements,
            n_color,
            elem_by_color,
            n_elem,
            n_dof,
            barycenter: barycenter_arr,
            dof_coordinates,
            grad_def_origin,
            sym_grad_def_origin,
            jacobian_origin,
            affine,
            loc2glob,
            glob2dg,
            dg2glob,
            p2_n_dof,
            p2_loc2glob,
            p2_glob2glob,
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    pub fn num_colors(&self) -> usize {
        self.n_color
    }
    pub fn num_elements_by_color(&self, color: usize) -> usize {
        self.elem_by_color[color].len()
    }
    pub fn num_elements(&self) -> usize {
        self.n_elem
    }
    pub fn num_dofs(&self) -> usize {
        self.n_dof
    }
    pub fn p2_num_dofs(&self) -> usize {
        self.p2_n_dof
    }
    pub fn num_loc_dofs(&self) -> usize {
        N_LOC_DOF
    }
    pub fn num_loc_p2_dofs(&self) -> usize {
        P2_N_LOC_DOF
    }
    pub fn is_affine(&self, i_elt_glob: usize) -> bool {
        self.affine[i_elt_glob]
    }

    pub fn elt_color_to_global(&self, color: usize, e: usize) -> usize {
        self.elem_by_color[color][e]
    }

    pub fn loc2glob(&self, i_elt_glob: usize, i_loc: usize) -> usize {
        self.loc2glob[i_elt_glob][i_loc]
    }

    pub fn p2_loc2glob(&self, i_elt_glob: usize, i_loc: usize) -> usize {
        self.p2_loc2glob[i_elt_glob][i_loc]
    }

    pub fn p2_glob2glob(&self, i: usize) -> usize {
        self.p2_glob2glob[i]
    }

    pub fn loc_dof_coordinates(&self, i_loc: usize) -> &[f64; 3] {
        &tet_15::POINTS[i_loc]
    }

    pub fn dof_coordinate(&self, i_glob: usize) -> &[f64; 3] {
        &self.dof_coordinates[i_glob]
    }

    pub fn p2_dof_coordinate(&self, i_glob_p2: usize) -> &[f64; 3] {
        &self.dof_coordinates[self.p2_glob2glob[i_glob_p2]]
    }

    pub fn barycenter(&self, i_elt_glob: usize) -> &[f64; 3] {
        &self.barycenter[i_elt_glob]
    }

    pub fn get_barycentric_coordinates(&self, uvw: &[f64; 3]) -> [f64; 4] {
        let (u, v, w) = (uvw[0], uvw[1], uvw[2]);
        [1.0 - u - v - w, u, v, w]
    }

    pub fn grad_def(&self, i_elt_glob: usize) -> &Matrix3<f64> {
        &self.grad_def_origin[i_elt_glob]
    }

    pub fn get_label(&self, i_elt_glob: usize) -> usize {
        self.mesh.get_element(self.elements[i_elt_glob]).label
    }

    // -----------------------------------------------------------------------
    // Assembly
    // -----------------------------------------------------------------------

    /// Assembles the global stiffness matrix as a sparse CSR matrix.
    ///
    /// K[I, J] = Σ_{el} Σ_{i,j} α · |J_el| · k_loc[i,j]
    ///
    /// where the sum over (i,j) runs over local DoF pairs with loc2glob(el,i)=I
    /// and loc2glob(el,j)=J, and k_loc is the local stiffness matrix on the
    /// reference element scaled by `sym_grad_def_origin` (which already absorbs
    /// the Jacobian factor).
    pub fn assemble_stiffness(&self, alpha: f64) -> CsMat<f64> {
        let mut triplets: TriMat<f64> = TriMat::new((self.n_dof, self.n_dof));

        for el in 0..self.n_elem {
            let k_loc = tet_15::build_local_stiffness(&self.sym_grad_def_origin[el]);
            let loc2glob = &self.loc2glob[el];
            for i in 0..N_LOC_DOF {
                for j in 0..N_LOC_DOF {
                    triplets.add_triplet(loc2glob[i], loc2glob[j], alpha * k_loc[(i, j)]);
                }
            }
        }

        triplets.to_csr()
    }

    pub fn assemble_mass(&self, mass: &mut Vec<f64>, alpha: f64) {
        for i_elt in 0..self.n_elem {
            let j = self.jacobian_origin[i_elt];
            for i_loc in 0..N_LOC_DOF {
                let i_glob = self.loc2glob[i_elt][i_loc];
                let weight = tet_15::WEIGHTS[i_loc];
                mass[i_glob] += alpha * j * weight;
            }
        }
    }

    pub fn mlt_stiffness(&self, u: &Vec<f64>, v: &mut Vec<f64>, alpha: f64) {
        for x in v.iter_mut() {
            *x = 0.0;
        }

        const N_QUAD: usize = 14;

        for i_color in 0..self.n_color {
            for &i_elt_glob in &self.elem_by_color[i_color] {
                let loc2glob = &self.loc2glob[i_elt_glob];

                let mut buf_u = [0.0f64; N_LOC_DOF];
                let mut buf_v = [0.0f64; N_LOC_DOF];
                let mut buf_grad_u = [0.0f64; N_QUAD * 3];
                let mut buf_grad_v = [0.0f64; N_QUAD * 3];

                for i_loc in 0..N_LOC_DOF {
                    buf_u[i_loc] = u[loc2glob[i_loc]];
                }

                tet_15_14::gradient_interpolation(&buf_u, &mut buf_grad_u);

                let sym_grad = &self.sym_grad_def_origin[i_elt_glob];

                for i_quad in 0..N_QUAD {
                    let col = Vector3::new(
                        buf_grad_u[i_quad * 3],
                        buf_grad_u[i_quad * 3 + 1],
                        buf_grad_u[i_quad * 3 + 2],
                    );
                    let result = sym_grad * col;
                    let weight = tet_14::WEIGHTS[i_quad];
                    buf_grad_v[i_quad * 3] = weight * result[0];
                    buf_grad_v[i_quad * 3 + 1] = weight * result[1];
                    buf_grad_v[i_quad * 3 + 2] = weight * result[2];
                }

                tet_15_14::transpose_gradient_interpolation(&buf_grad_v, &mut buf_v);

                for i_loc in 0..N_LOC_DOF {
                    v[loc2glob[i_loc]] += alpha * buf_v[i_loc];
                }
            }
        }
    }

    /// Sequential version of `mlt_stiffness_dg` (no Rayon), useful for
    /// profiling or single-threaded benchmarking.
    pub fn mlt_stiffness_dg_seq(&self, u: &Vec<f64>, v: &mut Vec<f64>, alpha: f64) {
        for el in 0..self.n_elem {
            let offset = el * N_LOC_DOF;
            let buf_u: &[f64; N_LOC_DOF] = u[offset..offset + N_LOC_DOF].try_into().unwrap();
            let mut buf_v = [0.0f64; N_LOC_DOF];

            tet_15::mlt_stiffness_opt(buf_u, &self.sym_grad_def_origin[el], &mut buf_v);

            for i_loc in 0..N_LOC_DOF {
                v[offset + i_loc] = alpha * buf_v[i_loc];
            }
        }
    }

    /// Like `mlt_stiffness`, but for a DG layout where `u` and `v` are stored
    /// element-by-element: DoF `i_loc` of element `el` lives at index
    /// `el * N_LOC_DOF + i_loc`.  Because every element owns its own slice of
    /// DoFs there is no global assembly and no coloring is required.
    pub fn mlt_stiffness_dg(&self, u: &Vec<f64>, v: &mut Vec<f64>, alpha: f64) {
        v.par_chunks_mut(N_LOC_DOF)
            .zip(u.par_chunks(N_LOC_DOF))
            .zip(self.sym_grad_def_origin.par_iter())
            .for_each(|((v_chunk, u_chunk), sym_grad)| {
                let buf_u: &[f64; N_LOC_DOF] = u_chunk.try_into().unwrap();
                let mut buf_v = [0.0f64; N_LOC_DOF];

                tet_15::mlt_stiffness_opt(buf_u, sym_grad, &mut buf_v);

                for i_loc in 0..N_LOC_DOF {
                    v_chunk[i_loc] = alpha * buf_v[i_loc];
                }
            });
    }

    /// Collects a DG vector (`n_elem * N_LOC_DOF`) into a CG vector (`n_dof`)
    /// by averaging all DG contributions that share each global DoF.
    /// Equivalent to `collect_dg` divided by the element-sharing count.
    pub fn collect_avg_dg(&self, dg: &Vec<f64>, cg: &mut Vec<f64>) {
        cg.par_iter_mut()
            .zip(self.glob2dg.par_iter())
            .for_each(|(cg_val, dg_indices)| {
                *cg_val = dg_indices.iter().map(|&k| dg[k]).sum::<f64>() / dg_indices.len() as f64;
            });
    }

    /// Collects a DG vector (`n_elem * N_LOC_DOF`) into a CG vector (`n_dof`)
    /// by summing all DG contributions that share each global DoF.
    /// Parallel over global DoFs: each task writes to one exclusive `cg[g]`
    /// and reads from its disjoint `glob2dg[g]` slice of `dg`.
    pub fn collect_dg(&self, dg: &Vec<f64>, cg: &mut Vec<f64>) {
        cg.par_iter_mut()
            .zip(self.glob2dg.par_iter())
            .for_each(|(cg_val, dg_indices)| {
                *cg_val = dg_indices.iter().map(|&k| dg[k]).sum();
            });
    }

    /// Applies the inverse lumped mass matrix in-place on a DG vector.
    pub fn mlt_inverse_mass(&self, dg: &mut Vec<f64>, mass: &Vec<f64>) {
        // let mut tmp_cg = vec![0.0_f64; mass.len()];
        // self.collect_dg(dg, &mut tmp_cg);
        // for i in 0..self.n_dof {
        //     tmp_cg[i] /= mass[i];
        // }
        // self.scatter_cg(&tmp_cg, dg);

        // SAFETY: glob2dg partitions the DG indices (each DG slot belongs to
        // exactly one CG dof), so parallel tasks access disjoint regions of `dg`.
        // The pointer ptr is cast to usize so the closure captures a Sync + Send type.
        let ptr = dg.as_mut_ptr() as usize;
        self.glob2dg
            .par_iter()
            .zip(mass.par_iter())
            .for_each(|(dg_indices, m)| {
                let cg_val: f64 = dg_indices.iter().map(|&k| dg[k]).sum::<f64>() / m;
                let ptr = ptr as *mut f64;
                for &k in dg_indices {
                    unsafe { *ptr.add(k) = cg_val };
                }
            });
    }

    /// Scatters a CG vector (`n_dof`) into a DG vector (`n_elem * N_LOC_DOF`)
    /// by copying each global DoF value to all its DG slots.
    /// Parallel over DG indices via `dg2glob`: each task writes to one exclusive slot.
    pub fn scatter_cg(&self, cg: &Vec<f64>, dg: &mut Vec<f64>) {
        dg.par_iter_mut()
            .zip(self.dg2glob.par_iter())
            .for_each(|(dg_val, &g)| {
                *dg_val = cg[g];
            });
    }
}
