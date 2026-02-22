use nalgebra::{Matrix3, Vector3};
use rouage::mesh::{DoFSet, Mesh, UnstructuredMesh, TETRAHEDRON};

use crate::lal::{DiagonalMatrix, Vector};
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

        // Element coloring (mesh.coloring is indexed by local 3-D element index)
        let n_color = mesh.n_colors;
        let mut elem_by_color: Vec<Vec<usize>> = vec![vec![]; n_color];
        for el in 0..n_elem {
            let color = mesh.coloring[el];
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

    pub fn assemble_mass(&self, mass: &mut DiagonalMatrix, alpha: f64) {
        for i_elt in 0..self.n_elem {
            let j = self.jacobian_origin[i_elt];
            for i_loc in 0..N_LOC_DOF {
                let i_glob = self.loc2glob[i_elt][i_loc];
                let weight = tet_15::WEIGHTS[i_loc];
                mass[i_glob] += alpha * j * weight;
            }
        }
    }

    pub fn mlt_stiffness(&self, u: &Vector, v: &mut Vector, alpha: f64) {
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
}
