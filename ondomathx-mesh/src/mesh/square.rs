pub use crate::mesh::*;

pub use crate::deformation_field::*;
pub use unstructured_mesh::*;

pub struct Square {
    pub origin: [f64; 2],
    pub lx: f64,
    pub ly: f64,
    pub nx: usize,
    pub ny: usize,
    pub deformation: Option<DeformationField>,
}

impl Square {
    pub fn new(
        origin: [f64; 2],
        lx: f64,
        ly: f64,
        nx: usize,
        ny: usize,
        deformation: Option<DeformationField>,
    ) -> Square {
        assert!(
            nx > 0,
            "Number of element in the x-direction should be greater than 0"
        );
        assert!(
            ny > 0,
            "Number of element in the y-direction should be greater than 0"
        );
        assert!(
            lx > 0.0,
            "The length of the square should be greater than 0"
        );
        assert!(
            ly > 0.0,
            "The length of the square should be greater than 0"
        );

        Square {
            origin: origin,
            lx: lx,
            ly: ly,
            nx: nx,
            ny: ny,
            deformation,
        }
    }

    fn get_numbering(&self, nlocdofx: usize, nlocdofy: usize) -> (Vec<Vec<usize>>, usize) {
        let mut numbering: Vec<Vec<usize>> = vec![vec![]; self.get_nelements()];

        let px = nlocdofx - 1;
        let py = nlocdofy - 1;
        let n_dof_x = self.nx * px + 1;
        let n_dof_y = self.ny * py + 1;
        let n_dof = n_dof_x * n_dof_y;

        for ix in 0..self.nx {
            for iy in 0..self.ny {
                for idofy in 0..nlocdofy {
                    for idofx in 0..nlocdofx {
                        numbering[iy * self.nx + ix]
                            .push((iy * py + idofy) * n_dof_x + (ix * px + idofx));
                    }
                }
            }
        }

        (numbering, n_dof)
    }
}

impl Mesh for Square {
    fn get_deformation(&self) -> Option<&DeformationField> {
        self.deformation.as_ref()
    }
    fn get_nnodes(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }
    fn get_node(&self, index: usize) -> [f64; 3] {
        let i = index % (self.nx + 1);
        let j = index / (self.nx + 1);

        let xyz = [
            self.origin[0] + (self.lx / self.nx as f64) * i as f64,
            self.origin[1] + (self.ly / self.ny as f64) * j as f64,
            0.0,
        ];

        xyz
    }
    fn get_nelements(&self) -> usize {
        self.nx * self.ny + 2 * self.nx + 2 * self.ny
    }
    fn get_element(&self, element_index: usize) -> MeshElement {
        debug_assert!(
            element_index < self.get_nelements(),
            "Element index out of bounds"
        );

        if element_index < self.nx * self.ny {
            let i = element_index % self.nx;
            let j = element_index / self.nx;

            MeshElement {
                geometric_element: &QUADRANGLE4,
                label: 0,
                indexes: vec![
                    (j + 0) * (self.nx + 1) + (i + 0),
                    (j + 0) * (self.nx + 1) + (i + 1),
                    (j + 1) * (self.nx + 1) + (i + 1),
                    (j + 1) * (self.nx + 1) + (i + 0),
                ],
            }
        } else if element_index < self.nx * self.ny + self.nx {
            let i = element_index - self.nx * self.ny;

            MeshElement {
                geometric_element: &EDGE2,
                label: 0,
                indexes: vec![(i + 0), (i + 1)],
            }
        } else if element_index < self.nx * self.ny + self.nx + self.ny {
            let j = element_index - (self.nx * self.ny + self.nx);

            MeshElement {
                geometric_element: &EDGE2,
                label: 1,
                indexes: vec![
                    (j + 0) * (self.nx + 1) + self.nx,
                    (j + 1) * (self.nx + 1) + self.nx,
                ],
            }
        } else if element_index < self.nx * self.ny + 2 * self.nx + self.ny {
            let i = (self.nx - 1) - (element_index - (self.nx * self.ny + self.nx + self.ny));

            MeshElement {
                geometric_element: &EDGE2,
                label: 2,
                indexes: vec![
                    self.ny * (self.nx + 1) + (i + 1),
                    self.ny * (self.nx + 1) + (i + 0),
                ],
            }
        } else {
            let j = (self.ny - 1) - (element_index - (self.nx * self.ny + 2 * self.nx + self.ny));

            MeshElement {
                geometric_element: &EDGE2,
                label: 3,
                indexes: vec![(j + 1) * (self.nx + 1), (j + 0) * (self.nx + 1)],
            }
        }
    }
    fn get_geometric_dimension(&self) -> usize {
        2
    }
    fn get_topologic_dimension(&self) -> usize {
        2
    }
    fn get_elements_by_dimension(&self, _dim: usize) -> Vec<usize> {
        assert!(false);
        vec![0]
    }
    fn get_elements_by_dimension_and_label(&self, _label: usize, _dim: usize) -> Vec<usize> {
        assert!(false);
        vec![0]
    }
    fn get_ncolors(&self) -> usize {
        4
    }
    fn get_nelements_by_color(&self, color_index: usize) -> usize {
        match color_index {
            0 => (self.nx / 2 + self.nx % 2) * (self.ny / 2 + self.ny % 2),
            1 => (self.nx / 2) * (self.ny / 2 + self.ny % 2),
            2 => (self.nx / 2 + self.nx % 2) * (self.ny / 2),
            3 => (self.nx / 2) * (self.ny / 2),
            _ => 0,
        }
    }
    fn get_element_color(&self, element_index: usize) -> usize {
        (element_index / self.nx) % 2 + 2 * ((element_index % self.nx) % 2)
    }
    fn get_element_index_by_color_to_element_index(
        &self,
        color_index: usize,
        element_index_by_color: usize,
    ) -> usize {
        let gex;
        let gey;

        debug_assert!(
            element_index_by_color < self.get_nelements_by_color(color_index),
            "Element index is out of bound"
        );

        match color_index {
            0 => {
                gex = 2 * element_index_by_color % (self.nx / 2 + self.nx % 2);
                gey = 2 * element_index_by_color % (self.nx / 2 + self.nx % 2);
            }
            1 => {
                gex = 2 * element_index_by_color % (self.nx / 2);
                gey = 2 * element_index_by_color % (self.nx / 2);
            }
            2 => {
                gex = 2 * element_index_by_color % (self.nx / 2 + self.nx % 2);
                gey = 2 * element_index_by_color % (self.nx / 2 + self.nx % 2);
            }
            3 => {
                gex = 2 * element_index_by_color % (self.nx / 2);
                gey = 2 * element_index_by_color % (self.nx / 2);
            }
            _ => {
                panic!("color index out of bounds")
            }
        }

        gey * self.nx + gex
    }
    fn get_element_grad_def(&self, _element_index: usize, _uvw: &[f64], grad_def: &mut [f64]) {
        let hx = self.lx / self.nx as f64;
        let hy = self.ly / self.ny as f64;

        grad_def[0] *= hx;
        grad_def[1] *= hy;
        grad_def[2] *= hx;
        grad_def[3] *= hy;
    }
}
