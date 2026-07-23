pub use crate::mesh::*;

pub use crate::diffeomorphism::*;
pub use unstructured_mesh::*;

pub struct Square {
    pub origin: [f64; 2],
    pub lx: f64,
    pub ly: f64,
    pub nx: usize,
    pub ny: usize,
    pub deformation: Option<Diffeomorphism>,
    //When true the mesh carries Q2 geometric nodes: the Q1 vertices keep
    //their numbering and the edge midpoints / cell centers are appended
    //after them; get_element then returns QUADRANGLE9 / EDGE3 elements.
    pub q2: bool,
}

impl Square {
    pub fn new(
        origin: [f64; 2],
        lx: f64,
        ly: f64,
        nx: usize,
        ny: usize,
        deformation: Option<Diffeomorphism>,
        q2: Option<bool>,
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
            origin,
            lx,
            ly,
            nx,
            ny,
            deformation,
            q2: q2.unwrap_or(false),
        }
    }

    //Global indexes of the Q2 geometric nodes, appended after the Q1
    //vertices: first the horizontal-edge midpoints (nx x (ny+1), row by
    //row), then the vertical-edge midpoints ((nx+1) x ny), then the cell
    //centers (nx x ny).
    fn hmid(&self, ix: usize, iy: usize) -> usize {
        (self.nx + 1) * (self.ny + 1) + iy * self.nx + ix
    }
    fn vmid(&self, ix: usize, iy: usize) -> usize {
        (self.nx + 1) * (self.ny + 1) + self.nx * (self.ny + 1) + iy * (self.nx + 1) + ix
    }
    fn center(&self, ix: usize, iy: usize) -> usize {
        (self.nx + 1) * (self.ny + 1)
            + self.nx * (self.ny + 1)
            + (self.nx + 1) * self.ny
            + iy * self.nx
            + ix
    }

    pub fn get_numbering(&self, nlocdofx: usize, nlocdofy: usize) -> (Vec<Vec<usize>>, usize) {
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
    fn get_deformation(&self) -> Option<&Diffeomorphism> {
        self.deformation.as_ref()
    }
    fn get_nnodes(&self) -> usize {
        if self.q2 {
            //vertices + horizontal-edge midpoints + vertical-edge midpoints
            //+ cell centers = (2*nx + 1)*(2*ny + 1)
            (2 * self.nx + 1) * (2 * self.ny + 1)
        } else {
            (self.nx + 1) * (self.ny + 1)
        }
    }
    fn get_node(&self, index: usize) -> [f64; 3] {
        debug_assert!(index < self.get_nnodes(), "Node index out of bounds");

        let hx = self.lx / self.nx as f64;
        let hy = self.ly / self.ny as f64;

        if self.q2 {
            //Q1 vertices first, then the Q2 geometric nodes: horizontal-edge
            //midpoints, vertical-edge midpoints, cell centers.
            let n_vert = (self.nx + 1) * (self.ny + 1);
            let n_hmid = self.nx * (self.ny + 1);
            let n_vmid = (self.nx + 1) * self.ny;

            if index < n_vert {
                let i = index % (self.nx + 1);
                let j = index / (self.nx + 1);
                [
                    self.origin[0] + hx * i as f64,
                    self.origin[1] + hy * j as f64,
                    0.0,
                ]
            } else if index < n_vert + n_hmid {
                let rel = index - n_vert;
                let i = rel % self.nx;
                let j = rel / self.nx;
                [
                    self.origin[0] + hx * (i as f64 + 0.5),
                    self.origin[1] + hy * j as f64,
                    0.0,
                ]
            } else if index < n_vert + n_hmid + n_vmid {
                let rel = index - n_vert - n_hmid;
                let i = rel % (self.nx + 1);
                let j = rel / (self.nx + 1);
                [
                    self.origin[0] + hx * i as f64,
                    self.origin[1] + hy * (j as f64 + 0.5),
                    0.0,
                ]
            } else {
                let rel = index - n_vert - n_hmid - n_vmid;
                let i = rel % self.nx;
                let j = rel / self.nx;
                [
                    self.origin[0] + hx * (i as f64 + 0.5),
                    self.origin[1] + hy * (j as f64 + 0.5),
                    0.0,
                ]
            }
        } else {
            let i = index % (self.nx + 1);
            let j = index / (self.nx + 1);

            [
                self.origin[0] + hx * i as f64,
                self.origin[1] + hy * j as f64,
                0.0,
            ]
        }
    }
    fn get_nelements(&self) -> usize {
        self.nx * self.ny + 2 * self.nx + 2 * self.ny
    }
    fn get_element(&self, element_index: usize) -> MeshElement {
        debug_assert!(
            element_index < self.get_nelements(),
            "Element index out of bounds"
        );

        if self.q2 {
            //Same elements and element numbering as the Q1 mesh, with the Q2
            //geometric nodes added: QUADRANGLE9 for the cells (midpoints
            //bottom, right, top, left, then center) and EDGE3 for the
            //boundary edges (midpoint last).
            return if element_index < self.nx * self.ny {
                let i = element_index % self.nx;
                let j = element_index / self.nx;

                MeshElement {
                    geometric_element: &QUADRANGLE9,
                    label: 0,
                    indexes: vec![
                        j * (self.nx + 1) + i,
                        j * (self.nx + 1) + (i + 1),
                        (j + 1) * (self.nx + 1) + (i + 1),
                        (j + 1) * (self.nx + 1) + i,
                        self.hmid(i, j),
                        self.vmid(i + 1, j),
                        self.hmid(i, j + 1),
                        self.vmid(i, j),
                        self.center(i, j),
                    ],
                }
            } else if element_index < self.nx * self.ny + self.nx {
                let i = element_index - self.nx * self.ny;

                MeshElement {
                    geometric_element: &EDGE3,
                    label: 0,
                    indexes: vec![i, (i + 1), self.hmid(i, 0)],
                }
            } else if element_index < self.nx * self.ny + self.nx + self.ny {
                let j = element_index - (self.nx * self.ny + self.nx);

                MeshElement {
                    geometric_element: &EDGE3,
                    label: 1,
                    indexes: vec![
                        j * (self.nx + 1) + self.nx,
                        (j + 1) * (self.nx + 1) + self.nx,
                        self.vmid(self.nx, j),
                    ],
                }
            } else if element_index < self.nx * self.ny + 2 * self.nx + self.ny {
                let i = (self.nx - 1) - (element_index - (self.nx * self.ny + self.nx + self.ny));

                MeshElement {
                    geometric_element: &EDGE3,
                    label: 2,
                    indexes: vec![
                        self.ny * (self.nx + 1) + (i + 1),
                        self.ny * (self.nx + 1) + i,
                        self.hmid(i, self.ny),
                    ],
                }
            } else {
                let j =
                    (self.ny - 1) - (element_index - (self.nx * self.ny + 2 * self.nx + self.ny));

                MeshElement {
                    geometric_element: &EDGE3,
                    label: 3,
                    indexes: vec![(j + 1) * (self.nx + 1), j * (self.nx + 1), self.vmid(0, j)],
                }
            };
        }

        if element_index < self.nx * self.ny {
            let i = element_index % self.nx;
            let j = element_index / self.nx;

            MeshElement {
                geometric_element: &QUADRANGLE4,
                label: 0,
                indexes: vec![
                    j * (self.nx + 1) + i,
                    j * (self.nx + 1) + (i + 1),
                    (j + 1) * (self.nx + 1) + (i + 1),
                    (j + 1) * (self.nx + 1) + i,
                ],
            }
        } else if element_index < self.nx * self.ny + self.nx {
            let i = element_index - self.nx * self.ny;

            MeshElement {
                geometric_element: &EDGE2,
                label: 0,
                indexes: vec![i, (i + 1)],
            }
        } else if element_index < self.nx * self.ny + self.nx + self.ny {
            let j = element_index - (self.nx * self.ny + self.nx);

            MeshElement {
                geometric_element: &EDGE2,
                label: 1,
                indexes: vec![
                    j * (self.nx + 1) + self.nx,
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
                    self.ny * (self.nx + 1) + i,
                ],
            }
        } else {
            let j = (self.ny - 1) - (element_index - (self.nx * self.ny + 2 * self.nx + self.ny));

            MeshElement {
                geometric_element: &EDGE2,
                label: 3,
                indexes: vec![(j + 1) * (self.nx + 1), j * (self.nx + 1)],
            }
        }
    }
    fn get_geometric_dimension(&self) -> usize {
        2
    }
    fn get_topologic_dimension(&self) -> usize {
        2
    }
    fn get_elements_by_dimension(&self, dim: usize) -> Vec<usize> {
        match dim {
            2 => (0..self.nx * self.ny).collect(),
            1 => (self.nx * self.ny..self.nx * self.ny + 2 * self.nx + 2 * self.ny).collect(),
            _ => vec![],
        }
    }
    fn get_elements_by_dimension_and_label(&self, label: usize, dim: usize) -> Vec<usize> {
        let quad_end = self.nx * self.ny;
        match (dim, label) {
            (2, 0) => (0..quad_end).collect(),
            (1, 0) => (quad_end..quad_end + self.nx).collect(),
            (1, 1) => (quad_end + self.nx..quad_end + self.nx + self.ny).collect(),
            (1, 2) => (quad_end + self.nx + self.ny..quad_end + 2 * self.nx + self.ny).collect(),
            (1, 3) => {
                (quad_end + 2 * self.nx + self.ny..quad_end + 2 * self.nx + 2 * self.ny).collect()
            }
            _ => vec![],
        }
    }
    fn get_ncolors(&self) -> usize {
        4
    }
    fn get_nelements_by_color(&self, color_index: usize) -> usize {
        // get_element_color = row_parity + 2*col_parity, so:
        //   color 0: even row, even col  → ceil(nx/2) * ceil(ny/2)
        //   color 1: odd row,  even col  → ceil(nx/2) * floor(ny/2)
        //   color 2: even row, odd col   → floor(nx/2) * ceil(ny/2)
        //   color 3: odd row,  odd col   → floor(nx/2) * floor(ny/2)
        match color_index {
            0 => (self.nx / 2 + self.nx % 2) * (self.ny / 2 + self.ny % 2),
            1 => (self.nx / 2 + self.nx % 2) * (self.ny / 2),
            2 => (self.nx / 2) * (self.ny / 2 + self.ny % 2),
            3 => (self.nx / 2) * (self.ny / 2),
            _ => 0,
        }
    }
    fn get_element_color(&self, element_index: usize) -> usize {
        debug_assert!(
            element_index < self.nx * self.ny,
            "Colors are only defined for elements of the highest topological dimension"
        );
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
                let stride = self.nx / 2 + self.nx % 2;
                let ex = element_index_by_color % stride;
                let ey = element_index_by_color / stride;
                gex = 2 * ex;
                gey = 2 * ey;
            }
            1 => {
                // odd row, even col: x-stride = ceil(nx/2), gex even, gey odd
                let stride = self.nx / 2 + self.nx % 2;
                let ex = element_index_by_color % stride;
                let ey = element_index_by_color / stride;
                gex = 2 * ex;
                gey = 2 * ey + 1;
            }
            2 => {
                // even row, odd col: x-stride = floor(nx/2), gex odd, gey even
                let stride = self.nx / 2;
                let ex = element_index_by_color % stride;
                let ey = element_index_by_color / stride;
                gex = 2 * ex + 1;
                gey = 2 * ey;
            }
            3 => {
                let stride = self.nx / 2;
                let ex = element_index_by_color % stride;
                let ey = element_index_by_color / stride;
                gex = 2 * ex + 1;
                gey = 2 * ey + 1;
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
