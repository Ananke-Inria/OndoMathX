pub use crate::mesh::*;

pub use crate::diffeomorphism::*;
pub use unstructured_mesh::*;

pub struct Cube {
    pub origin: [f64; 3],
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub deformation: Option<Diffeomorphism>,
    //When true the mesh carries Q2 geometric nodes: the Q1 vertices keep
    //their numbering and the edge midpoints / face centers / cell centers
    //are appended after them; get_element then returns HEXAHEDRON27 /
    //QUADRANGLE9 elements.
    pub q2: bool,
}

impl Cube {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        origin: [f64; 3],
        lx: f64,
        ly: f64,
        lz: f64,
        nx: usize,
        ny: usize,
        nz: usize,
        deformation: Option<Diffeomorphism>,
        q2: Option<bool>,
    ) -> Cube {
        assert!(
            nx > 0,
            "Number of element in the x-direction should be greater than 0"
        );
        assert!(
            ny > 0,
            "Number of element in the y-direction should be greater than 0"
        );
        assert!(
            nz > 0,
            "Number of element in the z-direction should be greater than 0"
        );
        assert!(lx > 0.0, "The length of the cube should be greater than 0");
        assert!(ly > 0.0, "The length of the cube should be greater than 0");
        assert!(lz > 0.0, "The length of the cube should be greater than 0");

        Cube {
            origin,
            lx,
            ly,
            lz,
            nx,
            ny,
            nz,
            deformation,
            q2: q2.unwrap_or(false),
        }
    }

    //Global indexes of the nodes. The Q1 vertices come first; the Q2
    //geometric nodes are appended after them in groups — x-edge midpoints,
    //y-edge midpoints, z-edge midpoints, z-normal face centers, y-normal
    //face centers, x-normal face centers, cell centers — each group numbered
    //x-fastest, then y, then z.
    fn vertex(&self, i: usize, j: usize, k: usize) -> usize {
        (k * (self.ny + 1) + j) * (self.nx + 1) + i
    }
    fn xmid(&self, i: usize, j: usize, k: usize) -> usize {
        let base = (self.nx + 1) * (self.ny + 1) * (self.nz + 1);
        base + (k * (self.ny + 1) + j) * self.nx + i
    }
    fn ymid(&self, i: usize, j: usize, k: usize) -> usize {
        let base =
            (self.nx + 1) * (self.ny + 1) * (self.nz + 1) + self.nx * (self.ny + 1) * (self.nz + 1);
        base + (k * self.ny + j) * (self.nx + 1) + i
    }
    fn zmid(&self, i: usize, j: usize, k: usize) -> usize {
        let base = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
            + self.nx * (self.ny + 1) * (self.nz + 1)
            + (self.nx + 1) * self.ny * (self.nz + 1);
        base + (k * (self.ny + 1) + j) * (self.nx + 1) + i
    }
    fn zface(&self, i: usize, j: usize, k: usize) -> usize {
        let base = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
            + self.nx * (self.ny + 1) * (self.nz + 1)
            + (self.nx + 1) * self.ny * (self.nz + 1)
            + (self.nx + 1) * (self.ny + 1) * self.nz;
        base + (k * self.ny + j) * self.nx + i
    }
    fn yface(&self, i: usize, j: usize, k: usize) -> usize {
        let base = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
            + self.nx * (self.ny + 1) * (self.nz + 1)
            + (self.nx + 1) * self.ny * (self.nz + 1)
            + (self.nx + 1) * (self.ny + 1) * self.nz
            + self.nx * self.ny * (self.nz + 1);
        base + (k * (self.ny + 1) + j) * self.nx + i
    }
    fn xface(&self, i: usize, j: usize, k: usize) -> usize {
        let base = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
            + self.nx * (self.ny + 1) * (self.nz + 1)
            + (self.nx + 1) * self.ny * (self.nz + 1)
            + (self.nx + 1) * (self.ny + 1) * self.nz
            + self.nx * self.ny * (self.nz + 1)
            + self.nx * (self.ny + 1) * self.nz;
        base + (k * self.ny + j) * (self.nx + 1) + i
    }
    fn center(&self, i: usize, j: usize, k: usize) -> usize {
        let base = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
            + self.nx * (self.ny + 1) * (self.nz + 1)
            + (self.nx + 1) * self.ny * (self.nz + 1)
            + (self.nx + 1) * (self.ny + 1) * self.nz
            + self.nx * self.ny * (self.nz + 1)
            + self.nx * (self.ny + 1) * self.nz
            + (self.nx + 1) * self.ny * self.nz;
        base + (k * self.ny + j) * self.nx + i
    }

    //Number of cells with even (odd = false) or odd parity along an axis of
    //n cells: the 2x2x2 checkerboard coloring slices each axis by parity.
    fn axis_count(n: usize, odd: bool) -> usize {
        if odd {
            n / 2
        } else {
            n / 2 + n % 2
        }
    }

    pub fn get_numbering(
        &self,
        nlocdofx: usize,
        nlocdofy: usize,
        nlocdofz: usize,
    ) -> (Vec<Vec<usize>>, usize) {
        let mut numbering: Vec<Vec<usize>> = vec![vec![]; self.get_nelements()];

        let px = nlocdofx - 1;
        let py = nlocdofy - 1;
        let pz = nlocdofz - 1;
        let n_dof_x = self.nx * px + 1;
        let n_dof_y = self.ny * py + 1;
        let n_dof_z = self.nz * pz + 1;
        let n_dof = n_dof_x * n_dof_y * n_dof_z;

        for iz in 0..self.nz {
            for iy in 0..self.ny {
                for ix in 0..self.nx {
                    let el = (iz * self.ny + iy) * self.nx + ix;

                    for idofz in 0..nlocdofz {
                        for idofy in 0..nlocdofy {
                            for idofx in 0..nlocdofx {
                                numbering[el].push(
                                    ((iz * pz + idofz) * n_dof_y + (iy * py + idofy)) * n_dof_x
                                        + (ix * px + idofx),
                                );
                            }
                        }
                    }
                }
            }
        }

        (numbering, n_dof)
    }
}

impl Mesh for Cube {
    //Structured numbering, any counts.
    fn get_tensor_numbering(&self, n: [usize; 3]) -> (Vec<Vec<usize>>, usize) {
        assert!(n.iter().all(|&k| k >= 2), "at least 2 DoFs per direction, got {n:?}");
        self.get_numbering(n[0], n[1], n[2])
    }

    fn get_deformation(&self) -> Option<&Diffeomorphism> {
        self.deformation.as_ref()
    }
    fn get_nnodes(&self) -> usize {
        if self.q2 {
            //vertices + edge midpoints + face centers + cell centers
            (2 * self.nx + 1) * (2 * self.ny + 1) * (2 * self.nz + 1)
        } else {
            (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        }
    }
    fn get_node(&self, index: usize) -> [f64; 3] {
        debug_assert!(index < self.get_nnodes(), "Node index out of bounds");

        let hx = self.lx / self.nx as f64;
        let hy = self.ly / self.ny as f64;
        let hz = self.lz / self.nz as f64;

        //Grid dimensions and half-step offsets of the node groups, in the
        //numbering order of the index helpers; a Q1 mesh only has the first.
        let groups: [([usize; 3], [f64; 3]); 8] = [
            ([self.nx + 1, self.ny + 1, self.nz + 1], [0.0, 0.0, 0.0]),
            ([self.nx, self.ny + 1, self.nz + 1], [0.5, 0.0, 0.0]),
            ([self.nx + 1, self.ny, self.nz + 1], [0.0, 0.5, 0.0]),
            ([self.nx + 1, self.ny + 1, self.nz], [0.0, 0.0, 0.5]),
            ([self.nx, self.ny, self.nz + 1], [0.5, 0.5, 0.0]),
            ([self.nx, self.ny + 1, self.nz], [0.5, 0.0, 0.5]),
            ([self.nx + 1, self.ny, self.nz], [0.0, 0.5, 0.5]),
            ([self.nx, self.ny, self.nz], [0.5, 0.5, 0.5]),
        ];

        let mut rel = index;
        for (dims, offset) in groups.iter() {
            let count = dims[0] * dims[1] * dims[2];

            if rel < count {
                let i = rel % dims[0];
                let j = (rel / dims[0]) % dims[1];
                let k = rel / (dims[0] * dims[1]);

                return [
                    self.origin[0] + hx * (i as f64 + offset[0]),
                    self.origin[1] + hy * (j as f64 + offset[1]),
                    self.origin[2] + hz * (k as f64 + offset[2]),
                ];
            }

            rel -= count;
        }

        unreachable!("Node index out of bounds");
    }
    fn get_nelements(&self) -> usize {
        self.nx * self.ny * self.nz
            + 2 * (self.nx * self.ny + self.nx * self.nz + self.ny * self.nz)
    }
    fn get_element(&self, element_index: usize) -> MeshElement {
        debug_assert!(
            element_index < self.get_nelements(),
            "Element index out of bounds"
        );

        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n_vol = nx * ny * nz;

        if element_index < n_vol {
            let i = element_index % nx;
            let j = (element_index / nx) % ny;
            let k = element_index / (nx * ny);

            let mut indexes = vec![
                self.vertex(i, j, k),
                self.vertex(i + 1, j, k),
                self.vertex(i + 1, j + 1, k),
                self.vertex(i, j + 1, k),
                self.vertex(i, j, k + 1),
                self.vertex(i + 1, j, k + 1),
                self.vertex(i + 1, j + 1, k + 1),
                self.vertex(i, j + 1, k + 1),
            ];

            if !self.q2 {
                return MeshElement {
                    geometric_element: &HEXAHEDRON8,
                    label: 0,
                    indexes,
                };
            }

            //Edge midpoints in HEXAHEDRON_EDGES_TOPOLOGY order, then face
            //centers in HEXAHEDRON_FACES_TOPOLOGY order, then the center
            indexes.extend_from_slice(&[
                self.xmid(i, j, k),         //[0,1]
                self.ymid(i + 1, j, k),     //[1,2]
                self.xmid(i, j + 1, k),     //[3,2]
                self.ymid(i, j, k),         //[0,3]
                self.zmid(i, j, k),         //[0,4]
                self.zmid(i + 1, j, k),     //[1,5]
                self.zmid(i + 1, j + 1, k), //[2,6]
                self.zmid(i, j + 1, k),     //[3,7]
                self.xmid(i, j, k + 1),     //[4,5]
                self.ymid(i + 1, j, k + 1), //[5,6]
                self.xmid(i, j + 1, k + 1), //[7,6]
                self.ymid(i, j, k + 1),     //[4,7]
                self.zface(i, j, k),        //[0,1,2,3]
                self.yface(i, j, k),        //[0,1,5,4]
                self.xface(i, j, k),        //[0,3,7,4]
                self.xface(i + 1, j, k),    //[1,2,6,5]
                self.yface(i, j + 1, k),    //[3,2,6,7]
                self.zface(i, j, k + 1),    //[4,5,6,7]
                self.center(i, j, k),
            ]);

            return MeshElement {
                geometric_element: &HEXAHEDRON27,
                label: 0,
                indexes,
            };
        }

        //Boundary faces in 6 labelled groups (0 bottom z=0, 1 front y=0,
        //2 left x=0, 3 right x=lx, 4 back y=ly, 5 top z=lz — the order of
        //the hexahedron face-table normals), corners ordered so that
        //(v1-v0)x(v3-v0) points outward. The Q2 nodes follow the
        //QUADRANGLE_EDGES_TOPOLOGY order, then the face center.
        let e = element_index - n_vol;

        let (label, v, q2_nodes) = if e < nx * ny {
            let (i, j) = (e % nx, e / nx);
            (
                0,
                [
                    self.vertex(i, j, 0),
                    self.vertex(i, j + 1, 0),
                    self.vertex(i + 1, j + 1, 0),
                    self.vertex(i + 1, j, 0),
                ],
                [
                    self.ymid(i, j, 0),
                    self.xmid(i, j + 1, 0),
                    self.ymid(i + 1, j, 0),
                    self.xmid(i, j, 0),
                    self.zface(i, j, 0),
                ],
            )
        } else if e < nx * ny + nx * nz {
            let rel = e - nx * ny;
            let (i, k) = (rel % nx, rel / nx);
            (
                1,
                [
                    self.vertex(i, 0, k),
                    self.vertex(i + 1, 0, k),
                    self.vertex(i + 1, 0, k + 1),
                    self.vertex(i, 0, k + 1),
                ],
                [
                    self.xmid(i, 0, k),
                    self.zmid(i + 1, 0, k),
                    self.xmid(i, 0, k + 1),
                    self.zmid(i, 0, k),
                    self.yface(i, 0, k),
                ],
            )
        } else if e < nx * ny + nx * nz + ny * nz {
            let rel = e - (nx * ny + nx * nz);
            let (j, k) = (rel % ny, rel / ny);
            (
                2,
                [
                    self.vertex(0, j, k),
                    self.vertex(0, j, k + 1),
                    self.vertex(0, j + 1, k + 1),
                    self.vertex(0, j + 1, k),
                ],
                [
                    self.zmid(0, j, k),
                    self.ymid(0, j, k + 1),
                    self.zmid(0, j + 1, k),
                    self.ymid(0, j, k),
                    self.xface(0, j, k),
                ],
            )
        } else if e < nx * ny + nx * nz + 2 * ny * nz {
            let rel = e - (nx * ny + nx * nz + ny * nz);
            let (j, k) = (rel % ny, rel / ny);
            (
                3,
                [
                    self.vertex(nx, j, k),
                    self.vertex(nx, j + 1, k),
                    self.vertex(nx, j + 1, k + 1),
                    self.vertex(nx, j, k + 1),
                ],
                [
                    self.ymid(nx, j, k),
                    self.zmid(nx, j + 1, k),
                    self.ymid(nx, j, k + 1),
                    self.zmid(nx, j, k),
                    self.xface(nx, j, k),
                ],
            )
        } else if e < nx * ny + 2 * nx * nz + 2 * ny * nz {
            let rel = e - (nx * ny + nx * nz + 2 * ny * nz);
            let (i, k) = (rel % nx, rel / nx);
            (
                4,
                [
                    self.vertex(i, ny, k),
                    self.vertex(i, ny, k + 1),
                    self.vertex(i + 1, ny, k + 1),
                    self.vertex(i + 1, ny, k),
                ],
                [
                    self.zmid(i, ny, k),
                    self.xmid(i, ny, k + 1),
                    self.zmid(i + 1, ny, k),
                    self.xmid(i, ny, k),
                    self.yface(i, ny, k),
                ],
            )
        } else {
            let rel = e - (nx * ny + 2 * nx * nz + 2 * ny * nz);
            let (i, j) = (rel % nx, rel / nx);
            (
                5,
                [
                    self.vertex(i, j, nz),
                    self.vertex(i + 1, j, nz),
                    self.vertex(i + 1, j + 1, nz),
                    self.vertex(i, j + 1, nz),
                ],
                [
                    self.xmid(i, j, nz),
                    self.ymid(i + 1, j, nz),
                    self.xmid(i, j + 1, nz),
                    self.ymid(i, j, nz),
                    self.zface(i, j, nz),
                ],
            )
        };

        if self.q2 {
            let mut indexes = v.to_vec();
            indexes.extend_from_slice(&q2_nodes);

            MeshElement {
                geometric_element: &QUADRANGLE9,
                label,
                indexes,
            }
        } else {
            MeshElement {
                geometric_element: &QUADRANGLE4,
                label,
                indexes: v.to_vec(),
            }
        }
    }
    fn get_geometric_dimension(&self) -> usize {
        3
    }
    fn get_topologic_dimension(&self) -> usize {
        3
    }
    fn get_elements_by_dimension(&self, dim: usize) -> Vec<usize> {
        let n_vol = self.nx * self.ny * self.nz;
        match dim {
            3 => (0..n_vol).collect(),
            2 => (n_vol..self.get_nelements()).collect(),
            _ => vec![],
        }
    }
    fn get_elements_by_dimension_and_label(&self, label: usize, dim: usize) -> Vec<usize> {
        let n_vol = self.nx * self.ny * self.nz;

        //Boundary group sizes in label order
        let sizes = [
            self.nx * self.ny,
            self.nx * self.nz,
            self.ny * self.nz,
            self.ny * self.nz,
            self.nx * self.nz,
            self.nx * self.ny,
        ];

        match (dim, label) {
            (3, 0) => (0..n_vol).collect(),
            (2, l) if l < 6 => {
                let start = n_vol + sizes[..l].iter().sum::<usize>();
                (start..start + sizes[l]).collect()
            }
            _ => vec![],
        }
    }
    fn get_ncolors(&self) -> usize {
        8
    }
    fn get_nelements_by_color(&self, color_index: usize) -> usize {
        //get_element_color = i%2 + 2*(j%2) + 4*(k%2): the bits of the color
        //are the cell parities along x, y, z.
        if color_index < 8 {
            Cube::axis_count(self.nx, color_index & 1 != 0)
                * Cube::axis_count(self.ny, color_index & 2 != 0)
                * Cube::axis_count(self.nz, color_index & 4 != 0)
        } else {
            0
        }
    }
    fn get_element_color(&self, element_index: usize) -> usize {
        debug_assert!(
            element_index < self.nx * self.ny * self.nz,
            "Colors are only defined for elements of the highest topological dimension"
        );

        let i = element_index % self.nx;
        let j = (element_index / self.nx) % self.ny;
        let k = element_index / (self.nx * self.ny);

        i % 2 + 2 * (j % 2) + 4 * (k % 2)
    }
    fn get_element_index_by_color_to_element_index(
        &self,
        color_index: usize,
        element_index_by_color: usize,
    ) -> usize {
        assert!(color_index < 8, "color index out of bounds");
        debug_assert!(
            element_index_by_color < self.get_nelements_by_color(color_index),
            "Element index is out of bound"
        );

        let bx = color_index & 1;
        let by = (color_index >> 1) & 1;
        let bz = (color_index >> 2) & 1;

        let sx = Cube::axis_count(self.nx, bx != 0);
        let sy = Cube::axis_count(self.ny, by != 0);

        let ex = element_index_by_color % sx;
        let ey = (element_index_by_color / sx) % sy;
        let ez = element_index_by_color / (sx * sy);

        let gi = 2 * ex + bx;
        let gj = 2 * ey + by;
        let gk = 2 * ez + bz;

        (gk * self.ny + gj) * self.nx + gi
    }
    //Writes the Jacobian of the analytic element map (composed with the
    //attached deformation if any), row-major: the cells are axis-aligned
    //boxes so the undeformed Jacobian is exactly diag(hx, hy, hz) (3×3); the
    //boundary faces are axis-aligned rectangles whose 3×2 Jacobian columns
    //are the corner differences (v1-v0, v3-v0) of get_element.
    fn get_element_grad_def(&self, element_index: usize, uvw: &[f64], grad_def: &mut [f64]) {
        debug_assert!(
            element_index < self.get_nelements(),
            "Element index out of bounds"
        );

        let hx = self.lx / self.nx as f64;
        let hy = self.ly / self.ny as f64;
        let hz = self.lz / self.nz as f64;
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n_vol = nx * ny * nz;

        //Undeformed Jacobian and (if a deformation is attached) the physical
        //image of uvw under the undeformed element map.
        let elem_dim;
        let mut xyz = [0.0; 3];

        if element_index < n_vol {
            //Cell (i, j, k): x(u,v,w) = origin + ((i+u)·hx, (j+v)·hy, (k+w)·hz)
            elem_dim = 3;
            grad_def[..9].copy_from_slice(&[hx, 0.0, 0.0, 0.0, hy, 0.0, 0.0, 0.0, hz]);

            let i = element_index % nx;
            let j = (element_index / nx) % ny;
            let k = element_index / (nx * ny);
            xyz[0] = self.origin[0] + (i as f64 + uvw[0]) * hx;
            xyz[1] = self.origin[1] + (j as f64 + uvw[1]) * hy;
            xyz[2] = self.origin[2] + (k as f64 + uvw[2]) * hz;
        } else {
            elem_dim = 2;
            let (u, v) = (uvw[0], uvw[1]);
            let e = element_index - n_vol;

            if e < nx * ny {
                //Bottom (label 0): x(u,v) = origin + ((i+v)·hx, (j+u)·hy, 0)
                let (i, j) = (e % nx, e / nx);
                grad_def[..6].copy_from_slice(&[0.0, hx, hy, 0.0, 0.0, 0.0]);
                xyz[0] = self.origin[0] + (i as f64 + v) * hx;
                xyz[1] = self.origin[1] + (j as f64 + u) * hy;
                xyz[2] = self.origin[2];
            } else if e < nx * ny + nx * nz {
                //Front (label 1): x(u,v) = origin + ((i+u)·hx, 0, (k+v)·hz)
                let rel = e - nx * ny;
                let (i, k) = (rel % nx, rel / nx);
                grad_def[..6].copy_from_slice(&[hx, 0.0, 0.0, 0.0, 0.0, hz]);
                xyz[0] = self.origin[0] + (i as f64 + u) * hx;
                xyz[1] = self.origin[1];
                xyz[2] = self.origin[2] + (k as f64 + v) * hz;
            } else if e < nx * ny + nx * nz + ny * nz {
                //Left (label 2): x(u,v) = origin + (0, (j+v)·hy, (k+u)·hz)
                let rel = e - (nx * ny + nx * nz);
                let (j, k) = (rel % ny, rel / ny);
                grad_def[..6].copy_from_slice(&[0.0, 0.0, 0.0, hy, hz, 0.0]);
                xyz[0] = self.origin[0];
                xyz[1] = self.origin[1] + (j as f64 + v) * hy;
                xyz[2] = self.origin[2] + (k as f64 + u) * hz;
            } else if e < nx * ny + nx * nz + 2 * ny * nz {
                //Right (label 3): x(u,v) = origin + (lx, (j+u)·hy, (k+v)·hz)
                let rel = e - (nx * ny + nx * nz + ny * nz);
                let (j, k) = (rel % ny, rel / ny);
                grad_def[..6].copy_from_slice(&[0.0, 0.0, hy, 0.0, 0.0, hz]);
                xyz[0] = self.origin[0] + self.lx;
                xyz[1] = self.origin[1] + (j as f64 + u) * hy;
                xyz[2] = self.origin[2] + (k as f64 + v) * hz;
            } else if e < nx * ny + 2 * nx * nz + 2 * ny * nz {
                //Back (label 4): x(u,v) = origin + ((i+v)·hx, ly, (k+u)·hz)
                let rel = e - (nx * ny + nx * nz + 2 * ny * nz);
                let (i, k) = (rel % nx, rel / nx);
                grad_def[..6].copy_from_slice(&[0.0, hx, 0.0, 0.0, hz, 0.0]);
                xyz[0] = self.origin[0] + (i as f64 + v) * hx;
                xyz[1] = self.origin[1] + self.ly;
                xyz[2] = self.origin[2] + (k as f64 + u) * hz;
            } else {
                //Top (label 5): x(u,v) = origin + ((i+u)·hx, (j+v)·hy, lz)
                let rel = e - (nx * ny + 2 * nx * nz + 2 * ny * nz);
                let (i, j) = (rel % nx, rel / nx);
                grad_def[..6].copy_from_slice(&[hx, 0.0, 0.0, hy, 0.0, 0.0]);
                xyz[0] = self.origin[0] + (i as f64 + u) * hx;
                xyz[1] = self.origin[1] + (j as f64 + v) * hy;
                xyz[2] = self.origin[2] + self.lz;
            }
        }

        //Chain rule with the attached deformation: grad_def ← Df(xyz)·grad_def.
        if let Some(deformation) = &self.deformation {
            deformation.compose_grad_def(&xyz, grad_def, elem_dim);
        }
    }
}
