pub use crate::diffeomorphism::*;
pub use crate::mesh::*;
pub use unstructured_mesh::*;

/// Returns `(n_x, n_rho, alpha)` satisfying `2·n_rho + n_x = n`.
/// `alpha` is chosen so that arm elements on the diagonal have the same
/// side length as the interior square elements: `alpha·√2/n_x = (1-alpha)/n_rho`.
fn optimal_params(n: usize) -> (usize, usize, f64) {
    let n_rho = n / 3;
    let n_x = n - 2 * n_rho;
    let alpha = n_x as f64 / (std::f64::consts::SQRT_2 * n_rho as f64 + n_x as f64);
    (n_x, n_rho, alpha)
}

// Maps the reference square [-r_ref, r_ref]² (r_ref = R/√2) to the disk of radius R.
// params: [radius, cx, cy, alpha].  5 branches: inner square (identity) + 4 arms
// (O-grid maps: the transverse coordinate drives the angle, see arm_map).

fn circle_selector(p: &[f64], xyz: &[f64]) -> usize {
    let r_ref = p[0] / std::f64::consts::SQRT_2;
    let a = p[3] * r_ref;
    let lx = xyz[0] - p[1];
    let ly = xyz[1] - p[2];
    if lx.abs() <= a && ly.abs() <= a {
        0
    }
    // inner square
    else if ly <= -lx.abs() {
        1
    }
    // bottom arm
    else if lx >= ly.abs() {
        2
    }
    // right arm
    else if ly >= lx.abs() {
        3
    }
    // top arm
    else {
        4
    } // left arm
}

fn circle_inner(_p: &[f64], xyz: &[f64], f_xyz: &mut [f64]) {
    f_xyz[0] = xyz[0];
    f_xyz[1] = xyz[1];
    f_xyz[2] = xyz[2];
}

// (cos, sin) of the rotation by k·90° that carries the bottom-arm frame to
// arm k (0 bottom, 1 right, 2 top, 3 left).
const ARM_ROT: [[f64; 2]; 4] = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];

// Shared O-grid arm map, written in the bottom-arm frame (arm k is first
// rotated into it). On the trapezoid {y ∈ [-r_ref, -a], |x| ≤ |y|} let
//     t = (|y| - a)/(r_ref - a) ∈ [0, 1],   s = x/|y| ∈ [-1, 1];
// each center ray s = const is sent to the straight segment joining its
// inner-square boundary point (s·a, -a) to the circle point at angle
// θ(s) = -π/2 + s·π/4:
//     f = (1-t)·(s·a, -a) + t·R·(sin θ, -cos θ).
// The transverse coordinate s drives the angle, so the Jacobian is nonzero
// on the whole closed arm — including the two corners where the diagonals
// meet the circle, where the previous angle-preserving radial blend
// r_new = r + t·(R - r) was degenerate (zero radial derivative).
#[inline]
fn arm_map(p: &[f64], xyz: &[f64], f_xyz: &mut [f64], k: usize) {
    let r_ref = p[0] / std::f64::consts::SQRT_2;
    let a = p[3] * r_ref;
    let [c, sr] = ARM_ROT[k];
    // Rotate by -k·90° into the bottom-arm frame.
    let lx = xyz[0] - p[1];
    let ly = xyz[1] - p[2];
    let bx = c * lx + sr * ly;
    let by = -sr * lx + c * ly;
    let t = (-by - a) / (r_ref - a);
    let s = -bx / by;
    let (sin_t, cos_t) = (s * std::f64::consts::FRAC_PI_4).sin_cos();
    let fx = (1.0 - t) * s * a + t * p[0] * sin_t;
    let fy = -(1.0 - t) * a - t * p[0] * cos_t;
    // Rotate back by k·90°.
    f_xyz[0] = c * fx - sr * fy + p[1];
    f_xyz[1] = sr * fx + c * fy + p[2];
    f_xyz[2] = xyz[2];
}

fn circle_bottom(p: &[f64], xyz: &[f64], f_xyz: &mut [f64]) {
    arm_map(p, xyz, f_xyz, 0);
}

fn circle_right(p: &[f64], xyz: &[f64], f_xyz: &mut [f64]) {
    arm_map(p, xyz, f_xyz, 1);
}

fn circle_top(p: &[f64], xyz: &[f64], f_xyz: &mut [f64]) {
    arm_map(p, xyz, f_xyz, 2);
}

fn circle_left(p: &[f64], xyz: &[f64], f_xyz: &mut [f64]) {
    arm_map(p, xyz, f_xyz, 3);
}

// Gradients of the circle-map branches: write the 2x2 Jacobian of the branch
// at `xyz` into `grad` (row-major: [∂f1/∂x, ∂f1/∂y, ∂f2/∂x, ∂f2/∂y]).

fn circle_inner_grad(_p: &[f64], _xyz: &[f64], grad: &mut [f64]) {
    grad[0] = 1.0;
    grad[1] = 0.0;
    grad[2] = 0.0;
    grad[3] = 1.0;
}

// Jacobian of arm_map: with f = (1-t)·p_in(s) + t·p_out(s),
//   ∂f/∂t = p_out - p_in,   ∂f/∂s = (1-t)·(a, 0) + t·R·(π/4)·(cos θ, sin θ),
//   ∂t/∂by = -1/(r_ref - a),   ∂s/∂bx = -1/by,   ∂s/∂by = bx/by²,
// conjugated by the frame rotation: G = Rk · B · Rkᵀ.
#[inline]
fn arm_map_grad(p: &[f64], xyz: &[f64], grad: &mut [f64], k: usize) {
    let r_ref = p[0] / std::f64::consts::SQRT_2;
    let a = p[3] * r_ref;
    let [c, sr] = ARM_ROT[k];
    let lx = xyz[0] - p[1];
    let ly = xyz[1] - p[2];
    let bx = c * lx + sr * ly;
    let by = -sr * lx + c * ly;
    let t = (-by - a) / (r_ref - a);
    let s = -bx / by;
    let (sin_t, cos_t) = (s * std::f64::consts::FRAC_PI_4).sin_cos();

    let dt_dby = -1.0 / (r_ref - a);
    let ds_dbx = -1.0 / by;
    let ds_dby = bx / (by * by);

    let df_dt = [p[0] * sin_t - s * a, a - p[0] * cos_t];
    let df_ds = [
        (1.0 - t) * a + t * p[0] * std::f64::consts::FRAC_PI_4 * cos_t,
        t * p[0] * std::f64::consts::FRAC_PI_4 * sin_t,
    ];

    // Jacobian in the bottom-arm frame.
    let b = [
        df_ds[0] * ds_dbx,
        df_dt[0] * dt_dby + df_ds[0] * ds_dby,
        df_ds[1] * ds_dbx,
        df_dt[1] * dt_dby + df_ds[1] * ds_dby,
    ];

    // G = Rk · B · Rkᵀ with Rk = [[c, -sr], [sr, c]].
    let brt = [
        b[0] * c - b[1] * sr,
        b[0] * sr + b[1] * c,
        b[2] * c - b[3] * sr,
        b[2] * sr + b[3] * c,
    ];
    grad[0] = c * brt[0] - sr * brt[2];
    grad[1] = c * brt[1] - sr * brt[3];
    grad[2] = sr * brt[0] + c * brt[2];
    grad[3] = sr * brt[1] + c * brt[3];
}

fn circle_bottom_grad(p: &[f64], xyz: &[f64], grad: &mut [f64]) {
    arm_map_grad(p, xyz, grad, 0);
}

fn circle_right_grad(p: &[f64], xyz: &[f64], grad: &mut [f64]) {
    arm_map_grad(p, xyz, grad, 1);
}

fn circle_top_grad(p: &[f64], xyz: &[f64], grad: &mut [f64]) {
    arm_map_grad(p, xyz, grad, 2);
}

fn circle_left_grad(p: &[f64], xyz: &[f64], grad: &mut [f64]) {
    arm_map_grad(p, xyz, grad, 3);
}

/// Structured quad mesh of a 2D disk of radius `R`.
///
/// The reference domain `[-r_ref, r_ref]²` (`r_ref = R/√2`) is split into an inner
/// square and four arms, then mapped to the disk by `circle_map`.
/// `n_x`, `n_rho`, `alpha` are computed automatically for near-square elements.
pub struct Circle {
    pub center: [f64; 2],
    pub radius: f64,
    pub n: usize,
    pub alpha: f64,
    pub deformation: Option<Diffeomorphism>,

    //When true the mesh carries Q2 geometric nodes: the Q1 vertices keep
    //their numbering and the edge midpoints / cell centers (images of the
    //reference-domain midpoints under circle_map) are appended after them;
    //get_element then returns QUADRANGLE9 / EDGE3 elements.
    pub q2: bool,

    pub n_x: usize,
    pub n_rho: usize,
    pub n_theta: usize,
    n_vert_square: usize,
    n_square_elem: usize,
    coloring: Vec<usize>,
    elements_by_color: Vec<Vec<usize>>,
}

impl Circle {
    pub fn new(
        center: [f64; 2],
        radius: f64,
        n: usize,
        deformation: Option<Diffeomorphism>,
        q2: Option<bool>,
    ) -> Self {
        assert!(n > 2, "n must be > 2");
        assert!(radius > 0.0, "radius must be > 0");

        let (n_x, n_rho, alpha) = optimal_params(n);
        let n_theta = 4 * n_x;
        let n_square_elem = n_x * n_x;

        // Hardcoded 4-coloring (elements of the same colour never share a node).
        // Inner square: 2×2 checkerboard ix%2 + 2*(iy%2), as in `Square`.
        // Annulus (transition layer = row 0, then the arm layers): per-quarter
        // colour tables indexed by [quarter][row parity][column parity]. The
        // tables are chosen so that the transition layer avoids the colours of
        // the square cells it touches and so that colours stay distinct across
        // the four quarter seams. The square-boundary colours seen by the
        // right/top quarters depend on the parity of n_x, hence two table sets.
        const TABLE_EVEN: [[[usize; 2]; 2]; 4] = [
            [[3, 2], [0, 1]], // bottom
            [[0, 2], [3, 1]], // right
            [[0, 1], [3, 2]], // top
            [[3, 1], [0, 2]], // left
        ];
        const TABLE_ODD: [[[usize; 2]; 2]; 4] = [
            [[2, 3], [1, 0]], // bottom
            [[3, 1], [0, 2]], // right
            [[2, 3], [1, 0]], // top
            [[3, 1], [0, 2]], // left
        ];
        let table = if n_x % 2 == 0 {
            &TABLE_EVEN
        } else {
            &TABLE_ODD
        };

        let mut coloring = Vec::with_capacity(n_square_elem + n_rho * n_theta);
        for iy in 0..n_x {
            for ix in 0..n_x {
                coloring.push(ix % 2 + 2 * (iy % 2));
            }
        }
        for i_rho in 0..n_rho {
            for i_theta in 0..n_theta {
                let i_quarter = i_theta / n_x;
                let i_x = i_theta % n_x;
                coloring.push(table[i_quarter][i_rho % 2][i_x % 2]);
            }
        }

        let mut elements_by_color: Vec<Vec<usize>> = vec![vec![]; 4];
        for (element, &color) in coloring.iter().enumerate() {
            elements_by_color[color].push(element);
        }

        Circle {
            center,
            radius,
            n,
            alpha,
            deformation,
            q2: q2.unwrap_or(false),
            n_x,
            n_rho,
            n_theta,
            n_vert_square: (n_x + 1) * (n_x + 1),
            n_square_elem,
            coloring,
            elements_by_color,
        }
    }

    //Number of Q1 nodes (the vertices of the mesh).
    fn n_q1_nodes(&self) -> usize {
        self.n_vert_square + self.n_rho * self.n_theta
    }

    //Global indexes of the Q2 geometric nodes, appended after the Q1
    //vertices in this order: square horizontal-edge midpoints (n_x*(n_x+1)),
    //square vertical-edge midpoints ((n_x+1)*n_x), radial-edge midpoints
    //(n_rho*n_theta, layer 0 connecting the square boundary to ring 1),
    //tangential ring-edge midpoints (n_rho*n_theta), cell centers.
    fn sq_hmid(&self, ix: usize, iy: usize) -> usize {
        self.n_q1_nodes() + iy * self.n_x + ix
    }
    fn sq_vmid(&self, ix: usize, iy: usize) -> usize {
        self.n_q1_nodes() + self.n_x * (self.n_x + 1) + iy * (self.n_x + 1) + ix
    }
    fn rad_mid(&self, r: usize, i_theta: usize) -> usize {
        self.n_q1_nodes() + 2 * self.n_x * (self.n_x + 1) + r * self.n_theta + i_theta
    }
    fn tan_mid(&self, l: usize, i_theta: usize) -> usize {
        self.n_q1_nodes()
            + 2 * self.n_x * (self.n_x + 1)
            + self.n_rho * self.n_theta
            + (l - 1) * self.n_theta
            + i_theta
    }
    fn cell_center(&self, element_index: usize) -> usize {
        self.n_q1_nodes()
            + 2 * self.n_x * (self.n_x + 1)
            + 2 * self.n_rho * self.n_theta
            + element_index
    }

    //Vertex of the inner-square boundary at angular position i_theta
    //(same numbering as the inner vertices of the transition layer).
    fn square_boundary_vertex(&self, i_theta: usize) -> usize {
        let i_quarter = i_theta / self.n_x;
        let i_x = i_theta % self.n_x;
        match i_quarter {
            0 => i_theta,
            1 => (self.n_x + 1) * i_x + self.n_x,
            2 => self.n_vert_square - 1 - i_x,
            3 => self.n_x * (self.n_x + 1) - i_x * (self.n_x + 1),
            _ => unreachable!(),
        }
    }

    //Midpoint of the inner-square boundary edge between the boundary
    //vertices at angular positions i_theta and i_theta+1.
    fn square_boundary_mid(&self, i_theta: usize) -> usize {
        let i_quarter = i_theta / self.n_x;
        let i_x = i_theta % self.n_x;
        match i_quarter {
            0 => self.sq_hmid(i_x, 0),
            1 => self.sq_vmid(self.n_x, i_x),
            2 => self.sq_hmid(self.n_x - 1 - i_x, self.n_x),
            3 => self.sq_vmid(0, self.n_x - 1 - i_x),
            _ => unreachable!(),
        }
    }

    //Vertex of ring l (1-indexed) at angular position i_theta.
    fn ring_vertex(&self, l: usize, i_theta: usize) -> usize {
        self.n_vert_square + (l - 1) * self.n_theta + i_theta
    }

    //Q1 corner vertices of a volume element (same numbering as get_element).
    fn cell_vertices(&self, element_index: usize) -> [usize; 4] {
        if element_index < self.n_square_elem {
            let ix = element_index % self.n_x;
            let iy = element_index / self.n_x;
            [
                iy * (self.n_x + 1) + ix,
                iy * (self.n_x + 1) + ix + 1,
                (iy + 1) * (self.n_x + 1) + ix + 1,
                (iy + 1) * (self.n_x + 1) + ix,
            ]
        } else if element_index < self.n_square_elem + self.n_theta {
            let i_theta = element_index - self.n_square_elem;
            [
                self.square_boundary_vertex(i_theta),
                self.ring_vertex(1, i_theta),
                self.ring_vertex(1, (i_theta + 1) % self.n_theta),
                self.square_boundary_vertex((i_theta + 1) % self.n_theta),
            ]
        } else {
            let i_rel = element_index - self.n_square_elem;
            let i_theta = i_rel % self.n_theta;
            let i_rho = i_rel / self.n_theta - 1;
            [
                self.ring_vertex(i_rho + 1, i_theta),
                self.ring_vertex(i_rho + 2, i_theta),
                self.ring_vertex(i_rho + 2, (i_theta + 1) % self.n_theta),
                self.ring_vertex(i_rho + 1, (i_theta + 1) % self.n_theta),
            ]
        }
    }

    //Image under circle_map of the reference-domain midpoint of the segment
    //between two Q1 vertices.
    fn edge_mid_node(&self, a: usize, b: usize) -> [f64; 3] {
        let ra = self.get_node_ref(a);
        let rb = self.get_node_ref(b);
        self.circle_map([
            0.5 * (ra[0] + rb[0]),
            0.5 * (ra[1] + rb[1]),
            0.5 * (ra[2] + rb[2]),
        ])
    }

    // Reference coordinates in arm `i_quarter` at (t_x, t_rho) ∈ [0,1]².
    fn arm_ref_coord(&self, t_x: f64, t_rho: f64, i_quarter: usize) -> (f64, f64) {
        let r_ref = self.radius / std::f64::consts::SQRT_2;
        let a = self.alpha * r_ref;
        let w = a + t_rho * (r_ref - a);
        let s = 2.0 * t_x - 1.0;
        match i_quarter {
            0 => (s * w, -w),
            1 => (w, s * w),
            2 => (-s * w, w),
            3 => (-w, -s * w),
            _ => unreachable!(),
        }
    }

    // Position of a node in the reference (square+arms) domain, before circle_map.
    // The mesh geometry is fully defined by get_node = circle_map ∘ get_node_ref.
    fn get_node_ref(&self, index: usize) -> [f64; 3] {
        let r_ref = self.radius / std::f64::consts::SQRT_2;
        let a = self.alpha * r_ref;
        let (ux, uy) = if index < self.n_vert_square {
            let ix = index % (self.n_x + 1);
            let iy = index / (self.n_x + 1);
            let x = -a + ix as f64 * 2.0 * a / self.n_x as f64;
            let y = -a + iy as f64 * 2.0 * a / self.n_x as f64;
            (x, y)
        } else {
            let i_rel = index - self.n_vert_square;
            let i_theta = i_rel % self.n_theta;
            let i_rho = i_rel / self.n_theta + 1; // 1-indexed
            let i_quarter = i_theta / self.n_x;
            let i_x = i_theta % self.n_x;
            // The arm map is affine along each center ray, so uniform t_rho
            // gives uniformly spaced ring layers.
            let t_rho = i_rho as f64 / self.n_rho as f64;
            let t_x = i_x as f64 / self.n_x as f64;
            self.arm_ref_coord(t_x, t_rho, i_quarter)
        };
        [ux + self.center[0], uy + self.center[1], 0.0]
    }

    // Applies the circle map (reference square → disk) to an absolute reference coordinate.
    fn circle_map(&self, xy: [f64; 3]) -> [f64; 3] {
        let p = [self.radius, self.center[0], self.center[1], self.alpha];
        let mut out = [0.0; 3];
        match circle_selector(&p, &xy) {
            0 => circle_inner(&p, &xy, &mut out),
            1 => circle_bottom(&p, &xy, &mut out),
            2 => circle_right(&p, &xy, &mut out),
            3 => circle_top(&p, &xy, &mut out),
            4 => circle_left(&p, &xy, &mut out),
            _ => unreachable!(),
        }
        out
    }
}

impl Mesh for Circle {
    fn get_deformation(&self) -> Option<&Diffeomorphism> {
        self.deformation.as_ref()
    }

    fn get_nnodes(&self) -> usize {
        if self.q2 {
            //Q1 vertices + square edge midpoints + radial and tangential
            //edge midpoints + cell centers
            self.n_q1_nodes()
                + 2 * self.n_x * (self.n_x + 1)
                + 2 * self.n_rho * self.n_theta
                + self.n_square_elem
                + self.n_rho * self.n_theta
        } else {
            self.n_q1_nodes()
        }
    }

    fn get_node(&self, index: usize) -> [f64; 3] {
        debug_assert!(index < self.get_nnodes(), "Node index out of bounds");

        if self.q2 && index >= self.n_q1_nodes() {
            //Q2 geometric node: image under circle_map of the
            //reference-domain midpoint of an edge (or center of a cell).
            let rel = index - self.n_q1_nodes();
            let n_sq_h = self.n_x * (self.n_x + 1);
            let n_rad = self.n_rho * self.n_theta;

            if rel < n_sq_h {
                //square horizontal-edge midpoint
                let ix = rel % self.n_x;
                let iy = rel / self.n_x;
                let a = iy * (self.n_x + 1) + ix;
                self.edge_mid_node(a, a + 1)
            } else if rel < 2 * n_sq_h {
                //square vertical-edge midpoint
                let rel = rel - n_sq_h;
                let ix = rel % (self.n_x + 1);
                let iy = rel / (self.n_x + 1);
                let a = iy * (self.n_x + 1) + ix;
                let b = (iy + 1) * (self.n_x + 1) + ix;
                self.edge_mid_node(a, b)
            } else if rel < 2 * n_sq_h + n_rad {
                //radial-edge midpoint (layer 0: square boundary -> ring 1)
                let rel = rel - 2 * n_sq_h;
                let i_theta = rel % self.n_theta;
                let r = rel / self.n_theta;
                let a = if r == 0 {
                    self.square_boundary_vertex(i_theta)
                } else {
                    self.ring_vertex(r, i_theta)
                };
                let b = self.ring_vertex(r + 1, i_theta);
                self.edge_mid_node(a, b)
            } else if rel < 2 * n_sq_h + 2 * n_rad {
                //tangential ring-edge midpoint
                let rel = rel - 2 * n_sq_h - n_rad;
                let i_theta = rel % self.n_theta;
                let l = rel / self.n_theta + 1;
                let a = self.ring_vertex(l, i_theta);
                let b = self.ring_vertex(l, (i_theta + 1) % self.n_theta);
                self.edge_mid_node(a, b)
            } else {
                //cell center
                let e = rel - 2 * n_sq_h - 2 * n_rad;
                let corners = self.cell_vertices(e);
                let refs: Vec<[f64; 3]> = corners.iter().map(|&v| self.get_node_ref(v)).collect();
                self.circle_map([
                    0.25 * (refs[0][0] + refs[1][0] + refs[2][0] + refs[3][0]),
                    0.25 * (refs[0][1] + refs[1][1] + refs[2][1] + refs[3][1]),
                    0.0,
                ])
            }
        } else {
            self.circle_map(self.get_node_ref(index))
        }
    }

    fn get_nelements(&self) -> usize {
        self.n_square_elem + self.n_rho * self.n_theta + self.n_theta
    }

    fn get_element(&self, element_index: usize) -> MeshElement {
        debug_assert!(element_index < self.get_nelements());
        let n_vol = self.n_square_elem + self.n_rho * self.n_theta;

        if self.q2 {
            //Same elements and element numbering as the Q1 mesh, with the Q2
            //geometric nodes added: QUADRANGLE9 for the cells (midpoints in
            //edge order m01, m12, m32, m03, then the center) and EDGE3 for
            //the boundary edges (midpoint last).
            return if element_index < n_vol {
                let [v0, v1, v2, v3] = self.cell_vertices(element_index);

                let midpoints = if element_index < self.n_square_elem {
                    let ix = element_index % self.n_x;
                    let iy = element_index / self.n_x;
                    [
                        self.sq_hmid(ix, iy),
                        self.sq_vmid(ix + 1, iy),
                        self.sq_hmid(ix, iy + 1),
                        self.sq_vmid(ix, iy),
                    ]
                } else if element_index < self.n_square_elem + self.n_theta {
                    let i_theta = element_index - self.n_square_elem;
                    [
                        self.rad_mid(0, i_theta),
                        self.tan_mid(1, i_theta),
                        self.rad_mid(0, (i_theta + 1) % self.n_theta),
                        self.square_boundary_mid(i_theta),
                    ]
                } else {
                    let i_rel = element_index - self.n_square_elem;
                    let i_theta = i_rel % self.n_theta;
                    let i_rho = i_rel / self.n_theta - 1;
                    [
                        self.rad_mid(i_rho + 1, i_theta),
                        self.tan_mid(i_rho + 2, i_theta),
                        self.rad_mid(i_rho + 1, (i_theta + 1) % self.n_theta),
                        self.tan_mid(i_rho + 1, i_theta),
                    ]
                };

                MeshElement {
                    geometric_element: &QUADRANGLE9,
                    label: 0,
                    indexes: vec![
                        v0,
                        v1,
                        v2,
                        v3,
                        midpoints[0],
                        midpoints[1],
                        midpoints[2],
                        midpoints[3],
                        self.cell_center(element_index),
                    ],
                }
            } else {
                let i_theta = element_index - n_vol;
                MeshElement {
                    geometric_element: &EDGE3,
                    label: 0,
                    indexes: vec![
                        self.ring_vertex(self.n_rho, i_theta),
                        self.ring_vertex(self.n_rho, (i_theta + 1) % self.n_theta),
                        self.tan_mid(self.n_rho, i_theta),
                    ],
                }
            };
        }

        if element_index < self.n_square_elem {
            let ix = element_index % self.n_x;
            let iy = element_index / self.n_x;
            MeshElement {
                geometric_element: &QUADRANGLE4,
                label: 0,
                indexes: vec![
                    iy * (self.n_x + 1) + ix,
                    iy * (self.n_x + 1) + ix + 1,
                    (iy + 1) * (self.n_x + 1) + ix + 1,
                    (iy + 1) * (self.n_x + 1) + ix,
                ],
            }
        } else if element_index < self.n_square_elem + self.n_theta {
            // Transition layer: connects inner square boundary to first arm layer.
            let i_rel = element_index - self.n_square_elem;
            let i_quarter = i_rel / self.n_x;
            let i_x = i_rel % self.n_x;
            let i_theta = i_rel;
            let (inner_0, inner_1) = match i_quarter {
                0 => (i_theta, i_theta + 1),
                1 => (
                    (self.n_x + 1) * i_x + self.n_x,
                    (self.n_x + 1) * (i_x + 1) + self.n_x,
                ),
                2 => (self.n_vert_square - 1 - i_x, self.n_vert_square - 2 - i_x),
                3 => (
                    self.n_x * (self.n_x + 1) - i_x * (self.n_x + 1),
                    self.n_x * (self.n_x + 1) - (i_x + 1) * (self.n_x + 1),
                ),
                _ => unreachable!(),
            };
            let outer_0 = self.n_vert_square + i_theta;
            let outer_1 = self.n_vert_square + (i_theta + 1) % self.n_theta;
            MeshElement {
                geometric_element: &QUADRANGLE4,
                label: 0,
                indexes: vec![inner_0, outer_0, outer_1, inner_1],
            }
        } else if element_index < n_vol {
            let i_rel = element_index - self.n_square_elem;
            let i_theta = i_rel % self.n_theta;
            let i_rho = i_rel / self.n_theta - 1;
            MeshElement {
                geometric_element: &QUADRANGLE4,
                label: 0,
                indexes: vec![
                    self.n_vert_square + i_rho * self.n_theta + i_theta,
                    self.n_vert_square + (i_rho + 1) * self.n_theta + i_theta,
                    self.n_vert_square + (i_rho + 1) * self.n_theta + (i_theta + 1) % self.n_theta,
                    self.n_vert_square + i_rho * self.n_theta + (i_theta + 1) % self.n_theta,
                ],
            }
        } else {
            let i_theta = element_index - n_vol;
            MeshElement {
                geometric_element: &EDGE2,
                label: 0,
                indexes: vec![
                    self.n_vert_square + (self.n_rho - 1) * self.n_theta + i_theta,
                    self.n_vert_square
                        + (self.n_rho - 1) * self.n_theta
                        + (i_theta + 1) % self.n_theta,
                ],
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
        let n_vol = self.n_square_elem + self.n_rho * self.n_theta;
        match dim {
            2 => (0..n_vol).collect(),
            1 => (n_vol..n_vol + self.n_theta).collect(),
            _ => vec![],
        }
    }

    fn get_elements_by_dimension_and_label(&self, label: usize, dim: usize) -> Vec<usize> {
        match (dim, label) {
            (2, 0) | (1, 0) => self.get_elements_by_dimension(dim),
            _ => vec![],
        }
    }

    // Writes the Jacobian of the analytic element map F = circle_map ∘ Q
    // (composed with the attached deformation if any), where Q is the
    // bilinear map from the reference element to the element's cell in the
    // reference (square+arms) domain, through the pre-circle_map corners.
    // J = D(circle_map)|_{Q(uvw)} · J_Q, row-major: 2×2 for the cells,
    // 2×1 (the tangent) for the boundary edges.
    fn get_element_grad_def(&self, element_index: usize, uvw: &[f64], grad_def: &mut [f64]) {
        debug_assert!(element_index < self.get_nelements());
        let n_vol = self.n_square_elem + self.n_rho * self.n_theta;

        // Branch of the circle map the element lives in (0 = inner square,
        // 1..=4 = bottom/right/top/left arm). Decided from the element index,
        // NOT from circle_selector: on the region interfaces (square boundary,
        // quarter diagonals) the map is only C0 and the element must use the
        // one-sided derivative of its own region.
        let region = if element_index < self.n_square_elem {
            0
        } else if element_index < n_vol {
            1 + ((element_index - self.n_square_elem) % self.n_theta) / self.n_x
        } else {
            1 + (element_index - n_vol) / self.n_x
        };

        let element = self.get_element(element_index);
        //Only the corner vertices define the bilinear map Q (with the q2
        //option the element also carries midpoint/center nodes).
        let nvertexes = element.geometric_element.reference_element.get_nvertexes();
        let corners: Vec<[f64; 3]> = element.indexes[..nvertexes]
            .iter()
            .map(|&i| self.get_node_ref(i))
            .collect();

        // Q(uvw) and its derivatives.
        let (q, dq_du, dq_dv) = if corners.len() == 4 {
            let (u, v) = (uvw[0], uvw[1]);
            let [c0, c1, c2, c3] = [corners[0], corners[1], corners[2], corners[3]];
            let q = [
                (1.0 - u) * (1.0 - v) * c0[0]
                    + u * (1.0 - v) * c1[0]
                    + u * v * c2[0]
                    + (1.0 - u) * v * c3[0],
                (1.0 - u) * (1.0 - v) * c0[1]
                    + u * (1.0 - v) * c1[1]
                    + u * v * c2[1]
                    + (1.0 - u) * v * c3[1],
                0.0,
            ];
            let dq_du = [
                (1.0 - v) * (c1[0] - c0[0]) + v * (c2[0] - c3[0]),
                (1.0 - v) * (c1[1] - c0[1]) + v * (c2[1] - c3[1]),
            ];
            let dq_dv = [
                (1.0 - u) * (c3[0] - c0[0]) + u * (c2[0] - c1[0]),
                (1.0 - u) * (c3[1] - c0[1]) + u * (c2[1] - c1[1]),
            ];
            (q, dq_du, dq_dv)
        } else {
            // Boundary EDGE2 element: linear map, single reference direction.
            let u = uvw[0];
            let [c0, c1] = [corners[0], corners[1]];
            let q = [
                (1.0 - u) * c0[0] + u * c1[0],
                (1.0 - u) * c0[1] + u * c1[1],
                0.0,
            ];
            let dq_du = [c1[0] - c0[0], c1[1] - c0[1]];
            (q, dq_du, [0.0, 0.0])
        };

        // Jacobian of the circle map at Q(uvw).
        let p = [self.radius, self.center[0], self.center[1], self.alpha];
        let mut d = [0.0; 4];
        match region {
            0 => circle_inner_grad(&p, &q, &mut d),
            1 => circle_bottom_grad(&p, &q, &mut d),
            2 => circle_right_grad(&p, &q, &mut d),
            3 => circle_top_grad(&p, &q, &mut d),
            4 => circle_left_grad(&p, &q, &mut d),
            _ => unreachable!(),
        }

        // J = D · [dq_du | dq_dv], row-major: 2×2 for the cells, 2×1 (the
        // tangent) for the boundary edges.
        let elem_dim;
        if corners.len() == 4 {
            elem_dim = 2;
            grad_def[0] = d[0] * dq_du[0] + d[1] * dq_du[1];
            grad_def[1] = d[0] * dq_dv[0] + d[1] * dq_dv[1];
            grad_def[2] = d[2] * dq_du[0] + d[3] * dq_du[1];
            grad_def[3] = d[2] * dq_dv[0] + d[3] * dq_dv[1];
        } else {
            elem_dim = 1;
            grad_def[0] = d[0] * dq_du[0] + d[1] * dq_du[1];
            grad_def[1] = d[2] * dq_du[0] + d[3] * dq_du[1];
        }

        //Chain rule with the attached deformation: grad_def ← Df(xyz)·grad_def,
        //where xyz = circle_map(Q(uvw)) is the physical (disk) image of uvw.
        if let Some(deformation) = &self.deformation {
            deformation.compose_grad_def(&self.circle_map(q), grad_def, elem_dim);
        }
    }

    fn get_ncolors(&self) -> usize {
        4
    }

    fn get_nelements_by_color(&self, color_index: usize) -> usize {
        self.elements_by_color[color_index].len()
    }

    fn get_element_color(&self, element_index: usize) -> usize {
        debug_assert!(
            element_index < self.coloring.len(),
            "Colors are only defined for elements of the highest topological dimension"
        );
        self.coloring[element_index]
    }

    fn get_element_index_by_color_to_element_index(
        &self,
        color_index: usize,
        element_index_by_color: usize,
    ) -> usize {
        self.elements_by_color[color_index][element_index_by_color]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Finite-difference validation of get_element_grad_def: the analytic
    // element map is F(uvw) = circle_map(Q(uvw)) with Q the bilinear map
    // through the element corners in the reference domain.
    #[test]
    fn grad_def_matches_finite_differences() {
        let h = 1e-6;
        for n in [3_usize, 6, 7, 10] {
            let c = Circle::new([0.5, -0.3], 2.0, n, None, None);

            for el in 0..c.get_nelements() {
                let element = c.get_element(el);
                let corners: Vec<[f64; 3]> =
                    element.indexes.iter().map(|&i| c.get_node_ref(i)).collect();

                // The analytic element map (same construction as
                // get_element_grad_def, evaluated instead of differentiated).
                let f = |u: f64, v: f64| -> [f64; 3] {
                    let q = if corners.len() == 4 {
                        [
                            (1.0 - u) * (1.0 - v) * corners[0][0]
                                + u * (1.0 - v) * corners[1][0]
                                + u * v * corners[2][0]
                                + (1.0 - u) * v * corners[3][0],
                            (1.0 - u) * (1.0 - v) * corners[0][1]
                                + u * (1.0 - v) * corners[1][1]
                                + u * v * corners[2][1]
                                + (1.0 - u) * v * corners[3][1],
                            0.0,
                        ]
                    } else {
                        [
                            (1.0 - u) * corners[0][0] + u * corners[1][0],
                            (1.0 - u) * corners[0][1] + u * corners[1][1],
                            0.0,
                        ]
                    };
                    c.circle_map(q)
                };

                // Interior points (the FD stencil must stay inside the element,
                // where the analytic map is smooth). The Jacobian is 2×2 for
                // the cells and 2×1 (the tangent) for the boundary edges.
                for &(u, v) in &[(0.3, 0.7), (0.5, 0.5), (0.05, 0.9)] {
                    let mut grad = [0.0; 4];
                    c.get_element_grad_def(el, &[u, v, 0.0], &mut grad);

                    let (fu1, fu0) = (f(u + h, v), f(u - h, v));
                    let fd: Vec<f64> = if corners.len() == 4 {
                        let (fv1, fv0) = (f(u, v + h), f(u, v - h));
                        vec![
                            (fu1[0] - fu0[0]) / (2.0 * h),
                            (fv1[0] - fv0[0]) / (2.0 * h),
                            (fu1[1] - fu0[1]) / (2.0 * h),
                            (fv1[1] - fv0[1]) / (2.0 * h),
                        ]
                    } else {
                        vec![(fu1[0] - fu0[0]) / (2.0 * h), (fu1[1] - fu0[1]) / (2.0 * h)]
                    };

                    for (k, fd_k) in fd.iter().enumerate() {
                        assert!(
                            (grad[k] - fd_k).abs() < 1e-5 * (1.0 + fd_k.abs()),
                            "n={} element {} at ({},{}): grad[{}]={} but FD gives {}",
                            n,
                            el,
                            u,
                            v,
                            k,
                            grad[k],
                            fd_k
                        );
                    }
                }
            }
        }
    }

    // On the outer boundary edges, the u-column of the Jacobian is tangent to
    // the circle: its radial component at the mapped point vanishes.
    #[test]
    fn grad_def_boundary_edges_tangent() {
        let center = [0.4, -0.2];
        let c = Circle::new(center, 2.0, 7, None, None);
        let n_vol = c.n_square_elem + c.n_rho * c.n_theta;
        for el in n_vol..c.get_nelements() {
            let element = c.get_element(el);
            let c0 = c.get_node_ref(element.indexes[0]);
            let c1 = c.get_node_ref(element.indexes[1]);
            for &u in &[0.0, 0.3, 0.5, 0.8, 1.0] {
                let q = [
                    (1.0 - u) * c0[0] + u * c1[0],
                    (1.0 - u) * c0[1] + u * c1[1],
                    0.0,
                ];
                let f = c.circle_map(q);
                let (rx, ry) = (f[0] - center[0], f[1] - center[1]);
                // The mapped point lies on the boundary circle
                assert!((rx.hypot(ry) - c.radius).abs() < 1e-12);

                //Boundary edges yield the 2×1 tangent [∂x/∂u, ∂y/∂u].
                let mut grad = [0.0; 2];
                c.get_element_grad_def(el, &[u, 0.0, 0.0], &mut grad);
                let (tx, ty) = (grad[0], grad[1]);
                let radial = (rx * tx + ry * ty) / (c.radius * tx.hypot(ty));
                assert!(
                    radial.abs() < 1e-10,
                    "el {} at u={}: radial component {}",
                    el,
                    u,
                    radial
                );
            }
        }
    }
}
