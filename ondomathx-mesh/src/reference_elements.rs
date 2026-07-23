use std::f64::consts::SQRT_2;

const SQRT_3: f64 = 1.7320508075688772;

use crate::eq;
use crate::gt;
use crate::leq;
use crate::vec_diff;
use crate::vec_eq;
use crate::vec_sp;

/**********************************************************************
 *   Vertex
 *
************************************************************************/

pub const VERTEX_TOPOLOGY: [usize; 4] = [
    0, // Number of faces
    0, // Number of edges
    1, // Number of vertexes
    0, // Dimension
];

pub const VERTEX_FACES_TOPOLOGY: [[usize; 4]; 0] = [];

pub const VERTEX_EDGES_TOPOLOGY: [[usize; 2]; 0] = [];

pub const VERTEX_VERTEXES: [[f64; 3]; 1] = [[0.0, 0.0, 0.0]];

pub const VERTEX_NORMALS: [[f64; 3]; 0] = [];

/**********************************************************************
 *   Edge
 *
 *   0----------1
 *
************************************************************************/

pub const EDGE_TOPOLOGY: [usize; 4] = [
    0, // Number of faces
    1, // Number of edges
    2, // Number of vertexes
    1, // Dimension
];

pub const EDGE_FACES_TOPOLOGY: [[usize; 4]; 0] = [];

pub const EDGE_EDGES_TOPOLOGY: [[usize; 2]; 1] = [[0, 1]];

pub const EDGE_VERTEXES: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

pub const EDGE_NORMALS: [[f64; 3]; 0] = [];

/**********************************************************************
* Triangle
*
*      2                       o
*      | \                     | \
*      |   \                   |   \
*      |     \                 2     1
*      |       \               |       \                             y
*      |         \             |         \                           |
*      0-----------1           o-----0-----o                         o--x
*
*
***********************************************************************/

pub const TRIANGLE_TOPOLOGY: [usize; 4] = [
    1, // Number of faces
    3, // Number of edges
    3, // Number of vertexes
    2, // Dimension
];

pub const TRIANGLE_EDGES_TOPOLOGY: [[usize; 2]; 3] = [[0, 1], [1, 2], [0, 2]];

pub const TRIANGLE_FACES_TOPOLOGY: [[usize; 4]; 1] = [[0, 1, 2, 2]];

pub const TRIANGLE_VERTEXES: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

pub const TRIANGLE_NORMALS: [[f64; 3]; 3] = [
    [0.0, -1.0, 0.0],
    [SQRT_2 / 2.0, SQRT_2 / 2.0, 0.0],
    [-1.0, 0.0, 0.0],
];

/************************************************************************
 * Quadrangle
 *
 *
 *   3-----------2         o-----2-----o
 *   |           |         |           |
 *   |           |         |           |
 *   |           |         3           1
 *   |           |         |           |                             y
 *   |           |         |           |                             |
 *   0-----------1         o-----0-----o                             o--x
 *
 *
 *************************************************************************/

pub const QUADRANGLE_TOPOLOGY: [usize; 4] = [
    1, // Number of faces
    4, // Number of edges
    4, // Number of vertexes
    2, // Dimension
];

pub const QUADRANGLE_EDGES_TOPOLOGY: [[usize; 2]; 4] = [[0, 1], [1, 2], [3, 2], [0, 3]];

pub const QUADRANGLE_FACES_TOPOLOGY: [[usize; 4]; 1] = [[0, 1, 2, 3]];

pub const QUADRANGLE_VERTEXES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
];

pub const QUADRANGLE_NORMALS: [[f64; 3]; 4] = [
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
];

/************************************************************************
* Hexahedron
*
*        7----------6              o----10----o              o----------o
*       /|         /|             /|         /|             /|         /|
*      / |        / |           11 7        9 6            / |  5     / |
*     /  |       /  |           /  |       /  |           /  |    4  /  |
*    4---+------5   |          o---+-8----o   |          o-2-+------o 3 |
*    |   3------+---2          |   o----2-+---o          |   o------+---o
*    |  /       |  /           4  /       5  /           |  /  1    |  /
*    | /        | /            | 3        | 1            | /    0   | /
*    |/         |/             |/         |/             |/         |/
*    0----------1              o-----0----o              o----------o
*
*
*
*
 *************************************************************************/

pub const HEXAHEDRON_TOPOLOGY: [usize; 4] = [
    6,  // Number of faces
    12, // Number of edges
    8,  // Number of vertexes
    3,  // Dimension
];

pub const HEXAHEDRON_EDGES_TOPOLOGY: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [3, 2],
    [0, 3],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    [4, 5],
    [5, 6],
    [7, 6],
    [4, 7],
];

pub const HEXAHEDRON_FACES_TOPOLOGY: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [3, 2, 6, 7],
    [4, 5, 6, 7],
];

pub const HEXAHEDRON_VERTEXES: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

pub const HEXAHEDRON_NORMALS: [[f64; 3]; 6] = [
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/**********************************************************************************
* Tetrahedron
 *
 *     3                            o                           o
 *     |\\                          |\\                         |\\
 *     | \ \                        | \ \                       | \ \
 *     |  \  \                      |  6  \                     |  \  \
 *     |   \   \                    |   \   \                   |   \   \
 *     |     2   \                  4     o   5                 | 2   o  3\
 *     |   /    \  \                |   /    \  \               |   /1   \  \
 *     |  /       \  \              |  3       2  \             |  /   0   \  \
 *     | /          \  \            | /          \  \           | /          \  \
 *     0----------------1           o-------1--------o          o----------------o
 *
 *
 *********************************************************************************/

pub const TETRAHEDRON_TOPOLOGY: [usize; 4] = [
    4, // Number of faces
    6, // Number of edges
    4, // Number of vertexes
    3, // Dimension
];

pub const TETRAHEDRON_FACES_TOPOLOGY: [[usize; 4]; 4] =
    [[0, 1, 2, 2], [0, 1, 3, 3], [0, 2, 3, 3], [1, 2, 3, 3]];

pub const TETRAHEDRON_EDGES_TOPOLOGY: [[usize; 2]; 6] =
    [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]];

pub const TETRAHEDRON_VERTEXES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

pub const TETRAHEDRON_NORMALS: [[f64; 3]; 4] = [
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [SQRT_3 / 3.0, SQRT_3 / 3.0, SQRT_3 / 3.0],
];

//////////////////////////////////////////////////////////////////////////

pub const ORIENTATIONS_TRIANGLE: [[usize; 3]; 6] = [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
    [0, 2, 1],
    [2, 1, 0],
    [1, 0, 2],
];

pub const ORIENTATIONS_QUADRANGLE: [[usize; 4]; 8] = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
    [0, 3, 2, 1],
    [3, 2, 1, 0],
    [2, 1, 0, 3],
    [1, 0, 3, 2],
];

//////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct ReferenceElement {
    pub topology: &'static [usize],
    pub faces: &'static [[usize; 4]],
    pub edges: &'static [[usize; 2]],
    pub vertexes: &'static [[f64; 3]],
    pub normals: &'static [[f64; 3]],
}

pub const VERTEX: ReferenceElement = ReferenceElement {
    topology: &VERTEX_TOPOLOGY,
    faces: &VERTEX_FACES_TOPOLOGY,
    edges: &VERTEX_EDGES_TOPOLOGY,
    vertexes: &VERTEX_VERTEXES,
    normals: &VERTEX_NORMALS,
};

pub const EDGE: ReferenceElement = ReferenceElement {
    topology: &EDGE_TOPOLOGY,
    faces: &EDGE_FACES_TOPOLOGY,
    edges: &EDGE_EDGES_TOPOLOGY,
    vertexes: &EDGE_VERTEXES,
    normals: &EDGE_NORMALS,
};

pub const TRIANGLE: ReferenceElement = ReferenceElement {
    topology: &TRIANGLE_TOPOLOGY,
    faces: &TRIANGLE_FACES_TOPOLOGY,
    edges: &TRIANGLE_EDGES_TOPOLOGY,
    vertexes: &TRIANGLE_VERTEXES,
    normals: &TRIANGLE_NORMALS,
};

pub const QUADRANGLE: ReferenceElement = ReferenceElement {
    topology: &QUADRANGLE_TOPOLOGY,
    faces: &QUADRANGLE_FACES_TOPOLOGY,
    edges: &QUADRANGLE_EDGES_TOPOLOGY,
    vertexes: &QUADRANGLE_VERTEXES,
    normals: &QUADRANGLE_NORMALS,
};

pub const HEXAHEDRON: ReferenceElement = ReferenceElement {
    topology: &HEXAHEDRON_TOPOLOGY,
    faces: &HEXAHEDRON_FACES_TOPOLOGY,
    edges: &HEXAHEDRON_EDGES_TOPOLOGY,
    vertexes: &HEXAHEDRON_VERTEXES,
    normals: &HEXAHEDRON_NORMALS,
};

pub const TETRAHEDRON: ReferenceElement = ReferenceElement {
    topology: &TETRAHEDRON_TOPOLOGY,
    faces: &TETRAHEDRON_FACES_TOPOLOGY,
    edges: &TETRAHEDRON_EDGES_TOPOLOGY,
    vertexes: &TETRAHEDRON_VERTEXES,
    normals: &TETRAHEDRON_NORMALS,
};

//***************************************************************************

impl ReferenceElement {
    pub fn get_nfaces(&self) -> usize {
        self.topology[0]
    }

    pub fn get_nedges(&self) -> usize {
        self.topology[1]
    }

    pub fn get_nvertexes(&self) -> usize {
        self.topology[2]
    }

    pub fn get_dim(&self) -> usize {
        self.topology[3]
    }

    pub fn is_face_triangular(&self, f: usize) -> bool {
        self.faces[f][2] == self.faces[f][3]
    }

    pub fn is_inside(&self, uvw: &[f64]) -> bool {
        match self.get_dim() {
            1 => {
                assert!(self.topology == EDGE_TOPOLOGY);
                leq!(uvw[0], 1.0) && leq!(0.0, uvw[0]) && eq!(uvw[1], 0.0) && eq!(uvw[2], 0.0)
            }
            2 => {
                if !eq!(uvw[2], 0.0) {
                    return false;
                }

                for e in 0..self.get_nedges() {
                    let n = self.normals[e];

                    let a = vec_diff!(uvw, self.vertexes[self.edges[e][0]]);

                    if gt!(vec_sp!(n, a), 0.0) {
                        return false;
                    }
                }

                true
            }
            3 => {
                for f in 0..self.get_nfaces() {
                    let n: [f64; 3] = self.normals[f];
                    let a: [f64; 3] = vec_diff!(uvw, self.vertexes[self.faces[f][0]]);

                    if gt!(vec_sp!(n, a), 0.0) {
                        return false;
                    }
                }

                true
            }
            _ => {
                panic!("Elements should have a dimension less or equal 3");
            }
        }
    }

    pub fn is_on_edge(&self, uvw: &[f64], e: usize) -> bool {
        let a = vec_diff!(uvw, self.vertexes[self.edges[e][0]]);

        let b = vec_diff!(
            self.vertexes[self.edges[e][1]],
            self.vertexes[self.edges[e][0]]
        );

        let a_b = vec_sp!(a, b);
        let a_a = vec_sp!(a, a);
        let b_b = vec_sp!(b, b);

        eq!(a_a * b_b, a_b * a_b) && leq!(a_a, b_b)
    }

    pub fn is_on_face(&self, uvw: &[f64], f: usize) -> bool {
        match self.get_dim() {
            1 => {
                panic!("This function is not supported for dimension 1 elements");
            }
            2 => self.is_inside(uvw),
            3 => {
                let n = self.normals[f];
                let a = vec_diff!(uvw, self.vertexes[self.faces[f][0]]);

                if !eq!(vec_sp!(a, n), 0.0) {
                    return false;
                }

                self.is_inside(uvw)
            }

            _ => {
                panic!("Elements should have a dimension less or equal 3");
            }
        }
    }

    pub fn is_on_vertex(&self, uvw: &[f64], v: usize) -> bool {
        vec_eq!(uvw, self.vertexes[v])
    }

    pub fn is_on_any_vertex(&self, uvw: &[f64]) -> bool {
        for v in 0..self.get_nvertexes() {
            if self.is_on_vertex(uvw, v) {
                return true;
            }
        }

        false
    }

    pub fn is_on_any_edge(&self, uvw: &[f64]) -> bool {
        for v in 0..self.get_nedges() {
            if self.is_on_edge(uvw, v) {
                return true;
            }
        }

        false
    }

    pub fn is_on_any_face(&self, uvw: &[f64]) -> bool {
        for v in 0..self.get_nfaces() {
            if self.is_on_face(uvw, v) {
                return true;
            }
        }

        false
    }

    pub fn uvw_edge_transform(&self, uvw: &[f64], e: usize, o: usize) -> [f64; 3] {
        let mut uvw_o = [uvw[0], uvw[1], uvw[2]];

        assert!(
            e < self.get_nedges(),
            "The index of the edge is larger than the number of edges"
        );

        assert!(
            self.is_on_edge(uvw, e),
            "The argument uvw should represents a coordinate along the edge e"
        );

        assert!(o <= 1, "There is only two type of orientations for edges");

        if o == 1 {
            //Compute barycentric coordinate l1 and l2
            let a = vec_diff!(uvw, self.vertexes[self.edges[e][0]]);

            let b = vec_diff!(
                self.vertexes[self.edges[e][1]],
                self.vertexes[self.edges[e][0]]
            );

            // a = l1 b and l1 + l2 = 1
            let l1 = vec_sp!(a, b) / vec_sp!(b, b);
            let l2 = 1.0 - l1;

            uvw_o = [
                l2 * self.vertexes[self.edges[e][1]][0] + l1 * self.vertexes[self.edges[e][0]][0],
                l2 * self.vertexes[self.edges[e][1]][1] + l1 * self.vertexes[self.edges[e][0]][1],
                l2 * self.vertexes[self.edges[e][1]][2] + l1 * self.vertexes[self.edges[e][0]][2],
            ];
        }

        uvw_o
    }

    pub fn uvw_face_transform(&self, uvw: &[f64], f: usize, o: usize) -> [f64; 3] {
        assert!(
            f < self.get_nfaces(),
            "The index of the face is larger than the number of faces"
        );

        assert!(
            self.is_on_face(uvw, f),
            "The argument uvw should represents a coordinate on the face e"
        );

        //Compute barycentric coordinate l1 and l2
        let a = [
            uvw[0] - self.vertexes[self.faces[f][0]][0],
            uvw[1] - self.vertexes[self.faces[f][0]][1],
            uvw[2] - self.vertexes[self.faces[f][0]][2],
        ];

        let d1 = vec_diff!(
            self.vertexes[self.faces[f][1]],
            self.vertexes[self.faces[f][0]]
        );

        // /!\ we use here the convention that the fourth index is repeated for triangle
        let d2 = vec_diff!(
            self.vertexes[self.faces[f][3]],
            self.vertexes[self.faces[f][0]]
        );

        //  a = l1 d1 + l2 d2
        // let b = Matrix3x2::new(d1[0], d2[0], d1[1], d2[1], d1[2], d2[2]);

        // let mut btb = b.transpose() * b;

        // let is_invertible = btb.try_inverse_mut();

        // assert!(
        //     is_invertible,
        //     "Inversion to compute coordinate transformation has failed"
        // );

        // let l = btb * b.transpose() * a;

        let l: [f64; 2];
        crate::vec_lsq!(a, d1, d2, l);

        let d1_o: [f64; 3];
        let d2_o: [f64; 3];
        let origin: [f64; 3];

        if self.is_face_triangular(f) {
            origin = self.vertexes[self.faces[f][ORIENTATIONS_TRIANGLE[o][0]]];

            d1_o = vec_diff!(
                self.vertexes[self.faces[f][ORIENTATIONS_TRIANGLE[o][1]]],
                origin
            );

            d2_o = vec_diff!(
                self.vertexes[self.faces[f][ORIENTATIONS_TRIANGLE[o][2]]],
                origin
            );
        } else {
            origin = self.vertexes[self.faces[f][ORIENTATIONS_QUADRANGLE[o][0]]];

            d1_o = vec_diff!(
                self.vertexes[self.faces[f][ORIENTATIONS_QUADRANGLE[o][1]]],
                origin
            );

            d2_o = vec_diff!(
                self.vertexes[self.faces[f][ORIENTATIONS_QUADRANGLE[o][3]]],
                origin
            );
        }

        [
            origin[0] + l[0] * d1_o[0] + l[1] * d2_o[0],
            origin[1] + l[0] * d1_o[1] + l[1] * d2_o[1],
            origin[2] + l[0] * d1_o[2] + l[1] * d2_o[2],
        ]
    }
}

//////////////////////////////////////////////////////////////////////////
