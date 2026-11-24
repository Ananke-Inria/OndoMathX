use crate::reference_elements::*;

//A scalar shape function and its gradient, evaluated at reference coordinates
pub type ShapeFunction = fn(&[f64]) -> f64;
pub type DShapeFunction = fn(&[f64]) -> [f64; 3];

const P2: [fn(f64) -> f64; 3] = [
    |u| 2.0 * (0.5 - u) * (1.0 - u),
    |u| 2.0 * (u - 0.5) * u,
    |u| 4.0 * u * (1.0 - u),
];

const DP2: [fn(f64) -> f64; 3] = [|u| 4.0 * u - 3.0, |u| 4.0 * u - 1.0, |u| 4.0 - 8.0 * u];

const EDGE_2_SHAPE_FUNCTION_VALUE: [ShapeFunction; 2] = [|uvw| 1.0 - uvw[0], |uvw| uvw[0]];

const EDGE_2_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 2] =
    [|_uvw| [-1.0, 0.0, 0.0], |_uvw| [1.0, 0.0, 0.0]];

const EDGE_3_SHAPE_FUNCTION_VALUE: [ShapeFunction; 3] = [
    |uvw| P2[0](uvw[0]),
    |uvw| P2[1](uvw[0]),
    |uvw| P2[2](uvw[0]),
];

const EDGE_3_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 3] = [
    |uvw| [DP2[0](uvw[0]), 0.0, 0.0],
    |uvw| [DP2[1](uvw[0]), 0.0, 0.0],
    |uvw| [DP2[2](uvw[0]), 0.0, 0.0],
];

const TRIANGLE_3_SHAPE_FUNCTION_VALUE: [ShapeFunction; 3] =
    [|uvw| 1.0 - uvw[0] - uvw[1], |uvw| uvw[0], |uvw| uvw[1]];

const TRIANGLE_3_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 3] = [
    |_uvw| [-1.0, -1.0, 0.0],
    |_uvw| [1.0, 0.0, 0.0],
    |_uvw| [0.0, 1.0, 0.0],
];

const TRIANGLE_6_SHAPE_FUNCTION_VALUE: [ShapeFunction; 6] = [
    |uvw| {
        let l0 = 1.0 - uvw[0] - uvw[1];
        l0 * (2.0 * l0 - 1.0)
    },
    |uvw| uvw[0] * (2.0 * uvw[0] - 1.0),
    |uvw| uvw[1] * (2.0 * uvw[1] - 1.0),
    |uvw| 4.0 * (1.0 - uvw[0] - uvw[1]) * uvw[0],
    |uvw| 4.0 * uvw[0] * uvw[1],
    |uvw| 4.0 * (1.0 - uvw[0] - uvw[1]) * uvw[1],
];

const TRIANGLE_6_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 6] = [
    |uvw| {
        let g = 1.0 - 4.0 * (1.0 - uvw[0] - uvw[1]);
        [g, g, 0.0]
    },
    |uvw| [4.0 * uvw[0] - 1.0, 0.0, 0.0],
    |uvw| [0.0, 4.0 * uvw[1] - 1.0, 0.0],
    |uvw| [4.0 * (1.0 - 2.0 * uvw[0] - uvw[1]), -4.0 * uvw[0], 0.0],
    |uvw| [4.0 * uvw[1], 4.0 * uvw[0], 0.0],
    |uvw| [-4.0 * uvw[1], 4.0 * (1.0 - uvw[0] - 2.0 * uvw[1]), 0.0],
];

const TETRAHEDRON_4_SHAPE_FUNCTION_VALUE: [ShapeFunction; 4] = [
    |uvw| 1.0 - uvw[0] - uvw[1] - uvw[2],
    |uvw| uvw[0],
    |uvw| uvw[1],
    |uvw| uvw[2],
];

const TETRAHEDRON_4_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 4] = [
    |_uvw| [-1.0, -1.0, -1.0],
    |_uvw| [1.0, 0.0, 0.0],
    |_uvw| [0.0, 1.0, 0.0],
    |_uvw| [0.0, 0.0, 1.0],
];

const TETRAHEDRON_10_SHAPE_FUNCTION_VALUE: [ShapeFunction; 10] = [
    |uvw| {
        let l0 = 1.0 - uvw[0] - uvw[1] - uvw[2];
        l0 * (2.0 * l0 - 1.0)
    },
    |uvw| uvw[0] * (2.0 * uvw[0] - 1.0),
    |uvw| uvw[1] * (2.0 * uvw[1] - 1.0),
    |uvw| uvw[2] * (2.0 * uvw[2] - 1.0),
    |uvw| 4.0 * (1.0 - uvw[0] - uvw[1] - uvw[2]) * uvw[0],
    |uvw| 4.0 * uvw[0] * uvw[1],
    |uvw| 4.0 * (1.0 - uvw[0] - uvw[1] - uvw[2]) * uvw[1],
    |uvw| 4.0 * (1.0 - uvw[0] - uvw[1] - uvw[2]) * uvw[2],
    |uvw| 4.0 * uvw[0] * uvw[2],
    |uvw| 4.0 * uvw[1] * uvw[2],
];

const TETRAHEDRON_10_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 10] = [
    |uvw| {
        let g = 4.0 * (uvw[0] + uvw[1] + uvw[2]) - 3.0;
        [g, g, g]
    },
    |uvw| [4.0 * uvw[0] - 1.0, 0.0, 0.0],
    |uvw| [0.0, 4.0 * uvw[1] - 1.0, 0.0],
    |uvw| [0.0, 0.0, 4.0 * uvw[2] - 1.0],
    |uvw| {
        [
            4.0 * (1.0 - 2.0 * uvw[0] - uvw[1] - uvw[2]),
            -4.0 * uvw[0],
            -4.0 * uvw[0],
        ]
    },
    |uvw| [4.0 * uvw[1], 4.0 * uvw[0], 0.0],
    |uvw| {
        [
            -4.0 * uvw[1],
            4.0 * (1.0 - uvw[0] - 2.0 * uvw[1] - uvw[2]),
            -4.0 * uvw[1],
        ]
    },
    |uvw| {
        [
            -4.0 * uvw[2],
            -4.0 * uvw[2],
            4.0 * (1.0 - uvw[0] - uvw[1] - 2.0 * uvw[2]),
        ]
    },
    |uvw| [4.0 * uvw[2], 0.0, 4.0 * uvw[0]],
    |uvw| [0.0, 4.0 * uvw[2], 4.0 * uvw[1]],
];

const QUADRANGLE_4_SHAPE_FUNCTION_VALUE: [ShapeFunction; 4] = [
    |uvw| (1.0 - uvw[0]) * (1.0 - uvw[1]),
    |uvw| uvw[0] * (1.0 - uvw[1]),
    |uvw| uvw[0] * uvw[1],
    |uvw| (1.0 - uvw[0]) * uvw[1],
];

const QUADRANGLE_4_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 4] = [
    |uvw| [uvw[1] - 1.0, uvw[0] - 1.0, 0.0],
    |uvw| [1.0 - uvw[1], -uvw[0], 0.0],
    |uvw| [uvw[1], uvw[0], 0.0],
    |uvw| [-uvw[1], 1.0 - uvw[0], 0.0],
];

const QUADRANGLE_9_SHAPE_FUNCTION_VALUE: [ShapeFunction; 9] = [
    |uvw| P2[0](uvw[0]) * P2[0](uvw[1]),
    |uvw| P2[1](uvw[0]) * P2[0](uvw[1]),
    |uvw| P2[1](uvw[0]) * P2[1](uvw[1]),
    |uvw| P2[0](uvw[0]) * P2[1](uvw[1]),
    |uvw| P2[2](uvw[0]) * P2[0](uvw[1]),
    |uvw| P2[1](uvw[0]) * P2[2](uvw[1]),
    |uvw| P2[2](uvw[0]) * P2[1](uvw[1]),
    |uvw| P2[0](uvw[0]) * P2[2](uvw[1]),
    |uvw| P2[2](uvw[0]) * P2[2](uvw[1]),
];

const QUADRANGLE_9_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 9] = [
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[0](uvw[1]),
            P2[0](uvw[0]) * DP2[0](uvw[1]),
            0.0,
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[0](uvw[1]),
            P2[1](uvw[0]) * DP2[0](uvw[1]),
            0.0,
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[1](uvw[1]),
            P2[1](uvw[0]) * DP2[1](uvw[1]),
            0.0,
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[1](uvw[1]),
            P2[0](uvw[0]) * DP2[1](uvw[1]),
            0.0,
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[0](uvw[1]),
            P2[2](uvw[0]) * DP2[0](uvw[1]),
            0.0,
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[2](uvw[1]),
            P2[1](uvw[0]) * DP2[2](uvw[1]),
            0.0,
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[1](uvw[1]),
            P2[2](uvw[0]) * DP2[1](uvw[1]),
            0.0,
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[2](uvw[1]),
            P2[0](uvw[0]) * DP2[2](uvw[1]),
            0.0,
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[2](uvw[1]),
            P2[2](uvw[0]) * DP2[2](uvw[1]),
            0.0,
        ]
    },
];

#[derive(Debug)]
pub struct GeometricElement {
    pub reference_element: &'static ReferenceElement,
    pub nshape_functions: usize,
    pub shape_functions: &'static [ShapeFunction],
    pub d_shape_functions: &'static [DShapeFunction],
    pub msh_id: usize,
}

impl GeometricElement {
    /// Values and reference gradients of all shape functions at `points`
    /// (reference coordinates), for evaluating the element map at a fixed set
    /// of points (e.g. quadrature points) many times.
    ///
    /// Point-major, shape function fastest, in the canonical shape-function
    /// order: with `n = nshape_functions`,
    /// `values[p * n + k] = φ_k(points[p])` and
    /// `gradients[p * n + k] = ∇φ_k(points[p])` (trailing zeros below
    /// dimension 3).
    ///
    /// Panics if a point is outside the reference element.
    pub fn tabulate(&self, points: &[[f64; 3]]) -> (Vec<f64>, Vec<[f64; 3]>) {
        let n = self.nshape_functions;
        let mut values = Vec::with_capacity(points.len() * n);
        let mut gradients = Vec::with_capacity(points.len() * n);
        for uvw in points {
            assert!(
                self.reference_element.is_inside(uvw),
                "tabulation point {uvw:?} is outside the reference element"
            );
            for k in 0..n {
                values.push((self.shape_functions[k])(uvw));
                gradients.push((self.d_shape_functions[k])(uvw));
            }
        }
        (values, gradients)
    }
}

pub const EDGE2: GeometricElement = GeometricElement {
    reference_element: &EDGE,
    nshape_functions: 2,
    shape_functions: &EDGE_2_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &EDGE_2_SHAPE_FUNCTION_D_VALUE,
    msh_id: 1,
};

pub const EDGE3: GeometricElement = GeometricElement {
    reference_element: &EDGE,
    nshape_functions: 3,
    shape_functions: &EDGE_3_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &EDGE_3_SHAPE_FUNCTION_D_VALUE,
    msh_id: 8,
};

pub const TRIANGLE3: GeometricElement = GeometricElement {
    reference_element: &TRIANGLE,
    nshape_functions: 3,
    shape_functions: &TRIANGLE_3_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &TRIANGLE_3_SHAPE_FUNCTION_D_VALUE,
    msh_id: 2,
};

pub const TRIANGLE6: GeometricElement = GeometricElement {
    reference_element: &TRIANGLE,
    nshape_functions: 6,
    shape_functions: &TRIANGLE_6_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &TRIANGLE_6_SHAPE_FUNCTION_D_VALUE,
    msh_id: 9,
};

pub const QUADRANGLE4: GeometricElement = GeometricElement {
    reference_element: &QUADRANGLE,
    nshape_functions: 4,
    shape_functions: &QUADRANGLE_4_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &QUADRANGLE_4_SHAPE_FUNCTION_D_VALUE,
    msh_id: 3,
};

pub const QUADRANGLE9: GeometricElement = GeometricElement {
    reference_element: &QUADRANGLE,
    nshape_functions: 9,
    shape_functions: &QUADRANGLE_9_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &QUADRANGLE_9_SHAPE_FUNCTION_D_VALUE,
    msh_id: 10,
};

pub const TETRAHEDRON4: GeometricElement = GeometricElement {
    reference_element: &TETRAHEDRON,
    nshape_functions: 4,
    shape_functions: &TETRAHEDRON_4_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &TETRAHEDRON_4_SHAPE_FUNCTION_D_VALUE,
    msh_id: 4,
};

pub const TETRAHEDRON10: GeometricElement = GeometricElement {
    reference_element: &TETRAHEDRON,
    nshape_functions: 10,
    shape_functions: &TETRAHEDRON_10_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &TETRAHEDRON_10_SHAPE_FUNCTION_D_VALUE,
    msh_id: 11,
};

const HEXAHEDRON_8_SHAPE_FUNCTION_VALUE: [ShapeFunction; 8] = [
    |uvw| (1.0 - uvw[0]) * (1.0 - uvw[1]) * (1.0 - uvw[2]),
    |uvw| uvw[0] * (1.0 - uvw[1]) * (1.0 - uvw[2]),
    |uvw| uvw[0] * uvw[1] * (1.0 - uvw[2]),
    |uvw| (1.0 - uvw[0]) * uvw[1] * (1.0 - uvw[2]),
    |uvw| (1.0 - uvw[0]) * (1.0 - uvw[1]) * uvw[2],
    |uvw| uvw[0] * (1.0 - uvw[1]) * uvw[2],
    |uvw| uvw[0] * uvw[1] * uvw[2],
    |uvw| (1.0 - uvw[0]) * uvw[1] * uvw[2],
];

const HEXAHEDRON_8_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 8] = [
    |uvw| {
        [
            -(1.0 - uvw[1]) * (1.0 - uvw[2]),
            -(1.0 - uvw[0]) * (1.0 - uvw[2]),
            -(1.0 - uvw[0]) * (1.0 - uvw[1]),
        ]
    },
    |uvw| {
        [
            (1.0 - uvw[1]) * (1.0 - uvw[2]),
            -uvw[0] * (1.0 - uvw[2]),
            -uvw[0] * (1.0 - uvw[1]),
        ]
    },
    |uvw| {
        [
            uvw[1] * (1.0 - uvw[2]),
            uvw[0] * (1.0 - uvw[2]),
            -uvw[0] * uvw[1],
        ]
    },
    |uvw| {
        [
            -uvw[1] * (1.0 - uvw[2]),
            (1.0 - uvw[0]) * (1.0 - uvw[2]),
            -(1.0 - uvw[0]) * uvw[1],
        ]
    },
    |uvw| {
        [
            -(1.0 - uvw[1]) * uvw[2],
            -(1.0 - uvw[0]) * uvw[2],
            (1.0 - uvw[0]) * (1.0 - uvw[1]),
        ]
    },
    |uvw| {
        [
            (1.0 - uvw[1]) * uvw[2],
            -uvw[0] * uvw[2],
            uvw[0] * (1.0 - uvw[1]),
        ]
    },
    |uvw| [uvw[1] * uvw[2], uvw[0] * uvw[2], uvw[0] * uvw[1]],
    |uvw| {
        [
            -uvw[1] * uvw[2],
            (1.0 - uvw[0]) * uvw[2],
            (1.0 - uvw[0]) * uvw[1],
        ]
    },
];

pub const HEXAHEDRON8: GeometricElement = GeometricElement {
    reference_element: &HEXAHEDRON,
    nshape_functions: 8,
    shape_functions: &HEXAHEDRON_8_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &HEXAHEDRON_8_SHAPE_FUNCTION_D_VALUE,
    msh_id: 5,
};

// Full tensor-product Q2 hexahedron built from the 1D quadratic Lagrange
// polynomials P2 on [0,1].
// Node order follows reference_elements.rs: 8 vertices (HEXAHEDRON_VERTEXES),
// then the 12 edge midpoints in HEXAHEDRON_EDGES_TOPOLOGY order, then the
// 6 face centers in HEXAHEDRON_FACES_TOPOLOGY order, then the center.
// NB: this differs from Gmsh's hexahedron27 node ordering — a permutation is
// needed when wiring this element into the .msh reader/writer.
const HEXAHEDRON_27_SHAPE_FUNCTION_VALUE: [ShapeFunction; 27] = [
    |uvw| P2[0](uvw[0]) * P2[0](uvw[1]) * P2[0](uvw[2]),
    |uvw| P2[1](uvw[0]) * P2[0](uvw[1]) * P2[0](uvw[2]),
    |uvw| P2[1](uvw[0]) * P2[1](uvw[1]) * P2[0](uvw[2]),
    |uvw| P2[0](uvw[0]) * P2[1](uvw[1]) * P2[0](uvw[2]),
    |uvw| P2[0](uvw[0]) * P2[0](uvw[1]) * P2[1](uvw[2]),
    |uvw| P2[1](uvw[0]) * P2[0](uvw[1]) * P2[1](uvw[2]),
    |uvw| P2[1](uvw[0]) * P2[1](uvw[1]) * P2[1](uvw[2]),
    |uvw| P2[0](uvw[0]) * P2[1](uvw[1]) * P2[1](uvw[2]),
    |uvw| P2[2](uvw[0]) * P2[0](uvw[1]) * P2[0](uvw[2]),
    |uvw| P2[1](uvw[0]) * P2[2](uvw[1]) * P2[0](uvw[2]),
    |uvw| P2[2](uvw[0]) * P2[1](uvw[1]) * P2[0](uvw[2]),
    |uvw| P2[0](uvw[0]) * P2[2](uvw[1]) * P2[0](uvw[2]),
    |uvw| P2[0](uvw[0]) * P2[0](uvw[1]) * P2[2](uvw[2]),
    |uvw| P2[1](uvw[0]) * P2[0](uvw[1]) * P2[2](uvw[2]),
    |uvw| P2[1](uvw[0]) * P2[1](uvw[1]) * P2[2](uvw[2]),
    |uvw| P2[0](uvw[0]) * P2[1](uvw[1]) * P2[2](uvw[2]),
    |uvw| P2[2](uvw[0]) * P2[0](uvw[1]) * P2[1](uvw[2]),
    |uvw| P2[1](uvw[0]) * P2[2](uvw[1]) * P2[1](uvw[2]),
    |uvw| P2[2](uvw[0]) * P2[1](uvw[1]) * P2[1](uvw[2]),
    |uvw| P2[0](uvw[0]) * P2[2](uvw[1]) * P2[1](uvw[2]),
    |uvw| P2[2](uvw[0]) * P2[2](uvw[1]) * P2[0](uvw[2]),
    |uvw| P2[2](uvw[0]) * P2[0](uvw[1]) * P2[2](uvw[2]),
    |uvw| P2[0](uvw[0]) * P2[2](uvw[1]) * P2[2](uvw[2]),
    |uvw| P2[1](uvw[0]) * P2[2](uvw[1]) * P2[2](uvw[2]),
    |uvw| P2[2](uvw[0]) * P2[1](uvw[1]) * P2[2](uvw[2]),
    |uvw| P2[2](uvw[0]) * P2[2](uvw[1]) * P2[1](uvw[2]),
    |uvw| P2[2](uvw[0]) * P2[2](uvw[1]) * P2[2](uvw[2]),
];

const HEXAHEDRON_27_SHAPE_FUNCTION_D_VALUE: [DShapeFunction; 27] = [
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[0](uvw[1]) * P2[0](uvw[2]),
            P2[0](uvw[0]) * DP2[0](uvw[1]) * P2[0](uvw[2]),
            P2[0](uvw[0]) * P2[0](uvw[1]) * DP2[0](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[0](uvw[1]) * P2[0](uvw[2]),
            P2[1](uvw[0]) * DP2[0](uvw[1]) * P2[0](uvw[2]),
            P2[1](uvw[0]) * P2[0](uvw[1]) * DP2[0](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[1](uvw[1]) * P2[0](uvw[2]),
            P2[1](uvw[0]) * DP2[1](uvw[1]) * P2[0](uvw[2]),
            P2[1](uvw[0]) * P2[1](uvw[1]) * DP2[0](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[1](uvw[1]) * P2[0](uvw[2]),
            P2[0](uvw[0]) * DP2[1](uvw[1]) * P2[0](uvw[2]),
            P2[0](uvw[0]) * P2[1](uvw[1]) * DP2[0](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[0](uvw[1]) * P2[1](uvw[2]),
            P2[0](uvw[0]) * DP2[0](uvw[1]) * P2[1](uvw[2]),
            P2[0](uvw[0]) * P2[0](uvw[1]) * DP2[1](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[0](uvw[1]) * P2[1](uvw[2]),
            P2[1](uvw[0]) * DP2[0](uvw[1]) * P2[1](uvw[2]),
            P2[1](uvw[0]) * P2[0](uvw[1]) * DP2[1](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[1](uvw[1]) * P2[1](uvw[2]),
            P2[1](uvw[0]) * DP2[1](uvw[1]) * P2[1](uvw[2]),
            P2[1](uvw[0]) * P2[1](uvw[1]) * DP2[1](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[1](uvw[1]) * P2[1](uvw[2]),
            P2[0](uvw[0]) * DP2[1](uvw[1]) * P2[1](uvw[2]),
            P2[0](uvw[0]) * P2[1](uvw[1]) * DP2[1](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[0](uvw[1]) * P2[0](uvw[2]),
            P2[2](uvw[0]) * DP2[0](uvw[1]) * P2[0](uvw[2]),
            P2[2](uvw[0]) * P2[0](uvw[1]) * DP2[0](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[2](uvw[1]) * P2[0](uvw[2]),
            P2[1](uvw[0]) * DP2[2](uvw[1]) * P2[0](uvw[2]),
            P2[1](uvw[0]) * P2[2](uvw[1]) * DP2[0](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[1](uvw[1]) * P2[0](uvw[2]),
            P2[2](uvw[0]) * DP2[1](uvw[1]) * P2[0](uvw[2]),
            P2[2](uvw[0]) * P2[1](uvw[1]) * DP2[0](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[2](uvw[1]) * P2[0](uvw[2]),
            P2[0](uvw[0]) * DP2[2](uvw[1]) * P2[0](uvw[2]),
            P2[0](uvw[0]) * P2[2](uvw[1]) * DP2[0](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[0](uvw[1]) * P2[2](uvw[2]),
            P2[0](uvw[0]) * DP2[0](uvw[1]) * P2[2](uvw[2]),
            P2[0](uvw[0]) * P2[0](uvw[1]) * DP2[2](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[0](uvw[1]) * P2[2](uvw[2]),
            P2[1](uvw[0]) * DP2[0](uvw[1]) * P2[2](uvw[2]),
            P2[1](uvw[0]) * P2[0](uvw[1]) * DP2[2](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[1](uvw[1]) * P2[2](uvw[2]),
            P2[1](uvw[0]) * DP2[1](uvw[1]) * P2[2](uvw[2]),
            P2[1](uvw[0]) * P2[1](uvw[1]) * DP2[2](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[1](uvw[1]) * P2[2](uvw[2]),
            P2[0](uvw[0]) * DP2[1](uvw[1]) * P2[2](uvw[2]),
            P2[0](uvw[0]) * P2[1](uvw[1]) * DP2[2](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[0](uvw[1]) * P2[1](uvw[2]),
            P2[2](uvw[0]) * DP2[0](uvw[1]) * P2[1](uvw[2]),
            P2[2](uvw[0]) * P2[0](uvw[1]) * DP2[1](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[2](uvw[1]) * P2[1](uvw[2]),
            P2[1](uvw[0]) * DP2[2](uvw[1]) * P2[1](uvw[2]),
            P2[1](uvw[0]) * P2[2](uvw[1]) * DP2[1](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[1](uvw[1]) * P2[1](uvw[2]),
            P2[2](uvw[0]) * DP2[1](uvw[1]) * P2[1](uvw[2]),
            P2[2](uvw[0]) * P2[1](uvw[1]) * DP2[1](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[2](uvw[1]) * P2[1](uvw[2]),
            P2[0](uvw[0]) * DP2[2](uvw[1]) * P2[1](uvw[2]),
            P2[0](uvw[0]) * P2[2](uvw[1]) * DP2[1](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[2](uvw[1]) * P2[0](uvw[2]),
            P2[2](uvw[0]) * DP2[2](uvw[1]) * P2[0](uvw[2]),
            P2[2](uvw[0]) * P2[2](uvw[1]) * DP2[0](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[0](uvw[1]) * P2[2](uvw[2]),
            P2[2](uvw[0]) * DP2[0](uvw[1]) * P2[2](uvw[2]),
            P2[2](uvw[0]) * P2[0](uvw[1]) * DP2[2](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[0](uvw[0]) * P2[2](uvw[1]) * P2[2](uvw[2]),
            P2[0](uvw[0]) * DP2[2](uvw[1]) * P2[2](uvw[2]),
            P2[0](uvw[0]) * P2[2](uvw[1]) * DP2[2](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[1](uvw[0]) * P2[2](uvw[1]) * P2[2](uvw[2]),
            P2[1](uvw[0]) * DP2[2](uvw[1]) * P2[2](uvw[2]),
            P2[1](uvw[0]) * P2[2](uvw[1]) * DP2[2](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[1](uvw[1]) * P2[2](uvw[2]),
            P2[2](uvw[0]) * DP2[1](uvw[1]) * P2[2](uvw[2]),
            P2[2](uvw[0]) * P2[1](uvw[1]) * DP2[2](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[2](uvw[1]) * P2[1](uvw[2]),
            P2[2](uvw[0]) * DP2[2](uvw[1]) * P2[1](uvw[2]),
            P2[2](uvw[0]) * P2[2](uvw[1]) * DP2[1](uvw[2]),
        ]
    },
    |uvw| {
        [
            DP2[2](uvw[0]) * P2[2](uvw[1]) * P2[2](uvw[2]),
            P2[2](uvw[0]) * DP2[2](uvw[1]) * P2[2](uvw[2]),
            P2[2](uvw[0]) * P2[2](uvw[1]) * DP2[2](uvw[2]),
        ]
    },
];

pub const HEXAHEDRON27: GeometricElement = GeometricElement {
    reference_element: &HEXAHEDRON,
    nshape_functions: 27,
    shape_functions: &HEXAHEDRON_27_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &HEXAHEDRON_27_SHAPE_FUNCTION_D_VALUE,
    msh_id: 12,
};
