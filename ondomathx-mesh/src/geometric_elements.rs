use crate::reference_elements::*;

const P2: [fn(f64) -> f64; 3] = [
    |u| 2.0 * (0.5 - u) * (1.0 - u),
    |u| 2.0 * (u - 0.5) * u,
    |u| 4.0 * u * (1.0 - u),
];

const DP2: [fn(f64) -> f64; 3] = [
    |u| return 4.0 * u - 3.0,
    |u| return 4.0 * u - 1.0,
    |u| return 4.0 - 8.0 * u,
];

const EDGE_2_SHAPE_FUNCTION_VALUE: [fn(&[f64]) -> f64; 2] =
    [|uvw| return 1.0 - uvw[0], |uvw| return uvw[0]];

const EDGE_2_SHAPE_FUNCTION_D_VALUE: [fn(uvw: &[f64]) -> [f64; 3]; 2] = [
    |_uvw| return [-1.0, 0.0, 0.0],
    |_uvw| return [1.0, 0.0, 0.0],
];

const EDGE_3_SHAPE_FUNCTION_VALUE: [fn(uvw: &[f64]) -> f64; 3] = [
    |uvw| P2[0](uvw[0]),
    |uvw| P2[1](uvw[0]),
    |uvw| P2[2](uvw[0]),
];

const EDGE_3_SHAPE_FUNCTION_D_VALUE: [fn(uvw: &[f64]) -> [f64; 3]; 3] = [
    |uvw| [DP2[0](uvw[0]), 0.0, 0.0],
    |uvw| [DP2[1](uvw[0]), 0.0, 0.0],
    |uvw| [DP2[2](uvw[0]), 0.0, 0.0],
];

const TRIANGLE_3_SHAPE_FUNCTION_VALUE: [fn(uvw: &[f64]) -> f64; 3] = [
    |uvw| return 1.0 - uvw[0] - uvw[1],
    |uvw| return uvw[0],
    |uvw| return uvw[1],
];

const TRIANGLE_3_SHAPE_FUNCTION_D_VALUE: [fn(uvw: &[f64]) -> [f64; 3]; 3] = [
    |_uvw| return [-1.0, -1.0, 0.0],
    |_uvw| return [1.0, 0.0, 0.0],
    |_uvw| return [0.0, 1.0, 0.0],
];

const TETRAHEDRON_4_SHAPE_FUNCTION_VALUE: [fn(&[f64]) -> f64; 4] = [
    |uvw| 1.0 - uvw[0] - uvw[1] - uvw[2],
    |uvw| uvw[0],
    |uvw| uvw[1],
    |uvw| uvw[2],
];

const TETRAHEDRON_4_SHAPE_FUNCTION_D_VALUE: [fn(&[f64]) -> [f64; 3]; 4] = [
    |_uvw| [-1.0, -1.0, -1.0],
    |_uvw| [1.0, 0.0, 0.0],
    |_uvw| [0.0, 1.0, 0.0],
    |_uvw| [0.0, 0.0, 1.0],
];

const QUADRANGLE_4_SHAPE_FUNCTION_VALUE: [fn(uvw: &[f64]) -> f64; 4] = [
    |uvw| return (1.0 - uvw[0]) * (1.0 - uvw[1]),
    |uvw| return uvw[0] * (1.0 - uvw[1]),
    |uvw| return uvw[0] * uvw[1],
    |uvw| return (1.0 - uvw[0]) * uvw[1],
];

const QUADRANGLE_4_SHAPE_FUNCTION_D_VALUE: [fn(uvw: &[f64]) -> [f64; 3]; 4] = [
    |uvw| return [uvw[1] - 1.0, uvw[0] - 1.0, 0.0],
    |uvw| return [1.0 - uvw[1], -uvw[0], 0.0],
    |uvw| return [uvw[1], uvw[0], 0.0],
    |uvw| return [-uvw[1], 1.0 - uvw[0], 0.0],
];

const QUADRANGLE_9_SHAPE_FUNCTION_VALUE: [fn(uvw: &[f64]) -> f64; 9] = [
    |uvw| return P2[0](uvw[0]) * P2[0](uvw[1]),
    |uvw| return P2[1](uvw[0]) * P2[0](uvw[1]),
    |uvw| return P2[1](uvw[0]) * P2[1](uvw[1]),
    |uvw| return P2[0](uvw[0]) * P2[1](uvw[1]),
    |uvw| return P2[2](uvw[0]) * P2[0](uvw[1]),
    |uvw| return P2[1](uvw[0]) * P2[2](uvw[1]),
    |uvw| return P2[2](uvw[0]) * P2[1](uvw[1]),
    |uvw| return P2[0](uvw[0]) * P2[2](uvw[1]),
    |uvw| return P2[2](uvw[0]) * P2[2](uvw[1]),
];

const QUADRANGLE_9_SHAPE_FUNCTION_D_VALUE: [fn(uvw: &[f64]) -> [f64; 3]; 9] = [
    |uvw| {
        return [
            DP2[0](uvw[0]) * P2[0](uvw[1]),
            P2[0](uvw[0]) * DP2[0](uvw[1]),
            0.0,
        ];
    },
    |uvw| {
        return [
            DP2[1](uvw[0]) * P2[0](uvw[1]),
            P2[1](uvw[0]) * DP2[0](uvw[1]),
            0.0,
        ];
    },
    |uvw| {
        return [
            DP2[1](uvw[0]) * P2[1](uvw[1]),
            P2[1](uvw[0]) * DP2[1](uvw[1]),
            0.0,
        ];
    },
    |uvw| {
        return [
            DP2[0](uvw[0]) * P2[1](uvw[1]),
            P2[0](uvw[0]) * DP2[1](uvw[1]),
            0.0,
        ];
    },
    |uvw| {
        return [
            DP2[2](uvw[0]) * P2[0](uvw[1]),
            P2[2](uvw[0]) * DP2[0](uvw[1]),
            0.0,
        ];
    },
    |uvw| {
        return [
            DP2[1](uvw[0]) * P2[2](uvw[1]),
            P2[1](uvw[0]) * DP2[2](uvw[1]),
            0.0,
        ];
    },
    |uvw| {
        return [
            DP2[2](uvw[0]) * P2[1](uvw[1]),
            P2[2](uvw[0]) * DP2[1](uvw[1]),
            0.0,
        ];
    },
    |uvw| {
        return [
            DP2[0](uvw[0]) * P2[2](uvw[1]),
            P2[0](uvw[0]) * DP2[2](uvw[1]),
            0.0,
        ];
    },
    |uvw| {
        return [
            DP2[2](uvw[0]) * P2[2](uvw[1]),
            P2[2](uvw[0]) * DP2[2](uvw[1]),
            0.0,
        ];
    },
];

#[derive(Debug)]
pub struct GeometricElement {
    pub reference_element: &'static ReferenceElement,
    pub nshape_functions: usize,
    pub shape_functions: &'static [fn(&[f64]) -> f64],
    pub d_shape_functions: &'static [fn(&[f64]) -> [f64; 3]],
    pub msh_id: usize,
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

const HEXAHEDRON_8_SHAPE_FUNCTION_VALUE: [fn(&[f64]) -> f64; 8] = [
    |uvw| (1.0 - uvw[0]) * (1.0 - uvw[1]) * (1.0 - uvw[2]),
    |uvw| uvw[0] * (1.0 - uvw[1]) * (1.0 - uvw[2]),
    |uvw| uvw[0] * uvw[1] * (1.0 - uvw[2]),
    |uvw| (1.0 - uvw[0]) * uvw[1] * (1.0 - uvw[2]),
    |uvw| (1.0 - uvw[0]) * (1.0 - uvw[1]) * uvw[2],
    |uvw| uvw[0] * (1.0 - uvw[1]) * uvw[2],
    |uvw| uvw[0] * uvw[1] * uvw[2],
    |uvw| (1.0 - uvw[0]) * uvw[1] * uvw[2],
];

const HEXAHEDRON_8_SHAPE_FUNCTION_D_VALUE: [fn(&[f64]) -> [f64; 3]; 8] = [
    |uvw| [-(1.0 - uvw[1]) * (1.0 - uvw[2]), -(1.0 - uvw[0]) * (1.0 - uvw[2]), -(1.0 - uvw[0]) * (1.0 - uvw[1])],
    |uvw| [(1.0 - uvw[1]) * (1.0 - uvw[2]), -uvw[0] * (1.0 - uvw[2]), -uvw[0] * (1.0 - uvw[1])],
    |uvw| [uvw[1] * (1.0 - uvw[2]), uvw[0] * (1.0 - uvw[2]), -uvw[0] * uvw[1]],
    |uvw| [-uvw[1] * (1.0 - uvw[2]), (1.0 - uvw[0]) * (1.0 - uvw[2]), -(1.0 - uvw[0]) * uvw[1]],
    |uvw| [-(1.0 - uvw[1]) * uvw[2], -(1.0 - uvw[0]) * uvw[2], (1.0 - uvw[0]) * (1.0 - uvw[1])],
    |uvw| [(1.0 - uvw[1]) * uvw[2], -uvw[0] * uvw[2], uvw[0] * (1.0 - uvw[1])],
    |uvw| [uvw[1] * uvw[2], uvw[0] * uvw[2], uvw[0] * uvw[1]],
    |uvw| [-uvw[1] * uvw[2], (1.0 - uvw[0]) * uvw[2], (1.0 - uvw[0]) * uvw[1]],
];

pub const HEXAHEDRON8: GeometricElement = GeometricElement {
    reference_element: &HEXAHEDRON,
    nshape_functions: 8,
    shape_functions: &HEXAHEDRON_8_SHAPE_FUNCTION_VALUE,
    d_shape_functions: &HEXAHEDRON_8_SHAPE_FUNCTION_D_VALUE,
    msh_id: 5,
};
