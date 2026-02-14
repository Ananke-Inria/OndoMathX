#![feature(more_float_constants)]
#![feature(concat_idents)]

//extern crate nalgebra as na;

pub mod macros;

pub mod geometric_elements;
pub mod reference_elements;

pub mod deformation_field;

pub mod dof_set;
pub mod topology;

pub mod mesh;

//#[derive(Serialize, Deserialize, Debug, Default)]
// pub enum RefElementType {
//     #[default]
//     None,
//     Edge,
//     Triangle,
//     Quadrangle,
//     Tetrahedron,
//     Hexahedron,
// }

// #[derive(Serialize, Deserialize, Debug)]
// pub enum GeoElementType {
//     Edge2,
//     Edge3,
//     Triangle3,
//     Triangle6,
//     Quadrangle4,
//     Quadrangle9,
//     Tetrahedron4,
//     Tetrahedron10,
//     Hexahedron8,
//     Hexahedron27,
// }
