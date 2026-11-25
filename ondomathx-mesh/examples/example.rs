extern crate ondomathx_mesh;

use ondomathx_mesh::mesh::*;
use unstructured_mesh::UnstructuredMesh;

fn main() {
    let mesh = UnstructuredMesh::new_from_msh("./assets/L.msh").unwrap();

    println!("Point lists: {:?}\n", mesh.get_nodes());

    println!("Element lists: {:?}\n", mesh.get_elements());

    let neighbour_edges: Vec<&[usize]> = (0..mesh.get_nelements())
        .map(|el| mesh.get_neighbour_edges(el))
        .collect();
    println!("topology lists: {:?}\n", neighbour_edges);

    let tri_data: &ReferenceElement = &TRIANGLE;

    println!("data: {:?}\n", tri_data);

    let dof_set_q1 = DoFSet::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        &QUADRANGLE,
    );

    println!("dof set: {:?}\n", dof_set_q1);

    let dof_set_q2 = DoFSet::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        &QUADRANGLE,
    );

    println!("dof set: {:?}\n", dof_set_q2);

    let (num, _n) = mesh.get_numbering(&dof_set_q1);

    println!("numbering: {:?}\n", num);

    let (num, _n) = mesh.get_numbering(&dof_set_q2);

    println!("numbering: {:?}\n", num);
}
