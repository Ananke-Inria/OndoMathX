extern crate rouage;

use rouage::mesh::*;
use unstructured_mesh::UnstructuredMesh;

fn main() {
    let mut mesh = UnstructuredMesh::new_from_msh("./assets/L.msh").unwrap();

    println!("Point lists: {:?}\n", mesh.nodes);

    println!("Element lists: {:?}\n", mesh.elements);

    println!("topology lists: {:?}\n", mesh.neighbour_edges);

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

    let (num, N) = mesh.get_numbering(&dof_set_q1);

    println!("numbering: {:?}\n", num);

    let (num, N) = mesh.get_numbering(&dof_set_q2);

    println!("numbering: {:?}\n", num);
}
