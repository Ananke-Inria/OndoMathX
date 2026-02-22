use rouage::mesh::{Mesh, UnstructuredMesh};

fn main() {
    let mesh = UnstructuredMesh::new_from_mesh("../Rouage/assets/Heart.mesh")
        .expect("Failed to load Heart.mesh");

    println!("TetriX - Heart mesh loaded");
    println!("  Nodes:    {}", mesh.get_nnodes());
    println!("  Elements: {}", mesh.get_nelements());
    println!("  Geometric dimension: {}", mesh.get_geometric_dimension());
    println!("  Topologic dimension: {}", mesh.get_topologic_dimension());
}
