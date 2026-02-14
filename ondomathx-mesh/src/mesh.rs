use itertools::Itertools;
use std::fs::File;
use std::io::{self, BufRead, Write};

pub mod square;
pub mod unstructured_mesh;

pub use crate::dof_set::*;
pub use crate::geometric_elements::*;
pub use crate::mesh::unstructured_mesh::*;
pub use crate::reference_elements::*;

#[derive(Debug, Clone)]
pub struct MeshElement {
    pub geometric_element: &'static GeometricElement,
    pub label: usize,
    pub indexes: Vec<usize>, //points indexes
}

pub trait Mesh {
    fn get_nnodes(&self) -> usize;
    fn get_node(&self, index: usize) -> [f64; 3];

    fn get_deformation(&self) -> Option<&DeformationField>;
    fn get_nelements(&self) -> usize;
    fn get_element(&self, element_index: usize) -> MeshElement;
    fn get_geometric_dimension(&self) -> usize;
    fn get_topologic_dimension(&self) -> usize;

    fn get_elements_by_dimension(&self, dim: usize) -> Vec<usize>;
    fn get_elements_by_dimension_and_label(&self, label: usize, dim: usize) -> Vec<usize>;

    fn get_element_grad_def(&self, element_index: usize, uvw: &[f64], grad_def: &mut [f64]);

    fn get_ncolors(&self) -> usize;
    fn get_nelements_by_color(&self, color_index: usize) -> usize;
    fn get_element_color(&self, element_index: usize) -> usize;
    fn get_element_index_by_color_to_element_index(
        &self,
        color_index: usize,
        element_index_by_color: usize,
    ) -> usize;

    fn get_unstructured_mesh(&self) -> UnstructuredMesh {
        let mut points = Vec::<[f64; 3]>::with_capacity(self.get_nnodes());
        let mut elements = Vec::<MeshElement>::with_capacity(self.get_nelements());

        for i in 0..self.get_nnodes() {
            points.push(self.get_node(i));
        }

        for i in 0..self.get_nelements() {
            elements.push(self.get_element(i));
        }

        UnstructuredMesh::new_from_points_and_elements(
            points,
            elements,
            self.get_deformation().cloned(),
        )
    }

    fn write_msh(&self, path: &str) -> std::io::Result<()> {
        // Creating file
        let mut mesh_file = File::create(path)?;

        let deformation = self.get_deformation();

        // Writing header.
        write!(mesh_file, "$MeshFormat\n")?;
        write!(mesh_file, "2.2 0 8\n")?;
        write!(mesh_file, "$EndMeshFormat\n")?;

        // Write the nodes
        write!(mesh_file, "$Nodes\n")?;
        let npnt = self.get_nnodes();
        write!(mesh_file, "{npnt}\n")?;

        if deformation.is_none() {
            for i in 1..npnt + 1 {
                let [x, y, z] = self.get_node(i - 1);
                write!(mesh_file, "{i} {x} {y} {z}\n")?;
            }
        } else {
            let deformation = deformation.unwrap();

            for i in 1..npnt + 1 {
                let p = self.get_node(i - 1);
                let mut f_p = [0.0; 3];
                deformation.eval(&p, &mut f_p);
                let [x, y, z] = f_p;
                write!(mesh_file, "{i} {x} {y} {z}\n")?;
            }
        }

        write!(mesh_file, "$EndNodes\n")?;

        // Write the elements
        write!(mesh_file, "$Elements\n")?;
        let nelem = self.get_nelements();
        write!(mesh_file, "{nelem}\n")?;

        for i in 1..nelem + 1 {
            let element = &self.get_element(i - 1);

            let msh_id = element.geometric_element.msh_id;
            let label = element.label;

            write!(mesh_file, "{i} {msh_id} 2 0 {label} ")?;

            let mut indexes = element.indexes.clone();
            indexes.iter_mut().for_each(|v| *v += 1);

            writeln!(mesh_file, "{}", indexes.iter().format(" "))?;
        }

        write!(mesh_file, "$EndElements\n")?;

        Ok(())
    }
}

impl MeshElement {
    pub fn get_vertexes(&self) -> Vec<usize> {
        return self.indexes.clone();
    }

    pub fn get_vertex(&self, k: usize) -> usize {
        return self.indexes[k];
    }

    pub fn get_edges(&self) -> Vec<Edge> {
        let mut edges: Vec<Edge> = Vec::new();
        let n_edges = self.geometric_element.reference_element.get_nedges();

        for k in 0..n_edges {
            edges.push(self.get_edge(k));
        }
        return edges;
    }

    pub fn get_edge(&self, k: usize) -> Edge {
        let edges_topology = self.geometric_element.reference_element.edges;

        return Edge::new([
            self.indexes[edges_topology[k][0]],
            self.indexes[edges_topology[k][1]],
        ]);
    }

    pub fn get_faces(&self) -> Vec<Face> {
        let mut faces: Vec<Face> = Vec::new();

        let n_faces = self.geometric_element.reference_element.get_nfaces();

        for k in 0..n_faces {
            faces.push(self.get_face(k));
        }

        return faces;
    }

    pub fn get_face(&self, k: usize) -> Face {
        let faces_topology = self.geometric_element.reference_element.faces;

        return Face::new([
            self.indexes[faces_topology[k][0]],
            self.indexes[faces_topology[k][1]],
            self.indexes[faces_topology[k][2]],
            self.indexes[faces_topology[k][3]],
        ]);
    }
}
