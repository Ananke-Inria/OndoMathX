use itertools::Itertools;
use std::fs::File;
use std::io::Write;

pub mod circle;
pub mod cube;
pub mod refine;
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

    fn get_deformation(&self) -> Option<&Diffeomorphism>;
    fn get_nelements(&self) -> usize;
    fn get_element(&self, element_index: usize) -> MeshElement;
    fn get_geometric_dimension(&self) -> usize;
    fn get_topologic_dimension(&self) -> usize;

    fn get_elements_by_dimension(&self, dim: usize) -> Vec<usize>;
    fn get_elements_by_dimension_and_label(&self, label: usize, dim: usize) -> Vec<usize>;

    //Writes into `grad_def` the Jacobian ∂x/∂u of the COMPLETE element map
    //(isoparametric or analytic map composed with any attached deformation)
    //evaluated at the reference coordinates `uvw`. `grad_def` is a pure
    //output, overwritten row-major with exact shape
    //geometric_dimension × (element reference dimension) — e.g. a quad cell
    //in a 2D mesh yields a 2×2, a boundary edge a 2×1 (the tangent vector).
    fn get_element_grad_def(&self, element_index: usize, uvw: &[f64], grad_def: &mut [f64]);

    //Writes into `def` the image x(uvw) of the COMPLETE element map (the map
    //differentiated by get_element_grad_def), as its first
    //geometric_dimension entries. The default interpolates the element's
    //geometric nodes and applies the attached deformation: exact for meshes
    //whose undeformed geometry is that interpolation (Square, Cube,
    //UnstructuredMesh). Meshes with an analytic map override it.
    fn get_element_def(&self, element_index: usize, uvw: &[f64], def: &mut [f64]) {
        let element = self.get_element(element_index);
        let geo = element.geometric_element;
        let mut x = [0.0; 3];
        for i in 0..geo.nshape_functions {
            let phi = geo.shape_functions[i](uvw);
            let node = self.get_node(element.indexes[i]);
            for a in 0..3 {
                x[a] += node[a] * phi;
            }
        }
        if let Some(deformation) = self.get_deformation() {
            let mut f_x = [0.0; 3];
            deformation.eval(&x, &mut f_x);
            x = f_x;
        }
        let geo_dim = self.get_geometric_dimension();
        def[..geo_dim].copy_from_slice(&x[..geo_dim]);
    }

    //Whether the complete map of the element is NOT given by its affine/Q1/Q2
    //geometric nodes alone: an analytic map (e.g. the arms of Circle) or an
    //attached deformation. When false, the map is the Lagrange interpolation
    //of the element's nodes. The default only checks the deformation.
    fn is_mapped_element(&self, element_index: usize) -> bool {
        let _ = element_index;
        self.get_deformation().is_some()
    }

    //Continuous numbering of a tensor-product DoF layout with n[d] DoFs per
    //direction of the reference element on every volume element, local index
    //x-fastest (`i + n0·j + n0·n1·k`): (numbering[element][local], n_dof).
    //Only the volume elements are guaranteed to be numbered. The default goes
    //through UnstructuredMesh::get_numbering with a tensor DoFSet, which
    //requires the same count in every direction (DoFSet symmetry); structured
    //meshes (Square, Cube) override it and accept anisotropic counts.
    fn get_tensor_numbering(&self, n: [usize; 3]) -> (Vec<Vec<usize>>, usize) {
        self.get_unstructured_mesh().get_tensor_numbering(n)
    }

    //Coloring API. Colors are defined only for elements of the highest
    //topological dimension (volume elements); querying the color of a
    //boundary element is a contract violation (implementations may panic).
    //All element indices exchanged through this API are global element
    //indices (valid for `get_element`), regardless of how the elements are
    //ordered in the mesh.
    fn get_ncolors(&self) -> usize;
    fn get_nelements_by_color(&self, color_index: usize) -> usize;
    fn get_element_color(&self, element_index: usize) -> usize;
    fn get_element_index_by_color_to_element_index(
        &self,
        color_index: usize,
        element_index_by_color: usize,
    ) -> usize;

    //NB: parametrized meshes (Square, Circle, ...) "lose" their
    //parametrization here: only the node positions and the element
    //connectivity are transferred. In the resulting UnstructuredMesh the
    //geometry is the piecewise (bi/tri)linear interpolation of the nodes —
    //in particular get_element_grad_def differentiates that interpolation,
    //not the analytic map of the original mesh, and curved boundaries become
    //polygonal. Only the optional Diffeomorphism is carried over.
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
        writeln!(mesh_file, "$MeshFormat")?;
        writeln!(mesh_file, "2.2 0 8")?;
        writeln!(mesh_file, "$EndMeshFormat")?;

        // Write the nodes
        writeln!(mesh_file, "$Nodes")?;
        let npnt = self.get_nnodes();
        writeln!(mesh_file, "{npnt}")?;

        if let Some(deformation) = deformation {
            for i in 1..npnt + 1 {
                let p = self.get_node(i - 1);
                let mut f_p = [0.0; 3];
                deformation.eval(&p, &mut f_p);
                let [x, y, z] = f_p;
                writeln!(mesh_file, "{i} {x} {y} {z}")?;
            }
        } else {
            for i in 1..npnt + 1 {
                let [x, y, z] = self.get_node(i - 1);
                writeln!(mesh_file, "{i} {x} {y} {z}")?;
            }
        }

        writeln!(mesh_file, "$EndNodes")?;

        // Write the elements
        writeln!(mesh_file, "$Elements")?;
        let nelem = self.get_nelements();
        writeln!(mesh_file, "{nelem}")?;

        //Node permutations applied when writing high-order elements: file
        //position j receives internal node TO[j]. These are the inverses of
        //the FROM permutations used by the reader in unstructured_mesh.rs.
        const TETRAHEDRON_10_TO_MSH: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8];
        const HEXAHEDRON_27_TO_MSH: [usize; 27] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 9, 13, 10, 14, 15, 16, 19, 17, 18, 20, 21, 22, 23,
            24, 25, 26,
        ];

        for i in 1..nelem + 1 {
            let element = &self.get_element(i - 1);

            let msh_id = element.geometric_element.msh_id;
            let label = element.label;

            //Two tags: physical group and elementary entity, both set to the
            //label so that reading the file back recovers it (the reader
            //takes the physical group as the label).
            write!(mesh_file, "{i} {msh_id} 2 {label} {label} ")?;

            let mut indexes = element.indexes.clone();
            indexes.iter_mut().for_each(|v| *v += 1);

            //High-order elements whose internal node ordering differs from
            //Gmsh's must be permuted back on write.
            let permutation: Option<&[usize]> = match msh_id {
                11 => Some(&TETRAHEDRON_10_TO_MSH),
                12 => Some(&HEXAHEDRON_27_TO_MSH),
                _ => None,
            };
            if let Some(perm) = permutation {
                indexes = perm.iter().map(|&p| indexes[p]).collect();
            }

            writeln!(mesh_file, "{}", indexes.iter().format(" "))?;
        }

        writeln!(mesh_file, "$EndElements")?;

        Ok(())
    }

    //Writes the mesh in the ASCII Medit .mesh format (readable by Vizir4 and
    //by UnstructuredMesh::new_from_mesh). Elements are grouped by type into
    //the Medit sections, so the global element ordering is not preserved on
    //a round-trip (labels are). The dimension is always written as 3 (z = 0
    //for 2D meshes) and vertex references are set to 0.
    fn write_mesh(&self, path: &str) -> std::io::Result<()> {
        let mut mesh_file = File::create(path)?;

        let deformation = self.get_deformation();

        // Writing header.
        writeln!(mesh_file, "MeshVersionFormatted 2")?;
        writeln!(mesh_file, "Dimension")?;
        writeln!(mesh_file, "3")?;

        // Write the vertices (deformed if a deformation is attached)
        writeln!(mesh_file, "Vertices")?;
        let npnt = self.get_nnodes();
        writeln!(mesh_file, "{npnt}")?;

        if let Some(deformation) = deformation {
            for i in 0..npnt {
                let p = self.get_node(i);
                let mut f_p = [0.0; 3];
                deformation.eval(&p, &mut f_p);
                let [x, y, z] = f_p;
                writeln!(mesh_file, "{x} {y} {z} 0")?;
            }
        } else {
            for i in 0..npnt {
                let [x, y, z] = self.get_node(i);
                writeln!(mesh_file, "{x} {y} {z} 0")?;
            }
        }

        //Medit sections in writing order, keyed by the Gmsh element-type id
        //of the corresponding GeometricElement.
        const SECTIONS: [(usize, &str); 10] = [
            (1, "Edges"),
            (2, "Triangles"),
            (3, "Quadrilaterals"),
            (4, "Tetrahedra"),
            (5, "Hexahedra"),
            (8, "EdgesP2"),
            (9, "TrianglesP2"),
            (10, "QuadrilateralsQ2"),
            (11, "TetrahedraP2"),
            (12, "HexahedraQ2"),
        ];

        //Internal -> file node permutation for HexahedraQ2 (inverse of the
        //permutation used by the reader; the node ordering of every other
        //section coincides with the internal one).
        const HEXAHEDRON_27_TO_MESH: [usize; 27] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15, 20, 25, 21, 23,
            24, 22, 26,
        ];

        // Group the elements by type into the Medit sections.
        let mut sections: Vec<Vec<String>> = vec![Vec::new(); SECTIONS.len()];

        for e in 0..self.get_nelements() {
            let element = self.get_element(e);
            let msh_id = element.geometric_element.msh_id;

            let section = SECTIONS
                .iter()
                .position(|&(id, _)| id == msh_id)
                .unwrap_or_else(|| {
                    panic!("Element type {msh_id} is not supported by the .mesh format")
                });

            let mut indexes = element.indexes.clone();
            indexes.iter_mut().for_each(|v| *v += 1);

            if msh_id == 12 {
                indexes = HEXAHEDRON_27_TO_MESH.iter().map(|&p| indexes[p]).collect();
            }

            sections[section].push(format!("{} {}", indexes.iter().format(" "), element.label));
        }

        for (k, &(_, name)) in SECTIONS.iter().enumerate() {
            if !sections[k].is_empty() {
                writeln!(mesh_file, "{name}")?;
                writeln!(mesh_file, "{}", sections[k].len())?;
                for line in sections[k].iter() {
                    writeln!(mesh_file, "{line}")?;
                }
            }
        }

        writeln!(mesh_file, "End")?;

        Ok(())
    }
}

impl MeshElement {
    pub fn get_edges(&self) -> Vec<Edge> {
        let mut edges: Vec<Edge> = Vec::new();
        let n_edges = self.geometric_element.reference_element.get_nedges();

        for k in 0..n_edges {
            edges.push(self.get_edge(k));
        }
        edges
    }

    pub fn get_edge(&self, k: usize) -> Edge {
        let edges_topology = self.geometric_element.reference_element.edges;

        Edge::new([
            self.indexes[edges_topology[k][0]],
            self.indexes[edges_topology[k][1]],
        ])
    }

    pub fn get_faces(&self) -> Vec<Face> {
        let mut faces: Vec<Face> = Vec::new();

        let n_faces = self.geometric_element.reference_element.get_nfaces();

        for k in 0..n_faces {
            faces.push(self.get_face(k));
        }

        faces
    }

    pub fn get_face(&self, k: usize) -> Face {
        let faces_topology = self.geometric_element.reference_element.faces;

        Face::new([
            self.indexes[faces_topology[k][0]],
            self.indexes[faces_topology[k][1]],
            self.indexes[faces_topology[k][2]],
            self.indexes[faces_topology[k][3]],
        ])
    }
}
