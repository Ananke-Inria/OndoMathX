use std::fs::File;
use std::io::{self, BufRead, Write};

use std::collections::BTreeMap;
use std::collections::HashMap;

pub use crate::deformation_field::*;
pub use crate::mesh::*;
pub use crate::topology::*;

#[derive(Default, Debug)]
pub struct UnstructuredMesh {
    pub elements: Vec<MeshElement>,

    //List of nodes
    pub nodes: Vec<[f64; 3]>,

    pub geometric_dimension: usize,
    pub topologic_dimension: usize,

    //Additionnal deformation function
    pub deformation: Option<DeformationField>,

    //Lists of elements sorted by dimension and label/////////

    //Element indices sorted by dimension
    pub elements_by_dim: Vec<Vec<usize>>,

    //Elements indices sorted by dimension and label
    pub elements_by_dim_and_label: Vec<HashMap<usize, Vec<usize>>>,

    //Topology ///////////////////////////////////////////////

    //Number of faces and  edges of the mesh
    pub nfaces: usize,
    pub nedges: usize,
    pub nvertexes: usize,

    //For each element, list of the neighbour faces
    pub neighbour_faces: Vec<Vec<usize>>,

    // For each face, the list of neighbour elements of highest dimension
    // faces_neigbour[k][i][0] corresponds to the index of the ith element connected to the face k.
    // faces_neigbour[k][i][1] corresponds to the index of the face in the element.
    pub faces_neigbour: Vec<Vec<[usize; 2]>>,

    //For each element, list of the neighbour edges
    pub neighbour_edges: Vec<Vec<usize>>,

    // For each edge, the list of neighbour elements of highest dimension
    // edges_neigbour[k][i][0] corresponds to the index of the ith element connected to the edge k.
    // edges_neigbour[k][i][1] corresponds to the index of the edge in the element.
    pub edges_neigbour: Vec<Vec<[usize; 2]>>,

    //For each element, list of the neighbour vertexes
    pub neighbour_vertexes: Vec<Vec<usize>>,

    // For each vertex , the list of neighbour elements of highest dimension
    // vertexes_neigbour[k][i][0] corresponds to the index of the ith element connected to the vertex k.
    // vertexes_neigbour[k][i][1] corresponds to the index of the vertex in the element.
    pub vertexes_neigbour: Vec<Vec<[usize; 2]>>,
    //////////////////////////////////////////////////////////

    //Coloring
    pub coloring: Vec<usize>,
    pub elements_by_color: Vec<Vec<usize>>,
    pub n_colors: usize,
}

impl Mesh for UnstructuredMesh {
    fn get_deformation(&self) -> Option<&DeformationField> {
        self.deformation.as_ref()
    }

    fn get_nnodes(&self) -> usize {
        self.nodes.len()
    }

    fn get_node(&self, index: usize) -> [f64; 3] {
        self.nodes[index]
    }

    fn get_nelements(&self) -> usize {
        self.elements.len()
    }

    fn get_element(&self, element_index: usize) -> MeshElement {
        self.elements[element_index].clone()
    }

    fn get_geometric_dimension(&self) -> usize {
        self.geometric_dimension
    }

    fn get_topologic_dimension(&self) -> usize {
        self.topologic_dimension
    }

    fn get_elements_by_dimension(&self, dim: usize) -> Vec<usize> {
        self.elements_by_dim[dim].clone()
    }

    fn get_elements_by_dimension_and_label(&self, label: usize, dim: usize) -> Vec<usize> {
        let vec = self.elements_by_dim_and_label[dim].get(&label);

        match vec {
            Some(v) => {
                return v.clone();
            }
            None => return vec![],
        }
    }

    fn get_ncolors(&self) -> usize {
        self.n_colors
    }

    fn get_nelements_by_color(&self, color_index: usize) -> usize {
        self.elements_by_color[color_index].len()
    }

    fn get_element_color(&self, element_index: usize) -> usize {
        self.coloring[element_index]
    }

    fn get_element_index_by_color_to_element_index(
        &self,
        color_index: usize,
        element_index_by_color: usize,
    ) -> usize {
        self.elements_by_color[color_index][element_index_by_color]
    }

    fn get_element_grad_def(&self, e: usize, uvw: &[f64], grad_def: &mut [f64]) {
        let element = &self.elements[e];
        let geo_element = element.geometric_element;

        let mut xyz = [0.0; 3];

        for i in 0..geo_element.nshape_functions {
            let inode = element.indexes[i];
            let node = &self.nodes[inode];
            let d = geo_element.d_shape_functions[i](uvw);

            if self.deformation.is_some() {
                let p = geo_element.shape_functions[i](uvw);

                if self.geometric_dimension == 3 {
                    xyz[0] += node[0] * p;
                    xyz[1] += node[1] * p;
                    xyz[2] += node[2] * p;
                } else if self.geometric_dimension == 2 {
                    xyz[0] += node[0] * p;
                    xyz[1] += node[1] * p;
                } else if self.geometric_dimension == 1 {
                    xyz[0] += node[0] * p;
                }
            }

            if self.geometric_dimension == 3 && self.topologic_dimension == 3 {
                grad_def[0] += node[0] * d[0];
                grad_def[1] += node[0] * d[1];
                grad_def[2] += node[0] * d[2];

                grad_def[3] += node[1] * d[0];
                grad_def[4] += node[1] * d[1];
                grad_def[5] += node[1] * d[2];

                grad_def[6] += node[2] * d[0];
                grad_def[7] += node[2] * d[1];
                grad_def[8] += node[2] * d[2];
            } else if self.geometric_dimension == 2 && self.topologic_dimension == 2 {
                grad_def[0] += node[0] * d[0];
                grad_def[1] += node[0] * d[1];

                grad_def[2] += node[1] * d[0];
                grad_def[3] += node[1] * d[1];
            } else if self.geometric_dimension == 3 && self.topologic_dimension == 2 {
                grad_def[0] += node[0] * d[0];
                grad_def[1] += node[0] * d[1];

                grad_def[2] += node[1] * d[0];
                grad_def[3] += node[1] * d[1];

                grad_def[4] += node[2] * d[0];
                grad_def[5] += node[2] * d[1];
            }
        }
    }
}

impl UnstructuredMesh {
    pub fn get_element_as_ref(&self, element_index: usize) -> &MeshElement {
        &self.elements[element_index]
    }

    pub fn get_element_as_mut(&mut self, element_index: usize) -> &mut MeshElement {
        &mut self.elements[element_index]
    }

    pub fn apply_deformation(&mut self) {
        if self.deformation.is_some() {
            let deformation: &DeformationField = self.deformation.as_ref().unwrap();

            for i in 0..self.nodes.len() {
                let mut p = [0.0; 3];
                deformation.eval(&self.nodes[i], &mut p);
                self.nodes[i] = p;
            }

            self.deformation = None;
        }
    }

    pub fn add_deformation(&mut self, deformation_field: DeformationField) {
        if self.deformation.is_some() {
            self.deformation
                .as_mut()
                .unwrap()
                .compose(deformation_field);
        } else {
            self.deformation = Some(deformation_field);
        }
    }
    //fn into_second_order(&mut self) {}

    fn compute_dimension(&mut self) {
        self.geometric_dimension = 1;

        for k in 0..(self.nodes.len()) {
            if self.nodes[k][1] != 0.0 {
                self.geometric_dimension = 2;
                break;
            }
        }

        for k in 0..(self.nodes.len()) {
            if self.nodes[k][2] != 0.0 {
                self.geometric_dimension = 3;
                break;
            }
        }
    }
    fn compute_indexing(&mut self) {
        self.elements_by_dim = vec![vec![]; 4];

        self.elements_by_dim_and_label = vec![HashMap::<usize, Vec<usize>>::new(); 4];

        for el in 0..self.elements.len() {
            let dim = self.elements[el]
                .geometric_element
                .reference_element
                .get_dim();

            let label = self.elements[el].label;

            self.elements_by_dim[dim].push(el);

            if self.elements_by_dim_and_label[dim].contains_key(&label) {
                self.elements_by_dim_and_label[dim]
                    .get_mut(&label)
                    .unwrap()
                    .push(el);
            } else {
                self.elements_by_dim_and_label[dim].insert(label, vec![el; 1]);
            }
        }

        self.topologic_dimension = 3;

        if self.elements_by_dim[3].len() == 0 {
            self.topologic_dimension -= 1;
        }

        if self.elements_by_dim[2].len() == 0 {
            self.topologic_dimension -= 1;
        }

        if self.elements_by_dim[1].len() == 0 {
            self.topologic_dimension -= 1;
        }
    }
    fn compute_topology(&mut self) {
        let nelements = self.elements.len();
        let nnodes = self.nodes.len();

        //Construct the array neighbour_vertexes
        {
            self.nvertexes = 0;

            let mut vertexes_inserted = vec![false; nnodes];
            let mut vertexes_index = vec![0; nnodes];

            self.neighbour_vertexes.resize(nelements, Vec::new());

            for el in 0..nelements {
                let element = &self.elements[el];
                let vertexes = element.get_vertexes();

                for v in vertexes {
                    //Number the vertex if it nas not been numbered yet
                    if vertexes_inserted[v] == false {
                        vertexes_index[v] = self.nvertexes;
                        self.nvertexes += 1;
                        vertexes_inserted[v] = true;
                    }
                    //Add the vertexes in the list
                    self.neighbour_vertexes[el].push(vertexes_index[v]);
                }
            }
        }

        //Construct the array neighbour_edges
        {
            self.nedges = 0;
            let mut map_edges: BTreeMap<Edge, usize> = BTreeMap::<Edge, usize>::new();
            self.neighbour_edges.resize(nelements, Vec::new());

            for el in 0..nelements {
                let element = &self.elements[el];
                let edges = element.get_edges();
                //Add the edges if they do not already exist
                for e in edges {
                    if !map_edges.contains_key(&e) {
                        map_edges.insert(e, self.nedges);

                        self.neighbour_edges[el].push(self.nedges);
                        self.nedges += 1;
                    } else {
                        self.neighbour_edges[el].push(*map_edges.get(&e).unwrap());
                    }
                }
            }
        }

        //Construct the array neighbour_faces
        {
            self.nfaces = 0;
            let mut map_faces: BTreeMap<Face, usize> = BTreeMap::<Face, usize>::new();
            self.neighbour_faces.resize(nelements, Vec::new());

            for el in 0..nelements {
                let element = &self.elements[el];
                let faces = element.get_faces();
                //Add the faces if they do not already exist
                for f in faces {
                    if !map_faces.contains_key(&f) {
                        map_faces.insert(f, self.nfaces);

                        self.neighbour_faces[el].push(self.nfaces);

                        self.nfaces += 1;
                    } else {
                        self.neighbour_faces[el].push(*map_faces.get(&f).unwrap());
                    }
                }
            }
        }

        //Construct the array vertexes_neigbour
        {
            self.vertexes_neigbour.resize(self.nvertexes, Vec::new());
            for el in self.elements_by_dim[self.topologic_dimension].iter() {
                let nv = self.neighbour_vertexes[*el].len();
                for v in 0..nv {
                    self.vertexes_neigbour[self.neighbour_vertexes[*el][v]].push([*el, v]);
                }
            }
        }

        //Construct the array edges_neigbour
        {
            self.edges_neigbour.resize(self.nedges, Vec::new());
            for el in self.elements_by_dim[self.topologic_dimension].iter() {
                let ne = self.neighbour_edges[*el].len();
                for e in 0..ne {
                    self.edges_neigbour[self.neighbour_edges[*el][e]].push([*el, e]);
                }
            }
        }

        //Construct the array faces_neigbour
        {
            if self.topologic_dimension == 3 {
                self.faces_neigbour.resize(self.nfaces, Vec::new());
                for el in self.elements_by_dim[self.topologic_dimension].iter() {
                    let nf = self.neighbour_faces[*el].len();
                    for f in 0..nf {
                        self.faces_neigbour[self.neighbour_faces[*el][f]].push([*el, f]);
                        // First the orientation index is set to 0
                    }
                }
            }
        }
    }

    pub fn compute_graph(&self) -> Vec<Vec<usize>> {
        let nelements = self.elements_by_dim[self.topologic_dimension].len();

        let mut graph = vec![vec![]; nelements];

        //Array that is a local numbering for all the element of dimension self.topologic_dimension
        let mut element_2_higest_dim_element = vec![0; self.elements.len()];
        for i in 0..nelements {
            let i_element = self.elements_by_dim[self.topologic_dimension][i];
            element_2_higest_dim_element[i_element] = i;
        }

        //Fill the graph matrix using vertexes
        for i in 0..nelements {
            let ielement = self.elements_by_dim[self.topologic_dimension][i];

            for v in self.neighbour_vertexes[ielement].iter() {
                for i_v_e in self.vertexes_neigbour[*v].iter() {
                    if i_v_e[0] != ielement {
                        graph[i].push(element_2_higest_dim_element[i_v_e[0]]);
                    }
                }
            }
        }

        //Sort and remove duplicate
        for i in 0..nelements {
            graph[i].sort();
            graph[i].dedup();
        }

        graph
    }
    pub fn compute_coloring(&mut self) {
        let graph = self.compute_graph();

        let mut order: Vec<_> = (0..graph.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(graph[i].len()));

        let mut coloring = vec![usize::MAX; graph.len()];
        let mut elements_by_color: Vec<Vec<usize>> = vec![vec![]];

        let mut n_color_max = 0;

        for i in order {
            let mut ncs = vec![];
            for n in graph[i].iter() {
                let nc = coloring[*n];
                if nc != usize::MAX {
                    if nc >= ncs.len() {
                        ncs.resize(nc + 1, false);
                    }
                    ncs[nc] = true;
                }
            }
            for c in 0.. {
                if c >= ncs.len() || !ncs[c] {
                    coloring[i] = c;

                    if c + 1 > elements_by_color.len() {
                        elements_by_color.resize(c + 1, vec![]);
                    }

                    elements_by_color[c].push(i);

                    if c > n_color_max {
                        n_color_max = c;
                    }

                    break;
                }
            }
        }

        self.elements_by_color = elements_by_color;
        self.coloring = coloring;
        self.n_colors = n_color_max + 1;
    }
    pub fn new_from_points_and_elements(
        nodes: Vec<[f64; 3]>,
        elements: Vec<MeshElement>,
        deformation: Option<DeformationField>,
    ) -> UnstructuredMesh {
        let mut mesh = UnstructuredMesh::default();
        mesh.nodes = nodes;
        mesh.elements = elements;
        mesh.deformation = deformation;

        mesh.compute_dimension();
        mesh.compute_indexing();
        mesh.compute_topology();
        mesh.compute_coloring();

        mesh
    }
    pub fn new_from_msh(path: &str) -> Result<UnstructuredMesh, std::io::Error> {
        let points = UnstructuredMesh::read_points_from_msh(path)?;

        let elements = UnstructuredMesh::read_elements_from_msh(path)?;

        Ok(UnstructuredMesh::new_from_points_and_elements(
            points, elements, None,
        ))
    }
    fn read_points_from_msh(path: &str) -> Result<Vec<[f64; 3]>, std::io::Error> {
        // Opening file & defining iterator over nodes segments.
        let file = std::fs::File::open(path)?;
        let file_buf = io::BufReader::new(file);
        let mut nodes_lines = file_buf
            .lines()
            .map(|line_res| line_res.unwrap())
            .skip_while(|line| line != "$Nodes")
            .take_while(|line| line != "$EndNodes")
            .skip(1);

        // Iterating over points.
        let mut points: Vec<[f64; 3]> = Vec::new();
        let npnt = nodes_lines.next().unwrap().parse::<usize>().unwrap();
        points.reserve(npnt);
        for node_line in nodes_lines {
            let split_node_line: Vec<&str> = node_line.split(' ').collect();
            points.push([
                split_node_line[1].parse::<f64>().unwrap(),
                split_node_line[2].parse::<f64>().unwrap(),
                split_node_line[3].parse::<f64>().unwrap(),
            ]);
        }
        return Ok(points);
    }
    fn read_elements_from_msh(path: &str) -> Result<Vec<MeshElement>, std::io::Error> {
        // Opening file & defining iterator over nodes segments.
        let file = std::fs::File::open(path)?;
        let file_buf = io::BufReader::new(file);
        let mut elements_lines = file_buf
            .lines()
            .map(|line_res| line_res.unwrap())
            .skip_while(|line| line != "$Elements")
            .take_while(|line| line != "$EndElements")
            .skip(1);

        // Iterating over elements.
        let mut elements: Vec<MeshElement> = Vec::new();
        let nelem = elements_lines.next().unwrap().parse::<usize>().unwrap();
        elements.reserve(nelem);

        for element_line in elements_lines {
            let split_element_line: Vec<&str> = element_line.split(' ').collect();

            match split_element_line[1] {
                "1" => {
                    // Edge2
                    let line_len = split_element_line.len();
                    let point_indexes = [
                        split_element_line[line_len - 2].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 1].parse::<usize>().unwrap() - 1,
                    ];

                    elements.push(MeshElement {
                        geometric_element: &EDGE2,
                        label: 0,
                        indexes: point_indexes.to_vec(),
                    })
                }
                "3" => {
                    // Quadrangle4
                    let line_len = split_element_line.len();
                    let point_indexes = [
                        split_element_line[line_len - 4].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 3].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 2].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 1].parse::<usize>().unwrap() - 1,
                    ];

                    elements.push(MeshElement {
                        geometric_element: &QUADRANGLE4,
                        label: 0,
                        indexes: point_indexes.to_vec(),
                    })
                }

                "15" => {}
                _ => panic!("Element type not supported yet"),
            }
        }

        return Ok(elements);
    }

    pub fn get_numbering(&self, dof_set: &DoFSet) -> (Vec<Vec<usize>>, usize) {
        let mut numbering: Vec<Vec<usize>> = vec![vec![]];

        numbering.resize(self.elements.len(), vec![]);

        let mut vertexes_numbering: Vec<Vec<usize>> = vec![vec![]; self.nvertexes];
        let mut edges_numbering: Vec<Vec<usize>> = vec![vec![]; self.nedges];
        let mut faces_numbering: Vec<Vec<usize>> = vec![vec![]; self.nfaces];

        let mut n_dof = 0;

        for el in self.elements_by_dim[self.topologic_dimension].iter() {
            numbering[*el].resize(dof_set.get_ndof(), 0);

            //numbering of the vertexes
            for v in 0..dof_set.reference_element.get_nvertexes() {
                //recover the index of the vertex
                let k = self.neighbour_vertexes[*el][v];

                let dof_on_vertexes = &dof_set.dof_on_vertexes[v];

                if dof_on_vertexes.len() > 0 {
                    //create the numbering on the node (if needed)
                    if vertexes_numbering[k].is_empty() {
                        vertexes_numbering[k].resize(dof_on_vertexes.len(), 0);

                        for n in 0..dof_on_vertexes.len() {
                            vertexes_numbering[k][n] = n_dof;
                            n_dof += 1;
                        }
                    }

                    //compute the local2global
                    for n in 0..dof_on_vertexes.len() {
                        numbering[*el][dof_on_vertexes[n]] = vertexes_numbering[k][n];
                    }
                }
            }

            //numbering of the interior of the edge
            if dof_set.reference_element.get_dim() >= 1 {
                for e in 0..dof_set.reference_element.get_nedges() {
                    //recover the index of the edge
                    let k = self.neighbour_edges[*el][e];
                    let o = self.elements[*el].get_edge(e).get_orientation();
                    let dof_in_edge = &dof_set.dof_strictly_in_edges[e][o];

                    if dof_in_edge.len() > 0 {
                        //create the numbering on the node (if needed)
                        if edges_numbering[k].is_empty() {
                            edges_numbering[k].resize(dof_in_edge.len(), 0);

                            for n in 0..dof_in_edge.len() {
                                edges_numbering[k][n] = n_dof;
                                n_dof += 1;
                            }
                        }

                        //compute the local2global array a.k.a. the numbering
                        for n in 0..dof_in_edge.len() {
                            numbering[*el][dof_in_edge[n]] = edges_numbering[k][n];
                        }
                    }
                }
            }

            //numbering of the interior of the face
            if dof_set.reference_element.get_dim() >= 2 {
                for f in 0..dof_set.reference_element.get_nfaces() {
                    //recover the index of the face
                    let k = self.neighbour_faces[*el][f];
                    let o = self.elements[*el].get_face(f).get_orientation();
                    let dof_in_face = &dof_set.dof_strictly_in_faces[f][o];

                    if dof_in_face.len() > 0 {
                        //create the numbering on the node (if needed)
                        if faces_numbering[k].is_empty() {
                            faces_numbering[k].resize(dof_in_face.len(), 0);

                            for n in 0..dof_in_face.len() {
                                faces_numbering[k][n] = n_dof;
                                n_dof += 1;
                            }
                        }

                        //compute the local2global array a.k.a. the numbering
                        for n in 0..dof_in_face.len() {
                            numbering[*el][dof_in_face[n]] = faces_numbering[k][n];
                        }
                    }
                }
            }

            //interior of the volume
            if dof_set.reference_element.get_dim() == 3 {
                let dof_in_volume = &dof_set.dof_strictly_in_volume;

                for n in 0..dof_in_volume.len() {
                    numbering[*el][dof_in_volume[n]] = n_dof;
                    n_dof += 1;
                }
            }
        }

        //Numbering of elements of dimension less than elements_dimension
        if 0 < self.topologic_dimension {
            for el in self.elements_by_dim[0].iter() {
                let i_vertex = self.elements[*el].indexes[0];

                assert!(vertexes_numbering[i_vertex].len()>0, "One vertex can not be numbered since it is not connected with any numbered highest dimensionnal element");

                numbering[*el] = vertexes_numbering[i_vertex].clone();
            }
        }

        if 1 < self.topologic_dimension {
            for el in self.elements_by_dim[1].iter() {
                let i_edge = self.neighbour_edges[*el][0];

                assert!(self.edges_neigbour[i_edge].len()>0, "One edge can not be numbered since it is not connected with any numbered highest dimensionnal element");

                let i_elem = self.edges_neigbour[i_edge][0][0];
                let i_interface = self.edges_neigbour[i_edge][0][1];

                let edge_elem = self.elements[i_elem].get_edge(i_interface);
                let edge = self.elements[*el].get_edge(0);
                let o = Edge::get_relative_orientation(&edge, &edge_elem);

                let dof = &dof_set.dof_on_edges[i_interface][o];

                numbering[*el].resize(dof.len(), 0);

                for n in 0..dof.len() {
                    numbering[*el][n] = numbering[i_elem][dof[n]];
                }
            }
        }

        if 2 < self.topologic_dimension {
            for el in self.elements_by_dim[2].iter() {
                let i_face = self.neighbour_faces[*el][0];

                assert!(self.faces_neigbour[i_face].len()>0, "One face can not be numbered since it is not connected with any numbered 3D dimensionnal element");

                let i_elem = self.edges_neigbour[i_face][0][0];
                let i_interface = self.edges_neigbour[i_face][0][1];

                let face_elem = self.elements[i_elem].get_face(i_interface);
                let face = self.elements[*el].get_face(0);
                let o = Face::get_relative_orientation(&face, &face_elem);

                let dof = &dof_set.dof_on_faces[i_interface][o];

                numbering[*el].resize(dof.len(), 0);

                for n in 0..dof.len() {
                    numbering[*el][n] = numbering[i_elem][dof[n]];
                }
            }
        }

        return (numbering, n_dof);
    }
}
