use std::io::{self, BufRead};

use std::collections::BTreeMap;
use std::collections::HashMap;

pub use crate::diffeomorphism::*;
pub use crate::mesh::*;
pub use crate::topology::*;

#[derive(Default, Debug)]
pub struct UnstructuredMesh {
    elements: Vec<MeshElement>,

    //List of nodes
    nodes: Vec<[f64; 3]>,

    geometric_dimension: usize,
    topologic_dimension: usize,

    //Additionnal deformation function
    deformation: Option<Diffeomorphism>,

    //Lists of elements sorted by dimension and label/////////

    //Element indices sorted by dimension
    elements_by_dim: Vec<Vec<usize>>,

    //Elements indices sorted by dimension and label
    elements_by_dim_and_label: Vec<HashMap<usize, Vec<usize>>>,

    //Topology ///////////////////////////////////////////////

    //Number of faces and  edges of the mesh
    nfaces: usize,
    nedges: usize,
    nvertexes: usize,

    //For each element, list of the neighbour faces
    neighbour_faces: Vec<Vec<usize>>,

    // For each face, the list of neighbour elements of highest dimension
    // faces_neighbour[k][i][0] corresponds to the index of the ith element connected to the face k.
    // faces_neighbour[k][i][1] corresponds to the index of the face in the element.
    faces_neighbour: Vec<Vec<[usize; 2]>>,

    //For each element, list of the neighbour edges
    neighbour_edges: Vec<Vec<usize>>,

    // For each edge, the list of neighbour elements of highest dimension
    // edges_neighbour[k][i][0] corresponds to the index of the ith element connected to the edge k.
    // edges_neighbour[k][i][1] corresponds to the index of the edge in the element.
    edges_neighbour: Vec<Vec<[usize; 2]>>,

    //For each element, list of the neighbour vertexes
    neighbour_vertexes: Vec<Vec<usize>>,

    // For each vertex , the list of neighbour elements of highest dimension
    // vertexes_neighbour[k][i][0] corresponds to the index of the ith element connected to the vertex k.
    // vertexes_neighbour[k][i][1] corresponds to the index of the vertex in the element.
    vertexes_neighbour: Vec<Vec<[usize; 2]>>,
    //////////////////////////////////////////////////////////

    //Coloring
    //`coloring` and `elements_by_color` are indexed by/store *ordinals*:
    //positions within elements_by_dim[topologic_dimension], NOT global element
    //indices (in meshes read from files, boundary elements may precede volume
    //elements in the global numbering). The `Mesh` trait accessors translate
    //between the two index spaces via `element_to_ordinal`.
    coloring: Vec<usize>,
    elements_by_color: Vec<Vec<usize>>,
    n_colors: usize,

    //For each element of the highest topological dimension, its ordinal
    //(position in elements_by_dim[topologic_dimension]); usize::MAX for
    //lower-dimensional elements.
    element_to_ordinal: Vec<usize>,
}

impl Mesh for UnstructuredMesh {
    fn get_deformation(&self) -> Option<&Diffeomorphism> {
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
            Some(v) => v.clone(),
            None => vec![],
        }
    }

    fn get_ncolors(&self) -> usize {
        self.n_colors
    }

    fn get_nelements_by_color(&self, color_index: usize) -> usize {
        self.elements_by_color[color_index].len()
    }

    fn get_element_color(&self, element_index: usize) -> usize {
        debug_assert!(
            self.element_to_ordinal[element_index] != usize::MAX,
            "Colors are only defined for elements of the highest topological dimension"
        );
        self.coloring[self.element_to_ordinal[element_index]]
    }

    fn get_element_index_by_color_to_element_index(
        &self,
        color_index: usize,
        element_index_by_color: usize,
    ) -> usize {
        //Translate the stored ordinal back to a global element index.
        self.elements_by_dim[self.topologic_dimension]
            [self.elements_by_color[color_index][element_index_by_color]]
    }

    fn get_element_grad_def(&self, e: usize, uvw: &[f64], grad_def: &mut [f64]) {
        let element = &self.elements[e];
        let geo_element = element.geometric_element;
        let elem_dim = geo_element.reference_element.get_dim();
        let geo_dim = self.geometric_dimension;

        //Jacobian of the isoparametric map, geo_dim × elem_dim row-major.
        grad_def[..geo_dim * elem_dim].fill(0.0);

        //Physical image of uvw, needed to evaluate the deformation Jacobian.
        let mut xyz = [0.0; 3];

        for i in 0..geo_element.nshape_functions {
            let node = &self.nodes[element.indexes[i]];
            let d = geo_element.d_shape_functions[i](uvw);

            if self.deformation.is_some() {
                let p = geo_element.shape_functions[i](uvw);
                for a in 0..geo_dim {
                    xyz[a] += node[a] * p;
                }
            }

            for a in 0..geo_dim {
                for b in 0..elem_dim {
                    grad_def[a * elem_dim + b] += node[a] * d[b];
                }
            }
        }

        //Chain rule with the attached deformation: grad_def ← Df(xyz)·grad_def.
        if let Some(deformation) = &self.deformation {
            deformation.compose_grad_def(&xyz, grad_def, elem_dim);
        }
    }
}

impl UnstructuredMesh {
    pub fn get_element_as_ref(&self, element_index: usize) -> &MeshElement {
        &self.elements[element_index]
    }

    pub fn get_nodes(&self) -> &[[f64; 3]] {
        &self.nodes
    }

    pub fn get_elements(&self) -> &[MeshElement] {
        &self.elements
    }

    //Changes the label of every element through
    //`new_label(element_index, &element) -> label`, then recomputes the
    //label-based indexing (elements_by_dim_and_label) so that
    //get_elements_by_dimension_and_label stays consistent.
    pub fn set_labels<F>(&mut self, mut new_label: F)
    where
        F: FnMut(usize, &MeshElement) -> usize,
    {
        for el in 0..self.elements.len() {
            let label = new_label(el, &self.elements[el]);
            self.elements[el].label = label;
        }

        self.compute_indexing();
    }

    //Topology accessors (zero-cost borrows over the private topology tables)

    pub fn get_nvertexes(&self) -> usize {
        self.nvertexes
    }

    pub fn get_nedges(&self) -> usize {
        self.nedges
    }

    pub fn get_nfaces(&self) -> usize {
        self.nfaces
    }

    //For an element, the global indices of its vertexes/edges/faces
    pub fn get_neighbour_vertexes(&self, element_index: usize) -> &[usize] {
        &self.neighbour_vertexes[element_index]
    }

    pub fn get_neighbour_edges(&self, element_index: usize) -> &[usize] {
        &self.neighbour_edges[element_index]
    }

    pub fn get_neighbour_faces(&self, element_index: usize) -> &[usize] {
        &self.neighbour_faces[element_index]
    }

    //For a vertex/edge/face, the list of neighbour elements of highest
    //dimension as [element_index, local_index_in_element] pairs
    pub fn get_vertex_neighbours(&self, vertex_index: usize) -> &[[usize; 2]] {
        &self.vertexes_neighbour[vertex_index]
    }

    pub fn get_edge_neighbours(&self, edge_index: usize) -> &[[usize; 2]] {
        &self.edges_neighbour[edge_index]
    }

    pub fn get_face_neighbours(&self, face_index: usize) -> &[[usize; 2]] {
        &self.faces_neighbour[face_index]
    }

    pub fn apply_deformation(&mut self) {
        if self.deformation.is_some() {
            let deformation: &Diffeomorphism = self.deformation.as_ref().unwrap();

            for i in 0..self.nodes.len() {
                let mut p = [0.0; 3];
                deformation.eval(&self.nodes[i], &mut p);
                self.nodes[i] = p;
            }

            self.deformation = None;
        }
    }

    /// Composes `diffeo` on top of any existing deformation,
    /// or sets it as the sole deformation if none is present yet.
    pub fn compose_deformation(&mut self, diffeo: Diffeomorphism) {
        if self.deformation.is_some() {
            self.deformation.as_mut().unwrap().compose(diffeo);
        } else {
            self.deformation = Some(diffeo);
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

            self.elements_by_dim_and_label[dim]
                .entry(label)
                .or_default()
                .push(el);
        }

        self.topologic_dimension = 0;
        for d in (1..=3).rev() {
            if !self.elements_by_dim[d].is_empty() {
                self.topologic_dimension = d;
                break;
            }
        }

        self.element_to_ordinal = vec![usize::MAX; self.elements.len()];
        for (ordinal, &el) in self.elements_by_dim[self.topologic_dimension]
            .iter()
            .enumerate()
        {
            self.element_to_ordinal[el] = ordinal;
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
                //The corner vertices are the first entries of `indexes`;
                //high-order nodes (midpoints, face/interior nodes) come after
                //and must not enter the vertex topology.
                let nvertexes = element.geometric_element.reference_element.get_nvertexes();

                for &v in &element.indexes[..nvertexes] {
                    //Number the vertex if it has not been numbered yet
                    if !vertexes_inserted[v] {
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
                    match map_edges.entry(e) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(self.nedges);
                            self.neighbour_edges[el].push(self.nedges);
                            self.nedges += 1;
                        }
                        std::collections::btree_map::Entry::Occupied(entry) => {
                            self.neighbour_edges[el].push(*entry.get());
                        }
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
                    match map_faces.entry(f) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(self.nfaces);
                            self.neighbour_faces[el].push(self.nfaces);
                            self.nfaces += 1;
                        }
                        std::collections::btree_map::Entry::Occupied(entry) => {
                            self.neighbour_faces[el].push(*entry.get());
                        }
                    }
                }
            }
        }

        //Construct the array vertexes_neighbour
        {
            self.vertexes_neighbour.resize(self.nvertexes, Vec::new());
            for el in self.elements_by_dim[self.topologic_dimension].iter() {
                let nv = self.neighbour_vertexes[*el].len();
                for v in 0..nv {
                    self.vertexes_neighbour[self.neighbour_vertexes[*el][v]].push([*el, v]);
                }
            }
        }

        //Construct the array edges_neighbour
        {
            self.edges_neighbour.resize(self.nedges, Vec::new());
            for el in self.elements_by_dim[self.topologic_dimension].iter() {
                let ne = self.neighbour_edges[*el].len();
                for e in 0..ne {
                    self.edges_neighbour[self.neighbour_edges[*el][e]].push([*el, e]);
                }
            }
        }

        //Construct the array faces_neighbour
        {
            if self.topologic_dimension == 3 {
                self.faces_neighbour.resize(self.nfaces, Vec::new());
                for el in self.elements_by_dim[self.topologic_dimension].iter() {
                    let nf = self.neighbour_faces[*el].len();
                    for f in 0..nf {
                        self.faces_neighbour[self.neighbour_faces[*el][f]].push([*el, f]);
                        // First the orientation index is set to 0
                    }
                }
            }
        }
    }

    //Every lower-dimensional element must be the trace (edge/face) of an
    //element of the highest topological dimension: the DOF numbering, the
    //boundary model and refinement all rely on this invariant, so a dangling
    //element (e.g. a geometry polyline meshed at a different resolution than
    //the cells) is rejected loudly at construction.
    fn check_dangling_elements(&self) {
        for d in 1..self.topologic_dimension {
            for &el in self.elements_by_dim[d].iter() {
                let dangling = if d == 1 {
                    self.edges_neighbour[self.neighbour_edges[el][0]].is_empty()
                } else {
                    self.faces_neighbour[self.neighbour_faces[el][0]].is_empty()
                };

                assert!(
                    !dangling,
                    "Element {el} of dimension {d} is dangling: it is not the trace of any element of dimension {}",
                    self.topologic_dimension
                );
            }
        }
    }

    pub fn compute_graph(&self) -> Vec<Vec<usize>> {
        let nelements = self.elements_by_dim[self.topologic_dimension].len();

        let mut graph = vec![vec![]; nelements];

        //Fill the graph matrix using vertexes
        for (i, graph_row) in graph.iter_mut().enumerate() {
            let ielement = self.elements_by_dim[self.topologic_dimension][i];

            for v in self.neighbour_vertexes[ielement].iter() {
                for i_v_e in self.vertexes_neighbour[*v].iter() {
                    if i_v_e[0] != ielement {
                        graph_row.push(self.element_to_ordinal[i_v_e[0]]);
                    }
                }
            }
        }

        //Sort and remove duplicate
        for graph_row in graph.iter_mut() {
            graph_row.sort();
            graph_row.dedup();
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
        deformation: Option<Diffeomorphism>,
    ) -> UnstructuredMesh {
        let mut mesh = UnstructuredMesh {
            nodes,
            elements,
            deformation,
            ..Default::default()
        };

        mesh.compute_dimension();
        mesh.compute_indexing();
        mesh.compute_topology();
        mesh.check_dangling_elements();
        mesh.compute_coloring();

        mesh
    }
    pub fn new_from_mesh(path: &str) -> Result<UnstructuredMesh, std::io::Error> {
        let points = UnstructuredMesh::read_points_from_mesh(path)?;
        let elements = UnstructuredMesh::read_elements_from_mesh(path)?;

        Ok(UnstructuredMesh::new_from_points_and_elements(
            points, elements, None,
        ))
    }
    fn read_points_from_mesh(path: &str) -> Result<Vec<[f64; 3]>, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let file_buf = io::BufReader::new(file);
        let mut vertices_lines = file_buf
            .lines()
            .map(|line_res| line_res.unwrap())
            .skip_while(|line| line.trim() != "Vertices")
            .skip(1);

        let npnt = vertices_lines
            .next()
            .unwrap()
            .trim()
            .parse::<usize>()
            .unwrap();
        let mut points: Vec<[f64; 3]> = Vec::with_capacity(npnt);

        for _ in 0..npnt {
            let line = vertices_lines.next().unwrap();
            let fields: Vec<&str> = line.split_whitespace().collect();
            points.push([
                fields[0].parse::<f64>().unwrap(),
                fields[1].parse::<f64>().unwrap(),
                fields[2].parse::<f64>().unwrap(),
            ]);
        }

        Ok(points)
    }
    fn read_elements_from_mesh(path: &str) -> Result<Vec<MeshElement>, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let file_buf = io::BufReader::new(file);
        let all_lines: Vec<String> = file_buf.lines().map(|line_res| line_res.unwrap()).collect();

        let mut elements: Vec<MeshElement> = Vec::new();

        //Medit "HexahedraQ2" section: 8 vertices, bottom edge ring (m01, m12, m23, m03), top edge
        //ring (m45, m56, m67, m47), vertical edges (m04, m15, m26, m37), face
        //centers (bottom, top, front, right, back, left), center.
        //Internal position i is read from file position PERM[i].
        const HEXAHEDRON_27_FROM_MESH: [usize; 27] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15, 20, 22, 25, 23,
            24, 21, 26,
        ];

        //(section name, element, number of nodes, file -> internal permutation)
        //EdgesP2/TrianglesP2/TetrahedraP2/QuadrilateralsQ2 follow the
        //Medit/libMeshb P2 convention: vertices, then the edge nodes in the
        //edge order (1,2),(2,3),(3,1)/(4,1)[, ...], then face/interior nodes —
        //which coincides with the internal ordering of reference_elements.rs,
        //hence no permutation.
        let section_specs: &[(&str, &GeometricElement, usize, Option<&[usize]>)] = &[
            ("Edges", &EDGE2, 2, None),
            ("Triangles", &TRIANGLE3, 3, None),
            ("Quadrilaterals", &QUADRANGLE4, 4, None),
            ("Tetrahedra", &TETRAHEDRON4, 4, None),
            ("Hexahedra", &HEXAHEDRON8, 8, None),
            ("EdgesP2", &EDGE3, 3, None),
            ("TrianglesP2", &TRIANGLE6, 6, None),
            ("QuadrilateralsQ2", &QUADRANGLE9, 9, None),
            ("TetrahedraP2", &TETRAHEDRON10, 10, None),
            (
                "HexahedraQ2",
                &HEXAHEDRON27,
                27,
                Some(&HEXAHEDRON_27_FROM_MESH),
            ),
        ];

        for &(section_name, geo_element, nnodes, permutation) in section_specs {
            if let Some(pos) = all_lines.iter().position(|l| l.trim() == section_name) {
                let nelem = all_lines[pos + 1].trim().parse::<usize>().unwrap();

                for k in 0..nelem {
                    let line = &all_lines[pos + 2 + k];
                    let fields: Vec<&str> = line.split_whitespace().collect();

                    let indexes: Vec<usize> = match permutation {
                        None => (0..nnodes)
                            .map(|i| fields[i].parse::<usize>().unwrap() - 1)
                            .collect(),
                        Some(perm) => perm
                            .iter()
                            .map(|&p| fields[p].parse::<usize>().unwrap() - 1)
                            .collect(),
                    };

                    let label = fields[nnodes].parse::<usize>().unwrap();

                    elements.push(MeshElement {
                        geometric_element: geo_element,
                        label,
                        indexes,
                    });
                }
            }
        }

        Ok(elements)
    }
    pub fn new_from_msh(path: &str) -> Result<UnstructuredMesh, std::io::Error> {
        // Check that the file is in Gmsh format 2.2
        let file = std::fs::File::open(path)?;
        let file_buf = io::BufReader::new(file);
        let mut lines = file_buf
            .lines()
            .map(|line_res| line_res.unwrap())
            .skip_while(|line| line != "$MeshFormat")
            .skip(1);
        //All 2.x versions share the same ASCII $Nodes/$Elements structure
        //(the writer always emits 2.2).
        let version_line = lines.next().expect("Missing $MeshFormat section");
        let mut fields = version_line.split(' ');
        let version = fields.next().unwrap();
        let file_type = fields.next().unwrap_or("0");
        assert!(
            version == "2" || version.starts_with("2."),
            "Only Gmsh format 2.x is supported, found {}",
            version
        );
        assert!(
            file_type == "0",
            "Only ASCII Gmsh files are supported, found file-type {}",
            file_type
        );

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
        Ok(points)
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

        //Node permutations for high-order elements whose Gmsh ordering differs
        //from the internal ordering of geometric_elements.rs (internal
        //position i is read from file position PERM[i]).

        //Gmsh tetrahedron10: the last two edge midpoints (m13, m23) are
        //swapped with respect to TETRAHEDRON_EDGES_TOPOLOGY.
        const TETRAHEDRON_10_FROM_MSH: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8];

        //Gmsh hexahedron27: edge midpoints in Gmsh's edge order
        //(0,1),(0,3),(0,4),(1,2),(1,5),(2,3),(2,6),(3,7),(4,5),(4,7),(5,6),(6,7);
        //face centers and center coincide with the internal ordering.
        const HEXAHEDRON_27_FROM_MSH: [usize; 27] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 10, 12, 14, 15, 16, 18, 19, 17, 20, 21, 22, 23,
            24, 25, 26,
        ];

        for element_line in elements_lines {
            let split_element_line: Vec<&str> = element_line.split(' ').collect();

            // Parse number of tags and extract label (physical group)
            let ntags = split_element_line[2].parse::<usize>().unwrap();
            let label = if ntags > 0 {
                split_element_line[3].parse::<usize>().unwrap()
            } else {
                0
            };

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
                        label,
                        indexes: point_indexes.to_vec(),
                    })
                }
                "2" => {
                    // Triangle3
                    let line_len = split_element_line.len();
                    let point_indexes = [
                        split_element_line[line_len - 3].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 2].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 1].parse::<usize>().unwrap() - 1,
                    ];

                    elements.push(MeshElement {
                        geometric_element: &TRIANGLE3,
                        label,
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
                        label,
                        indexes: point_indexes.to_vec(),
                    })
                }
                "4" => {
                    // Tetrahedron4
                    let line_len = split_element_line.len();
                    let point_indexes = [
                        split_element_line[line_len - 4].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 3].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 2].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 1].parse::<usize>().unwrap() - 1,
                    ];

                    elements.push(MeshElement {
                        geometric_element: &TETRAHEDRON4,
                        label,
                        indexes: point_indexes.to_vec(),
                    })
                }
                "5" => {
                    // Hexahedron8
                    let line_len = split_element_line.len();
                    let point_indexes = [
                        split_element_line[line_len - 8].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 7].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 6].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 5].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 4].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 3].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 2].parse::<usize>().unwrap() - 1,
                        split_element_line[line_len - 1].parse::<usize>().unwrap() - 1,
                    ];

                    elements.push(MeshElement {
                        geometric_element: &HEXAHEDRON8,
                        label,
                        indexes: point_indexes.to_vec(),
                    })
                }

                "8" => {
                    // Edge3 (Gmsh node order coincides with the internal one)
                    let base = split_element_line.len() - 3;
                    let indexes: Vec<usize> = (0..3)
                        .map(|i| split_element_line[base + i].parse::<usize>().unwrap() - 1)
                        .collect();

                    elements.push(MeshElement {
                        geometric_element: &EDGE3,
                        label,
                        indexes,
                    })
                }
                "9" => {
                    // Triangle6 (Gmsh node order coincides with the internal one)
                    let base = split_element_line.len() - 6;
                    let indexes: Vec<usize> = (0..6)
                        .map(|i| split_element_line[base + i].parse::<usize>().unwrap() - 1)
                        .collect();

                    elements.push(MeshElement {
                        geometric_element: &TRIANGLE6,
                        label,
                        indexes,
                    })
                }
                "10" => {
                    // Quadrangle9 (Gmsh node order coincides with the internal one)
                    let base = split_element_line.len() - 9;
                    let indexes: Vec<usize> = (0..9)
                        .map(|i| split_element_line[base + i].parse::<usize>().unwrap() - 1)
                        .collect();

                    elements.push(MeshElement {
                        geometric_element: &QUADRANGLE9,
                        label,
                        indexes,
                    })
                }
                "11" => {
                    // Tetrahedron10 (permuted from the Gmsh node ordering)
                    let base = split_element_line.len() - 10;
                    let indexes: Vec<usize> = TETRAHEDRON_10_FROM_MSH
                        .iter()
                        .map(|&p| split_element_line[base + p].parse::<usize>().unwrap() - 1)
                        .collect();

                    elements.push(MeshElement {
                        geometric_element: &TETRAHEDRON10,
                        label,
                        indexes,
                    })
                }
                "12" => {
                    // Hexahedron27 (permuted from the Gmsh node ordering)
                    let base = split_element_line.len() - 27;
                    let indexes: Vec<usize> = HEXAHEDRON_27_FROM_MSH
                        .iter()
                        .map(|&p| split_element_line[base + p].parse::<usize>().unwrap() - 1)
                        .collect();

                    elements.push(MeshElement {
                        geometric_element: &HEXAHEDRON27,
                        label,
                        indexes,
                    })
                }

                "15" => {}
                _ => panic!("Element type {} not supported yet", split_element_line[1]),
            }
        }

        Ok(elements)
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

                if !dof_on_vertexes.is_empty() {
                    //create the numbering on the node (if needed)
                    if vertexes_numbering[k].is_empty() {
                        vertexes_numbering[k].resize(dof_on_vertexes.len(), 0);

                        for num in vertexes_numbering[k].iter_mut() {
                            *num = n_dof;
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

                    if !dof_in_edge.is_empty() {
                        //create the numbering on the node (if needed)
                        if edges_numbering[k].is_empty() {
                            edges_numbering[k].resize(dof_in_edge.len(), 0);

                            for num in edges_numbering[k].iter_mut() {
                                *num = n_dof;
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

                    if !dof_in_face.is_empty() {
                        //create the numbering on the node (if needed)
                        if faces_numbering[k].is_empty() {
                            faces_numbering[k].resize(dof_in_face.len(), 0);

                            for num in faces_numbering[k].iter_mut() {
                                *num = n_dof;
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

                assert!(!vertexes_numbering[i_vertex].is_empty(), "One vertex can not be numbered since it is not connected with any numbered highest dimensionnal element");

                numbering[*el] = vertexes_numbering[i_vertex].clone();
            }
        }

        if 1 < self.topologic_dimension {
            for el in self.elements_by_dim[1].iter() {
                let i_edge = self.neighbour_edges[*el][0];

                assert!(!self.edges_neighbour[i_edge].is_empty(), "One edge can not be numbered since it is not connected with any numbered highest dimensionnal element");

                let i_elem = self.edges_neighbour[i_edge][0][0];
                let i_interface = self.edges_neighbour[i_edge][0][1];

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

                assert!(!self.faces_neighbour[i_face].is_empty(), "One face can not be numbered since it is not connected with any numbered 3D dimensionnal element");

                let i_elem = self.faces_neighbour[i_face][0][0];
                let i_interface = self.faces_neighbour[i_face][0][1];

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

        (numbering, n_dof)
    }
}
