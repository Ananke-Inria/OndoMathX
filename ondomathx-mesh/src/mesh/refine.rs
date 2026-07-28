use std::collections::BTreeSet;

use crate::mesh::*;

//Geometric order of an element type: 1 for P1/Q1, 2 for P2/Q2.
fn get_geometric_order(geometric_element: &GeometricElement) -> usize {
    match geometric_element.msh_id {
        1..=5 => 1,
        8..=12 => 2,
        id => panic!("Uniform refinement is not supported for element type with msh id {id}"),
    }
}

fn mid(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        0.5 * (a[0] + b[0]),
        0.5 * (a[1] + b[1]),
        0.5 * (a[2] + b[2]),
    ]
}

//Corner reference coordinates of the n^dim children of a uniformly refined
//cell. Edges, quadrangles and hexahedra split into the n^dim scaled
//translates of the cell; triangles into the up/down triangles of the
//principal lattice; tetrahedra follow the Freudenthal-Kuhn subdivision: the
//reference tetrahedron is mapped onto the Kuhn simplex {x >= y >= z}, the
//dilation by n of which is tiled by exactly n^3 path-tetrahedra of the Kuhn
//triangulation of the integer grid. The Kuhn subdivision restricted to a
//boundary face is the principal triangulation of that face, so volume and
//boundary children stay conforming; every child has a positive Jacobian
//(odd permutation paths are reordered).
fn get_children_corners(reference_element: &ReferenceElement, n: usize) -> Vec<Vec<[f64; 3]>> {
    let v = reference_element.vertexes;
    let nv = reference_element.get_nvertexes();
    let dim = reference_element.get_dim();
    let h = 1.0 / n as f64;

    let mut children = Vec::<Vec<[f64; 3]>>::new();

    match (dim, nv) {
        //Edge, quadrangle, hexahedron: scaled translates, corners in the
        //vertex-table order of the parent
        (1, 2) | (2, 4) | (3, 8) => {
            for i in 0..n {
                for j in 0..(if dim >= 2 { n } else { 1 }) {
                    for k in 0..(if dim == 3 { n } else { 1 }) {
                        children.push(
                            (0..nv)
                                .map(|w| {
                                    [
                                        (i as f64 + v[w][0]) * h,
                                        (j as f64 + v[w][1]) * h,
                                        (k as f64 + v[w][2]) * h,
                                    ]
                                })
                                .collect(),
                        );
                    }
                }
            }
        }
        //Triangle: principal lattice, n(n+1)/2 upward + n(n-1)/2 downward
        (2, 3) => {
            for j in 0..n {
                for i in 0..(n - j) {
                    let (x, y) = (i as f64 * h, j as f64 * h);

                    children.push(vec![[x, y, 0.0], [x + h, y, 0.0], [x, y + h, 0.0]]);

                    if i + j + 2 <= n {
                        children.push(vec![[x + h, y, 0.0], [x + h, y + h, 0.0], [x, y + h, 0.0]]);
                    }
                }
            }
        }
        //Tetrahedron: Freudenthal-Kuhn. The affine map sending the reference
        //vertices (0, e1, e2, e3) to the Kuhn vertices (0,0,0), (1,0,0),
        //(1,1,0), (1,1,1) has inverse (x,y,z) -> (x-y, y-z, z), and both have
        //a positive determinant, so orientations transfer.
        (3, 4) => {
            //(axis permutation, parity is odd)
            const PATHS: [([usize; 3], bool); 6] = [
                ([0, 1, 2], false),
                ([0, 2, 1], true),
                ([1, 0, 2], true),
                ([1, 2, 0], false),
                ([2, 0, 1], false),
                ([2, 1, 0], true),
            ];

            let m = n as i64;
            for i in 0..m {
                for j in 0..m {
                    for k in 0..m {
                        for (perm, odd) in PATHS.iter() {
                            let mut points = [[i, j, k]; 4];
                            for s in 0..3 {
                                points[s + 1] = points[s];
                                points[s + 1][perm[s]] += 1;
                            }

                            //Keep the path tetrahedra inside the Kuhn simplex
                            if points.iter().all(|p| p[0] >= p[1] && p[1] >= p[2]) {
                                let mut corners: Vec<[f64; 3]> = points
                                    .iter()
                                    .map(|p| {
                                        [
                                            (p[0] - p[1]) as f64 * h,
                                            (p[1] - p[2]) as f64 * h,
                                            p[2] as f64 * h,
                                        ]
                                    })
                                    .collect();

                                if *odd {
                                    corners.swap(2, 3);
                                }

                                children.push(corners);
                            }
                        }
                    }
                }
            }
        }
        _ => panic!("Uniform refinement is not supported for this cell type"),
    }

    assert!(children.len() == n.pow(dim as u32));

    children
}

//Reference positions of the nodes of one child, in the canonical internal
//ordering (corners, then edge midpoints in edge-table order, then
//quadrangular-face centroids in face-table order, then the hexahedron
//center) so that they match the node layout of the child geometric element.
fn get_child_node_positions(
    corners: &[[f64; 3]],
    reference_element: &ReferenceElement,
    order: usize,
) -> Vec<[f64; 3]> {
    let mut positions = corners.to_vec();

    if order == 2 {
        for e in reference_element.edges.iter() {
            positions.push(mid(&corners[e[0]], &corners[e[1]]));
        }

        for f in 0..reference_element.get_nfaces() {
            if !reference_element.is_face_triangular(f) {
                let face = &reference_element.faces[f];
                let mut c = [0.0; 3];
                for &i in face.iter() {
                    for a in 0..3 {
                        c[a] += 0.25 * corners[i][a];
                    }
                }
                positions.push(c);
            }
        }

        if reference_element.get_dim() == 3 && reference_element.get_nvertexes() == 8 {
            let mut c = [0.0; 3];
            for corner in corners.iter() {
                for a in 0..3 {
                    c[a] += 0.125 * corner[a];
                }
            }
            positions.push(c);
        }
    }

    positions
}

impl UnstructuredMesh {
    /// Uniform refinement by a factor 2: every triangle/quadrangle splits
    /// into 4 children, every tetrahedron/hexahedron into 8, every boundary
    /// edge into 2 and every boundary face into 4. See `uniform_refine_by`.
    pub fn uniform_refine(&self) -> (UnstructuredMesh, Vec<Vec<usize>>) {
        self.uniform_refine_by(2)
    }

    /// Uniformly refines the mesh by a factor `n` into a NEW mesh: every
    /// triangle/quadrangle splits into n^2 children, every
    /// tetrahedron/hexahedron into n^3 (Freudenthal-Kuhn subdivision for
    /// tetrahedra), every boundary edge into n and every boundary face into
    /// n^2 — element types, labels and the attached deformation are
    /// preserved. New node positions are evaluated through the parent
    /// element maps, so Q2 meshes keep their curved geometry (node indices
    /// are NOT preserved). Also returns, for every element of the original
    /// mesh, the indices of its children in the refined mesh.
    ///
    /// Requires all highest-dimensional elements to share a single geometric
    /// element type, and all elements to share a single geometric order.
    pub fn uniform_refine_by(&self, n: usize) -> (UnstructuredMesh, Vec<Vec<usize>>) {
        assert!(n >= 1, "The refinement factor must be at least 1");
        assert!(
            n <= 100,
            "The refinement factor {n} is unreasonably large (at most 100)"
        );

        let topologic_dimension = self.get_topologic_dimension();
        assert!(
            topologic_dimension >= 1,
            "Uniform refinement requires elements of dimension >= 1"
        );

        let volume_elements = self.get_elements_by_dimension(topologic_dimension);
        let volume_geo = self
            .get_element_as_ref(volume_elements[0])
            .geometric_element;
        let order = get_geometric_order(volume_geo);

        for &el in volume_elements.iter() {
            assert!(
                self.get_element_as_ref(el).geometric_element.msh_id == volume_geo.msh_id,
                "Uniform refinement requires a single volume element type"
            );
        }

        for el in 0..self.get_nelements() {
            let geo = self.get_element_as_ref(el).geometric_element;
            if geo.reference_element.get_dim() > 0 {
                assert!(
                    get_geometric_order(geo) == order,
                    "Uniform refinement requires all elements to have the same geometric order"
                );
            }
        }

        let reference_element = volume_geo.reference_element;

        //Fine lattice: union of the node positions of all children,
        //deduplicated in reference coordinates; child_tables[c] lists the
        //nodes of child c as lattice indices.
        let mut tree = BTreeSet::<DoF>::new();
        let mut lattice = Vec::<[f64; 3]>::new();
        let mut child_tables = Vec::<Vec<usize>>::new();

        for corners in get_children_corners(reference_element, n) {
            let mut table = Vec::new();

            for uvw in get_child_node_positions(&corners, reference_element, order) {
                let index = match tree.get(&DoF { uvw, index: 0 }) {
                    Some(dof) => dof.index,
                    None => {
                        lattice.push(uvw);
                        tree.insert(DoF {
                            uvw,
                            index: lattice.len() - 1,
                        });
                        lattice.len() - 1
                    }
                };
                table.push(index);
            }

            assert!(table.len() == volume_geo.nshape_functions);
            child_tables.push(table);
        }

        //The lattice seen as a DoFSet gives, through get_numbering, a
        //conforming orientation-aware global numbering of the refined nodes.
        let fine_dof_set = DoFSet::new(lattice.clone(), reference_element);
        let (numbering, n_dof) = self.get_numbering(&fine_dof_set);

        //New node positions, evaluated through the parent element maps (the
        //undeformed isoparametric maps: the attached deformation is carried
        //over and keeps being composed at evaluation time). Nodes shared by
        //several parents are written several times with the same value, since
        //the parents share their interpolated edge/face geometry.
        let nodes = self.get_nodes();
        let mut new_nodes = vec![[0.0f64; 3]; n_dof];

        for &el in volume_elements.iter() {
            let element = self.get_element_as_ref(el);

            for (i, uvw) in lattice.iter().enumerate() {
                let mut xyz = [0.0; 3];

                for k in 0..volume_geo.nshape_functions {
                    let p = volume_geo.shape_functions[k](uvw);
                    let node = &nodes[element.indexes[k]];

                    for a in 0..3 {
                        xyz[a] += p * node[a];
                    }
                }

                new_nodes[numbering[el][i]] = xyz;
            }
        }

        let mut new_elements = Vec::<MeshElement>::new();
        let mut parent_to_children = vec![Vec::<usize>::new(); self.get_nelements()];

        for &el in volume_elements.iter() {
            let element = self.get_element_as_ref(el);

            for table in child_tables.iter() {
                parent_to_children[el].push(new_elements.len());

                new_elements.push(MeshElement {
                    geometric_element: element.geometric_element,
                    label: element.label,
                    indexes: table.iter().map(|&i| numbering[el][i]).collect(),
                });
            }
        }

        //Children of the lower-dimensional (boundary) elements: their nodes
        //already exist in the refined volume elements. Each child node
        //position is mapped into the reference element of a neighbouring
        //volume element — an affine map determined by the relative
        //orientation of the shared edge/face — and looked up in the lattice.
        for d in (1..topologic_dimension).rev() {
            for &el in self.get_elements_by_dimension(d).iter() {
                let element = self.get_element_as_ref(el);
                let boundary_geo = element.geometric_element;
                let boundary_ref = boundary_geo.reference_element;

                let (i_elem, origin, d1, d2) = match d {
                    1 => {
                        let i_edge = self.get_neighbour_edges(el)[0];
                        let neighbours = self.get_edge_neighbours(i_edge);
                        assert!(!neighbours.is_empty(), "A boundary edge can not be refined since it is not connected with any volume element");

                        let [i_elem, i_local] = neighbours[0];
                        let edge_vol = self.get_element_as_ref(i_elem).get_edge(i_local);
                        let o = Edge::get_relative_orientation(&element.get_edge(0), &edge_vol);

                        let e = reference_element.edges[i_local];
                        let a = reference_element.vertexes[e[0]];
                        let b = reference_element.vertexes[e[1]];

                        let (origin, end) = if o == 0 { (a, b) } else { (b, a) };
                        let d1 = [end[0] - origin[0], end[1] - origin[1], end[2] - origin[2]];

                        (i_elem, origin, d1, [0.0; 3])
                    }
                    2 => {
                        let i_face = self.get_neighbour_faces(el)[0];
                        let neighbours = self.get_face_neighbours(i_face);
                        assert!(!neighbours.is_empty(), "A boundary face can not be refined since it is not connected with any volume element");

                        let [i_elem, i_local] = neighbours[0];
                        let face_vol = self.get_element_as_ref(i_elem).get_face(i_local);
                        let o = Face::get_relative_orientation(&element.get_face(0), &face_vol);

                        let f = reference_element.faces[i_local];
                        let (i0, i1, i2) = if reference_element.is_face_triangular(i_local) {
                            (
                                ORIENTATIONS_TRIANGLE[o][0],
                                ORIENTATIONS_TRIANGLE[o][1],
                                ORIENTATIONS_TRIANGLE[o][2],
                            )
                        } else {
                            (
                                ORIENTATIONS_QUADRANGLE[o][0],
                                ORIENTATIONS_QUADRANGLE[o][1],
                                ORIENTATIONS_QUADRANGLE[o][3],
                            )
                        };

                        let origin = reference_element.vertexes[f[i0]];
                        let p1 = reference_element.vertexes[f[i1]];
                        let p2 = reference_element.vertexes[f[i2]];

                        let d1 = [p1[0] - origin[0], p1[1] - origin[1], p1[2] - origin[2]];
                        let d2 = [p2[0] - origin[0], p2[1] - origin[1], p2[2] - origin[2]];

                        (i_elem, origin, d1, d2)
                    }
                    _ => unreachable!(),
                };

                for corners in get_children_corners(boundary_ref, n) {
                    let positions = get_child_node_positions(&corners, boundary_ref, order);
                    assert!(positions.len() == boundary_geo.nshape_functions);

                    let indexes: Vec<usize> = positions
                        .iter()
                        .map(|p| {
                            let uvw = [
                                origin[0] + p[0] * d1[0] + p[1] * d2[0],
                                origin[1] + p[0] * d1[1] + p[1] * d2[1],
                                origin[2] + p[0] * d1[2] + p[1] * d2[2],
                            ];

                            let dof = fine_dof_set
                                .dof_tree_set
                                .get(&DoF { uvw, index: 0 })
                                .expect(
                                    "A refined boundary node is missing from the volume lattice",
                                );

                            numbering[i_elem][dof.index]
                        })
                        .collect();

                    parent_to_children[el].push(new_elements.len());

                    new_elements.push(MeshElement {
                        geometric_element: boundary_geo,
                        label: element.label,
                        indexes,
                    });
                }
            }
        }

        //Dimension-0 elements are carried over unchanged, with their vertex
        //node renumbered through the backfilled numbering row.
        for &el in self.get_elements_by_dimension(0).iter() {
            let element = self.get_element_as_ref(el);

            parent_to_children[el].push(new_elements.len());

            new_elements.push(MeshElement {
                geometric_element: element.geometric_element,
                label: element.label,
                indexes: vec![numbering[el][0]],
            });
        }

        let refined = UnstructuredMesh::new_from_points_and_elements(
            new_nodes,
            new_elements,
            self.get_deformation().cloned(),
        );

        (refined, parent_to_children)
    }
}
