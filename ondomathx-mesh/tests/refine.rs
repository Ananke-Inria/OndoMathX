extern crate ondomathx_mesh;

#[cfg(test)]
mod tests {
    use ondomathx_mesh::dof_set::*;
    use ondomathx_mesh::geometric_elements::*;
    use ondomathx_mesh::mesh::circle::*;
    use ondomathx_mesh::mesh::square::*;
    use ondomathx_mesh::mesh::Mesh;

    fn mid(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [
            0.5 * (a[0] + b[0]),
            0.5 * (a[1] + b[1]),
            0.5 * (a[2] + b[2]),
        ]
    }

    //Every new element must be the child of exactly one parent
    fn check_partition(parent_to_children: &[Vec<usize>], nelements: usize) {
        let mut all: Vec<usize> = parent_to_children.iter().flatten().copied().collect();
        all.sort();
        assert_eq!(all, (0..nelements).collect::<Vec<usize>>());
    }

    fn eval_map(mesh: &UnstructuredMesh, el: usize, uvw: &[f64]) -> [f64; 3] {
        let element = mesh.get_element_as_ref(el);
        let geo = element.geometric_element;

        let mut xyz = [0.0; 3];
        for k in 0..geo.nshape_functions {
            let p = geo.shape_functions[k](uvw);
            let node = mesh.get_node(element.indexes[k]);
            for a in 0..3 {
                xyz[a] += p * node[a];
            }
        }
        xyz
    }

    fn tet_signed_volume(mesh: &UnstructuredMesh, el: usize) -> f64 {
        let indexes = &mesh.get_element_as_ref(el).indexes;
        let p: Vec<[f64; 3]> = indexes[..4].iter().map(|&i| mesh.get_node(i)).collect();

        let mut m = [[0.0; 3]; 3];
        for k in 0..3 {
            for a in 0..3 {
                m[k][a] = p[k + 1][a] - p[0][a];
            }
        }

        (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            / 6.0
    }

    #[test]
    fn square_q1() {
        let mesh = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None, None).get_unstructured_mesh();
        let (refined, parent_to_children) = mesh.uniform_refine();

        assert_eq!(refined.get_nnodes(), 25);
        assert_eq!(refined.get_elements_by_dimension(2).len(), 16);
        assert_eq!(refined.get_elements_by_dimension(1).len(), 16);
        check_partition(&parent_to_children, refined.get_nelements());

        for (el, children) in parent_to_children.iter().enumerate() {
            let dim = mesh
                .get_element_as_ref(el)
                .geometric_element
                .reference_element
                .get_dim();

            assert_eq!(children.len(), if dim == 2 { 4 } else { 2 });

            for &child in children.iter() {
                assert_eq!(
                    refined.get_element_as_ref(child).label,
                    mesh.get_element_as_ref(el).label
                );
            }
        }

        //Each of the 4 labelled boundary groups doubles its element count
        for label in 0..4 {
            assert_eq!(
                refined.get_elements_by_dimension_and_label(label, 1).len(),
                4
            );
        }

        //Total area, by the shoelace formula on the (straight) quad cells
        let mut area = 0.0;
        for &el in refined.get_elements_by_dimension(2).iter() {
            let indexes = &refined.get_element_as_ref(el).indexes;
            let p: Vec<[f64; 3]> = indexes[..4].iter().map(|&i| refined.get_node(i)).collect();

            for k in 0..4 {
                let q = (k + 1) % 4;
                area += 0.5 * (p[k][0] * p[q][1] - p[q][0] * p[k][1]);
            }
        }
        assert!((area - 1.0).abs() < 1e-12);
    }

    #[test]
    fn square_q2() {
        let mesh =
            Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None, Some(true)).get_unstructured_mesh();
        let (refined, parent_to_children) = mesh.uniform_refine();

        assert_eq!(refined.get_nnodes(), 81);
        assert_eq!(refined.get_elements_by_dimension(2).len(), 16);
        assert_eq!(refined.get_elements_by_dimension(1).len(), 16);
        check_partition(&parent_to_children, refined.get_nelements());

        for &el in refined.get_elements_by_dimension(2).iter() {
            assert_eq!(refined.get_element_as_ref(el).geometric_element.msh_id, 10);
        }
        for &el in refined.get_elements_by_dimension(1).iter() {
            assert_eq!(refined.get_element_as_ref(el).geometric_element.msh_id, 8);
        }

        //On the unit square all refined Q2 nodes sit on the uniform 1/8 grid
        for i in 0..refined.get_nnodes() {
            let p = refined.get_node(i);
            for c in p.iter().take(2) {
                let s = c * 8.0;
                assert!((s - s.round()).abs() < 1e-12);
            }
            assert_eq!(p[2], 0.0);
        }
    }

    #[test]
    fn circle_q2_preserves_curved_boundary() {
        let mesh = Circle::new([0.0, 0.0], 1.0, 4, None, Some(true)).get_unstructured_mesh();
        let (refined, parent_to_children) = mesh.uniform_refine();

        check_partition(&parent_to_children, refined.get_nelements());

        //Each boundary EDGE3 child must lie exactly on the parent's Q2 curve:
        //child 0 covers [0, 1/2] of the parent, child 1 covers [1/2, 1], and
        //the EDGE3 nodes are ordered (v0, v1, midpoint).
        for &el in mesh.get_elements_by_dimension(1).iter() {
            let children = &parent_to_children[el];
            assert_eq!(children.len(), 2);

            for (c, &child) in children.iter().enumerate() {
                let offset = 0.5 * c as f64;

                for (n, t_child) in [0.0, 1.0, 0.5].iter().enumerate() {
                    let t_parent = offset + 0.5 * t_child;
                    let expected = eval_map(&mesh, el, &[t_parent, 0.0, 0.0]);
                    let node = refined.get_node(refined.get_element_as_ref(child).indexes[n]);

                    for a in 0..3 {
                        assert!((node[a] - expected[a]).abs() < 1e-12);
                    }
                }
            }
        }

        //The refined mesh is conforming: an isoparametric Q2 numbering has
        //exactly one DoF per node
        let mut positions: Vec<[f64; 3]> = QUADRANGLE_VERTEXES.to_vec();
        for e in QUADRANGLE_EDGES_TOPOLOGY.iter() {
            positions.push(mid(&QUADRANGLE_VERTEXES[e[0]], &QUADRANGLE_VERTEXES[e[1]]));
        }
        positions.push([0.5, 0.5, 0.0]);

        let dof_set = DoFSet::new(positions, &QUADRANGLE);
        let (_, n_dof) = refined.get_numbering(&dof_set);
        assert_eq!(n_dof, refined.get_nnodes());

        refined
            .write_msh("./tests/results/circle_q2_refined.msh")
            .unwrap();
    }

    //Large quad mesh (816 quads + 116 boundary edges); writes the refined
    //mesh under tests/results/ for visual inspection
    #[test]
    fn potatoe_q1() {
        let mesh = UnstructuredMesh::new_from_msh("./assets/Potatoe.msh").unwrap();
        let (refined, parent_to_children) = mesh.uniform_refine();

        assert_eq!(refined.get_elements_by_dimension(2).len(), 4 * 816);
        assert_eq!(refined.get_elements_by_dimension(1).len(), 2 * 116);
        assert_eq!(
            refined.get_nnodes(),
            mesh.get_nvertexes() + mesh.get_nedges() + 816
        );
        check_partition(&parent_to_children, refined.get_nelements());

        //Total area preserved (straight quads, shoelace formula)
        let area = |m: &UnstructuredMesh| -> f64 {
            m.get_elements_by_dimension(2)
                .iter()
                .map(|&el| {
                    let indexes = &m.get_element_as_ref(el).indexes;
                    let p: Vec<[f64; 3]> = indexes[..4].iter().map(|&i| m.get_node(i)).collect();

                    (0..4)
                        .map(|k| {
                            let q = (k + 1) % 4;
                            0.5 * (p[k][0] * p[q][1] - p[q][0] * p[k][1])
                        })
                        .sum::<f64>()
                })
                .sum()
        };

        let a = area(&mesh);
        assert!((area(&refined) - a).abs() < 1e-12 * a.abs());

        refined
            .write_msh("./tests/results/Potatoe_refined.msh")
            .unwrap();

        //Second output for visual inspection of the children grouping:
        //every child labelled with its parent's element index
        let mut parent_of = vec![0; refined.get_nelements()];
        for (parent, children) in parent_to_children.iter().enumerate() {
            for &child in children.iter() {
                parent_of[child] = parent;
            }
        }

        let mut relabelled = refined;
        relabelled.set_labels(|el, _| parent_of[el]);
        relabelled
            .write_msh("./tests/results/Potatoe_refined_by_parent.msh")
            .unwrap();
    }

    //Gmsh 2.1 file (exercises the 2.x reader compatibility); large triangle
    //mesh: 4724 triangles + 648 boundary edges, points skipped by the reader
    #[test]
    fn fractals_triangles_p1() {
        let mesh = UnstructuredMesh::new_from_msh("./assets/Fractals.msh").unwrap();

        assert_eq!(mesh.get_elements_by_dimension(2).len(), 4724);
        assert_eq!(mesh.get_elements_by_dimension(1).len(), 648);

        let (refined, parent_to_children) = mesh.uniform_refine();

        assert_eq!(refined.get_elements_by_dimension(2).len(), 4 * 4724);
        assert_eq!(refined.get_elements_by_dimension(1).len(), 2 * 648);
        assert_eq!(
            refined.get_nnodes(),
            mesh.get_nvertexes() + mesh.get_nedges()
        );
        check_partition(&parent_to_children, refined.get_nelements());

        //Total area preserved (straight triangles)
        let area = |m: &UnstructuredMesh| -> f64 {
            m.get_elements_by_dimension(2)
                .iter()
                .map(|&el| {
                    let indexes = &m.get_element_as_ref(el).indexes;
                    let p: Vec<[f64; 3]> = indexes.iter().map(|&i| m.get_node(i)).collect();

                    0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                        - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
                })
                .sum()
        };

        let a = area(&mesh);
        assert!((area(&refined) - a).abs() < 1e-12 * a.abs());

        refined
            .write_msh("./tests/results/Fractals_refined.msh")
            .unwrap();
    }

    #[test]
    fn cube_tet_p1() {
        let mesh = UnstructuredMesh::new_from_mesh("./assets/Cube.mesh").unwrap();
        assert_eq!(mesh.get_elements_by_dimension(3).len(), 6);

        let (refined, parent_to_children) = mesh.uniform_refine();

        assert_eq!(refined.get_elements_by_dimension(3).len(), 48);
        assert_eq!(
            refined.get_nnodes(),
            mesh.get_nvertexes() + mesh.get_nedges()
        );
        check_partition(&parent_to_children, refined.get_nelements());

        //Each parent splits into 8 children of the same orientation whose
        //volumes sum to the parent volume (validates the red-refinement
        //corner tables)
        for &el in mesh.get_elements_by_dimension(3).iter() {
            let parent_volume = tet_signed_volume(&mesh, el);
            assert_eq!(parent_to_children[el].len(), 8);

            let mut children_volume = 0.0;
            for &child in parent_to_children[el].iter() {
                let v = tet_signed_volume(&refined, child);
                assert!(v * parent_volume > 0.0);
                children_volume += v;
            }

            assert!((children_volume - parent_volume).abs() < 1e-12);
        }

        //Output for visual inspection, children labelled with their parent's
        //element index to show the 8-child groups
        let mut parent_of = vec![0; refined.get_nelements()];
        for (parent, children) in parent_to_children.iter().enumerate() {
            for &child in children.iter() {
                parent_of[child] = parent;
            }
        }

        let mut relabelled = refined;
        relabelled.set_labels(|el, _| parent_of[el]);
        relabelled
            .write_msh("./tests/results/Cube_tet_refined_by_parent.msh")
            .unwrap();
    }

    #[test]
    fn hexahedron_q2_with_boundary_face() {
        //Unit-cube HEXAHEDRON27 built from the canonical internal ordering:
        //vertices, edge midpoints (edge-table order), face centroids
        //(face-table order), center.
        let mut nodes: Vec<[f64; 3]> = HEXAHEDRON_VERTEXES.to_vec();

        for e in HEXAHEDRON_EDGES_TOPOLOGY.iter() {
            nodes.push(mid(&HEXAHEDRON_VERTEXES[e[0]], &HEXAHEDRON_VERTEXES[e[1]]));
        }

        for f in HEXAHEDRON_FACES_TOPOLOGY.iter() {
            let mut c = [0.0; 3];
            for &i in f.iter() {
                for a in 0..3 {
                    c[a] += 0.25 * HEXAHEDRON_VERTEXES[i][a];
                }
            }
            nodes.push(c);
        }

        nodes.push([0.5, 0.5, 0.5]);

        let elements = vec![
            MeshElement {
                geometric_element: &HEXAHEDRON27,
                label: 0,
                indexes: (0..27).collect(),
            },
            //Bottom face (z = 0) as a QUADRANGLE9, listed rotated relative to
            //the hexahedron's local face to exercise the orientation mapping
            MeshElement {
                geometric_element: &QUADRANGLE9,
                label: 7,
                indexes: vec![1, 2, 3, 0, 9, 10, 11, 8, 20],
            },
        ];

        let mesh = UnstructuredMesh::new_from_points_and_elements(nodes, elements, None);
        let (refined, parent_to_children) = mesh.uniform_refine();

        assert_eq!(refined.get_nnodes(), 125);
        assert_eq!(refined.get_elements_by_dimension(3).len(), 8);
        assert_eq!(refined.get_elements_by_dimension(2).len(), 4);
        assert_eq!(parent_to_children[0].len(), 8);
        assert_eq!(parent_to_children[1].len(), 4);
        check_partition(&parent_to_children, refined.get_nelements());

        //All refined nodes sit on the uniform 1/4 grid of the unit cube
        for i in 0..refined.get_nnodes() {
            let p = refined.get_node(i);
            for c in p.iter() {
                let s = c * 4.0;
                assert!((s - s.round()).abs() < 1e-12);
                assert!(*c > -1e-12 && *c < 1.0 + 1e-12);
            }
        }

        //Boundary children are QUADRANGLE9, keep their label and stay on the
        //z = 0 face
        for &child in parent_to_children[1].iter() {
            let element = refined.get_element_as_ref(child);

            assert_eq!(element.geometric_element.msh_id, 10);
            assert_eq!(element.label, 7);

            for &i in element.indexes.iter() {
                assert!(refined.get_node(i)[2].abs() < 1e-13);
            }
        }

        //Same mesh, factor 3: 27 children on the 1/6 grid
        let (refined3, parent_to_children3) = mesh.uniform_refine_by(3);

        assert_eq!(refined3.get_nnodes(), 343);
        assert_eq!(refined3.get_elements_by_dimension(3).len(), 27);
        assert_eq!(refined3.get_elements_by_dimension(2).len(), 9);
        assert_eq!(parent_to_children3[0].len(), 27);
        assert_eq!(parent_to_children3[1].len(), 9);
        check_partition(&parent_to_children3, refined3.get_nelements());

        for i in 0..refined3.get_nnodes() {
            let p = refined3.get_node(i);
            for c in p.iter() {
                let s = c * 6.0;
                assert!((s - s.round()).abs() < 1e-12);
            }
        }

        for &child in parent_to_children3[1].iter() {
            let element = refined3.get_element_as_ref(child);
            assert_eq!(element.label, 7);
            for &i in element.indexes.iter() {
                assert!(refined3.get_node(i)[2].abs() < 1e-13);
            }
        }

        //Outputs for visual inspection
        refined
            .write_msh("./tests/results/Hexahedron_q2_refined.msh")
            .unwrap();
        refined3
            .write_msh("./tests/results/Hexahedron_q2_refined_by_3.msh")
            .unwrap();
    }

    #[test]
    fn square_q1_by_3() {
        let mesh = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None, None).get_unstructured_mesh();
        let (refined, parent_to_children) = mesh.uniform_refine_by(3);

        assert_eq!(refined.get_nnodes(), 49);
        assert_eq!(refined.get_elements_by_dimension(2).len(), 36);
        assert_eq!(refined.get_elements_by_dimension(1).len(), 24);
        check_partition(&parent_to_children, refined.get_nelements());

        for (el, children) in parent_to_children.iter().enumerate() {
            let dim = mesh
                .get_element_as_ref(el)
                .geometric_element
                .reference_element
                .get_dim();

            assert_eq!(children.len(), if dim == 2 { 9 } else { 3 });
        }

        //Total area, by the shoelace formula on the (straight) quad cells
        let mut area = 0.0;
        for &el in refined.get_elements_by_dimension(2).iter() {
            let indexes = &refined.get_element_as_ref(el).indexes;
            let p: Vec<[f64; 3]> = indexes[..4].iter().map(|&i| refined.get_node(i)).collect();

            for k in 0..4 {
                let q = (k + 1) % 4;
                area += 0.5 * (p[k][0] * p[q][1] - p[q][0] * p[k][1]);
            }
        }
        assert!((area - 1.0).abs() < 1e-12);

        refined
            .write_msh("./tests/results/Square_refined_by_3.msh")
            .unwrap();
    }

    #[test]
    fn circle_q2_by_3_preserves_curved_boundary() {
        let mesh = Circle::new([0.0, 0.0], 1.0, 4, None, Some(true)).get_unstructured_mesh();
        let (refined, parent_to_children) = mesh.uniform_refine_by(3);

        check_partition(&parent_to_children, refined.get_nelements());

        //Boundary EDGE3 child c covers [c/3, (c+1)/3] of its parent and must
        //lie exactly on the parent's Q2 curve
        for &el in mesh.get_elements_by_dimension(1).iter() {
            let children = &parent_to_children[el];
            assert_eq!(children.len(), 3);

            for (c, &child) in children.iter().enumerate() {
                for (n, t_child) in [0.0, 1.0, 0.5].iter().enumerate() {
                    let t_parent = (c as f64 + t_child) / 3.0;
                    let expected = eval_map(&mesh, el, &[t_parent, 0.0, 0.0]);
                    let node = refined.get_node(refined.get_element_as_ref(child).indexes[n]);

                    for a in 0..3 {
                        assert!((node[a] - expected[a]).abs() < 1e-12);
                    }
                }
            }
        }

        //Conformity: an isoparametric Q2 numbering has one DoF per node
        let mut positions: Vec<[f64; 3]> = QUADRANGLE_VERTEXES.to_vec();
        for e in QUADRANGLE_EDGES_TOPOLOGY.iter() {
            positions.push(mid(&QUADRANGLE_VERTEXES[e[0]], &QUADRANGLE_VERTEXES[e[1]]));
        }
        positions.push([0.5, 0.5, 0.0]);

        let dof_set = DoFSet::new(positions, &QUADRANGLE);
        let (_, n_dof) = refined.get_numbering(&dof_set);
        assert_eq!(n_dof, refined.get_nnodes());

        refined
            .write_msh("./tests/results/circle_q2_refined_by_3.msh")
            .unwrap();
    }

    //Single reference TETRAHEDRON10 with its four TRIANGLE6 faces, factor 3:
    //stresses the Freudenthal-Kuhn subdivision and its conformity with the
    //principal triangulation of the boundary faces (P2 lattice included)
    #[test]
    fn tetrahedron10_by_3() {
        let mut nodes: Vec<[f64; 3]> = TETRAHEDRON_VERTEXES.to_vec();
        for e in TETRAHEDRON_EDGES_TOPOLOGY.iter() {
            nodes.push(mid(
                &TETRAHEDRON_VERTEXES[e[0]],
                &TETRAHEDRON_VERTEXES[e[1]],
            ));
        }

        //TRIANGLE6 boundary faces: face vertices then the midpoints of the
        //face edges (0,1), (1,2), (0,2), using the global edge-midpoint ids
        let faces: [[usize; 6]; 4] = [
            [0, 1, 2, 4, 5, 6],
            [0, 1, 3, 4, 8, 7],
            [0, 2, 3, 6, 9, 7],
            [1, 2, 3, 5, 9, 8],
        ];

        let mut elements = vec![MeshElement {
            geometric_element: &TETRAHEDRON10,
            label: 0,
            indexes: (0..10).collect(),
        }];
        for (k, f) in faces.iter().enumerate() {
            elements.push(MeshElement {
                geometric_element: &TRIANGLE6,
                label: k + 1,
                indexes: f.to_vec(),
            });
        }

        let mesh = UnstructuredMesh::new_from_points_and_elements(nodes, elements, None);
        let (refined, parent_to_children) = mesh.uniform_refine_by(3);

        assert_eq!(refined.get_elements_by_dimension(3).len(), 27);
        assert_eq!(refined.get_elements_by_dimension(2).len(), 4 * 9);
        assert_eq!(parent_to_children[0].len(), 27);
        check_partition(&parent_to_children, refined.get_nelements());

        //All children are positively oriented and tile the parent volume
        let mut volume = 0.0;
        for &child in parent_to_children[0].iter() {
            let v = tet_signed_volume(&refined, child);
            assert!(v > 0.0);
            volume += v;
        }
        assert!((volume - 1.0 / 6.0).abs() < 1e-12);

        //Conformity: an isoparametric P2 numbering has one DoF per node
        let mut positions: Vec<[f64; 3]> = TETRAHEDRON_VERTEXES.to_vec();
        for e in TETRAHEDRON_EDGES_TOPOLOGY.iter() {
            positions.push(mid(
                &TETRAHEDRON_VERTEXES[e[0]],
                &TETRAHEDRON_VERTEXES[e[1]],
            ));
        }

        let dof_set = DoFSet::new(positions, &TETRAHEDRON);
        let (_, n_dof) = refined.get_numbering(&dof_set);
        assert_eq!(n_dof, refined.get_nnodes());

        //Boundary children keep their face label
        for (el, children) in parent_to_children.iter().enumerate().skip(1) {
            assert_eq!(children.len(), 9);
            for &child in children.iter() {
                assert_eq!(refined.get_element_as_ref(child).label, el);
            }
        }

        refined
            .write_msh("./tests/results/Tetrahedron10_refined_by_3.msh")
            .unwrap();
    }

    #[test]
    fn refine_by_one_is_identity() {
        let mesh = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None, None).get_unstructured_mesh();
        let (refined, parent_to_children) = mesh.uniform_refine_by(1);

        assert_eq!(refined.get_nnodes(), mesh.get_nnodes());
        assert_eq!(refined.get_nelements(), mesh.get_nelements());

        for children in parent_to_children.iter() {
            assert_eq!(children.len(), 1);
        }
    }

    #[test]
    #[should_panic(expected = "unreasonably large")]
    fn refine_factor_too_large_panics() {
        let mesh = Square::new([0.0, 0.0], 1.0, 1.0, 1, 1, None, None).get_unstructured_mesh();
        mesh.uniform_refine_by(101);
    }

    //L.msh lists its 4 boundary edges BEFORE its 12 quads: the parent index
    //bookkeeping must follow the global element ordering, not the dimension
    #[test]
    fn l_shape_q1_boundary_first() {
        let mesh = UnstructuredMesh::new_from_msh("./assets/L.msh").unwrap();
        let (refined, parent_to_children) = mesh.uniform_refine();

        assert_eq!(refined.get_elements_by_dimension(2).len(), 48);
        assert_eq!(refined.get_elements_by_dimension(1).len(), 8);
        assert_eq!(
            refined.get_nnodes(),
            mesh.get_nvertexes() + mesh.get_nedges() + 12
        );
        check_partition(&parent_to_children, refined.get_nelements());

        for (el, children) in parent_to_children.iter().enumerate() {
            let element = mesh.get_element_as_ref(el);
            let dim = element.geometric_element.reference_element.get_dim();

            assert_eq!(children.len(), if dim == 2 { 4 } else { 2 });

            for &child in children.iter() {
                let child_element = refined.get_element_as_ref(child);
                assert_eq!(child_element.label, element.label);
                assert_eq!(
                    child_element.geometric_element.msh_id,
                    element.geometric_element.msh_id
                );
            }
        }
    }

    //Two TRIANGLE6 sharing a CURVED edge, listed in opposite directions by
    //the two elements, plus one boundary EDGE3
    #[test]
    fn triangle6_conforming_curved_interface() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.55, 0.55, 0.0], //curved midpoint of the shared edge (1,2)
            [0.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 0.5, 0.0],
        ];

        let elements = vec![
            MeshElement {
                geometric_element: &TRIANGLE6,
                label: 0,
                indexes: vec![0, 1, 2, 4, 5, 6],
            },
            //Shared edge seen as (2, 1) from this side
            MeshElement {
                geometric_element: &TRIANGLE6,
                label: 1,
                indexes: vec![3, 2, 1, 7, 5, 8],
            },
            MeshElement {
                geometric_element: &EDGE3,
                label: 3,
                indexes: vec![0, 1, 4],
            },
        ];

        let mesh = UnstructuredMesh::new_from_points_and_elements(nodes, elements, None);
        let (refined, parent_to_children) = mesh.uniform_refine();

        assert_eq!(refined.get_elements_by_dimension(2).len(), 8);
        assert_eq!(refined.get_elements_by_dimension(1).len(), 2);
        assert_eq!(refined.get_nnodes(), 25);
        check_partition(&parent_to_children, refined.get_nelements());

        //The refined mesh is conforming: an isoparametric P2 numbering has
        //exactly one DoF per node
        let mut positions: Vec<[f64; 3]> = TRIANGLE_VERTEXES.to_vec();
        for e in TRIANGLE_EDGES_TOPOLOGY.iter() {
            positions.push(mid(&TRIANGLE_VERTEXES[e[0]], &TRIANGLE_VERTEXES[e[1]]));
        }

        let dof_set = DoFSet::new(positions, &TRIANGLE);
        let (_, n_dof) = refined.get_numbering(&dof_set);
        assert_eq!(n_dof, refined.get_nnodes());

        //The refined nodes on the shared edge lie exactly on the parent's
        //quadratic curve, despite the orientation mismatch: element 0 sees
        //the edge (1,2) as uvw = (1 - t, t)
        for t in [0.25, 0.5, 0.75] {
            let expected = eval_map(&mesh, 0, &[1.0 - t, t, 0.0]);

            let found = (0..refined.get_nnodes()).any(|i| {
                let p = refined.get_node(i);
                (0..3).all(|a| (p[a] - expected[a]).abs() < 1e-12)
            });
            assert!(found);
        }

        refined
            .write_msh("./tests/results/Triangle6_curved_refined.msh")
            .unwrap();
    }

    #[test]
    fn deformation_and_double_refine() {
        let def = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_p, _xyz, grad| {
                grad[0] = 2.0;
                grad[1] = 0.0;
                grad[2] = 0.0;
                grad[3] = 1.0;
            },
            inner_deformation: None,
        };

        let mesh = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, Some(def), None).get_unstructured_mesh();
        let (refined, _) = mesh.uniform_refine();

        assert!(refined.get_deformation().is_some());

        let (refined_twice, parent_to_children) = refined.uniform_refine();

        assert_eq!(refined_twice.get_elements_by_dimension(2).len(), 64);
        assert_eq!(refined_twice.get_elements_by_dimension(1).len(), 32);
        assert_eq!(refined_twice.get_nnodes(), 81);
        assert!(refined_twice.get_deformation().is_some());
        check_partition(&parent_to_children, refined_twice.get_nelements());
    }

    #[test]
    #[should_panic(expected = "same geometric order")]
    fn mixed_order_panics() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
        ];

        let elements = vec![
            MeshElement {
                geometric_element: &QUADRANGLE4,
                label: 0,
                indexes: vec![0, 1, 2, 3],
            },
            MeshElement {
                geometric_element: &EDGE3,
                label: 1,
                indexes: vec![0, 1, 4],
            },
        ];

        let mesh = UnstructuredMesh::new_from_points_and_elements(nodes, elements, None);
        mesh.uniform_refine();
    }
}
