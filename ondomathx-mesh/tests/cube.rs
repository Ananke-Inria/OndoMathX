extern crate ondomathx_mesh;

#[cfg(test)]
mod tests {
    use ondomathx_mesh::dof_set::*;
    use ondomathx_mesh::mesh::cube::*;
    use ondomathx_mesh::mesh::Mesh;

    fn mid(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [
            0.5 * (a[0] + b[0]),
            0.5 * (a[1] + b[1]),
            0.5 * (a[2] + b[2]),
        ]
    }

    // === Construction ===

    #[test]
    fn construction() {
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 2.0, 3.0, 2, 3, 4, None, None);
        assert_eq!(cube.nx, 2);
        assert_eq!(cube.ny, 3);
        assert_eq!(cube.nz, 4);
        assert!((cube.lx - 1.0).abs() < 1e-12);
        assert!((cube.ly - 2.0).abs() < 1e-12);
        assert!((cube.lz - 3.0).abs() < 1e-12);
        assert!(!cube.q2);
    }

    // === Q1 nodes and elements ===

    #[test]
    fn q1_nodes_and_counts() {
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 2, 2, 2, None, None);
        assert_eq!(cube.get_nnodes(), 27);
        //8 cells + 2*(4 + 4 + 4) boundary faces
        assert_eq!(cube.get_nelements(), 32);
        assert_eq!(cube.get_elements_by_dimension(3).len(), 8);
        assert_eq!(cube.get_elements_by_dimension(2).len(), 24);
        assert_eq!(cube.get_geometric_dimension(), 3);
        assert_eq!(cube.get_topologic_dimension(), 3);

        //Anisotropic case
        let cube = Cube::new([-1.0, 2.0, 0.5], 3.0, 2.0, 1.0, 3, 2, 1, None, None);
        assert_eq!(cube.get_nnodes(), 24);
        assert_eq!(cube.get_nelements(), 6 + 2 * (6 + 3 + 2));

        //Corner nodes
        let p0 = cube.get_node(0);
        assert!((p0[0] + 1.0).abs() < 1e-12);
        assert!((p0[1] - 2.0).abs() < 1e-12);
        assert!((p0[2] - 0.5).abs() < 1e-12);

        let plast = cube.get_node(cube.get_nnodes() - 1);
        assert!((plast[0] - 2.0).abs() < 1e-12);
        assert!((plast[1] - 4.0).abs() < 1e-12);
        assert!((plast[2] - 1.5).abs() < 1e-12);
    }

    // === Q2 nodes ===

    #[test]
    fn q2_node_groups() {
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 2, 2, 2, None, Some(true));
        assert_eq!(cube.get_nnodes(), 125);

        //First node of every Q2 group (2x2x2 unit cube, h = 0.5): x-edge
        //midpoints at 27, y at 45, z at 63, z-normal face centers at 81,
        //y-normal at 93, x-normal at 105, cell centers at 117.
        let expected = [
            (27, [0.25, 0.0, 0.0]),
            (45, [0.0, 0.25, 0.0]),
            (63, [0.0, 0.0, 0.25]),
            (81, [0.25, 0.25, 0.0]),
            (93, [0.25, 0.0, 0.25]),
            (105, [0.0, 0.25, 0.25]),
            (117, [0.25, 0.25, 0.25]),
        ];

        for (index, xyz) in expected.iter() {
            let p = cube.get_node(*index);
            for a in 0..3 {
                assert!((p[a] - xyz[a]).abs() < 1e-12);
            }
        }
    }

    //Every Q2 node of every element must be the centroid of its entity
    //corners (the cells are axis-aligned boxes, so the Q2 geometry is affine)
    #[test]
    fn q2_element_nodes_are_entity_centroids() {
        let cube = Cube::new([-1.0, 2.0, 0.5], 3.0, 2.0, 1.0, 3, 2, 2, None, Some(true));

        for el in 0..cube.get_nelements() {
            let element = cube.get_element(el);
            let reference_element = element.geometric_element.reference_element;
            let nv = reference_element.get_nvertexes();
            let p: Vec<[f64; 3]> = element.indexes.iter().map(|&i| cube.get_node(i)).collect();

            //Edge midpoints
            for (t, e) in reference_element.edges.iter().enumerate() {
                let m = mid(&p[e[0]], &p[e[1]]);
                for a in 0..3 {
                    assert!((p[nv + t][a] - m[a]).abs() < 1e-12);
                }
            }

            //Face centers, then the hexahedron center
            let ne = reference_element.get_nedges();
            for f in 0..reference_element.get_nfaces() {
                let face = &reference_element.faces[f];
                let mut c = [0.0; 3];
                for &i in face.iter() {
                    for a in 0..3 {
                        c[a] += 0.25 * p[i][a];
                    }
                }
                for a in 0..3 {
                    assert!((p[nv + ne + f][a] - c[a]).abs() < 1e-12);
                }
            }

            if nv == 8 {
                let mut c = [0.0; 3];
                for corner in p.iter().take(8) {
                    for a in 0..3 {
                        c[a] += 0.125 * corner[a];
                    }
                }
                for a in 0..3 {
                    assert!((p[26][a] - c[a]).abs() < 1e-12);
                }
            }
        }
    }

    // === Boundary faces ===

    #[test]
    fn boundary_faces_on_plane_with_outward_normal() {
        let origin = [-1.0, 2.0, 0.5];
        let (lx, ly, lz) = (3.0, 2.0, 1.0);
        let cube = Cube::new(origin, lx, ly, lz, 3, 2, 2, None, None);

        //(fixed coordinate axis, plane value, outward sign) per label
        let planes = [
            (2, origin[2], -1.0),
            (1, origin[1], -1.0),
            (0, origin[0], -1.0),
            (0, origin[0] + lx, 1.0),
            (1, origin[1] + ly, 1.0),
            (2, origin[2] + lz, 1.0),
        ];

        for &el in cube.get_elements_by_dimension(2).iter() {
            let element = cube.get_element(el);
            let (axis, value, sign) = planes[element.label];

            let p: Vec<[f64; 3]> = element.indexes.iter().map(|&i| cube.get_node(i)).collect();

            //All corners on the plane of the group
            for corner in p.iter() {
                assert!((corner[axis] - value).abs() < 1e-12);
            }

            //(v1-v0) x (v3-v0) points outward
            let d1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let d2 = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
            let normal = [
                d1[1] * d2[2] - d1[2] * d2[1],
                d1[2] * d2[0] - d1[0] * d2[2],
                d1[0] * d2[1] - d1[1] * d2[0],
            ];

            assert!(sign * normal[axis] > 0.0);
        }
    }

    #[test]
    fn boundary_groups_and_labels() {
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 3, 2, 2, None, None);

        let sizes = [6, 6, 4, 4, 6, 6]; //nx*ny, nx*nz, ny*nz, ny*nz, nx*nz, nx*ny

        for (label, &size) in sizes.iter().enumerate() {
            let group = cube.get_elements_by_dimension_and_label(label, 2);
            assert_eq!(group.len(), size);

            for &el in group.iter() {
                assert_eq!(cube.get_element(el).label, label);
            }
        }

        assert_eq!(
            cube.get_elements_by_dimension_and_label(0, 3).len(),
            3 * 2 * 2
        );
    }

    // === Coloring ===

    #[test]
    fn coloring_is_a_valid_partition() {
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 3, 2, 2, None, None);
        let n_vol = 3 * 2 * 2;

        assert_eq!(cube.get_ncolors(), 8);

        //The by-color indexing is a partition of the cells consistent with
        //get_element_color
        let mut all: Vec<usize> = Vec::new();
        for color in 0..8 {
            for idx in 0..cube.get_nelements_by_color(color) {
                let el = cube.get_element_index_by_color_to_element_index(color, idx);
                assert_eq!(cube.get_element_color(el), color);
                all.push(el);
            }
        }
        all.sort();
        assert_eq!(all, (0..n_vol).collect::<Vec<usize>>());

        //Two cells of the same color never share a node
        for e1 in 0..n_vol {
            for e2 in (e1 + 1)..n_vol {
                if cube.get_element_color(e1) == cube.get_element_color(e2) {
                    let i1 = cube.get_element(e1).indexes;
                    let i2 = cube.get_element(e2).indexes;
                    assert!(i1.iter().all(|i| !i2.contains(i)));
                }
            }
        }
    }

    // === Unstructured conversion (also runs the dangling-element check) ===

    #[test]
    fn unstructured_topology_counts() {
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 2, 2, 2, None, None);
        let mesh = cube.get_unstructured_mesh();

        assert_eq!(mesh.get_nvertexes(), 27);
        assert_eq!(mesh.get_nedges(), 54);
        assert_eq!(mesh.get_nfaces(), 36);
        assert_eq!(mesh.get_elements_by_dimension(3).len(), 8);
        assert_eq!(mesh.get_elements_by_dimension(2).len(), 24);
    }

    #[test]
    fn q2_numbering_is_conforming() {
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 2, 2, 2, None, Some(true));
        let mesh = cube.get_unstructured_mesh();

        //Isoparametric Q2 DoFSet on the hexahedron: one DoF per node
        let dof_set = DoFSet::new(hex_q2_positions(), &HEXAHEDRON);
        let (_, n_dof) = mesh.get_numbering(&dof_set);
        assert_eq!(n_dof, 125);
        assert_eq!(mesh.get_nnodes(), 125);

        //The structured tensor numbering agrees on the DoF count
        let (_, n_dof_structured) = cube.get_numbering(3, 3, 3);
        assert_eq!(n_dof_structured, 125);
    }

    fn hex_q2_positions() -> Vec<[f64; 3]> {
        let mut positions: Vec<[f64; 3]> = HEXAHEDRON_VERTEXES.to_vec();
        for e in HEXAHEDRON_EDGES_TOPOLOGY.iter() {
            positions.push(mid(&HEXAHEDRON_VERTEXES[e[0]], &HEXAHEDRON_VERTEXES[e[1]]));
        }
        for f in HEXAHEDRON_FACES_TOPOLOGY.iter() {
            let mut c = [0.0; 3];
            for &i in f.iter() {
                for a in 0..3 {
                    c[a] += 0.25 * HEXAHEDRON_VERTEXES[i][a];
                }
            }
            positions.push(c);
        }
        positions.push([0.5, 0.5, 0.5]);
        positions
    }

    // === grad_def ===

    #[test]
    fn grad_def_matches_element_corners() {
        let cube = Cube::new([-1.0, 2.0, 0.5], 3.0, 2.0, 1.0, 3, 2, 2, None, None);
        let (hx, hy, hz) = (1.0, 1.0, 0.5);

        let mut grad = [0.0; 9];

        for &el in cube.get_elements_by_dimension(3).iter() {
            cube.get_element_grad_def(el, &[0.3, 0.4, 0.5], &mut grad);
            let expected = [hx, 0.0, 0.0, 0.0, hy, 0.0, 0.0, 0.0, hz];
            for a in 0..9 {
                assert!((grad[a] - expected[a]).abs() < 1e-13);
            }
        }

        //Boundary faces: 3x2 Jacobian columns = (v1-v0, v3-v0)
        for &el in cube.get_elements_by_dimension(2).iter() {
            cube.get_element_grad_def(el, &[0.3, 0.4, 0.0], &mut grad);

            let element = cube.get_element(el);
            let p: Vec<[f64; 3]> = element.indexes.iter().map(|&i| cube.get_node(i)).collect();

            for a in 0..3 {
                assert!((grad[2 * a] - (p[1][a] - p[0][a])).abs() < 1e-13);
                assert!((grad[2 * a + 1] - (p[3][a] - p[0][a])).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn grad_def_with_deformation() {
        let def = Diffeomorphism {
            inner_dim: 3,
            outer_dim: 3,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = xyz[1];
                f_xyz[2] = xyz[2];
            },
            grad_diffeo: |_p, _xyz, grad| {
                grad.copy_from_slice(&[2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            },
            inner_deformation: None,
        };

        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 2, 2, 2, Some(def), None);
        let h = 0.5;

        let mut grad = [0.0; 9];
        cube.get_element_grad_def(0, &[0.5, 0.5, 0.5], &mut grad);

        let expected = [2.0 * h, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, h];
        for a in 0..9 {
            assert!((grad[a] - expected[a]).abs() < 1e-13);
        }
    }

    // === Uniform refinement of the Q2 cube ===

    #[test]
    fn q2_uniform_refine() {
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 2, 2, 2, None, Some(true));
        let mesh = cube.get_unstructured_mesh();

        let (refined, parent_to_children) = mesh.uniform_refine();

        assert_eq!(refined.get_elements_by_dimension(3).len(), 64);
        assert_eq!(refined.get_elements_by_dimension(2).len(), 96);
        assert_eq!(refined.get_nnodes(), 729);

        for &el in mesh.get_elements_by_dimension(3).iter() {
            assert_eq!(parent_to_children[el].len(), 8);
        }

        //Conformity of the refined Q2 mesh
        let dof_set = DoFSet::new(hex_q2_positions(), &HEXAHEDRON);
        let (_, n_dof) = refined.get_numbering(&dof_set);
        assert_eq!(n_dof, 729);

        //Outputs for visual inspection
        cube.write_msh("./tests/results/Cube_q2.msh").unwrap();
        refined
            .write_msh("./tests/results/Cube_q2_refined.msh")
            .unwrap();
    }
}
