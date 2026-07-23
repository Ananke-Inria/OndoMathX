extern crate anamesh;

#[cfg(test)]
mod tests {
    use anamesh::mesh::unstructured_mesh::UnstructuredMesh;
    use anamesh::mesh::Mesh;

    #[test]
    fn unstructured_mesh_from_msh() {
        let mesh = UnstructuredMesh::new_from_msh("./assets/Potatoe.msh").unwrap();

        // Potatoe.msh has 901 nodes and 951 elements (135 edges + 816 quads)
        assert_eq!(mesh.get_nnodes(), 901);
        assert_eq!(mesh.get_nelements(), 951);

        // It is a 2D mesh
        assert_eq!(mesh.get_geometric_dimension(), 2);
        assert_eq!(mesh.get_topologic_dimension(), 2);

        // Check element counts per dimension
        let edges = mesh.get_elements_by_dimension(1);
        let faces = mesh.get_elements_by_dimension(2);
        assert_eq!(edges.len(), 135);
        assert_eq!(faces.len(), 816);
        assert_eq!(edges.len() + faces.len(), mesh.get_nelements());
    }

    #[test]
    fn unstructured_mesh_topology() {
        let mesh = UnstructuredMesh::new_from_msh("./assets/Potatoe.msh").unwrap();

        // Expected counts for Potatoe.msh
        assert_eq!(mesh.get_nvertexes(), 893);
        assert_eq!(mesh.get_nedges(), 1709);

        // Every quad element must have exactly 4 vertices and 4 edges
        let faces = mesh.get_elements_by_dimension(2);
        for &el in &faces {
            assert_eq!(
                mesh.get_neighbour_vertexes(el).len(),
                4,
                "element {} should have 4 vertices",
                el
            );
            assert_eq!(
                mesh.get_neighbour_edges(el).len(),
                4,
                "element {} should have 4 edges",
                el
            );
        }

        // Each edge is shared by at most 2 top-dimension elements
        for k in 0..mesh.get_nedges() {
            let neighbours = mesh.get_edge_neighbours(k);
            assert!(
                neighbours.len() <= 2,
                "edge {} has {} top-dim neighbours, expected at most 2",
                k,
                neighbours.len()
            );
        }

        // Symmetry: if edge k lists element el, then element el lists edge k
        for k in 0..mesh.get_nedges() {
            let neighbours = mesh.get_edge_neighbours(k);
            for entry in neighbours {
                let el = entry[0];
                assert!(
                    mesh.get_neighbour_edges(el).contains(&k),
                    "edge {} references element {}, but element does not reference edge back",
                    k,
                    el
                );
            }
        }
    }

    #[test]
    fn unstructured_mesh_coloring_validity() {
        let mesh = UnstructuredMesh::new_from_msh("./assets/Potatoe.msh").unwrap();

        assert!(mesh.get_ncolors() > 0);

        // Verify no two adjacent elements share the same color, going through
        // the trait accessor (global element indices).
        let vec_elem = mesh.get_elements_by_dimension(2);
        let graph = mesh.compute_graph();
        for (i, neighbors) in graph.iter().enumerate() {
            for &n in neighbors {
                assert_ne!(
                    mesh.get_element_color(vec_elem[i]),
                    mesh.get_element_color(vec_elem[n]),
                    "Elements {} and {} are neighbors but share color {}",
                    vec_elem[i],
                    vec_elem[n],
                    mesh.get_element_color(vec_elem[i])
                );
            }
        }
    }

    // Regression test: in meshes read from Gmsh files, boundary elements can
    // precede volume elements in the global numbering (L.msh lists 4 edges
    // before its 12 quads). The coloring API must speak global element
    // indices nonetheless.
    #[test]
    fn coloring_api_uses_global_indices() {
        let mesh = UnstructuredMesh::new_from_msh("./assets/L.msh").unwrap();

        let vec_elem = mesh.get_elements_by_dimension(2);

        // Sanity check of the premise: the first volume element is NOT global
        // element 0 (otherwise this test does not exercise the translation).
        assert!(
            vec_elem[0] > 0,
            "L.msh no longer lists boundary edges first"
        );

        // Colors must be readable for every volume element (global index).
        for &e in &vec_elem {
            assert!(mesh.get_element_color(e) < mesh.get_ncolors());
        }

        // The (color, index-by-color) -> element map must return global
        // indices of volume elements, partitioning them exactly.
        let mut seen = std::collections::HashSet::new();
        for c in 0..mesh.get_ncolors() {
            for k in 0..mesh.get_nelements_by_color(c) {
                let e = mesh.get_element_index_by_color_to_element_index(c, k);
                assert!(vec_elem.contains(&e), "{} is not a volume element", e);
                assert_eq!(mesh.get_element_color(e), c);
                assert!(seen.insert(e), "element {} listed twice", e);
            }
        }
        assert_eq!(seen.len(), vec_elem.len());
    }

    // === High-order (P2/Q2) element reading ===

    // Internal node positions (reference_elements.rs ordering).
    const TET10_INTERNAL: [[f64; 3]; 10] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];

    const HEX27_INTERNAL: [[f64; 3]; 27] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.5, 0.0, 1.0],
        [1.0, 0.5, 1.0],
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5],
    ];

    fn assert_element_nodes(
        mesh: &UnstructuredMesh,
        element_index: usize,
        expected: &[[f64; 3]],
        context: &str,
    ) {
        let element = mesh.get_element(element_index);
        assert_eq!(element.indexes.len(), expected.len(), "{}", context);
        for (i, exp) in expected.iter().enumerate() {
            let node = mesh.get_node(element.indexes[i]);
            for c in 0..3 {
                assert!(
                    (node[c] - exp[c]).abs() < 1e-12,
                    "{}: local node {} is at {:?}, expected {:?}",
                    context,
                    i,
                    node,
                    exp
                );
            }
        }
    }

    // A tetrahedron10 and one of its triangle6 faces, with the nodes listed
    // in Gmsh ordering; the reader must permute the tet midpoints into the
    // internal ordering.
    #[test]
    fn read_msh_p2_tetrahedron_and_triangle() {
        let msh = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
$Nodes\n10\n\
1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n\
5 0.5 0 0\n6 0.5 0.5 0\n7 0 0.5 0\n8 0 0 0.5\n\
9 0 0.5 0.5\n10 0.5 0 0.5\n\
$EndNodes\n\
$Elements\n2\n\
1 9 2 0 1 1 2 3 5 6 7\n\
2 11 2 0 1 1 2 3 4 5 6 7 8 9 10\n\
$EndElements\n";
        let path = "./tests/results/p2_tet.msh";
        std::fs::write(path, msh).unwrap();

        let mesh = UnstructuredMesh::new_from_msh(path).unwrap();
        assert_eq!(mesh.get_nnodes(), 10);
        assert_eq!(mesh.get_nelements(), 2);
        assert_eq!(mesh.get_topologic_dimension(), 3);
        // Only the 4 corners are topological vertices, not the midpoints.
        assert_eq!(mesh.get_nvertexes(), 4);
        assert_eq!(mesh.get_nedges(), 6);

        let tet = mesh.get_elements_by_dimension(3);
        let tri = mesh.get_elements_by_dimension(2);
        assert_eq!(tet.len(), 1);
        assert_eq!(tri.len(), 1);

        assert_element_nodes(&mesh, tet[0], &TET10_INTERNAL, "tetrahedron10");
        // Triangle6: Gmsh and internal orderings coincide.
        assert_element_nodes(
            &mesh,
            tri[0],
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
            ],
            "triangle6",
        );
    }

    // A single hexahedron27 with the nodes listed in Gmsh ordering.
    #[test]
    fn read_msh_q2_hexahedron() {
        // Nodes 1..27 at the Gmsh-ordered reference positions.
        let gmsh_nodes = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.0, 0.0], // m01
            [0.0, 0.5, 0.0], // m03
            [0.0, 0.0, 0.5], // m04
            [1.0, 0.5, 0.0], // m12
            [1.0, 0.0, 0.5], // m15
            [0.5, 1.0, 0.0], // m23
            [1.0, 1.0, 0.5], // m26
            [0.0, 1.0, 0.5], // m37
            [0.5, 0.0, 1.0], // m45
            [0.0, 0.5, 1.0], // m47
            [1.0, 0.5, 1.0], // m56
            [0.5, 1.0, 1.0], // m67
            [0.5, 0.5, 0.0], // bottom
            [0.5, 0.0, 0.5], // front
            [0.0, 0.5, 0.5], // left
            [1.0, 0.5, 0.5], // right
            [0.5, 1.0, 0.5], // back
            [0.5, 0.5, 1.0], // top
            [0.5, 0.5, 0.5], // center
        ];
        let mut msh = String::from("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n27\n");
        for (i, p) in gmsh_nodes.iter().enumerate() {
            msh.push_str(&format!("{} {} {} {}\n", i + 1, p[0], p[1], p[2]));
        }
        msh.push_str("$EndNodes\n$Elements\n1\n1 12 2 0 1");
        for i in 1..=27 {
            msh.push_str(&format!(" {}", i));
        }
        msh.push_str("\n$EndElements\n");
        let path = "./tests/results/q2_hex.msh";
        std::fs::write(path, msh).unwrap();

        let mesh = UnstructuredMesh::new_from_msh(path).unwrap();
        assert_eq!(mesh.get_nnodes(), 27);
        assert_eq!(mesh.get_nelements(), 1);
        assert_eq!(mesh.get_nvertexes(), 8);
        assert_eq!(mesh.get_nedges(), 12);
        assert_eq!(mesh.get_nfaces(), 6);
        assert_element_nodes(&mesh, 0, &HEX27_INTERNAL, "hexahedron27 (.msh)");
    }

    // A single HexahedraQ2 in the Medit .mesh convention of the C++ IO_MESH
    // writer: vertices, bottom edge ring, top edge ring, vertical edges,
    // face centers (bottom, top, front, right, back, left), center.
    #[test]
    fn read_mesh_q2_hexahedron() {
        let file_nodes = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.0, 0.0], // m01
            [1.0, 0.5, 0.0], // m12
            [0.5, 1.0, 0.0], // m23
            [0.0, 0.5, 0.0], // m03
            [0.5, 0.0, 1.0], // m45
            [1.0, 0.5, 1.0], // m56
            [0.5, 1.0, 1.0], // m67
            [0.0, 0.5, 1.0], // m47
            [0.0, 0.0, 0.5], // m04
            [1.0, 0.0, 0.5], // m15
            [1.0, 1.0, 0.5], // m26
            [0.0, 1.0, 0.5], // m37
            [0.5, 0.5, 0.0], // bottom
            [0.5, 0.5, 1.0], // top
            [0.5, 0.0, 0.5], // front
            [1.0, 0.5, 0.5], // right
            [0.5, 1.0, 0.5], // back
            [0.0, 0.5, 0.5], // left
            [0.5, 0.5, 0.5], // center
        ];
        let mut mesh_file = String::from("MeshVersionFormatted 2\nDimension\n3\nVertices\n27\n");
        for p in file_nodes.iter() {
            mesh_file.push_str(&format!("{} {} {} 0\n", p[0], p[1], p[2]));
        }
        mesh_file.push_str("HexahedraQ2\n1\n");
        for i in 1..=27 {
            mesh_file.push_str(&format!("{} ", i));
        }
        mesh_file.push_str("0\nEnd\n");
        let path = "./tests/results/q2_hex.mesh";
        std::fs::write(path, mesh_file).unwrap();

        let mesh = UnstructuredMesh::new_from_mesh(path).unwrap();
        assert_eq!(mesh.get_nnodes(), 27);
        assert_eq!(mesh.get_nelements(), 1);
        assert_eq!(mesh.get_nvertexes(), 8);
        assert_eq!(mesh.get_nedges(), 12);
        assert_eq!(mesh.get_nfaces(), 6);
        assert_element_nodes(&mesh, 0, &HEX27_INTERNAL, "hexahedron27 (.mesh)");
    }

    // A TetrahedraP2 and one of its TrianglesP2 faces in the Medit/libMeshb
    // P2 convention (vertices, then edge nodes on edges (1,2),(2,3),(3,1),
    // (1,4),(2,4),(3,4)), which coincides with the internal ordering.
    #[test]
    fn read_mesh_p2_tetrahedron_and_triangle() {
        let mut mesh_file = String::from("MeshVersionFormatted 2\nDimension\n3\nVertices\n10\n");
        for p in TET10_INTERNAL.iter() {
            mesh_file.push_str(&format!("{} {} {} 0\n", p[0], p[1], p[2]));
        }
        // Face (0,1,2) of the tet: vertices 1 2 3, midpoints m01 m12 m02 = 5 6 7.
        mesh_file.push_str("TrianglesP2\n1\n1 2 3 5 6 7 1\n");
        mesh_file.push_str("TetrahedraP2\n1\n1 2 3 4 5 6 7 8 9 10 1\n");
        mesh_file.push_str("End\n");
        let path = "./tests/results/p2_tet.mesh";
        std::fs::write(path, mesh_file).unwrap();

        let mesh = UnstructuredMesh::new_from_mesh(path).unwrap();
        assert_eq!(mesh.get_nnodes(), 10);
        assert_eq!(mesh.get_nelements(), 2);
        assert_eq!(mesh.get_topologic_dimension(), 3);
        assert_eq!(mesh.get_nvertexes(), 4);
        assert_eq!(mesh.get_nedges(), 6);

        let tet = mesh.get_elements_by_dimension(3);
        let tri = mesh.get_elements_by_dimension(2);
        assert_eq!(tet.len(), 1);
        assert_eq!(tri.len(), 1);

        assert_element_nodes(&mesh, tet[0], &TET10_INTERNAL, "tetrahedron10 (.mesh)");
        assert_element_nodes(
            &mesh,
            tri[0],
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
            ],
            "triangle6 (.mesh)",
        );
    }

    // A QuadrilateralsQ2 and one of its EdgesP2 boundary edges in the
    // Medit/libMeshb convention (vertices, edge midpoints on edges
    // (1,2),(2,3),(3,4),(4,1), center), which coincides with the internal
    // ordering.
    #[test]
    fn read_mesh_q2_quadrangle_and_p2_edge() {
        let quad9_internal = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ];
        let mut mesh_file = String::from("MeshVersionFormatted 2\nDimension\n3\nVertices\n9\n");
        for p in quad9_internal.iter() {
            mesh_file.push_str(&format!("{} {} {} 0\n", p[0], p[1], p[2]));
        }
        // Bottom edge (v0, v1) with its midpoint m01 = node 5.
        mesh_file.push_str("EdgesP2\n1\n1 2 5 1\n");
        mesh_file.push_str("QuadrilateralsQ2\n1\n1 2 3 4 5 6 7 8 9 1\n");
        mesh_file.push_str("End\n");
        let path = "./tests/results/q2_quad.mesh";
        std::fs::write(path, mesh_file).unwrap();

        let mesh = UnstructuredMesh::new_from_mesh(path).unwrap();
        assert_eq!(mesh.get_nnodes(), 9);
        assert_eq!(mesh.get_nelements(), 2);
        assert_eq!(mesh.get_topologic_dimension(), 2);
        assert_eq!(mesh.get_nvertexes(), 4);
        assert_eq!(mesh.get_nedges(), 4);

        let quad = mesh.get_elements_by_dimension(2);
        let edge = mesh.get_elements_by_dimension(1);
        assert_eq!(quad.len(), 1);
        assert_eq!(edge.len(), 1);

        assert_element_nodes(&mesh, quad[0], &quad9_internal, "quadrangle9 (.mesh)");
        assert_element_nodes(
            &mesh,
            edge[0],
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            "edge3 (.mesh)",
        );
    }

    // write_msh applies the internal -> Gmsh node permutation for
    // Tetrahedron10/Hexahedron27, so writing and re-reading is the identity.
    #[test]
    fn write_msh_high_order_roundtrip() {
        use anamesh::geometric_elements::{HEXAHEDRON27, TETRAHEDRON10};
        use anamesh::mesh::MeshElement;

        // Tetrahedron10 with nodes at the internal reference positions.
        let tet_mesh = UnstructuredMesh::new_from_points_and_elements(
            TET10_INTERNAL.to_vec(),
            vec![MeshElement {
                geometric_element: &TETRAHEDRON10,
                label: 1,
                indexes: (0..10).collect(),
            }],
            None,
        );
        let path = "./tests/results/tet10_roundtrip.msh";
        tet_mesh.write_msh(path).unwrap();
        let reread = UnstructuredMesh::new_from_msh(path).unwrap();
        assert_element_nodes(&reread, 0, &TET10_INTERNAL, "tetrahedron10 roundtrip");

        // Hexahedron27 with nodes at the internal reference positions.
        let hex_mesh = UnstructuredMesh::new_from_points_and_elements(
            HEX27_INTERNAL.to_vec(),
            vec![MeshElement {
                geometric_element: &HEXAHEDRON27,
                label: 1,
                indexes: (0..27).collect(),
            }],
            None,
        );
        let path = "./tests/results/hex27_roundtrip.msh";
        hex_mesh.write_msh(path).unwrap();
        let reread = UnstructuredMesh::new_from_msh(path).unwrap();
        assert_element_nodes(&reread, 0, &HEX27_INTERNAL, "hexahedron27 roundtrip");
    }

    // write_mesh (Medit/Vizir output): the high-order node orderings survive
    // a write + reread (TetrahedraP2 is the identity, HexahedraQ2 goes
    // through the C++ IO_MESH permutation and back).
    #[test]
    fn write_mesh_high_order_roundtrip() {
        use anamesh::geometric_elements::{HEXAHEDRON27, TETRAHEDRON10};
        use anamesh::mesh::MeshElement;

        let tet_mesh = UnstructuredMesh::new_from_points_and_elements(
            TET10_INTERNAL.to_vec(),
            vec![MeshElement {
                geometric_element: &TETRAHEDRON10,
                label: 1,
                indexes: (0..10).collect(),
            }],
            None,
        );
        let path = "./tests/results/tet10_roundtrip.mesh";
        tet_mesh.write_mesh(path).unwrap();
        let reread = UnstructuredMesh::new_from_mesh(path).unwrap();
        assert_element_nodes(&reread, 0, &TET10_INTERNAL, "tetrahedron10 .mesh roundtrip");

        let hex_mesh = UnstructuredMesh::new_from_points_and_elements(
            HEX27_INTERNAL.to_vec(),
            vec![MeshElement {
                geometric_element: &HEXAHEDRON27,
                label: 1,
                indexes: (0..27).collect(),
            }],
            None,
        );
        let path = "./tests/results/hex27_roundtrip.mesh";
        hex_mesh.write_mesh(path).unwrap();
        let reread = UnstructuredMesh::new_from_mesh(path).unwrap();
        assert_element_nodes(&reread, 0, &HEX27_INTERNAL, "hexahedron27 .mesh roundtrip");
    }

    // write_mesh on a mixed P1 mesh: the elements are regrouped by section
    // but their connectivity and labels are preserved.
    #[test]
    fn write_mesh_roundtrip_preserves_elements() {
        let mesh = UnstructuredMesh::new_from_msh("./assets/L.msh").unwrap();
        let path = "./tests/results/l_roundtrip.mesh";
        mesh.write_mesh(path).unwrap();
        let reread = UnstructuredMesh::new_from_mesh(path).unwrap();

        assert_eq!(reread.get_nnodes(), mesh.get_nnodes());
        assert_eq!(reread.get_nelements(), mesh.get_nelements());

        // Same elements per dimension, up to reordering.
        for dim in 1..=2 {
            let collect = |m: &UnstructuredMesh| {
                let mut v: Vec<(Vec<usize>, usize)> = m
                    .get_elements_by_dimension(dim)
                    .iter()
                    .map(|&e| {
                        let el = m.get_element(e);
                        (el.indexes.clone(), el.label)
                    })
                    .collect();
                v.sort();
                v
            };
            assert_eq!(collect(&mesh), collect(&reread), "dim {}", dim);
        }
    }

    // A Q2 circle written to the Medit format for Vizir: the curved Q2
    // elements (QuadrilateralsQ2 / EdgesP2 sections) survive the roundtrip
    // and the boundary midpoints stay on the circle.
    #[test]
    fn write_mesh_q2_circle() {
        use anamesh::geometric_elements::{EDGE3, QUADRANGLE9};
        use anamesh::mesh::circle::Circle;

        let radius = 1.5;
        let center = [0.2, -0.3];
        let circle = Circle::new(center, radius, 7, None, Some(true));
        let path = "./tests/results/circle_q2.mesh";
        circle.write_mesh(path).unwrap();

        let reread = UnstructuredMesh::new_from_mesh(path).unwrap();
        assert_eq!(reread.get_nnodes(), circle.get_nnodes());
        assert_eq!(reread.get_nelements(), circle.get_nelements());

        // Element types are preserved through the QuadrilateralsQ2 / EdgesP2
        // sections.
        let quads = reread.get_elements_by_dimension(2);
        let edges = reread.get_elements_by_dimension(1);
        assert_eq!(quads.len(), circle.get_elements_by_dimension(2).len());
        assert_eq!(edges.len(), circle.get_elements_by_dimension(1).len());
        for &e in &quads {
            assert_eq!(
                reread.get_element(e).geometric_element.msh_id,
                QUADRANGLE9.msh_id
            );
        }
        for &e in &edges {
            assert_eq!(reread.get_element(e).geometric_element.msh_id, EDGE3.msh_id);
        }

        // All boundary nodes (midpoints included) lie on the circle.
        for &e in &edges {
            for &i in &reread.get_element(e).indexes {
                let [x, y, _] = reread.get_node(i);
                let r = (x - center[0]).hypot(y - center[1]);
                assert!((r - radius).abs() < 1e-12, "node {} at distance {}", i, r);
            }
        }
    }

    // === Tests for .mesh format (INRIA Medit) ===

    #[test]
    fn unstructured_mesh_from_mesh() {
        let mesh = UnstructuredMesh::new_from_mesh("./assets/Heart.mesh").unwrap();

        // Heart.mesh has 10598 nodes and 59005 elements (11288 triangles + 47717 tetrahedra)
        assert_eq!(mesh.get_nnodes(), 10598);
        assert_eq!(mesh.get_nelements(), 59005);

        // It is a 3D tet mesh
        assert_eq!(mesh.get_geometric_dimension(), 3);
        assert_eq!(mesh.get_topologic_dimension(), 3);

        // Check element counts per dimension
        let triangles = mesh.get_elements_by_dimension(2);
        let tetrahedra = mesh.get_elements_by_dimension(3);
        assert_eq!(triangles.len(), 11288);
        assert_eq!(tetrahedra.len(), 47717);
        assert_eq!(triangles.len() + tetrahedra.len(), mesh.get_nelements());
    }

    #[test]
    fn unstructured_mesh_from_mesh_topology() {
        let mesh = UnstructuredMesh::new_from_mesh("./assets/Heart.mesh").unwrap();

        // Expected counts for Heart.mesh
        assert_eq!(mesh.get_nvertexes(), 10598);
        assert_eq!(mesh.get_nedges(), 63426);
        assert_eq!(mesh.get_nfaces(), 100544);

        // Every tet must have exactly 4 vertices, 6 edges, and 4 faces
        let tetrahedra = mesh.get_elements_by_dimension(3);
        for &el in &tetrahedra {
            assert_eq!(
                mesh.get_neighbour_vertexes(el).len(),
                4,
                "tet {} should have 4 vertices",
                el
            );
            assert_eq!(
                mesh.get_neighbour_edges(el).len(),
                6,
                "tet {} should have 6 edges",
                el
            );
            assert_eq!(
                mesh.get_neighbour_faces(el).len(),
                4,
                "tet {} should have 4 faces",
                el
            );
        }

        // Each face is shared by at most 2 top-dimension elements
        for k in 0..mesh.get_nfaces() {
            let neighbours = mesh.get_face_neighbours(k);
            assert!(
                neighbours.len() <= 2,
                "face {} has {} top-dim neighbours, expected at most 2",
                k,
                neighbours.len()
            );
        }

        // Symmetry: if face k lists element el, then element el lists face k
        for k in 0..mesh.get_nfaces() {
            let neighbours = mesh.get_face_neighbours(k);
            for entry in neighbours {
                let el = entry[0];
                assert!(
                    mesh.get_neighbour_faces(el).contains(&k),
                    "face {} references element {}, but element does not reference face back",
                    k,
                    el
                );
            }
        }
    }

    // === Tests for hexahedral meshes (Cylinder) ===

    #[test]
    fn unstructured_mesh_from_msh_hexahedra() {
        let mesh = UnstructuredMesh::new_from_msh("./assets/Cylinder.msh").unwrap();

        // Cylinder.msh in Gmsh 2.2 format (5940 nodes, no isolated vertices)
        assert_eq!(mesh.get_nnodes(), 5940);
        assert_eq!(mesh.get_nelements(), 8136);
        assert_eq!(mesh.get_geometric_dimension(), 3);
        assert_eq!(mesh.get_topologic_dimension(), 3);

        let edges = mesh.get_elements_by_dimension(1);
        let quads = mesh.get_elements_by_dimension(2);
        let hexahedra = mesh.get_elements_by_dimension(3);
        assert_eq!(edges.len(), 448);
        assert_eq!(quads.len(), 2648);
        assert_eq!(hexahedra.len(), 5040);
    }

    #[test]
    fn unstructured_mesh_from_mesh_hexahedra() {
        let mesh = UnstructuredMesh::new_from_mesh("./assets/Cylinder.mesh").unwrap();

        // Cylinder.mesh: 5940 nodes, 8136 elements (448 edges + 2648 quads + 5040 hexahedra)
        assert_eq!(mesh.get_nnodes(), 5940);
        assert_eq!(mesh.get_nelements(), 8136);

        // 3D hex mesh
        assert_eq!(mesh.get_geometric_dimension(), 3);
        assert_eq!(mesh.get_topologic_dimension(), 3);

        // Check element counts per dimension
        let edges = mesh.get_elements_by_dimension(1);
        let quads = mesh.get_elements_by_dimension(2);
        let hexahedra = mesh.get_elements_by_dimension(3);
        assert_eq!(edges.len(), 448);
        assert_eq!(quads.len(), 2648);
        assert_eq!(hexahedra.len(), 5040);
        assert_eq!(
            edges.len() + quads.len() + hexahedra.len(),
            mesh.get_nelements()
        );
    }

    #[test]
    fn unstructured_mesh_from_mesh_hexahedra_topology() {
        let mesh = UnstructuredMesh::new_from_mesh("./assets/Cylinder.mesh").unwrap();

        // Expected counts for Cylinder.mesh
        assert_eq!(mesh.get_nvertexes(), 5940);
        assert_eq!(mesh.get_nedges(), 16884);
        assert_eq!(mesh.get_nfaces(), 15984);

        // Every hex must have exactly 8 vertices, 12 edges, and 6 faces
        let hexahedra = mesh.get_elements_by_dimension(3);
        for &el in &hexahedra {
            assert_eq!(
                mesh.get_neighbour_vertexes(el).len(),
                8,
                "hex {} should have 8 vertices",
                el
            );
            assert_eq!(
                mesh.get_neighbour_edges(el).len(),
                12,
                "hex {} should have 12 edges",
                el
            );
            assert_eq!(
                mesh.get_neighbour_faces(el).len(),
                6,
                "hex {} should have 6 faces",
                el
            );
        }

        // Each face is shared by at most 2 top-dimension elements
        for k in 0..mesh.get_nfaces() {
            let neighbours = mesh.get_face_neighbours(k);
            assert!(
                neighbours.len() <= 2,
                "face {} has {} top-dim neighbours, expected at most 2",
                k,
                neighbours.len()
            );
        }

        // Symmetry: if face k lists element el, then element el lists face k
        for k in 0..mesh.get_nfaces() {
            let neighbours = mesh.get_face_neighbours(k);
            for entry in neighbours {
                let el = entry[0];
                assert!(
                    mesh.get_neighbour_faces(el).contains(&k),
                    "face {} references element {}, but element does not reference face back",
                    k,
                    el
                );
            }
        }
    }
}
