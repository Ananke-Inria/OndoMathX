extern crate rouage;

#[cfg(test)]
mod tests {
    use rouage::mesh::unstructured_mesh::UnstructuredMesh;
    use rouage::mesh::Mesh;

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
        assert_eq!(mesh.nvertexes, 893);
        assert_eq!(mesh.nedges, 1709);

        // Every quad element must have exactly 4 vertices and 4 edges
        let faces = mesh.get_elements_by_dimension(2);
        for &el in &faces {
            assert_eq!(
                mesh.neighbour_vertexes[el].len(),
                4,
                "element {} should have 4 vertices",
                el
            );
            assert_eq!(
                mesh.neighbour_edges[el].len(),
                4,
                "element {} should have 4 edges",
                el
            );
        }

        // Each edge is shared by at most 2 top-dimension elements
        for (k, neighbours) in mesh.edges_neighbour.iter().enumerate() {
            assert!(
                neighbours.len() <= 2,
                "edge {} has {} top-dim neighbours, expected at most 2",
                k,
                neighbours.len()
            );
        }

        // Symmetry: if edge k lists element el, then element el lists edge k
        for (k, neighbours) in mesh.edges_neighbour.iter().enumerate() {
            for entry in neighbours {
                let el = entry[0];
                assert!(
                    mesh.neighbour_edges[el].contains(&k),
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

        assert!(mesh.n_colors > 0);

        // Verify no two adjacent elements share the same color
        let graph = mesh.compute_graph();
        for (i, neighbors) in graph.iter().enumerate() {
            for &n in neighbors {
                assert_ne!(
                    mesh.coloring[i], mesh.coloring[n],
                    "Elements {} and {} are neighbors but share color {}",
                    i, n, mesh.coloring[i]
                );
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
        assert_eq!(mesh.nvertexes, 10598);
        assert_eq!(mesh.nedges, 63426);
        assert_eq!(mesh.nfaces, 100544);

        // Every tet must have exactly 4 vertices, 6 edges, and 4 faces
        let tetrahedra = mesh.get_elements_by_dimension(3);
        for &el in &tetrahedra {
            assert_eq!(
                mesh.neighbour_vertexes[el].len(),
                4,
                "tet {} should have 4 vertices",
                el
            );
            assert_eq!(
                mesh.neighbour_edges[el].len(),
                6,
                "tet {} should have 6 edges",
                el
            );
            assert_eq!(
                mesh.neighbour_faces[el].len(),
                4,
                "tet {} should have 4 faces",
                el
            );
        }

        // Each face is shared by at most 2 top-dimension elements
        for (k, neighbours) in mesh.faces_neighbour.iter().enumerate() {
            assert!(
                neighbours.len() <= 2,
                "face {} has {} top-dim neighbours, expected at most 2",
                k,
                neighbours.len()
            );
        }

        // Symmetry: if face k lists element el, then element el lists face k
        for (k, neighbours) in mesh.faces_neighbour.iter().enumerate() {
            for entry in neighbours {
                let el = entry[0];
                assert!(
                    mesh.neighbour_faces[el].contains(&k),
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
        assert_eq!(mesh.nvertexes, 5940);
        assert_eq!(mesh.nedges, 16884);
        assert_eq!(mesh.nfaces, 15984);

        // Every hex must have exactly 8 vertices, 12 edges, and 6 faces
        let hexahedra = mesh.get_elements_by_dimension(3);
        for &el in &hexahedra {
            assert_eq!(
                mesh.neighbour_vertexes[el].len(),
                8,
                "hex {} should have 8 vertices",
                el
            );
            assert_eq!(
                mesh.neighbour_edges[el].len(),
                12,
                "hex {} should have 12 edges",
                el
            );
            assert_eq!(
                mesh.neighbour_faces[el].len(),
                6,
                "hex {} should have 6 faces",
                el
            );
        }

        // Each face is shared by at most 2 top-dimension elements
        for (k, neighbours) in mesh.faces_neighbour.iter().enumerate() {
            assert!(
                neighbours.len() <= 2,
                "face {} has {} top-dim neighbours, expected at most 2",
                k,
                neighbours.len()
            );
        }

        // Symmetry: if face k lists element el, then element el lists face k
        for (k, neighbours) in mesh.faces_neighbour.iter().enumerate() {
            for entry in neighbours {
                let el = entry[0];
                assert!(
                    mesh.neighbour_faces[el].contains(&k),
                    "face {} references element {}, but element does not reference face back",
                    k,
                    el
                );
            }
        }
    }
}
