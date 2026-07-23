extern crate anamesh;

#[cfg(test)]
mod tests {
    use anamesh::dof_set::*;
    use anamesh::reference_elements::*;

    #[test]
    fn dof_set_edge_p1() {
        // 2 DoFs at the vertices of the edge
        let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

        let dof_set = DoFSet::new(pos, &EDGE);

        assert_eq!(dof_set.get_ndof(), 2);
        assert_eq!(dof_set.dof_on_vertexes[0].len(), 1);
        assert_eq!(dof_set.dof_on_vertexes[1].len(), 1);
    }

    #[test]
    fn dof_set_edge_p2() {
        // 3 DoFs: vertices + midpoint
        let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]];

        let dof_set = DoFSet::new(pos, &EDGE);

        assert_eq!(dof_set.get_ndof(), 3);
        // Each vertex has 1 DoF
        assert_eq!(dof_set.dof_on_vertexes[0].len(), 1);
        assert_eq!(dof_set.dof_on_vertexes[1].len(), 1);
        // 1 DoF strictly in the edge (midpoint)
        assert_eq!(dof_set.dof_strictly_in_edges[0][0].len(), 1);
    }

    #[test]
    fn dof_set_triangle_p1() {
        // 3 DoFs at the vertices of the triangle
        let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let dof_set = DoFSet::new(pos, &TRIANGLE);

        assert_eq!(dof_set.get_ndof(), 3);
        // Each vertex has exactly 1 DoF
        for v in 0..3 {
            assert_eq!(dof_set.dof_on_vertexes[v].len(), 1);
        }
        // No DoFs strictly in edges
        for e in 0..3 {
            assert_eq!(dof_set.dof_strictly_in_edges[e][0].len(), 0);
        }
    }

    #[test]
    fn dof_set_triangle_p2() {
        // 6 DoFs: 3 vertices + 3 edge midpoints
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0], // mid edge 0
            [0.5, 0.5, 0.0], // mid edge 1
            [0.0, 0.5, 0.0], // mid edge 2
        ];

        let dof_set = DoFSet::new(pos, &TRIANGLE);

        assert_eq!(dof_set.get_ndof(), 6);
        // Each vertex has 1 DoF
        for v in 0..3 {
            assert_eq!(dof_set.dof_on_vertexes[v].len(), 1);
        }
        // Each edge has 1 interior DoF
        for e in 0..3 {
            assert_eq!(dof_set.dof_strictly_in_edges[e][0].len(), 1);
        }
    }

    #[test]
    fn dof_set_quadrangle_q1() {
        // 4 DoFs at the vertices of the quadrangle
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let dof_set = DoFSet::new(pos, &QUADRANGLE);

        assert_eq!(dof_set.get_ndof(), 4);
        for v in 0..4 {
            assert_eq!(dof_set.dof_on_vertexes[v].len(), 1);
        }
    }

    #[test]
    fn dof_set_quadrangle_q2() {
        // 9 DoFs: 4 vertices + 4 edge midpoints + 1 center
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0], // mid edge 0
            [1.0, 0.5, 0.0], // mid edge 1
            [0.5, 1.0, 0.0], // mid edge 2
            [0.0, 0.5, 0.0], // mid edge 3
            [0.5, 0.5, 0.0], // center
        ];

        let dof_set = DoFSet::new(pos, &QUADRANGLE);

        assert_eq!(dof_set.get_ndof(), 9);
        for v in 0..4 {
            assert_eq!(dof_set.dof_on_vertexes[v].len(), 1);
        }
        for e in 0..4 {
            assert_eq!(dof_set.dof_strictly_in_edges[e][0].len(), 1);
        }
    }
}
