extern crate anamesh;

//Validation of Mesh::get_tensor_numbering: a continuous numbering of a
//tensor DoF layout (n[d] DoFs per direction, local index x-fastest) on the
//volume elements. Checked geometrically: all copies of a global DoF sit at
//the same physical point, distinct DoFs at distinct points.
#[cfg(test)]
mod tests {
    use anamesh::mesh::circle::Circle;
    use anamesh::mesh::cube::Cube;
    use anamesh::mesh::square::Square;
    use anamesh::mesh::{DoFSet, Mesh};

    //Tensor reference points (equispaced, x fastest) of a layout n.
    fn tensor_points(n: [usize; 3], dim: usize) -> Vec<[f64; 3]> {
        let nloc: usize = n[..dim].iter().product();
        (0..nloc)
            .map(|l| {
                let mut p = [0.0; 3];
                let mut k = l;
                for d in 0..dim {
                    p[d] = (k % n[d]) as f64 / (n[d] - 1) as f64;
                    k /= n[d];
                }
                p
            })
            .collect()
    }

    //Copies of a global DoF coincide, distinct DoFs are distinct points
    //(compared on a 1e-9 grid, the meshes being O(1)).
    fn check_numbering(mesh: &impl Mesh, n: [usize; 3]) -> usize {
        let dim = mesh.get_topologic_dimension();
        let geo_dim = mesh.get_geometric_dimension();
        let (numbering, n_dof) = mesh.get_tensor_numbering(n);
        let points = tensor_points(n, dim);
        let mut position: Vec<Option<[f64; 3]>> = vec![None; n_dof];
        for el in mesh.get_elements_by_dimension(dim) {
            assert_eq!(numbering[el].len(), points.len(), "element {el}");
            for (l, uvw) in points.iter().enumerate() {
                let mut x = [0.0; 3];
                mesh.get_element_def(el, uvw, &mut x);
                let g = numbering[el][l];
                match position[g] {
                    None => position[g] = Some(x),
                    Some(y) => assert!(
                        (0..geo_dim).all(|a| (x[a] - y[a]).abs() < 1e-10),
                        "global DoF {g}: {x:?} vs {y:?} (element {el}, local {l})"
                    ),
                }
            }
        }
        assert!(position.iter().all(|p| p.is_some()), "unused global DoF");
        let mut keys: Vec<[i64; 3]> = position
            .iter()
            .map(|p| p.unwrap().map(|v| (v * 1e9).round() as i64))
            .collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), n_dof, "two global DoFs at the same point");
        n_dof
    }

    #[test]
    fn square_anisotropic() {
        let square = Square::new([0.0, 0.0], 2.0, 1.0, 4, 3, None, None);
        for n in [[2, 2, 1], [3, 5, 1], [6, 2, 1], [4, 4, 1]] {
            let n_dof = check_numbering(&square, n);
            assert_eq!(n_dof, (4 * (n[0] - 1) + 1) * (3 * (n[1] - 1) + 1));
        }
    }

    #[test]
    fn cube_anisotropic() {
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 2.0, 1.0, 2, 3, 2, None, None);
        for n in [[2, 2, 2], [3, 5, 2], [4, 2, 3]] {
            let n_dof = check_numbering(&cube, n);
            assert_eq!(n_dof, (2 * (n[0] - 1) + 1) * (3 * (n[1] - 1) + 1) * (2 * (n[2] - 1) + 1));
        }
    }

    #[test]
    fn unstructured_and_circle_isotropic() {
        let square = Square::new([0.0, 0.0], 2.0, 1.0, 4, 3, None, Some(true));
        let mesh = square.get_unstructured_mesh();
        for k in [2, 3, 5] {
            //Same DoF count as the structured numbering of the same mesh.
            assert_eq!(check_numbering(&mesh, [k, k, 1]), square.get_tensor_numbering([k, k, 1]).1);
        }
        let circle = Circle::new([0.0, 0.0], 1.0, 7, None, Some(true));
        for k in [2, 4] {
            check_numbering(&circle, [k, k, 1]);
        }
        let cube = Cube::new([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, 2, 2, 2, None, None).get_unstructured_mesh();
        check_numbering(&cube, [3, 3, 3]);
    }

    #[test]
    fn default_equals_get_numbering() {
        //The default is UnstructuredMesh::get_numbering with a tensor DoFSet
        //(the numbering does not depend on where the symmetric points sit).
        let mesh = Square::new([0.0, 0.0], 1.0, 1.0, 3, 2, None, None).get_unstructured_mesh();
        let xi = [0.0, 0.17, 0.5, 0.83, 1.0];
        let pos: Vec<[f64; 3]> = (0..25).map(|l| [xi[l % 5], xi[l / 5], 0.0]).collect();
        let reference = mesh.get_element(mesh.get_elements_by_dimension(2)[0]).geometric_element.reference_element;
        let (a, na) = mesh.get_numbering(&DoFSet::new(pos, reference));
        let (b, nb) = mesh.get_tensor_numbering([5, 5, 1]);
        assert_eq!(na, nb);
        for el in mesh.get_elements_by_dimension(2) {
            assert_eq!(a[el], b[el]);
        }
    }

    #[test]
    #[should_panic(expected = "same count")]
    fn unstructured_anisotropic_panics() {
        let mesh = Square::new([0.0, 0.0], 1.0, 1.0, 2, 2, None, None).get_unstructured_mesh();
        let _ = mesh.get_tensor_numbering([3, 4, 1]);
    }
}
