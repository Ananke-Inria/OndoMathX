extern crate anamesh;

//Validation of the get_element_grad_def contract: grad_def is overwritten
//with the Jacobian of the COMPLETE element map (undeformed map composed
//with any attached deformation), row-major, shape geo_dim × elem_dim.
#[cfg(test)]
mod tests {
    use anamesh::diffeomorphism::Diffeomorphism;
    use anamesh::mesh::circle::Circle;
    use anamesh::mesh::square::Square;
    use anamesh::mesh::Mesh;

    //Nonlinear deformation with a smooth known Jacobian:
    //f(x, y) = (x + 0.1·sin(y), y + 0.2·x²), Df = [[1, 0.1·cos y], [0.4·x, 1]]
    fn bending() -> Diffeomorphism {
        Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0] + 0.1 * xyz[1].sin();
                f_xyz[1] = xyz[1] + 0.2 * xyz[0] * xyz[0];
            },
            grad_diffeo: |_p, xyz, grad| {
                grad[0] = 1.0;
                grad[1] = 0.1 * xyz[1].cos();
                grad[2] = 0.4 * xyz[0];
                grad[3] = 1.0;
            },
            inner_deformation: None,
        }
    }

    //Rigid rotation by 0.5 rad (area preserving: det Df = 1).
    fn rotation() -> Diffeomorphism {
        Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![0.5],
            diffeo: |p, xyz, f_xyz| {
                let (c, s) = (p[0].cos(), p[0].sin());
                f_xyz[0] = c * xyz[0] - s * xyz[1];
                f_xyz[1] = s * xyz[0] + c * xyz[1];
            },
            grad_diffeo: |p, _xyz, grad| {
                let (c, s) = (p[0].cos(), p[0].sin());
                grad[0] = c;
                grad[1] = -s;
                grad[2] = s;
                grad[3] = c;
            },
            inner_deformation: None,
        }
    }

    //Full element map by node interpolation composed with the attached
    //deformation. Exact for meshes whose undeformed geometry IS the
    //interpolation of their nodes (Square, UnstructuredMesh, and the
    //inner-square cells of Circle where the circle map is the identity).
    fn eval_map(mesh: &impl Mesh, el: usize, uvw: &[f64]) -> [f64; 3] {
        let element = mesh.get_element(el);
        let geo = element.geometric_element;

        let mut x = [0.0; 3];
        for i in 0..geo.nshape_functions {
            let p = geo.shape_functions[i](uvw);
            let node = mesh.get_node(element.indexes[i]);
            for (x_a, node_a) in x.iter_mut().zip(node.iter()) {
                *x_a += node_a * p;
            }
        }

        match mesh.get_deformation() {
            Some(def) => {
                let mut f_x = [0.0; 3];
                def.eval(&x, &mut f_x);
                f_x
            }
            None => x,
        }
    }

    //Compares get_element_grad_def with centred finite differences of
    //eval_map at `uvw`.
    fn check_fd(mesh: &impl Mesh, el: usize, uvw: &[f64]) {
        let elem_dim = mesh
            .get_element(el)
            .geometric_element
            .reference_element
            .get_dim();
        let geo_dim = mesh.get_geometric_dimension();
        let h = 1e-5;

        let mut jac = [0.0; 9];
        mesh.get_element_grad_def(el, uvw, &mut jac);

        for b in 0..elem_dim {
            let mut up = [uvw[0], uvw[1], uvw[2]];
            let mut um = up;
            up[b] += h;
            um[b] -= h;
            let f_p = eval_map(mesh, el, &up);
            let f_m = eval_map(mesh, el, &um);
            for a in 0..geo_dim {
                let fd = (f_p[a] - f_m[a]) / (2.0 * h);
                assert!(
                    (jac[a * elem_dim + b] - fd).abs() < 1e-7,
                    "element {el}, J[{a}][{b}] = {} but finite differences give {}",
                    jac[a * elem_dim + b],
                    fd
                );
            }
        }
    }

    fn check_fd_all(mesh: &impl Mesh, elements: impl Iterator<Item = usize>) {
        for el in elements {
            let elem_dim = mesh
                .get_element(el)
                .geometric_element
                .reference_element
                .get_dim();
            let points: &[[f64; 3]] = if elem_dim == 2 {
                &[[0.3, 0.7, 0.0], [0.55, 0.15, 0.0]]
            } else {
                &[[0.3, 0.0, 0.0], [0.85, 0.0, 0.0]]
            };
            for uvw in points {
                check_fd(mesh, el, uvw);
            }
        }
    }

    // === Square ===

    #[test]
    fn square_cells_are_diag() {
        let square = Square::new([0.0, 0.0], 2.0, 1.5, 4, 3, None, None);
        let (hx, hy) = (2.0 / 4.0, 1.5 / 3.0);

        for el in square.get_elements_by_dimension(2) {
            let mut jac = [0.0; 4];
            square.get_element_grad_def(el, &[0.3, 0.6, 0.0], &mut jac);
            assert!((jac[0] - hx).abs() < 1e-12, "el {el}: {jac:?}");
            assert!(jac[1].abs() < 1e-12, "el {el}: {jac:?}");
            assert!(jac[2].abs() < 1e-12, "el {el}: {jac:?}");
            assert!((jac[3] - hy).abs() < 1e-12, "el {el}: {jac:?}");
        }
    }

    #[test]
    fn square_boundary_edges_tangent() {
        let square = Square::new([0.0, 0.0], 2.0, 1.5, 4, 3, None, None);
        let (hx, hy) = (2.0 / 4.0, 1.5 / 3.0);

        //Expected 2×1 tangents per boundary group; the top (2) and left (3)
        //groups run reversed.
        let expected = [[hx, 0.0], [0.0, hy], [-hx, 0.0], [0.0, -hy]];

        for (label, expected) in expected.iter().enumerate() {
            let edges = square.get_elements_by_dimension_and_label(label, 1);
            assert!(!edges.is_empty());
            for el in edges {
                let mut jac = [0.0; 2];
                square.get_element_grad_def(el, &[0.4, 0.0, 0.0], &mut jac);
                assert!((jac[0] - expected[0]).abs() < 1e-12, "el {el}: {jac:?}");
                assert!((jac[1] - expected[1]).abs() < 1e-12, "el {el}: {jac:?}");
            }
        }
    }

    #[test]
    fn square_matches_fd() {
        let square = Square::new([-0.5, 0.25], 2.0, 1.5, 4, 3, None, None);
        check_fd_all(&square, 0..square.get_nelements());
    }

    #[test]
    fn square_deformed_matches_fd() {
        //Regression: the attached deformation used to be ignored by
        //get_element_grad_def.
        let square = Square::new([-0.5, 0.25], 2.0, 1.5, 4, 3, Some(bending()), None);
        check_fd_all(&square, 0..square.get_nelements());
    }

    // === UnstructuredMesh ===

    #[test]
    fn unstructured_matches_fd() {
        let mesh = Square::new([-0.5, 0.25], 2.0, 1.5, 3, 2, None, None).get_unstructured_mesh();
        check_fd_all(&mesh, 0..mesh.get_nelements());
    }

    #[test]
    fn unstructured_deformed_matches_fd() {
        //Regression: the deformation Jacobian used to be silently dropped
        //(the physical point xyz was computed but Df was never applied).
        let mut mesh =
            Square::new([-0.5, 0.25], 2.0, 1.5, 3, 2, None, None).get_unstructured_mesh();
        mesh.compose_deformation(bending());
        check_fd_all(&mesh, 0..mesh.get_nelements());
    }

    #[test]
    fn unstructured_q2_deformed_matches_fd() {
        //Q2 geometric elements (QUADRANGLE9 / EDGE3) with a deformation.
        let mut mesh =
            Square::new([-0.5, 0.25], 2.0, 1.5, 3, 2, None, Some(true)).get_unstructured_mesh();
        mesh.compose_deformation(bending());
        check_fd_all(&mesh, 0..mesh.get_nelements());
    }

    // === Circle ===

    #[test]
    fn circle_inner_square_deformed_matches_fd() {
        //Regression: Circle never composed a user-attached deformation into
        //its Jacobian. On the inner-square cells the circle map is the
        //identity, so the node interpolation of eval_map is the exact
        //undeformed map and finite differences apply.
        let circle = Circle::new([0.5, 0.25], 2.0, 6, Some(bending()), None);
        check_fd_all(&circle, 0..circle.n_x * circle.n_x);
    }

    #[test]
    fn circle_rotated_area() {
        //A rigid rotation preserves det J, so integrating det J over the
        //rotated disk must still give π·R². Exercises the deformation
        //composition on ALL cells (inner square and arms).
        let radius = 2.0;
        let circle = Circle::new([0.5, 0.25], radius, 8, Some(rotation()), None);

        //3-point Gauss-Legendre on [0, 1]
        let d = (3.0_f64 / 5.0).sqrt() / 2.0;
        let pts = [0.5 - d, 0.5, 0.5 + d];
        let wts = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0];

        let mut area = 0.0;
        for el in circle.get_elements_by_dimension(2) {
            for i in 0..3 {
                for j in 0..3 {
                    let mut jac = [0.0; 4];
                    circle.get_element_grad_def(el, &[pts[i], pts[j], 0.0], &mut jac);
                    area += wts[i] * wts[j] * (jac[0] * jac[3] - jac[1] * jac[2]);
                }
            }
        }

        let exact = std::f64::consts::PI * radius * radius;
        assert!(
            (area - exact).abs() < 1e-3 * exact,
            "area = {area}, exact = {exact}"
        );
    }

    // === get_element_def: the map differentiated by get_element_grad_def ===

    use anamesh::mesh::cube::Cube;

    //3D map with a smooth known Jacobian.
    fn twist_3d() -> Diffeomorphism {
        Diffeomorphism {
            inner_dim: 3,
            outer_dim: 3,
            params: vec![],
            diffeo: |_p, x, f| {
                f[0] = x[0] + 0.1 * x[2].sin();
                f[1] = x[1] + 0.1 * x[0] * x[0];
                f[2] = x[2] + 0.05 * x[0] * x[1];
            },
            grad_diffeo: |_p, x, g| {
                g.copy_from_slice(&[1.0, 0.0, 0.1 * x[2].cos(), 0.2 * x[0], 1.0, 0.0, 0.05 * x[1], 0.05 * x[0], 1.0]);
            },
            inner_deformation: None,
        }
    }

    //Centred finite differences of get_element_def against
    //get_element_grad_def, at interior and boundary-touching points.
    fn check_def_fd(mesh: &impl Mesh, elements: impl Iterator<Item = usize>) {
        let geo_dim = mesh.get_geometric_dimension();
        let h = 1e-5;
        for el in elements {
            let elem_dim = mesh.get_element(el).geometric_element.reference_element.get_dim();
            let points: Vec<[f64; 3]> = match elem_dim {
                3 => vec![[0.3, 0.7, 0.2], [0.55, 0.15, 0.9]],
                2 => vec![[0.3, 0.7, 0.0], [0.55, 0.15, 0.0]],
                _ => vec![[0.3, 0.0, 0.0], [0.85, 0.0, 0.0]],
            };
            for uvw in &points {
                let mut jac = [0.0; 9];
                mesh.get_element_grad_def(el, uvw, &mut jac);
                for b in 0..elem_dim {
                    let (mut up, mut um) = (*uvw, *uvw);
                    up[b] += h;
                    um[b] -= h;
                    let (mut fp, mut fm) = ([0.0; 3], [0.0; 3]);
                    mesh.get_element_def(el, &up, &mut fp);
                    mesh.get_element_def(el, &um, &mut fm);
                    for a in 0..geo_dim {
                        let fd = (fp[a] - fm[a]) / (2.0 * h);
                        assert!(
                            (jac[a * elem_dim + b] - fd).abs() < 1e-7,
                            "element {el}, J[{a}][{b}] = {} but FD of get_element_def gives {fd}",
                            jac[a * elem_dim + b]
                        );
                    }
                }
            }
        }
    }

    //At the corner vertices the map interpolates the (deformed) nodes.
    fn check_def_vertices(mesh: &impl Mesh) {
        let geo_dim = mesh.get_geometric_dimension();
        for el in mesh.get_elements_by_dimension(mesh.get_topologic_dimension()) {
            let element = mesh.get_element(el);
            let reference = element.geometric_element.reference_element;
            let corners: Vec<[f64; 3]> = match reference.get_dim() {
                2 => vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                _ => vec![
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                ],
            };
            for (v, uvw) in corners.iter().enumerate() {
                let mut node = mesh.get_node(element.indexes[v]);
                if let Some(def) = mesh.get_deformation() {
                    let mut f = [0.0; 3];
                    def.eval(&node, &mut f);
                    node = f;
                }
                let mut x = [0.0; 3];
                mesh.get_element_def(el, uvw, &mut x);
                for a in 0..geo_dim {
                    assert!((x[a] - node[a]).abs() < 1e-12, "element {el} vertex {v}: {x:?} vs {node:?}");
                }
            }
        }
    }

    #[test]
    fn def_matches_grad_def_square_and_unstructured() {
        for q2 in [None, Some(true)] {
            for deformation in [None, Some(bending())] {
                let square = Square::new([-0.5, 0.25], 2.0, 1.5, 4, 3, deformation.clone(), q2);
                check_def_fd(&square, 0..square.get_nelements());
                check_def_vertices(&square);
                let mesh = square.get_unstructured_mesh();
                check_def_fd(&mesh, 0..mesh.get_nelements());
                check_def_vertices(&mesh);
                //The default is the node interpolation of eval_map.
                for el in 0..mesh.get_nelements() {
                    let mut x = [0.0; 3];
                    mesh.get_element_def(el, &[0.3, 0.6, 0.0], &mut x);
                    let y = eval_map(&mesh, el, &[0.3, 0.6, 0.0]);
                    assert_eq!(x[..2], y[..2]);
                }
            }
        }
    }

    #[test]
    fn def_matches_grad_def_cube() {
        for q2 in [None, Some(true)] {
            for deformation in [None, Some(twist_3d())] {
                let cube = Cube::new([0.1, -0.2, 0.3], 1.0, 2.0, 1.5, 2, 3, 2, deformation.clone(), q2);
                check_def_fd(&cube, 0..cube.get_nelements());
                check_def_vertices(&cube);
            }
        }
    }

    #[test]
    fn def_matches_grad_def_circle() {
        //All cells and boundary edges, arms included (analytic map).
        for q2 in [None, Some(true)] {
            for deformation in [None, Some(bending())] {
                let circle = Circle::new([0.5, 0.25], 2.0, 7, deformation.clone(), q2);
                check_def_fd(&circle, 0..circle.get_nelements());
                check_def_vertices(&circle);
            }
        }
    }

    #[test]
    fn circle_boundary_is_exact() {
        //Points of the boundary edges lie on the circle, at any u.
        let (center, radius) = ([0.5, 0.25], 2.0);
        let circle = Circle::new(center, radius, 7, None, Some(true));
        for el in circle.get_elements_by_dimension(1) {
            for u in [0.0, 0.21, 0.5, 0.77, 1.0] {
                let mut x = [0.0; 3];
                circle.get_element_def(el, &[u, 0.0, 0.0], &mut x);
                let r = ((x[0] - center[0]).powi(2) + (x[1] - center[1]).powi(2)).sqrt();
                assert!((r - radius).abs() < 1e-12, "edge {el}, u = {u}: r = {r}");
            }
        }
    }

    #[test]
    fn mapped_elements() {
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 3, None, None);
        assert!((0..square.get_nelements()).all(|e| !square.is_mapped_element(e)));
        let square = Square::new([0.0, 0.0], 1.0, 1.0, 3, 3, Some(bending()), None);
        assert!((0..square.get_nelements()).all(|e| square.is_mapped_element(e)));

        let circle = Circle::new([0.0, 0.0], 1.0, 7, None, None);
        let n_square = circle.n_x * circle.n_x;
        for e in circle.get_elements_by_dimension(2) {
            assert_eq!(circle.is_mapped_element(e), e >= n_square, "cell {e}");
        }
        //Converting to an UnstructuredMesh keeps only the nodes: nothing is mapped.
        let mesh = circle.get_unstructured_mesh();
        assert!((0..mesh.get_nelements()).all(|e| !mesh.is_mapped_element(e)));
        let circle = Circle::new([0.0, 0.0], 1.0, 7, Some(rotation()), None);
        assert!((0..circle.get_nelements()).all(|e| circle.is_mapped_element(e)));
    }
}
