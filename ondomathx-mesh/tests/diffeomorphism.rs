extern crate ondomathx_mesh;

//grad_diffeo contract: WRITE the Jacobian Df(xyz), row-major
//outer_dim × inner_dim (never read the previous content of the buffer);
//chain-rule products are performed by the library in grad_eval.
#[cfg(test)]
mod tests {
    use ondomathx_mesh::diffeomorphism::*;

    #[test]
    fn identity_deformation() {
        let def = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_p, _xyz, grad| {
                grad[0] = 1.0;
                grad[1] = 0.0;
                grad[2] = 0.0;
                grad[3] = 1.0;
            },
            inner_deformation: None,
        };

        let input = [3.0, 7.0];
        let mut output = [0.0; 2];
        def.eval(&input, &mut output);

        assert!((output[0] - 3.0).abs() < 1e-12);
        assert!((output[1] - 7.0).abs() < 1e-12);
        assert!(def.test_grad(&input, None, None));
    }

    #[test]
    fn scaling_deformation() {
        let def = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = 3.0 * xyz[1];
            },
            grad_diffeo: |_p, _xyz, grad| {
                grad[0] = 2.0;
                grad[1] = 0.0;
                grad[2] = 0.0;
                grad[3] = 3.0;
            },
            inner_deformation: None,
        };

        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        def.eval(&input, &mut output);

        assert!((output[0] - 2.0).abs() < 1e-12);
        assert!((output[1] - 6.0).abs() < 1e-12);
        assert!(def.test_grad(&input, None, None));
    }

    #[test]
    fn scaling_with_params() {
        let def = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![2.0, 3.0],
            diffeo: |p, xyz, f_xyz| {
                f_xyz[0] = p[0] * xyz[0];
                f_xyz[1] = p[1] * xyz[1];
            },
            grad_diffeo: |p, _xyz, grad| {
                grad[0] = p[0];
                grad[1] = 0.0;
                grad[2] = 0.0;
                grad[3] = p[1];
            },
            inner_deformation: None,
        };

        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        def.eval(&input, &mut output);

        assert!((output[0] - 2.0).abs() < 1e-12);
        assert!((output[1] - 6.0).abs() < 1e-12);
        assert!(def.test_grad(&input, None, None));
    }

    #[test]
    fn compose_deformations() {
        // inner: g(x, y) = (2x, y)
        let inner = Diffeomorphism {
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

        // outer: f(x, y) = (x + 1, y + 1)
        let mut outer = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0] + 1.0;
                f_xyz[1] = xyz[1] + 1.0;
            },
            grad_diffeo: |_p, _xyz, grad| {
                grad[0] = 1.0;
                grad[1] = 0.0;
                grad[2] = 0.0;
                grad[3] = 1.0;
            },
            inner_deformation: None,
        };

        outer.compose(inner);

        // f(g(x,y)) = (2x + 1, y + 1)
        let input = [3.0, 5.0];
        let mut output = [0.0; 2];
        outer.eval(&input, &mut output);

        assert!((output[0] - 7.0).abs() < 1e-12);
        assert!((output[1] - 6.0).abs() < 1e-12);
        assert!(outer.test_grad(&input, None, None));
    }

    #[test]
    fn deformation_with_inner_diffeo() {
        // inner: g(x,y) = (x^2, y)
        let inner = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0] * xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_p, xyz, grad| {
                grad[0] = 2.0 * xyz[0];
                grad[1] = 0.0;
                grad[2] = 0.0;
                grad[3] = 1.0;
            },
            inner_deformation: None,
        };

        // outer: f(x,y) = (x + 10, y + 10)
        let outer = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0] + 10.0;
                f_xyz[1] = xyz[1] + 10.0;
            },
            grad_diffeo: |_p, _xyz, grad| {
                grad[0] = 1.0;
                grad[1] = 0.0;
                grad[2] = 0.0;
                grad[3] = 1.0;
            },
            inner_deformation: Some(Box::new(inner)),
        };

        let input = [3.0, 2.0];
        let mut output = [0.0; 2];
        outer.eval(&input, &mut output);

        assert!((output[0] - 19.0).abs() < 1e-12);
        assert!((output[1] - 12.0).abs() < 1e-12);
        assert!(outer.test_grad(&input, None, None));
    }

    #[test]
    fn deformation_3d() {
        let def = Diffeomorphism {
            inner_dim: 3,
            outer_dim: 3,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0];
                f_xyz[1] = xyz[1];
                f_xyz[2] = xyz[2] + 1.0;
            },
            grad_diffeo: |_p, _xyz, grad| {
                grad.copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            },
            inner_deformation: None,
        };

        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        def.eval(&input, &mut output);

        assert!((output[0] - 1.0).abs() < 1e-12);
        assert!((output[1] - 2.0).abs() < 1e-12);
        assert!((output[2] - 4.0).abs() < 1e-12);
        assert!(def.test_grad(&input, None, None));
    }

    #[test]
    fn rotation_deformation() {
        let def = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                let c = 0.5;
                let s = 3.0_f64.sqrt() / 2.0;
                f_xyz[0] = c * xyz[0] - s * xyz[1];
                f_xyz[1] = s * xyz[0] + c * xyz[1];
            },
            grad_diffeo: |_p, _xyz, grad| {
                let c = 0.5;
                let s = 3.0_f64.sqrt() / 2.0;
                grad[0] = c;
                grad[1] = -s;
                grad[2] = s;
                grad[3] = c;
            },
            inner_deformation: None,
        };

        let input = [1.0, 2.0];
        let c = 0.5;
        let s = 3.0_f64.sqrt() / 2.0;
        let mut output = [0.0; 2];
        def.eval(&input, &mut output);

        assert!((output[0] - (c - 2.0 * s)).abs() < 1e-12);
        assert!((output[1] - (s + 2.0 * c)).abs() < 1e-12);
        assert!(def.test_grad(&input, None, None));
    }

    #[test]
    fn polar_map() {
        let def = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                let (r, theta) = (xyz[0], xyz[1]);
                f_xyz[0] = r * theta.cos();
                f_xyz[1] = r * theta.sin();
            },
            grad_diffeo: |_p, xyz, grad| {
                let (r, theta) = (xyz[0], xyz[1]);
                grad[0] = theta.cos();
                grad[1] = -r * theta.sin();
                grad[2] = theta.sin();
                grad[3] = r * theta.cos();
            },
            inner_deformation: None,
        };

        let input = [2.0, std::f64::consts::PI / 4.0];
        let mut output = [0.0; 2];
        def.eval(&input, &mut output);

        assert!((output[0] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((output[1] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!(def.test_grad(&input, None, None));
    }

    #[test]
    fn bilinear_product_3d() {
        let def = Diffeomorphism {
            inner_dim: 3,
            outer_dim: 3,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0] * xyz[1];
                f_xyz[1] = xyz[1] * xyz[2];
                f_xyz[2] = xyz[0] * xyz[2];
            },
            grad_diffeo: |_p, xyz, grad| {
                let (x, y, z) = (xyz[0], xyz[1], xyz[2]);
                grad.copy_from_slice(&[y, x, 0.0, 0.0, z, y, z, 0.0, x]);
            },
            inner_deformation: None,
        };

        let input = [2.0, 3.0, 5.0];
        let mut output = [0.0; 3];
        def.eval(&input, &mut output);

        assert!((output[0] - 6.0).abs() < 1e-12);
        assert!((output[1] - 15.0).abs() < 1e-12);
        assert!((output[2] - 10.0).abs() < 1e-12);
        assert!(def.test_grad(&input, None, None));
    }

    #[test]
    fn composed_nonlinear() {
        // inner: g(x,y) = (x + y^2, y)
        let inner = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0] + xyz[1] * xyz[1];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_p, xyz, grad| {
                grad[0] = 1.0;
                grad[1] = 2.0 * xyz[1];
                grad[2] = 0.0;
                grad[3] = 1.0;
            },
            inner_deformation: None,
        };

        // outer: f(u,v) = (u*v, u^2 - v)
        let mut outer = Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0] * xyz[1];
                f_xyz[1] = xyz[0] * xyz[0] - xyz[1];
            },
            grad_diffeo: |_p, xyz, grad| {
                let (u, v) = (xyz[0], xyz[1]);
                grad[0] = v;
                grad[1] = u;
                grad[2] = 2.0 * u;
                grad[3] = -1.0;
            },
            inner_deformation: None,
        };

        outer.compose(inner);

        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        outer.eval(&input, &mut output);

        assert!((output[0] - 10.0).abs() < 1e-12);
        assert!((output[1] - 23.0).abs() < 1e-12);
        assert!(outer.test_grad(&input, None, None));
    }

    #[test]
    fn composed_nonlinear_3d() {
        // inner: g(x,y,z) = (x*y, y+z, x-z)
        let inner = Diffeomorphism {
            inner_dim: 3,
            outer_dim: 3,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0] * xyz[1];
                f_xyz[1] = xyz[1] + xyz[2];
                f_xyz[2] = xyz[0] - xyz[2];
            },
            grad_diffeo: |_p, xyz, grad| {
                let (x, y, _z) = (xyz[0], xyz[1], xyz[2]);
                grad.copy_from_slice(&[y, x, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0]);
            },
            inner_deformation: None,
        };

        // outer: f(u,v,w) = (u^2 + v, v*w, u + w^2)
        let mut outer = Diffeomorphism {
            inner_dim: 3,
            outer_dim: 3,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = xyz[0] * xyz[0] + xyz[1];
                f_xyz[1] = xyz[1] * xyz[2];
                f_xyz[2] = xyz[0] + xyz[2] * xyz[2];
            },
            grad_diffeo: |_p, xyz, grad| {
                let (u, v, w) = (xyz[0], xyz[1], xyz[2]);
                grad.copy_from_slice(&[2.0 * u, 1.0, 0.0, 0.0, w, v, 1.0, 0.0, 2.0 * w]);
            },
            inner_deformation: None,
        };

        outer.compose(inner);

        let input = [2.0, 3.0, 1.0];
        let mut output = [0.0; 3];
        outer.eval(&input, &mut output);

        assert!((output[0] - 40.0).abs() < 1e-12);
        assert!((output[1] - 4.0).abs() < 1e-12);
        assert!((output[2] - 7.0).abs() < 1e-12);
        assert!(outer.test_grad(&input, None, None));
    }

    // === compose_grad_def ===

    fn scaling() -> Diffeomorphism {
        // f(x, y) = (2x, 3y), Df = [[2, 0], [0, 3]]
        Diffeomorphism {
            inner_dim: 2,
            outer_dim: 2,
            params: vec![],
            diffeo: |_p, xyz, f_xyz| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = 3.0 * xyz[1];
            },
            grad_diffeo: |_p, _xyz, grad| {
                grad[0] = 2.0;
                grad[1] = 0.0;
                grad[2] = 0.0;
                grad[3] = 3.0;
            },
            inner_deformation: None,
        }
    }

    #[test]
    fn compose_grad_def_matrix() {
        // Df · J = [[2,0],[0,3]] · [[1,2],[3,4]] = [[2,4],[9,12]]
        let def = scaling();
        let mut grad_def = [1.0, 2.0, 3.0, 4.0];
        def.compose_grad_def(&[0.5, 0.5], &mut grad_def, 2);

        assert!((grad_def[0] - 2.0).abs() < 1e-12);
        assert!((grad_def[1] - 4.0).abs() < 1e-12);
        assert!((grad_def[2] - 9.0).abs() < 1e-12);
        assert!((grad_def[3] - 12.0).abs() < 1e-12);
    }

    #[test]
    fn compose_grad_def_tangent() {
        // Df · t = [[2,0],[0,3]] · [1,3]ᵀ = [2,9]ᵀ (elem_dim = 1)
        let def = scaling();
        let mut grad_def = [1.0, 3.0];
        def.compose_grad_def(&[0.5, 0.5], &mut grad_def, 1);

        assert!((grad_def[0] - 2.0).abs() < 1e-12);
        assert!((grad_def[1] - 9.0).abs() < 1e-12);
    }
}
