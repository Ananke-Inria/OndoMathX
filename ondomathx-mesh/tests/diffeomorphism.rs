extern crate anamesh;

#[cfg(test)]
mod tests {
    use anamesh::diffeomorphism::*;

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
            grad_diffeo: |_p, _xyz, _grad| {},
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
                grad[0] *= 2.0;
                grad[1] *= 3.0;
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
                grad[0] *= p[0];
                grad[1] *= p[1];
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
                grad[0] *= 2.0;
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
            grad_diffeo: |_p, _xyz, _grad| {},
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
                grad[0] *= 2.0 * xyz[0];
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
            grad_diffeo: |_p, _xyz, _grad| {},
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
            grad_diffeo: |_p, _xyz, _grad| {},
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
                let (g0, g1) = (grad[0], grad[1]);
                grad[0] = c * g0 - s * g1;
                grad[1] = s * g0 + c * g1;
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
                let (g0, g1) = (grad[0], grad[1]);
                grad[0] = theta.cos() * g0 - r * theta.sin() * g1;
                grad[1] = theta.sin() * g0 + r * theta.cos() * g1;
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
                let (g0, g1, g2) = (grad[0], grad[1], grad[2]);
                grad[0] = y * g0 + x * g1;
                grad[1] = z * g1 + y * g2;
                grad[2] = z * g0 + x * g2;
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
                grad[0] += 2.0 * xyz[1] * grad[1];
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
                let (g0, g1) = (grad[0], grad[1]);
                grad[0] = v * g0 + u * g1;
                grad[1] = 2.0 * u * g0 - g1;
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
                let (g0, g1, g2) = (grad[0], grad[1], grad[2]);
                grad[0] = y * g0 + x * g1;
                grad[1] = g1 + g2;
                grad[2] = g0 - g2;
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
                let (g0, g1, g2) = (grad[0], grad[1], grad[2]);
                grad[0] = 2.0 * u * g0 + g1;
                grad[1] = w * g1 + v * g2;
                grad[2] = g0 + 2.0 * w * g2;
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
}
