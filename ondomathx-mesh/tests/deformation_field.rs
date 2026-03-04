extern crate rouage;

#[cfg(test)]
mod tests {
    use rouage::deformation_field::*;

    #[test]
    fn identity_deformation() {
        // f(x,y) = (x,y), Df = I
        let def = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], _grad: &mut [f64]| {},
            inner_diffeo: None,
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
        // f(x,y) = (2x, 3y), Df = diag(2, 3)
        let def = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = 3.0 * xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], grad: &mut [f64]| {
                grad[0] = 2.0 * grad[0];
                grad[1] = 3.0 * grad[1];
            },
            inner_diffeo: None,
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
        // inner: g(x, y) = (2x, y), Dg = diag(2, 1)
        let inner = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], grad: &mut [f64]| {
                grad[0] = 2.0 * grad[0];
            },
            inner_diffeo: None,
        };

        // outer: f(x, y) = (x + 1, y + 1), Df = I
        let mut outer = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] + 1.0;
                f_xyz[1] = xyz[1] + 1.0;
            },
            grad_diffeo: |_xyz: &[f64], _grad: &mut [f64]| {},
            inner_diffeo: None,
        };

        outer.compose(inner);

        // f(g(x,y)) = (2x + 1, y + 1)
        let input = [3.0, 5.0];
        let mut output = [0.0; 2];
        outer.eval(&input, &mut output);

        assert!((output[0] - 7.0).abs() < 1e-12); // 2*3 + 1
        assert!((output[1] - 6.0).abs() < 1e-12); // 5 + 1
        assert!(outer.test_grad(&input, None, None));
    }

    #[test]
    fn deformation_with_inner_diffeo() {
        // inner: g(x,y) = (x^2, y), Dg(x,y) = diag(2x, 1)
        let inner = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] * xyz[0]; // x^2
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |xyz: &[f64], grad: &mut [f64]| {
                grad[0] = 2.0 * xyz[0] * grad[0];
            },
            inner_diffeo: None,
        };

        // outer: f(x,y) = (x + 10, y + 10), Df = I
        let outer = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] + 10.0;
                f_xyz[1] = xyz[1] + 10.0;
            },
            grad_diffeo: |_xyz: &[f64], _grad: &mut [f64]| {},
            inner_diffeo: Some(Box::new(inner)),
        };

        let input = [3.0, 2.0];
        let mut output = [0.0; 2];
        outer.eval(&input, &mut output);

        assert!((output[0] - 19.0).abs() < 1e-12); // 3^2 + 10
        assert!((output[1] - 12.0).abs() < 1e-12); // 2 + 10
        assert!(outer.test_grad(&input, None, None));
    }

    #[test]
    fn deformation_3d() {
        // f(x,y,z) = (x, y, z+1), Df = I
        let def = DeformationField {
            inner_dim: 3,
            outer_dim: 3,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0];
                f_xyz[1] = xyz[1];
                f_xyz[2] = xyz[2] + 1.0;
            },
            grad_diffeo: |_xyz: &[f64], _grad: &mut [f64]| {},
            inner_diffeo: None,
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
        // f(x,y) = R(pi/3) * (x,y), Df = R(pi/3) (constant rotation matrix)
        let def = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                let c = 0.5;
                let s = 3.0_f64.sqrt() / 2.0;
                f_xyz[0] = c * xyz[0] - s * xyz[1];
                f_xyz[1] = s * xyz[0] + c * xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], grad: &mut [f64]| {
                let c = 0.5;
                let s = 3.0_f64.sqrt() / 2.0;
                let (g0, g1) = (grad[0], grad[1]);
                grad[0] = c * g0 - s * g1;
                grad[1] = s * g0 + c * g1;
            },
            inner_diffeo: None,
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
        // f(r, theta) = (r cos theta, r sin theta)
        // Df = [[cos theta, -r sin theta], [sin theta, r cos theta]]
        let def = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                let (r, theta) = (xyz[0], xyz[1]);
                f_xyz[0] = r * theta.cos();
                f_xyz[1] = r * theta.sin();
            },
            grad_diffeo: |xyz: &[f64], grad: &mut [f64]| {
                let (r, theta) = (xyz[0], xyz[1]);
                let (g0, g1) = (grad[0], grad[1]);
                grad[0] = theta.cos() * g0 - r * theta.sin() * g1;
                grad[1] = theta.sin() * g0 + r * theta.cos() * g1;
            },
            inner_diffeo: None,
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
        // f(x,y,z) = (xy, yz, xz)
        // Df = [[y, x, 0], [0, z, y], [z, 0, x]]
        let def = DeformationField {
            inner_dim: 3,
            outer_dim: 3,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] * xyz[1];
                f_xyz[1] = xyz[1] * xyz[2];
                f_xyz[2] = xyz[0] * xyz[2];
            },
            grad_diffeo: |xyz: &[f64], grad: &mut [f64]| {
                let (x, y, z) = (xyz[0], xyz[1], xyz[2]);
                let (g0, g1, g2) = (grad[0], grad[1], grad[2]);
                grad[0] = y * g0 + x * g1;
                grad[1] = z * g1 + y * g2;
                grad[2] = z * g0 + x * g2;
            },
            inner_diffeo: None,
        };

        let input = [2.0, 3.0, 5.0];
        let mut output = [0.0; 3];
        def.eval(&input, &mut output);

        assert!((output[0] - 6.0).abs() < 1e-12);  // 2*3
        assert!((output[1] - 15.0).abs() < 1e-12); // 3*5
        assert!((output[2] - 10.0).abs() < 1e-12); // 2*5
        assert!(def.test_grad(&input, None, None));
    }

    #[test]
    fn composed_nonlinear() {
        // inner: g(x,y) = (x + y^2, y), Dg = [[1, 2y], [0, 1]]
        let inner = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] + xyz[1] * xyz[1];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |xyz: &[f64], grad: &mut [f64]| {
                grad[0] = grad[0] + 2.0 * xyz[1] * grad[1];
            },
            inner_diffeo: None,
        };

        // outer: f(u,v) = (u*v, u^2 - v), Df = [[v, u], [2u, -1]]
        let mut outer = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] * xyz[1];
                f_xyz[1] = xyz[0] * xyz[0] - xyz[1];
            },
            grad_diffeo: |xyz: &[f64], grad: &mut [f64]| {
                let (u, v) = (xyz[0], xyz[1]);
                let (g0, g1) = (grad[0], grad[1]);
                grad[0] = v * g0 + u * g1;
                grad[1] = 2.0 * u * g0 - g1;
            },
            inner_diffeo: None,
        };

        outer.compose(inner);

        // g(1,2) = (5, 2), f(5,2) = (10, 23)
        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        outer.eval(&input, &mut output);

        assert!((output[0] - 10.0).abs() < 1e-12); // 5*2
        assert!((output[1] - 23.0).abs() < 1e-12); // 25 - 2
        assert!(outer.test_grad(&input, None, None));
    }

    #[test]
    fn composed_nonlinear_3d() {
        // inner: g(x,y,z) = (x*y, y+z, x-z)
        // Dg = [[y, x, 0], [0, 1, 1], [1, 0, -1]]
        let inner = DeformationField {
            inner_dim: 3,
            outer_dim: 3,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] * xyz[1];
                f_xyz[1] = xyz[1] + xyz[2];
                f_xyz[2] = xyz[0] - xyz[2];
            },
            grad_diffeo: |xyz: &[f64], grad: &mut [f64]| {
                let (x, y, _z) = (xyz[0], xyz[1], xyz[2]);
                let (g0, g1, g2) = (grad[0], grad[1], grad[2]);
                grad[0] = y * g0 + x * g1;
                grad[1] = g1 + g2;
                grad[2] = g0 - g2;
            },
            inner_diffeo: None,
        };

        // outer: f(u,v,w) = (u^2 + v, v*w, u + w^2)
        // Df = [[2u, 1, 0], [0, w, v], [1, 0, 2w]]
        let mut outer = DeformationField {
            inner_dim: 3,
            outer_dim: 3,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] * xyz[0] + xyz[1];
                f_xyz[1] = xyz[1] * xyz[2];
                f_xyz[2] = xyz[0] + xyz[2] * xyz[2];
            },
            grad_diffeo: |xyz: &[f64], grad: &mut [f64]| {
                let (u, v, w) = (xyz[0], xyz[1], xyz[2]);
                let (g0, g1, g2) = (grad[0], grad[1], grad[2]);
                grad[0] = 2.0 * u * g0 + g1;
                grad[1] = w * g1 + v * g2;
                grad[2] = g0 + 2.0 * w * g2;
            },
            inner_diffeo: None,
        };

        outer.compose(inner);

        // g(2,3,1) = (6, 4, 1), f(6,4,1) = (40, 4, 7)
        let input = [2.0, 3.0, 1.0];
        let mut output = [0.0; 3];
        outer.eval(&input, &mut output);

        assert!((output[0] - 40.0).abs() < 1e-12); // 36 + 4
        assert!((output[1] - 4.0).abs() < 1e-12);  // 4*1
        assert!((output[2] - 7.0).abs() < 1e-12);  // 6 + 1
        assert!(outer.test_grad(&input, None, None));
    }
}
