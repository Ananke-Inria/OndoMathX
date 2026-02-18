extern crate rouage;

#[cfg(test)]
mod tests {
    use rouage::deformation_field::*;

    #[test]
    fn identity_deformation() {
        let def = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], _f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        let input = [3.0, 7.0];
        let mut output = [0.0; 2];
        def.eval(&input, &mut output);

        assert!((output[0] - 3.0).abs() < 1e-12);
        assert!((output[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn scaling_deformation() {
        let def = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = 3.0 * xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], _f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        def.eval(&input, &mut output);

        assert!((output[0] - 2.0).abs() < 1e-12);
        assert!((output[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn compose_deformations() {
        // inner: (x, y) -> (2x, y)
        let inner = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = 2.0 * xyz[0];
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], _f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        // outer: (x, y) -> (x + 1, y + 1)
        let mut outer = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] + 1.0;
                f_xyz[1] = xyz[1] + 1.0;
            },
            grad_diffeo: |_xyz: &[f64], _f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        outer.compose(inner);

        // Result: (x, y) -> outer(inner(x, y)) = outer(2x, y) = (2x + 1, y + 1)
        let input = [3.0, 5.0];
        let mut output = [0.0; 2];
        outer.eval(&input, &mut output);

        assert!((output[0] - 7.0).abs() < 1e-12); // 2*3 + 1
        assert!((output[1] - 6.0).abs() < 1e-12); // 5 + 1
    }

    #[test]
    fn deformation_with_inner_diffeo() {
        // Build composed deformation via constructor
        let inner = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] * xyz[0]; // x^2
                f_xyz[1] = xyz[1];
            },
            grad_diffeo: |_xyz: &[f64], _f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        let outer = DeformationField {
            inner_dim: 2,
            outer_dim: 2,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0] + 10.0;
                f_xyz[1] = xyz[1] + 10.0;
            },
            grad_diffeo: |_xyz: &[f64], _f_xyz: &mut [f64]| {},
            inner_diffeo: Some(Box::new(inner)),
        };

        let input = [3.0, 2.0];
        let mut output = [0.0; 2];
        outer.eval(&input, &mut output);

        assert!((output[0] - 19.0).abs() < 1e-12); // 3^2 + 10
        assert!((output[1] - 12.0).abs() < 1e-12); // 2 + 10
    }

    #[test]
    fn deformation_3d() {
        let def = DeformationField {
            inner_dim: 3,
            outer_dim: 3,
            diffeo: |xyz: &[f64], f_xyz: &mut [f64]| {
                f_xyz[0] = xyz[0];
                f_xyz[1] = xyz[1];
                f_xyz[2] = xyz[2] + 1.0;
            },
            grad_diffeo: |_xyz: &[f64], _f_xyz: &mut [f64]| {},
            inner_diffeo: None,
        };

        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        def.eval(&input, &mut output);

        assert!((output[0] - 1.0).abs() < 1e-12);
        assert!((output[1] - 2.0).abs() < 1e-12);
        assert!((output[2] - 4.0).abs() < 1e-12);
    }
}
