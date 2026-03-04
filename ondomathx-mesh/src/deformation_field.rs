/// Represents a smooth deformation field f: R^inner_dim -> R^outer_dim,
/// optionally composed with an inner deformation g, yielding f o g.
///
/// - `diffeo`: evaluates the map f(x) -> f_xyz.
/// - `grad_diffeo`: multiplies a vector in-place by Df(x), i.e. grad <- Df(x) * grad.
/// - `inner_diffeo`: optional inner deformation g for composition f o g.
#[derive(Debug, Clone)]
pub struct DeformationField {
    pub inner_dim: usize,
    pub outer_dim: usize,
    pub diffeo: fn(&[f64], &mut [f64]),
    pub grad_diffeo: fn(&[f64], &mut [f64]),
    pub inner_diffeo: Option<Box<DeformationField>>,
}

impl DeformationField {
    /// Evaluates the deformation field at `xyz`.
    /// If an inner deformation g is set, computes f(g(xyz)).
    pub fn eval(&self, xyz: &[f64], f_xyz: &mut [f64]) {
        if self.inner_diffeo.is_none() {
            (self.diffeo)(xyz, f_xyz);
        } else {
            let inner_diffeo = self.inner_diffeo.as_ref().unwrap().as_ref();
            let mut g_xyz = vec![0.0; self.inner_dim];

            inner_diffeo.eval(xyz, &mut g_xyz);
            (self.diffeo)(&g_xyz, f_xyz);
        }
    }

    /// Multiplies `grad` in-place by the Jacobian of the deformation at `xyz`.
    /// If an inner deformation g is set, applies the chain rule:
    /// grad <- Df(g(xyz)) * Dg(xyz) * grad.
    pub fn grad_eval(&self, xyz: &[f64], grad: &mut [f64]) {
        if self.inner_diffeo.is_none() {
            (self.grad_diffeo)(xyz, grad);
        } else {
            let inner_diffeo = self.inner_diffeo.as_ref().unwrap().as_ref();
            let mut g_xyz = vec![0.0; self.inner_dim];

            inner_diffeo.eval(xyz, &mut g_xyz);
            inner_diffeo.grad_eval(xyz, grad);
            (self.grad_diffeo)(&g_xyz, grad);
        }
    }

    /// Validates `grad_eval` against centered finite differences at point `xyz`.
    /// For each canonical direction e_j, compares Df(xyz) * e_j with
    /// (f(xyz + h*e_j) - f(xyz - h*e_j)) / (2h).
    /// Returns `true` if all entries match within tolerance.
    ///
    /// - `tol`: comparison tolerance (default: 1e-6).
    /// - `step`: finite difference step h (default: 1e-7).
    pub fn test_grad(&self, xyz: &[f64], tol: Option<f64>, step: Option<f64>) -> bool {
        let tol = tol.unwrap_or(1e-6);
        let h = step.unwrap_or(1e-7);

        let input_dim = xyz.len();

        for j in 0..input_dim {
            let mut e_j = vec![0.0; input_dim];
            e_j[j] = 1.0;

            // Df * e_j
            let mut grad = e_j;
            self.grad_eval(xyz, &mut grad);

            // (f(x + h*e_j) - f(x - h*e_j)) / (2h)
            let mut xyz_ph = xyz.to_vec();
            let mut xyz_mh = xyz.to_vec();
            xyz_ph[j] += h;
            xyz_mh[j] -= h;

            let mut f_ph = vec![0.0; self.outer_dim];
            let mut f_mh = vec![0.0; self.outer_dim];
            self.eval(&xyz_ph, &mut f_ph);
            self.eval(&xyz_mh, &mut f_mh);

            for i in 0..self.outer_dim {
                let fd = (f_ph[i] - f_mh[i]) / (2.0 * h);
                if (grad[i] - fd).abs() > tol {
                    return false;
                }
            }
        }

        true
    }

    /// Sets the inner deformation g so that this field becomes f o g.
    pub fn compose(&mut self, inner: DeformationField) {
        self.inner_diffeo = Some(Box::new(inner));
    }
}
