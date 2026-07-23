/// A smooth deformation f: R^inner_dim -> R^outer_dim,
/// optionally composed with an inner deformation g, yielding f ∘ g.
///
/// - `params`: auxiliary parameters passed to the functions.
/// - `diffeo`: evaluates the map: `(params, xyz) -> f_xyz`.
/// - `grad_diffeo`: applies Df in-place: `grad <- Df(xyz) · grad`.
/// - `inner_deformation`: optional inner deformation g for composition f ∘ g.
#[derive(Debug, Clone)]
pub struct Diffeomorphism {
    pub inner_dim: usize,
    pub outer_dim: usize,
    pub params: Vec<f64>,
    pub diffeo: fn(&[f64], &[f64], &mut [f64]),
    pub grad_diffeo: fn(&[f64], &[f64], &mut [f64]),
    pub inner_deformation: Option<Box<Diffeomorphism>>,
}

impl Diffeomorphism {
    /// Evaluates the deformation at `xyz`.
    /// If an inner deformation g is set, computes f(g(xyz)).
    pub fn eval(&self, xyz: &[f64], f_xyz: &mut [f64]) {
        let input = if let Some(inner) = &self.inner_deformation {
            let mut g = vec![0.0; self.inner_dim];
            inner.eval(xyz, &mut g);
            g
        } else {
            xyz.to_vec()
        };
        (self.diffeo)(&self.params, &input, f_xyz);
    }

    /// Applies the Jacobian in-place: `grad <- Df(xyz) · grad`.
    /// If an inner deformation g is set, applies the chain rule:
    /// `grad <- Df(g(x)) · Dg(x) · grad`.
    pub fn grad_eval(&self, xyz: &[f64], grad: &mut [f64]) {
        let input = if let Some(inner) = &self.inner_deformation {
            let mut g = vec![0.0; self.inner_dim];
            inner.eval(xyz, &mut g);
            inner.grad_eval(xyz, grad);
            g
        } else {
            xyz.to_vec()
        };
        (self.grad_diffeo)(&self.params, &input, grad);
    }

    /// Validates `grad_eval` against centred finite differences at `xyz`.
    /// Returns `true` if all entries match within `tol` (default 1e-6).
    pub fn test_grad(&self, xyz: &[f64], tol: Option<f64>, step: Option<f64>) -> bool {
        let tol = tol.unwrap_or(1e-6);
        let h = step.unwrap_or(1e-7);
        let input_dim = xyz.len();
        for j in 0..input_dim {
            let mut e_j = vec![0.0; input_dim];
            e_j[j] = 1.0;
            let mut grad = e_j;
            self.grad_eval(xyz, &mut grad);
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

    /// Sets the inner deformation g so that this field becomes f ∘ g.
    pub fn compose(&mut self, inner: Diffeomorphism) {
        self.inner_deformation = Some(Box::new(inner));
    }
}
