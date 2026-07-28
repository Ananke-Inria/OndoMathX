/// A smooth deformation f: R^inner_dim -> R^outer_dim,
/// optionally composed with an inner deformation g, yielding f ∘ g.
///
/// - `params`: auxiliary parameters passed to the functions.
/// - `diffeo`: evaluates the map: `(params, xyz) -> f_xyz`.
/// - `grad_diffeo`: writes the Jacobian Df(xyz) into its output slice,
///   row-major `outer_dim × inner_dim`. It must overwrite every entry and
///   never read the previous content; any chain-rule products are performed
///   by the library in `grad_eval`.
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
    /// Dimension of the domain of the full (possibly composed) map:
    /// the `inner_dim` of the innermost deformation.
    pub fn input_dim(&self) -> usize {
        match &self.inner_deformation {
            Some(inner) => inner.input_dim(),
            None => self.inner_dim,
        }
    }

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

    /// Writes the Jacobian of the full map at `xyz` into `jac`, row-major
    /// `outer_dim × input_dim()`. If an inner deformation g is set, the
    /// chain rule D(f ∘ g)(xyz) = Df(g(xyz)) · Dg(xyz) is applied here.
    pub fn grad_eval(&self, xyz: &[f64], jac: &mut [f64]) {
        if let Some(inner) = &self.inner_deformation {
            let input_dim = inner.input_dim();

            let mut g = vec![0.0; self.inner_dim];
            inner.eval(xyz, &mut g);

            let mut dg = vec![0.0; inner.outer_dim * input_dim];
            inner.grad_eval(xyz, &mut dg);

            let mut df = vec![0.0; self.outer_dim * self.inner_dim];
            (self.grad_diffeo)(&self.params, &g, &mut df);

            // jac = Df (outer_dim × inner_dim) · Dg (inner_dim × input_dim)
            for i in 0..self.outer_dim {
                for j in 0..input_dim {
                    let mut s = 0.0;
                    for k in 0..self.inner_dim {
                        s += df[i * self.inner_dim + k] * dg[k * input_dim + j];
                    }
                    jac[i * input_dim + j] = s;
                }
            }
        } else {
            (self.grad_diffeo)(&self.params, xyz, jac);
        }
    }

    /// Composes this deformation into an element Jacobian in place:
    /// `grad_def ← Df(xyz) · grad_def`, where `grad_def` is row-major
    /// `input_dim() × elem_dim` on entry and `outer_dim × elem_dim` on exit
    /// (`xyz` is the point the element map sends `uvw` to, i.e. where Df is
    /// evaluated). This is the chain rule shared by every mesh type's
    /// `get_element_grad_def`.
    pub fn compose_grad_def(&self, xyz: &[f64], grad_def: &mut [f64], elem_dim: usize) {
        let in_dim = self.input_dim();

        let mut df = [0.0; 9];
        self.grad_eval(xyz, &mut df[..self.outer_dim * in_dim]);

        let mut jac = [0.0; 9];
        for a in 0..self.outer_dim {
            for b in 0..elem_dim {
                let mut s = 0.0;
                for k in 0..in_dim {
                    s += df[a * in_dim + k] * grad_def[k * elem_dim + b];
                }
                jac[a * elem_dim + b] = s;
            }
        }
        grad_def[..self.outer_dim * elem_dim].copy_from_slice(&jac[..self.outer_dim * elem_dim]);
    }

    /// Validates `grad_eval` against centred finite differences at `xyz`.
    /// Returns `true` if all entries match within `tol` (default 1e-6).
    pub fn test_grad(&self, xyz: &[f64], tol: Option<f64>, step: Option<f64>) -> bool {
        let tol = tol.unwrap_or(1e-6);
        let h = step.unwrap_or(1e-7);
        let input_dim = self.input_dim();

        let mut jac = vec![0.0; self.outer_dim * input_dim];
        self.grad_eval(xyz, &mut jac);

        for j in 0..input_dim {
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
                if (jac[i * input_dim + j] - fd).abs() > tol {
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
