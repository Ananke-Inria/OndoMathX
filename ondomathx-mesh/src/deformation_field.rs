#[derive(Debug, Clone)]
pub struct DeformationField {
    pub inner_dim: usize,
    pub outer_dim: usize,
    pub diffeo: fn(&[f64], &mut [f64]),
    pub grad_diffeo: fn(&[f64], &mut [f64]),
    pub inner_diffeo: Option<Box<DeformationField>>,
}

impl DeformationField {
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

    pub fn compose(&mut self, inner: DeformationField) {
        self.inner_diffeo = Some(Box::new(inner));
    }
}
