//! The map between per-element ("DG") storage and the global DoFs of a
//! continuous space, shared by `ondomathx-tetrix` and `ondomathx-sem`.
//!
//! A DG vector stores every element's nodes contiguously: with `C`
//! components and `nloc` nodes per element, component `c` of local node `i`
//! of element `e` is at `e·C·nloc + c·nloc + i` (`[elem][comp][node]`). A
//! global DoF shared by `k` elements has `k` copies. A global vector stores
//! one value per DoF and component, `g·C + c` (`[dof][comp]`).
//!
//! Copies are always visited in increasing DG index, so every sum has a
//! fixed order (results do not depend on the number of threads).

use rayon::prelude::*;

/// DG ↔ global map of a continuous space: for each global DoF its copies
/// (`glob2dg`, CSR), for each copy its DoF (`dg2glob`), and the DoFs with at
/// least two copies (the shared groups, the only ones a shared-DoF sum
/// touches). Indices are `u32` (at most `u32::MAX` copies).
pub struct DgMap {
    nloc: usize,
    n_dof: usize,
    dg2glob: Vec<u32>,
    glob_offsets: Vec<u32>,
    glob_indices: Vec<u32>,
    shared_offsets: Vec<u32>,
    shared_indices: Vec<u32>,
}

/// A raw pointer that may be shared between the tasks of a parallel loop
/// whose writes are disjoint (see the SAFETY comments).
#[derive(Clone, Copy)]
struct SharedPtr(*mut f64);
unsafe impl Send for SharedPtr {}
unsafe impl Sync for SharedPtr {}

impl SharedPtr {
    /// The pointer (a method, so that closures capture the whole wrapper).
    fn get(self) -> *mut f64 {
        self.0
    }
}

impl DgMap {
    /// From a flat table in storage order: `loc2glob[e·nloc + i]` is the
    /// global DoF of local node `i` of element `e`, in `0..n_dof`. Panics if
    /// a DoF has no copy or if there are more than `u32::MAX` copies.
    pub fn new(loc2glob: &[usize], nloc: usize, n_dof: usize) -> Self {
        assert!(nloc > 0, "nloc must be positive");
        assert!(
            loc2glob.len().is_multiple_of(nloc),
            "table length {} is not a multiple of nloc = {nloc}",
            loc2glob.len()
        );
        let n_dg = loc2glob.len();
        assert!(n_dg <= u32::MAX as usize, "{n_dg} copies exceed u32 indices");
        let mut count = vec![0u32; n_dof];
        for &g in loc2glob {
            assert!(g < n_dof, "DoF {g} out of range 0..{n_dof}");
            count[g] += 1;
        }
        if let Some(g) = count.iter().position(|&c| c == 0) {
            panic!("DoF {g} has no copy");
        }
        let mut glob_offsets = Vec::with_capacity(n_dof + 1);
        glob_offsets.push(0u32);
        for &c in &count {
            glob_offsets.push(glob_offsets.last().unwrap() + c);
        }
        let mut next: Vec<u32> = glob_offsets[..n_dof].to_vec();
        let mut glob_indices = vec![0u32; n_dg];
        for (k, &g) in loc2glob.iter().enumerate() {
            glob_indices[next[g] as usize] = k as u32;
            next[g] += 1;
        }
        let mut shared_offsets = vec![0u32];
        let mut shared_indices = Vec::new();
        for g in 0..n_dof {
            let copies = &glob_indices[glob_offsets[g] as usize..glob_offsets[g + 1] as usize];
            if copies.len() >= 2 {
                shared_indices.extend_from_slice(copies);
                shared_offsets.push(shared_indices.len() as u32);
            }
        }
        DgMap {
            nloc,
            n_dof,
            dg2glob: loc2glob.iter().map(|&g| g as u32).collect(),
            glob_offsets,
            glob_indices,
            shared_offsets,
            shared_indices,
        }
    }

    /// From the result of a mesh numbering (`UnstructuredMesh::get_numbering`
    /// or `Mesh::get_tensor_numbering`: one row of global DoFs per mesh
    /// element, and their count), keeping the mesh elements `elements` in
    /// this order (the space's storage order). Every kept row has the same
    /// length, `nloc`.
    pub fn from_numbering(numbering: &[Vec<usize>], n_dof: usize, elements: &[usize]) -> Self {
        assert!(!elements.is_empty(), "no element");
        let nloc = numbering[elements[0]].len();
        let mut flat = Vec::with_capacity(elements.len() * nloc);
        for &m in elements {
            assert_eq!(numbering[m].len(), nloc, "element {m}: not {nloc} DoFs");
            flat.extend_from_slice(&numbering[m]);
        }
        // DoFs numbered only on elements that are not kept get no copy:
        // renumber the kept ones contiguously.
        let mut new_id = vec![usize::MAX; n_dof];
        let mut n = 0;
        for g in flat.iter_mut() {
            if new_id[*g] == usize::MAX {
                new_id[*g] = n;
                n += 1;
            }
            *g = new_id[*g];
        }
        if n == n_dof {
            // Already contiguous: keep the mesh's numbering.
            let mut flat = Vec::with_capacity(elements.len() * nloc);
            for &m in elements {
                flat.extend_from_slice(&numbering[m]);
            }
            return Self::new(&flat, nloc, n_dof);
        }
        Self::new(&flat, nloc, n)
    }

    /// Nodes per element.
    pub fn num_loc_dofs(&self) -> usize {
        self.nloc
    }

    /// Number of global DoFs.
    pub fn num_dofs(&self) -> usize {
        self.n_dof
    }

    /// Number of copies (DG size of one component), `num_elements · nloc`.
    pub fn num_dg_dofs(&self) -> usize {
        self.dg2glob.len()
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.dg2glob.len() / self.nloc
    }

    /// Global DoF of copy `k` (`k = e·nloc + i`).
    pub fn dg2glob(&self, k: usize) -> usize {
        self.dg2glob[k] as usize
    }

    /// The copies of global DoF `g`, in increasing order.
    pub fn glob2dg(&self, g: usize) -> &[u32] {
        &self.glob_indices[self.glob_offsets[g] as usize..self.glob_offsets[g + 1] as usize]
    }

    /// Number of copies of global DoF `g`.
    pub fn multiplicity(&self, g: usize) -> usize {
        (self.glob_offsets[g + 1] - self.glob_offsets[g]) as usize
    }

    /// Bytes held by the map.
    pub fn memory_bytes(&self) -> usize {
        4 * (self.dg2glob.len()
            + self.glob_offsets.len()
            + self.glob_indices.len()
            + self.shared_offsets.len()
            + self.shared_indices.len())
    }

    /// Position of component `c` of copy `k` in a `C`-component DG vector.
    #[inline]
    fn at<const C: usize>(&self, k: usize, c: usize) -> usize {
        if C == 1 {
            k
        } else {
            (k / self.nloc) * C * self.nloc + c * self.nloc + k % self.nloc
        }
    }

    fn check<const C: usize>(&self, dg: usize, cg: Option<usize>) {
        assert_eq!(dg, self.num_dg_dofs() * C, "DG vector length");
        if let Some(cg) = cg {
            assert_eq!(cg, self.n_dof * C, "global vector length");
        }
    }

    /// `cg[g] = Σ` of the copies of `g` (DG → global, the assembly).
    pub fn collect_dg<const C: usize>(&self, dg: &[f64], cg: &mut [f64]) {
        self.check::<C>(dg.len(), Some(cg.len()));
        cg.par_chunks_mut(C).enumerate().for_each(|(g, out)| {
            for (c, o) in out.iter_mut().enumerate() {
                *o = self.glob2dg(g).iter().map(|&k| dg[self.at::<C>(k as usize, c)]).sum();
            }
        });
    }

    /// `cg[g] =` the average of the copies of `g`.
    pub fn collect_avg_dg<const C: usize>(&self, dg: &[f64], cg: &mut [f64]) {
        self.check::<C>(dg.len(), Some(cg.len()));
        cg.par_chunks_mut(C).enumerate().for_each(|(g, out)| {
            let copies = self.glob2dg(g);
            for (c, o) in out.iter_mut().enumerate() {
                *o = copies.iter().map(|&k| dg[self.at::<C>(k as usize, c)]).sum::<f64>()
                    / copies.len() as f64;
            }
        });
    }

    /// Every copy of `g` receives `cg[g]` (global → DG).
    pub fn scatter_cg<const C: usize>(&self, cg: &[f64], dg: &mut [f64]) {
        self.check::<C>(dg.len(), Some(cg.len()));
        let nloc = self.nloc;
        dg.par_chunks_mut(C * nloc).enumerate().for_each(|(e, block)| {
            for (c, comp) in block.chunks_exact_mut(nloc).enumerate() {
                for (i, v) in comp.iter_mut().enumerate() {
                    *v = cg[self.dg2glob[e * nloc + i] as usize * C + c];
                }
            }
        });
    }

    /// Shared-DoF sum in place (DG → DG): every copy of a shared DoF
    /// receives the sum of its copies; DoFs with one copy are not touched.
    pub fn sum_shared<const C: usize>(&self, dg: &mut [f64]) {
        self.check::<C>(dg.len(), None);
        let ptr = SharedPtr(dg.as_mut_ptr());
        (0..self.shared_offsets.len() - 1).into_par_iter().for_each(|s| {
            let ptr = ptr.get();
            let copies = &self.shared_indices
                [self.shared_offsets[s] as usize..self.shared_offsets[s + 1] as usize];
            for c in 0..C {
                // SAFETY: the shared groups are disjoint sets of copies, so
                // the tasks write disjoint entries; every index is below
                // dg.len() (checked above) and `dg` is exclusively borrowed.
                unsafe {
                    let sum: f64 = copies.iter().map(|&k| *ptr.add(self.at::<C>(k as usize, c))).sum();
                    for &k in copies {
                        *ptr.add(self.at::<C>(k as usize, c)) = sum;
                    }
                }
            }
        });
    }

    /// Assembly and inverse lumped mass in one pass (TetriX's
    /// `mlt_inverse_mass`): every copy of `g` receives
    /// `(Σ copies of g) / mass[g]`, `mass` one value per global DoF shared by
    /// the components.
    pub fn mlt_inverse_mass<const C: usize>(&self, dg: &mut [f64], mass: &[f64]) {
        self.check::<C>(dg.len(), None);
        assert_eq!(mass.len(), self.n_dof, "mass length");
        let ptr = SharedPtr(dg.as_mut_ptr());
        (0..self.n_dof).into_par_iter().for_each(|g| {
            let ptr = ptr.get();
            let copies = self.glob2dg(g);
            for c in 0..C {
                // SAFETY: the copies of different DoFs are disjoint, so the
                // tasks write disjoint entries; every index is below
                // dg.len() (checked above) and `dg` is exclusively borrowed.
                unsafe {
                    let sum: f64 = copies.iter().map(|&k| *ptr.add(self.at::<C>(k as usize, c))).sum();
                    let v = sum / mass[g];
                    for &k in copies {
                        *ptr.add(self.at::<C>(k as usize, c)) = v;
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three 1D P2 elements in a chain, `0-1-2`, `2-3-4`, `4-5-6`: DoFs 2 and
    /// 4 have two copies, the others one.
    fn chain() -> DgMap {
        DgMap::new(&[0, 1, 2, 2, 3, 4, 4, 5, 6], 3, 7)
    }

    #[test]
    fn the_map_of_a_chain() {
        let m = chain();
        assert_eq!((m.num_dofs(), m.num_dg_dofs(), m.num_elements(), m.num_loc_dofs()), (7, 9, 3, 3));
        assert_eq!(m.glob2dg(2), &[2, 3]);
        assert_eq!(m.glob2dg(3), &[4]);
        assert_eq!((m.multiplicity(4), m.multiplicity(5)), (2, 1));
        assert_eq!(m.dg2glob(6), 4);
    }

    #[test]
    fn transfers_on_one_component() {
        let m = chain();
        let dg: Vec<f64> = (0..9).map(|k| k as f64).collect();
        let mut cg = vec![0.0; 7];
        m.collect_dg::<1>(&dg, &mut cg);
        assert_eq!(cg, [0.0, 1.0, 5.0, 4.0, 11.0, 7.0, 8.0]);
        m.collect_avg_dg::<1>(&dg, &mut cg);
        assert_eq!(cg, [0.0, 1.0, 2.5, 4.0, 5.5, 7.0, 8.0]);
        let mut back = vec![f64::NAN; 9];
        m.scatter_cg::<1>(&cg, &mut back);
        assert_eq!(back, [0.0, 1.0, 2.5, 2.5, 4.0, 5.5, 5.5, 7.0, 8.0]);
        let mut s = dg.clone();
        m.sum_shared::<1>(&mut s);
        assert_eq!(s, [0.0, 1.0, 5.0, 5.0, 4.0, 11.0, 11.0, 7.0, 8.0]);
        let mass = [1.0, 2.0, 4.0, 1.0, 2.0, 1.0, 4.0];
        let mut v = dg.clone();
        m.mlt_inverse_mass::<1>(&mut v, &mass);
        assert_eq!(v, [0.0, 0.5, 1.25, 1.25, 4.0, 5.5, 5.5, 7.0, 2.0]);
    }

    /// Two components: each is transferred on its own, with the DG layout
    /// `[elem][comp][node]` and the global layout `[dof][comp]`.
    #[test]
    fn transfers_on_two_components() {
        let m = chain();
        let one: Vec<f64> = (0..9).map(|k| k as f64).collect();
        let two: Vec<f64> = (0..9).map(|k| 100.0 + k as f64).collect();
        let mut dg = vec![0.0; 18];
        for e in 0..3 {
            for i in 0..3 {
                dg[e * 6 + i] = one[e * 3 + i];
                dg[e * 6 + 3 + i] = two[e * 3 + i];
            }
        }
        let (mut c1, mut c2, mut cg) = (vec![0.0; 7], vec![0.0; 7], vec![0.0; 14]);
        m.collect_dg::<1>(&one, &mut c1);
        m.collect_dg::<1>(&two, &mut c2);
        m.collect_dg::<2>(&dg, &mut cg);
        for g in 0..7 {
            assert_eq!((cg[2 * g], cg[2 * g + 1]), (c1[g], c2[g]));
        }
        let mut s = dg.clone();
        m.sum_shared::<2>(&mut s);
        let mut back = vec![0.0; 18];
        m.scatter_cg::<2>(&cg, &mut back);
        assert_eq!(s, back, "a shared sum is collect then scatter");
        let mass = [1.0, 2.0, 4.0, 1.0, 2.0, 1.0, 4.0];
        let mut v = dg.clone();
        m.mlt_inverse_mass::<2>(&mut v, &mass);
        let scaled: Vec<f64> = (0..14).map(|i| cg[i] / mass[i / 2]).collect();
        m.scatter_cg::<2>(&scaled, &mut back);
        assert_eq!(v, back);
    }

    /// The numbering of a mesh with an element that is not kept: its own DoF
    /// is dropped and the kept ones renumbered contiguously.
    #[test]
    fn from_numbering_keeps_the_given_elements() {
        let numbering = vec![vec![0, 1, 2], vec![7], vec![2, 3, 4]];
        let m = DgMap::from_numbering(&numbering, 8, &[2, 0]);
        assert_eq!((m.num_dofs(), m.num_elements()), (5, 2));
        let mut cg = vec![0.0; 5];
        m.collect_dg::<1>(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &mut cg);
        assert_eq!(cg.iter().filter(|&&v| v == 2.0).count(), 1, "one shared DoF");
        let full = DgMap::from_numbering(&[vec![0, 1, 2], vec![2, 3, 4]], 5, &[0, 1]);
        assert_eq!((0..6).map(|k| full.dg2glob(k)).collect::<Vec<_>>(), [0, 1, 2, 2, 3, 4]);
    }
}
