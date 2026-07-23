use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::reference_elements::*;

use crate::vec_eq;
use crate::vec_lt;

#[derive(Default, Debug)]
pub struct DoF {
    pub uvw: [f64; 3],
    pub index: usize,
}

impl Ord for DoF {
    fn cmp(&self, other: &Self) -> Ordering {
        let o: Ordering;

        if vec_eq!(self.uvw, other.uvw) {
            o = Ordering::Equal;
        } else if vec_lt!(self.uvw, other.uvw) {
            o = Ordering::Less;
        } else {
            o = Ordering::Greater;
        }

        o
    }
}

impl PartialOrd for DoF {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DoF {
    fn eq(&self, other: &Self) -> bool {
        vec_eq!(self.uvw, other.uvw)
    }
}

impl Eq for DoF {}

#[derive(Debug)]
pub struct DoFSet {
    pub dof_tree_set: BTreeSet<DoF>,
    pub dof_list: Vec<DoF>,

    pub dof_on_vertexes: Vec<Vec<usize>>,

    pub dof_strictly_in_edges: Vec<Vec<Vec<usize>>>,
    pub dof_strictly_in_faces: Vec<Vec<Vec<usize>>>,
    pub dof_strictly_in_volume: Vec<usize>,

    pub dof_on_edges: Vec<Vec<Vec<usize>>>,
    pub dof_on_faces: Vec<Vec<Vec<usize>>>,

    pub reference_element: &'static ReferenceElement,
}

impl DoFSet {
    pub fn get_ndof(&self) -> usize {
        self.dof_list.len()
    }

    pub fn new(pos_list: Vec<[f64; 3]>, reference_element: &'static ReferenceElement) -> DoFSet {
        let mut dof_tree_set = BTreeSet::<DoF>::new();
        let mut dof_list = Vec::<DoF>::new();

        let mut dof_on_vertexes = vec![vec![]; reference_element.get_nvertexes()];

        let mut dof_on_edges = vec![vec![vec![]; 2]; reference_element.get_nedges()];
        let mut dof_strictly_in_edges = vec![vec![vec![]; 2]; reference_element.get_nedges()];

        let mut dof_on_faces = vec![vec![vec![]]; reference_element.get_nfaces()];
        let mut dof_strictly_in_faces = vec![vec![vec![]]; reference_element.get_nfaces()];

        let mut dof_strictly_in_volume = Vec::new();

        for f in 0..reference_element.get_nfaces() {
            let no = if reference_element.is_face_triangular(f) {
                6
            } else {
                8
            };

            dof_on_faces[f].resize(no, vec![]);
            dof_strictly_in_faces[f].resize(no, vec![]);
        }

        for (i, pos) in pos_list.iter().enumerate() {
            assert!(
                reference_element.is_inside(pos),
                "The DoF is not inside the element"
            );

            dof_list.push(DoF {
                uvw: *pos,
                index: i,
            });

            dof_tree_set.insert(DoF {
                uvw: *pos,
                index: i,
            });
        }

        for dof in dof_list.iter() {
            // If the on a vertex
            for (v, dof_on_vertex) in dof_on_vertexes.iter_mut().enumerate() {
                if reference_element.is_on_vertex(&dof.uvw, v) {
                    dof_on_vertex.push(dof.index);
                }
            }

            for e in 0..reference_element.get_nedges() {
                if reference_element.is_on_edge(&dof.uvw, e) {
                    dof_on_edges[e][0].push(dof.index);

                    if !reference_element.is_on_any_vertex(&dof.uvw) {
                        dof_strictly_in_edges[e][0].push(dof.index);
                    }

                    let t_uvw = reference_element.uvw_edge_transform(&dof.uvw, e, 1);

                    let t_dof = dof_tree_set.get(&DoF {
                        uvw: t_uvw,
                        index: 0,
                    });

                    if let Some(t_dof) = t_dof {
                        dof_on_edges[e][1].push(t_dof.index);

                        if !reference_element.is_on_any_vertex(&t_uvw) {
                            dof_strictly_in_edges[e][1].push(t_dof.index);
                        }
                    } else {
                        panic!("Symmetry problem in the d.o.f list");
                    }
                }
            }

            for f in 0..reference_element.get_nfaces() {
                if reference_element.is_on_face(&dof.uvw, f) {
                    let no = if reference_element.is_face_triangular(f) {
                        6
                    } else {
                        8
                    };

                    for o in 0..no {
                        let t_uvw = reference_element.uvw_face_transform(&dof.uvw, f, o);

                        let t_dof = dof_tree_set.get(&DoF {
                            uvw: t_uvw,
                            index: 0,
                        });

                        if let Some(t_dof) = t_dof {
                            dof_on_faces[f][o].push(t_dof.index);

                            if !reference_element.is_on_any_edge(&t_uvw) {
                                dof_strictly_in_faces[f][o].push(t_dof.index);
                            }
                        } else {
                            panic!("Symmetry problem in the d.o.f list");
                        }
                    }
                }
            }

            if reference_element.get_dim() == 3 && !reference_element.is_on_any_face(&dof.uvw) {
                dof_strictly_in_volume.push(dof.index);
            }
        }

        //Various check on the d.o.f list
        for v in 0..reference_element.get_nvertexes() {
            assert!(
                dof_on_vertexes[v].len() == dof_on_vertexes[0].len(),
                "Symmetry problem in the d.o.f list"
            );
        }

        for e in 0..reference_element.get_nedges() {
            assert!(
                dof_on_edges[e].len() == dof_on_edges[0].len(),
                "Symmetry problem in the d.o.f list"
            );

            assert!(
                dof_strictly_in_edges[e].len() == dof_strictly_in_edges[0].len(),
                "Symmetry problem in the d.o.f list"
            );
        }

        for f in 0..reference_element.get_nfaces() {
            assert!(
                dof_strictly_in_faces[f].len() == dof_strictly_in_faces[0].len(),
                "Symmetry problem in the d.o.f list"
            );
        }

        DoFSet {
            dof_tree_set,
            dof_list,
            dof_on_vertexes,
            dof_strictly_in_edges,
            dof_strictly_in_faces,
            dof_strictly_in_volume,
            dof_on_edges,
            dof_on_faces,
            reference_element,
        }
    }
}
