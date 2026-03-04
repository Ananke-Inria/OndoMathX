/// 5-point quadrature rule for the reference tetrahedron, exact for P2.
///
/// Uses the 4 face barycenters and the tetrahedron barycenter as nodes.
/// The weights are uniquely determined by imposing exactness for polynomials
/// of degree ≤ 2 and the tetrahedral symmetry (all face nodes share one weight).
///
/// Points in (u, v, w) coordinates (λ0 = 1−u−v−w):
///
///   k=0: (1/3, 1/3,   0) — barycenter of face λ3=0  (V0 V1 V2)
///   k=1: (1/3,   0, 1/3) — barycenter of face λ2=0  (V0 V1 V3)
///   k=2: (  0, 1/3, 1/3) — barycenter of face λ1=0  (V0 V2 V3)
///   k=3: (1/3, 1/3, 1/3) — barycenter of face λ0=0  (V1 V2 V3)
///   k=4: (1/4, 1/4, 1/4) — barycenter of the tetrahedron
///
/// Derivation of weights (V = 1/6, symmetry: w0=w1=w2=w3 =: w_f, w4 =: w_c):
///
///   degree-0 constraint:  4 w_f + w_c = 1/6
///   degree-2 constraint for u² vs uv gives the linear system:
///     w_f/3  + w_c/16 = 1/60     (from ∫u²)
///     2w_f/9 + w_c/16 = 1/120    (from ∫uv)
///   Solution: w_f = 3/40,  w_c = −2/15.
///
/// Note: w_c is negative.  This is unavoidable for this node placement.
/// Degree of precision: 2.
/// Weights sum to 1/6 (volume of the reference tetrahedron).

pub const W_FACE: f64 = 3.0 / 40.0;
pub const W_INTERIOR: f64 = -2.0 / 15.0;

pub const POINTS: [[f64; 3]; 5] = [
    [1.0 / 3.0, 1.0 / 3.0, 0.0],
    [1.0 / 3.0, 0.0, 1.0 / 3.0],
    [0.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [0.25, 0.25, 0.25],
];

pub const WEIGHTS: [f64; 5] = [W_FACE, W_FACE, W_FACE, W_FACE, W_INTERIOR];
