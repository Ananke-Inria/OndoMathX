/// 4-point quadrature rule for the reference tetrahedron, exact for P2.
///
/// Uses the orbit (a, a, a, b) in barycentric coordinates and its 3 permutations,
/// where a = (5 − √5) / 20 and b = 1 − 3a.  All four weights are equal: w = 1/24.
///
/// Degree of precision: 2.
/// Weights sum to 1/6 (volume of the reference tetrahedron).

// a = (5 − √5) / 20
pub const A: f64 = 0.1381966011250105;
// b = (5 + 3√5) / 20 = 1 − 3a
pub const B: f64 = 0.5854101966249685;

pub const W: f64 = 1.0 / 24.0;

pub const POINTS: [[f64; 3]; 4] = [
    [A, A, B],
    [A, B, A],
    [B, A, A],
    [A, A, A],
];

pub const WEIGHTS: [f64; 4] = [W, W, W, W];
