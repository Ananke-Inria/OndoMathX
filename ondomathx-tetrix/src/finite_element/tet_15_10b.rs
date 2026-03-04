// Scalar coefficients for the change-of-basis (exact rational values).
//
//   N9  = −1/9  : P2 vertex  value at a face centroid  (e.g. λ0(2λ0−1) at (1/3,1/3,0))
//   P9  =  4/9  : P2 edge    value at a face centroid  (e.g. 4λ0λ1      at (1/3,1/3,0))
//   N8  = −1/8  : P2 vertex  value at interior centroid (λ0(2λ0−1) at (1/4,1/4,1/4))
//   P4  =  1/4  : P2 edge    value at interior centroid (4λ0λ1      at (1/4,1/4,1/4))
//   B64 = 27/64 : face bubble value at interior centroid (27λ0λ1λ2  at (1/4,1/4,1/4))
const N9: f64 = -1.0 / 9.0;
const P9: f64 = 4.0 / 9.0;
const N8: f64 = -1.0 / 8.0;
const P4: f64 = 1.0 / 4.0;
const B64: f64 = 27.0 / 64.0;

/// Convert coefficients from the `tet_10b` basis to the `tet_15` basis.
///
/// Both bases span the same 15-dimensional function space. Since `POINTS_10b`
/// and `POINTS_15` are identical, the evaluation matrix
///
///   E[i, j] = φ_j^{10b}(POINTS_10b[i])
///
/// has an identity block for the first 10 rows (Lagrange property + bubbles
/// vanish at P2 nodes). The 5 bubble rows are sparse: each face centroid is
/// reached by only 3 vertex and 3 edge functions (7 non-zeros out of 15).
///
///   a[i] = b[i]  for i < 10
///
///   a[10] = N9·(b[0]+b[1]+b[2]) + P9·(b[4]+b[5]+b[6]) + b[10]  // face V0V1V2
///   a[11] = N9·(b[0]+b[1]+b[3]) + P9·(b[4]+b[7]+b[8]) + b[11]  // face V0V1V3
///   a[12] = N9·(b[0]+b[2]+b[3]) + P9·(b[6]+b[7]+b[9]) + b[12]  // face V0V2V3
///   a[13] = N9·(b[1]+b[2]+b[3]) + P9·(b[5]+b[8]+b[9]) + b[13]  // face V1V2V3
///   a[14] = N8·(b[0]+…+b[3]) + P4·(b[4]+…+b[9]) + B64·(b[10]+…+b[13]) + b[14]
pub fn tet_10b_to_tet_15(b: &[f64; 15], a: &mut [f64; 15]) {
    a[..10].copy_from_slice(&b[..10]);

    a[10] = N9 * (b[0] + b[1] + b[2]) + P9 * (b[4] + b[5] + b[6]) + b[10];
    a[11] = N9 * (b[0] + b[1] + b[3]) + P9 * (b[4] + b[7] + b[8]) + b[11];
    a[12] = N9 * (b[0] + b[2] + b[3]) + P9 * (b[6] + b[7] + b[9]) + b[12];
    a[13] = N9 * (b[1] + b[2] + b[3]) + P9 * (b[5] + b[8] + b[9]) + b[13];
    a[14] = N8 * (b[0] + b[1] + b[2] + b[3])
        + P4 * (b[4] + b[5] + b[6] + b[7] + b[8] + b[9])
        + B64 * (b[10] + b[11] + b[12] + b[13])
        + b[14];
}

/// Convert coefficients from the `tet_15` basis to the `tet_10b` basis.
///
/// Uses the fact that `POINTS_10b == POINTS_15` (so `val[j] = a[j]` for all j
/// by the Lagrange property of `tet_15`), and that the evaluation matrix is
/// **lower triangular with diagonal 1**, enabling forward substitution.
///
/// The first 10 rows are the identity. For the bubble rows the cross-bubble
/// terms in BUBBLE_MAT[j−10][k] vanish for k = 10..j−1 (rows 10–13), so those
/// rows depend only on a[0..9]. Row 14 additionally depends on b[10..13].
///
///   b[i] = a[i]  for i < 10
///
///   b[10] = a[10] − N9·(a[0]+a[1]+a[2]) − P9·(a[4]+a[5]+a[6])
///   b[11] = a[11] − N9·(a[0]+a[1]+a[3]) − P9·(a[4]+a[7]+a[8])
///   b[12] = a[12] − N9·(a[0]+a[2]+a[3]) − P9·(a[6]+a[7]+a[9])
///   b[13] = a[13] − N9·(a[1]+a[2]+a[3]) − P9·(a[5]+a[8]+a[9])
///   b[14] = a[14] − N8·(a[0]+…+a[3]) − P4·(a[4]+…+a[9]) − B64·(b[10]+…+b[13])
pub fn tet_15_to_tet_10b(a: &[f64; 15], b: &mut [f64; 15]) {
    b[..10].copy_from_slice(&a[..10]);

    b[10] = a[10] - N9 * (a[0] + a[1] + a[2]) - P9 * (a[4] + a[5] + a[6]);
    b[11] = a[11] - N9 * (a[0] + a[1] + a[3]) - P9 * (a[4] + a[7] + a[8]);
    b[12] = a[12] - N9 * (a[0] + a[2] + a[3]) - P9 * (a[6] + a[7] + a[9]);
    b[13] = a[13] - N9 * (a[1] + a[2] + a[3]) - P9 * (a[5] + a[8] + a[9]);
    b[14] = a[14]
        - N8 * (a[0] + a[1] + a[2] + a[3])
        - P4 * (a[4] + a[5] + a[6] + a[7] + a[8] + a[9])
        - B64 * (b[10] + b[11] + b[12] + b[13]);
}

/// Apply the transpose of `tet_15_to_tet_10b` (matrix P^T) to a vector.
///
/// If P denotes the matrix of `tet_15_to_tet_10b` (so that b = P·a), this
/// function computes a = P^T·c.
///
/// Derivation: P^T[j,i] = P[i,j].
///
/// - Vertices (j = 0..3): each vertex appears in 3 face bubbles (rows 10..13)
///   and in the interior bubble (row 14, after substituting rows 10..13).
/// - Edges   (j = 4..9): each edge appears in 2 face bubbles; row-14 coefficient
///   is -P4 + 2·B64·P9.
/// - Face bubbles (j = 10..13): only the interior-bubble row contributes: -B64.
/// - Interior bubble (j = 14): diagonal only.
pub fn tet_15_to_tet_10b_transpose(c: &[f64; 15], a: &mut [f64; 15]) {
    // Precomputed row-14 coefficients (after substituting b[10..13] into b[14]):
    //   P[14, vertex] = -N8 + 3·B64·N9
    //   P[14, edge]   = -P4 + 2·B64·P9
    let p14_v = -N8 + 3.0 * B64 * N9;
    let p14_e = -P4 + 2.0 * B64 * P9;

    // Vertices: each is in 3 face bubbles
    a[0] = c[0] - N9 * (c[10] + c[11] + c[12]) + p14_v * c[14]; // faces V0V1V2, V0V1V3, V0V2V3
    a[1] = c[1] - N9 * (c[10] + c[11] + c[13]) + p14_v * c[14]; // faces V0V1V2, V0V1V3, V1V2V3
    a[2] = c[2] - N9 * (c[10] + c[12] + c[13]) + p14_v * c[14]; // faces V0V1V2, V0V2V3, V1V2V3
    a[3] = c[3] - N9 * (c[11] + c[12] + c[13]) + p14_v * c[14]; // faces V0V1V3, V0V2V3, V1V2V3

    // Edges: each is in 2 face bubbles
    a[4] = c[4] - P9 * (c[10] + c[11]) + p14_e * c[14]; // E01 in V0V1V2, V0V1V3
    a[5] = c[5] - P9 * (c[10] + c[13]) + p14_e * c[14]; // E12 in V0V1V2, V1V2V3
    a[6] = c[6] - P9 * (c[10] + c[12]) + p14_e * c[14]; // E02 in V0V1V2, V0V2V3
    a[7] = c[7] - P9 * (c[11] + c[12]) + p14_e * c[14]; // E03 in V0V1V3, V0V2V3
    a[8] = c[8] - P9 * (c[11] + c[13]) + p14_e * c[14]; // E13 in V0V1V3, V1V2V3
    a[9] = c[9] - P9 * (c[12] + c[13]) + p14_e * c[14]; // E23 in V0V2V3, V1V2V3

    // Face bubbles: only the interior-bubble row contributes
    a[10] = c[10] - B64 * c[14];
    a[11] = c[11] - B64 * c[14];
    a[12] = c[12] - B64 * c[14];
    a[13] = c[13] - B64 * c[14];

    // Interior bubble: diagonal only
    a[14] = c[14];
}
