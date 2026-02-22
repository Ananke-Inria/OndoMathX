// Scalar coefficients for the change-of-basis (exact rational values).
//
//   N9  = −1/9  : P2 vertex  value at a face centroid  (e.g. λ0(2λ0−1) at (1/3,1/3,0))
//   P9  =  4/9  : P2 edge    value at a face centroid  (e.g. 4λ0λ1      at (1/3,1/3,0))
//   N8  = −1/8  : P2 vertex  value at interior centroid (λ0(2λ0−1) at (1/4,1/4,1/4))
//   P4  =  1/4  : P2 edge    value at interior centroid (4λ0λ1      at (1/4,1/4,1/4))
//   B64 = 27/64 : face bubble value at interior centroid (27λ0λ1λ2  at (1/4,1/4,1/4))
const N9:  f64 = -1.0 / 9.0;
const P9:  f64 =  4.0 / 9.0;
const N8:  f64 = -1.0 / 8.0;
const P4:  f64 =  1.0 / 4.0;
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
    a[14] = N8  * (b[0] + b[1] + b[2] + b[3])
          + P4  * (b[4] + b[5] + b[6] + b[7] + b[8] + b[9])
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
    b[14] = a[14] - N8  * (a[0] + a[1] + a[2] + a[3])
                  - P4  * (a[4] + a[5] + a[6] + a[7] + a[8] + a[9])
                  - B64 * (b[10] + b[11] + b[12] + b[13]);
}
