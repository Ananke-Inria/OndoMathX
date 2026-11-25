pub const C1: f64 = 0.09273525031089123;
pub const C2: f64 = 0.3108859192633006;
pub const D: f64 = 0.04550370412564965;

pub const WC1: f64 = 0.01224884051939366;
pub const WC2: f64 = 0.01878132095300264;
pub const WD: f64 = 0.007091003462846911;

pub const POINTS: [[f64; 3]; 14] = [
    [C1, C1, C1],
    [1.0 - 3.0 * C1, C1, C1],
    [C1, 1.0 - 3.0 * C1, C1],
    [C1, C1, 1.0 - 3.0 * C1],
    [C2, C2, C2],
    [1.0 - 3.0 * C2, C2, C2],
    [C2, 1.0 - 3.0 * C2, C2],
    [C2, C2, 1.0 - 3.0 * C2],
    [0.5 - D, D, D],
    [D, 0.5 - D, D],
    [D, D, 0.5 - D],
    [0.5 - D, 0.5 - D, D],
    [D, 0.5 - D, 0.5 - D],
    [0.5 - D, D, 0.5 - D],
];

pub const WEIGHTS: [f64; 14] = [
    WC1, WC1, WC1, WC1, WC2, WC2, WC2, WC2, WD, WD, WD, WD, WD, WD,
];
