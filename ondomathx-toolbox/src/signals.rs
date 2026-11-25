//! Source signals for the examples and drivers: the Ricker wavelet and the
//! Gaussian, and a C∞ bump with compact support.

use std::f64::consts::PI;

/// Ricker wavelet (normalised, peak `+1` at `t0`) of peak frequency `f0`:
/// `(1 − 2τ²) e^{−τ²}` with `τ = π f0 (t − t0)`. The usual delay is
/// [`ricker_delay`]`(f0) = 1/f0`.
pub fn ricker(t: f64, f0: f64, t0: f64) -> f64 {
    let tau = PI * f0 * (t - t0);
    (1.0 - 2.0 * tau * tau) * (-tau * tau).exp()
}

/// The usual delay of a Ricker wavelet of peak frequency `f0`: `1/f0`, so
/// that the signal starts close to zero.
pub fn ricker_delay(f0: f64) -> f64 {
    1.0 / f0
}

/// Gaussian `e^{−r²/σ²}` of a distance (or time offset) `r`, `sigma2 = σ²`.
pub fn gaussian(r: f64, sigma2: f64) -> f64 {
    (-r * r / sigma2).exp()
}

/// C∞ bump with compact support (C++ `CInfty0`): `1` at `center`, `0` from
/// the distance `r0` on, `exp(s0·r0² / (r² − r0²) + s0)` in between.
pub fn bump<const D: usize>(x: [f64; D], center: [f64; D], r0: f64, s0: f64) -> f64 {
    let r2: f64 = (0..D).map(|d| (x[d] - center[d]).powi(2)).sum();
    if r2 >= r0 * r0 {
        0.0
    } else {
        (s0 * r0 * r0 / (r2 - r0 * r0) + s0).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ricker_peaks_at_its_delay() {
        assert_eq!(ricker(0.2, 7.0, 0.2), 1.0);
        assert!(ricker(0.2 + 0.01, 7.0, 0.2) < 1.0);
        assert!(ricker(0.0, 7.0, ricker_delay(7.0)).abs() < 1e-3);
    }

    #[test]
    fn bump_is_one_at_the_centre_and_zero_outside() {
        assert_eq!(bump([0.5, 0.5], [0.5, 0.5], 0.1, 1.0), 1.0);
        assert_eq!(bump([0.5, 0.61], [0.5, 0.5], 0.1, 1.0), 0.0);
        let inside = bump([0.5, 0.55, 0.5], [0.5, 0.5, 0.5], 0.1, 1.0);
        assert!(inside > 0.0 && inside < 1.0);
    }

    #[test]
    fn gaussian_is_one_at_zero() {
        assert_eq!(gaussian(0.0, 100.0), 1.0);
        assert!((gaussian(10.0, 100.0) - (-1.0f64).exp()).abs() < 1e-15);
    }
}
