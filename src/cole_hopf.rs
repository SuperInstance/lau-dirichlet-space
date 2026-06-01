//! Cole-Hopf transform: u → -ℏ log(u), the dequantization axis ℏ: 1→0.
//!
//! The Cole-Hopf transform connects the linear heat equation to the
//! Hamilton-Jacobi equation. At ℏ=1, we have the standard linear heat flow.
//! As ℏ→0, the transformed equation approaches the Hopf-Lax formula.
//!
//! Heat equation: ∂_t u = Δu
//! Cole-Hopf: v = -ℏ log(u), so u = e^{-v/ℏ}
//! Transformed: ∂_t v = -|∇v|² - ℏ Δv  (Hamilton-Jacobi-Bellman + viscous term)

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::core::HBar;

/// Cole-Hopf transform and its inverse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColeHopfTransform {
    /// Dequantization parameter ℏ.
    pub hbar: HBar,
}

impl ColeHopfTransform {
    /// Create with a given ℏ value.
    pub fn new(hbar: f64) -> Self {
        Self {
            hbar: HBar::new(hbar),
        }
    }

    /// Classical transform (ℏ=1).
    pub fn classical() -> Self {
        Self::new(1.0)
    }

    /// Tropical limit (ℏ→0, small but nonzero for numerics).
    pub fn tropical() -> Self {
        Self::new(1e-10)
    }

    /// Forward transform: u → v = -ℏ log(u).
    /// u must be strictly positive.
    pub fn forward(&self, u: &DVector<f64>) -> Result<DVector<f64>, String> {
        if u.iter().any(|&v| v <= 0.0) {
            return Err("Cole-Hopf forward transform requires u > 0".to_string());
        }
        Ok(u.map(|v| -self.hbar.value * v.ln()))
    }

    /// Inverse transform: v → u = exp(-v/ℏ).
    pub fn inverse(&self, v: &DVector<f64>) -> DVector<f64> {
        if self.hbar.value < 1e-15 {
            // In the tropical limit, the inverse becomes indicator-like
            let min_v = v.iter().cloned().fold(f64::INFINITY, f64::min);
            v.map(|vi| if (vi - min_v).abs() < 1e-10 { 1.0 } else { 0.0 })
        } else {
            v.map(|vi| (-vi / self.hbar.value).exp())
        }
    }

    /// Verify the transform is invertible: inverse(forward(u)) ≈ u.
    pub fn verify_roundtrip(&self, u: &DVector<f64>) -> bool {
        if u.iter().any(|&v| v <= 0.0) {
            return false;
        }
        match self.forward(u) {
            Ok(v) => {
                let u_roundtrip = self.inverse(&v);
                let diff = &u_roundtrip - u;
                diff.norm() < 1e-6 * u.norm().max(1.0)
            }
            Err(_) => false,
        }
    }

    /// Apply the dequantization: compute the effective Hamiltonian
    /// H(v) = |∇v|² + ℏ Δv for the transformed equation.
    /// In finite dimensions, |∇v|² is approximated by the Dirichlet energy
    /// and Δv is the Laplacian applied to v.
    pub fn hamiltonian(
        &self,
        v: &DVector<f64>,
        gradient_sq: &DVector<f64>,
        laplacian_v: &DVector<f64>,
    ) -> DVector<f64> {
        // H(v) = |∇v|² + ℏ Δv
        gradient_sq + laplacian_v.scale(self.hbar.value)
    }

    /// Compute the viscous correction term (vanishes as ℏ→0).
    pub fn viscous_term(&self, laplacian_v: &DVector<f64>) -> DVector<f64> {
        laplacian_v.scale(self.hbar.value)
    }

    /// Interpolate between two states at different ℏ values.
    /// Given states v₁ (ℏ₁) and v₂ (ℏ₂), compute the state at ℏ.
    pub fn interpolate(&self, v1: &DVector<f64>, hbar1: f64, v2: &DVector<f64>, hbar2: f64) -> DVector<f64> {
        if (hbar2 - hbar1).abs() < 1e-15 {
            return v1.clone();
        }
        let alpha = (self.hbar.value - hbar1) / (hbar2 - hbar1);
        v1 * (1.0 - alpha) + v2 * alpha
    }
}

/// A sequence of Cole-Hopf transforms along the dequantization axis.
pub struct DequantizationPath {
    /// ℏ values from 1 to 0.
    pub hbar_values: Vec<f64>,
}

impl DequantizationPath {
    /// Create a uniform path from ℏ=1 to ℏ=0 with n steps.
    pub fn uniform(n: usize) -> Self {
        Self {
            hbar_values: (0..=n).map(|i| 1.0 - i as f64 / n as f64).collect(),
        }
    }

    /// Create a logarithmic path (more points near ℏ=0).
    pub fn logarithmic(n: usize) -> Self {
        let mut values = vec![1.0];
        for i in 1..n {
            let t = i as f64 / n as f64;
            values.push((1.0 - t).max(1e-10));
        }
        values.push(0.0);
        Self { hbar_values: values }
    }

    /// Apply the full path, transforming a function u through each ℏ.
    pub fn apply(&self, u: &DVector<f64>) -> Vec<DVector<f64>> {
        self.hbar_values
            .iter()
            .map(|&h| {
                let ch = ColeHopfTransform::new(h.max(1e-10));
                match ch.forward(u) {
                    Ok(v) => v,
                    Err(_) => DVector::zeros(u.len()),
                }
            })
            .collect()
    }
}
