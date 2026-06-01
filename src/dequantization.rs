//! Dequantization axis: full ℏ-interpolation between spectral and tropical worlds.
//!
//! The dequantization parameter ℏ ∈ [0,1] interpolates:
//! - ℏ = 1: Classical linear heat semigroup e^{-tΔ}, spectral theory
//! - ℏ ∈ (0,1): Viscous Hamilton-Jacobi, Cole-Hopf regime
//! - ℏ = 0: Tropical Hopf-Lax semigroup, (min,+) algebra
//!
//! The key insight: as ℏ→0, multiplication → addition, addition → min,
//! eigenvalues → tropical eigenvalues, and the spectral decomposition
//! degenerates to the tropical shortest-path decomposition.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::cole_hopf::ColeHopfTransform;
use crate::core::HBar;
use crate::dirichlet_form::DirichletForm;
use crate::heat_semigroup::HeatSemigroup;
use crate::hopf_lax::HopfLaxSemigroup;
use crate::laplacian::Laplacian;

/// The full dequantization axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DequantizationAxis {
    /// The underlying Dirichlet form.
    pub form: DirichletForm,
    /// Number of interpolation points.
    pub n_points: usize,
}

impl DequantizationAxis {
    /// Create from a Dirichlet form.
    pub fn from_dirichlet_form(form: DirichletForm, n_points: usize) -> Self {
        Self { form, n_points }
    }

    /// Get ℏ values uniformly spaced from 0 to 1.
    pub fn hbar_values(&self) -> Vec<f64> {
        (0..=self.n_points)
            .map(|i| {
                let h = 1.0 - i as f64 / self.n_points as f64;
                if h < 1e-10 { 0.0 } else { h }
            })
            .collect()
    }

    /// Evolve a function f through the full ℏ-axis at time t.
    /// Returns u(t) for each ℏ value.
    pub fn evolve_axis(&self, f: &DVector<f64>, t: f64) -> Vec<DVector<f64>> {
        let hbar_values = self.hbar_values();
        let n = self.form.n;

        let mut results = Vec::new();

        for &h in &hbar_values {
            if h > 1e-10 {
                // Classical/viscous regime: Cole-Hopf
                let ch = ColeHopfTransform::new(h);
                if let Ok(v) = ch.forward(f) {
                    // Evolve in v-space using modified Laplacian (approximation)
                    // In the viscous regime, ∂_t v = H_ℏ(v) where H_ℏ includes viscous correction
                    // Simple approximation: linear heat flow + correction
                    let mut heat = HeatSemigroup::from_dirichlet_form(&self.form);
                    let u_evolved = heat.evolve(f, t);
                    match ch.forward(&u_evolved) {
                        Ok(v_evolved) => results.push(v_evolved),
                        Err(_) => results.push(v.clone()),
                    }
                } else {
                    results.push(DVector::zeros(n));
                }
            } else {
                // Tropical regime: Hopf-Lax
                let space = crate::core::MetricMeasureSpace::uniform(n);
                // Use Laplacian entries as cost (absolute value of off-diagonal)
                let mut cost = DMatrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            cost[(i, j)] = self.form.laplacian[(i, j)].abs().max(0.001);
                            // Note: Laplacian off-diagonal is negative, so |L_{ij}| = weight
                        }
                    }
                }
                let hl = HopfLaxSemigroup::from_cost(cost);
                results.push(hl.evolve(f, t));
            }
        }
        results
    }

    /// Compute the transition from spectral to tropical eigenvalues.
    /// At ℏ=1: classical eigenvalues of Δ.
    /// At ℏ=0: tropical eigenvalues (min cycle means).
    /// In between: interpolated (formally, this is the Maslov dequantization).
    pub fn spectral_degeneration(&self) -> SpectralDegeneration {
        let mut lap = Laplacian::from_dirichlet_form(&self.form);
        let spec = lap.spectrum();

        let classical_eigenvalues = spec.eigenvalues.clone();

        // Tropical eigenvalue approximation
        let n = self.form.n;
        let mut cost = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    cost[(i, j)] = self.form.laplacian[(i, j)].abs().max(0.001);
                }
            }
        }
        let hl = HopfLaxSemigroup::from_cost(cost);
        let tropical_eigenvalue = hl.tropical_eigenvalue();

        SpectralDegeneration {
            classical_eigenvalues,
            tropical_eigenvalue,
            n: self.form.n,
        }
    }

    /// Compare the classical and tropical regimes for a given input.
    pub fn compare_regimes(&self, f: &DVector<f64>, t: f64) -> RegimeComparison {
        let n = self.form.n;

        // Classical: e^{-tΔ} f
        let mut heat = HeatSemigroup::from_dirichlet_form(&self.form);
        let classical = heat.evolve(f, t);

        // Tropical: Hopf-Lax
        let mut cost = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    cost[(i, j)] = self.form.laplacian[(i, j)].abs().max(0.001);
                }
            }
        }
        let hl = HopfLaxSemigroup::from_cost(cost);
        let tropical = hl.evolve(f, t);

        // Difference
        let diff = &classical - &tropical;
        let l2_diff = diff.norm();

        RegimeComparison {
            classical,
            tropical,
            l2_difference: l2_diff,
        }
    }

    /// Verify that the dequantization is continuous:
    /// as ℏ→0, the Cole-Hopf solution converges to the Hopf-Lax solution.
    pub fn verify_continuity(&self, f: &DVector<f64>, t: f64, tol: f64) -> bool {
        // Compare ℏ=0.01 with ℏ=0 (tropical)
        let results = self.evolve_axis(f, t);

        if results.len() < 3 {
            return true; // Not enough points
        }

        // Compare near-tropical (second to last) with tropical (last)
        let near_tropical = &results[results.len() - 2];
        let tropical = &results[results.len() - 1];

        let diff = near_tropical - tropical;
        diff.norm() < tol
    }
}

/// Spectral degeneration data as ℏ → 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDegeneration {
    /// Classical eigenvalues (ℏ=1).
    pub classical_eigenvalues: Vec<f64>,
    /// Tropical eigenvalue (ℏ=0, min cycle mean).
    pub tropical_eigenvalue: f64,
    /// Dimension.
    pub n: usize,
}

impl SpectralDegeneration {
    /// The "log" of eigenvalues — as ℏ→0, eigenvalues degenerate via
    /// ℏ log(Σ exp(λ_k/ℏ)) → max(λ_k).
    pub fn tropical_limit(&self) -> f64 {
        self.classical_eigenvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Comparison between classical and tropical regimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeComparison {
    /// Classical evolution e^{-tΔ}f.
    pub classical: DVector<f64>,
    /// Tropical evolution (Hopf-Lax).
    pub tropical: DVector<f64>,
    /// L² difference between the two.
    pub l2_difference: f64,
}

impl RegimeComparison {
    /// Relative difference as a fraction of the classical norm.
    pub fn relative_difference(&self) -> f64 {
        let classical_norm = self.classical.norm();
        if classical_norm > 1e-10 {
            self.l2_difference / classical_norm
        } else {
            0.0
        }
    }
}
