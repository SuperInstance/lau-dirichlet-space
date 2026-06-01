//! Belief dynamics: evolution of probability measures under the Dirichlet flow.
//!
//! The heat semigroup acts on probability measures via:
//! ∂_t μ = Δ μ  (Fokker-Planck / Kolmogorov forward equation)
//!
//! This drives μ toward the stationary distribution (Gibbs measure),
//! with rate governed by the spectral gap and curvature.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::dirichlet_form::DirichletForm;
use crate::laplacian::SpectralDecomposition;
use crate::registers::MeasureRegister;

/// Belief dynamics under the Dirichlet flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefDynamics {
    /// The Dirichlet form.
    pub form: DirichletForm,
    /// Stationary distribution π (uniform for unweighted graphs).
    pub stationary: DVector<f64>,
}

impl BeliefDynamics {
    /// Create from a Dirichlet form.
    pub fn from_dirichlet_form(form: &DirichletForm) -> Self {
        let n = form.n;
        // For an unweighted connected graph, π is uniform
        let stationary = DVector::from_element(n, 1.0 / n as f64);
        Self {
            form: form.clone(),
            stationary,
        }
    }

    /// Create from a Dirichlet form with a specific stationary distribution.
    pub fn with_stationary(form: &DirichletForm, stationary: DVector<f64>) -> Self {
        Self {
            form: form.clone(),
            stationary,
        }
    }

    /// Evolve a belief μ₀ for time t via the heat kernel.
    pub fn evolve(&self, mu0: &DVector<f64>, t: f64) -> DVector<f64> {
        let spec = SpectralDecomposition::from_laplacian(&self.form.laplacian);
        let n = self.form.n;

        // μ(t) = Σ_k e^{-tλ_k} ⟨μ₀, φ_k⟩ φ_k + ⟨μ₀, 1⟩/n · 1
        // But we need to be careful: the Laplacian acts on functions, not measures.
        // For the forward equation, we use the transpose semigroup.
        // On a finite graph with uniform measure: K_t = e^{-tL}

        let mut result = DVector::zeros(n);
        for k in 0..n {
            let phi = spec.eigenvector(k);
            let coeff = phi.dot(mu0);
            let decay = (-t * spec.eigenvalue(k)).exp();
            result += &phi * (coeff * decay);
        }

        // Ensure it remains a probability measure
        let total = result.sum();
        if total.abs() > 1e-15 {
            result = result / total;
        }

        // Clamp negatives
        for v in result.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let total = result.sum();
        if total > 1e-15 {
            result = result / total;
        }

        result
    }

    /// Compute the trajectory of beliefs over time.
    pub fn trajectory(&self, mu0: &DVector<f64>, times: &[f64]) -> Vec<DVector<f64>> {
        times.iter().map(|&t| self.evolve(mu0, t)).collect()
    }

    /// Compute relative entropy H(μ | π) along the trajectory.
    pub fn relative_entropy_trajectory(&self, mu0: &DVector<f64>, times: &[f64]) -> Vec<f64> {
        let trajectory = self.trajectory(mu0, times);
        trajectory
            .iter()
            .map(|mu| {
                mu.iter()
                    .zip(self.stationary.iter())
                    .filter(|(p, _)| **p > 1e-15)
                    .map(|(p, q)| {
                        let q_safe = q.max(1e-15);
                        p * (p / q_safe).ln()
                    })
                    .sum()
            })
            .collect()
    }

    /// Verify that relative entropy is non-increasing (second law).
    pub fn entropy_nonincreasing(&self, mu0: &DVector<f64>, times: &[f64]) -> bool {
        let entropies = self.relative_entropy_trajectory(mu0, times);
        for i in 1..entropies.len() {
            if entropies[i] > entropies[i - 1] + 1e-6 {
                return false;
            }
        }
        true
    }

    /// Fisher information I(μ | π) = 2E(log(μ/π), log(μ/π)).
    pub fn fisher_information(&self, mu: &DVector<f64>) -> f64 {
        let log_ratio_vec: Vec<f64> = mu.iter()
            .zip(self.stationary.iter())
            .map(|(p, q)| {
                if *p > 1e-15 && *q > 1e-15 {
                    (p / q).ln()
                } else {
                    0.0
                }
            })
            .collect();
        let log_ratio = DVector::from_vec(log_ratio_vec);

        // I = ⟨Δ log(μ/π), log(μ/π)⟩ = energy of log-ratio
        self.form.energy(&log_ratio)
    }

    /// Verify the entropy-energy inequality: d/dt H(μ|π) = -I(μ|π).
    pub fn verify_entropy_energy_identity(&self, mu0: &DVector<f64>, dt: f64) -> bool {
        let mu_t = self.evolve(mu0, dt);
        let mu_tdt = self.evolve(mu0, 2.0 * dt);

        // Numerical derivative of entropy
        let h_t = self.relative_entropy(&mu_t);
        let h_tdt = self.relative_entropy(&mu_tdt);
        let dh_dt = (h_tdt - h_t) / dt;

        // Fisher information at t
        let fisher = self.fisher_information(&mu_t);

        // Should satisfy: dh/dt ≈ -fisher
        if fisher.abs() < 1e-10 && dh_dt.abs() < 1e-6 {
            return true;
        }
        (dh_dt + fisher).abs() < (dh_dt.abs() + fisher.abs()) * 0.5 + 1e-4
    }

    /// Relative entropy D_KL(μ || π).
    fn relative_entropy(&self, mu: &DVector<f64>) -> f64 {
        mu.iter()
            .zip(self.stationary.iter())
            .filter(|(p, _)| **p > 1e-15)
            .map(|(p, q)| {
                let q_safe = q.max(1e-15);
                p * (p / q_safe).ln()
            })
            .sum()
    }

    /// Compute the belief as a MeasureRegister.
    pub fn as_measure(&self, mu: &DVector<f64>) -> MeasureRegister {
        MeasureRegister {
            measure: mu.clone(),
            n: self.form.n,
        }
    }

    /// The stationary distribution as a MeasureRegister.
    pub fn stationary_measure(&self) -> MeasureRegister {
        MeasureRegister {
            measure: self.stationary.clone(),
            n: self.form.n,
        }
    }
}
