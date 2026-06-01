//! Spectral gap analysis: λ₁ > 0 ⟹ ergodic convergence, entropy decay.
//!
//! For a connected graph, λ₁ > 0 (spectral gap) implies:
//! 1. Exponential convergence: ‖S_t f - π‖ ≤ e^{-λ₁ t} ‖f - π‖
//! 2. Log-Sobolev inequality: controls entropy
//! 3. Poincaré inequality: Var_π(f) ≤ E(f,f) / λ₁

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::dirichlet_form::DirichletForm;
use crate::laplacian::SpectralDecomposition;

/// Spectral gap analysis for a Dirichlet form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralGapAnalysis {
    /// The spectral gap λ₁.
    pub lambda1: f64,
    /// All eigenvalues (sorted).
    pub eigenvalues: Vec<f64>,
    /// Is the graph connected (λ₁ > 0)?
    pub is_connected: bool,
    /// Mixing time estimate (informal).
    pub mixing_time: Option<f64>,
}

impl SpectralGapAnalysis {
    /// Analyze the spectral gap of a Dirichlet form.
    pub fn from_dirichlet_form(form: &DirichletForm) -> Self {
        let spec = SpectralDecomposition::from_laplacian(&form.laplacian);
        let mut eigenvalues = spec.eigenvalues.clone();

        // Find λ₁
        let lambda1 = spec.spectral_gap();
        let is_connected = lambda1 > 1e-10;
        let mixing_time = if is_connected {
            Some(1.0 / lambda1)
        } else {
            None
        };

        Self {
            lambda1,
            eigenvalues,
            is_connected,
            mixing_time,
        }
    }

    /// Exponential convergence rate for the heat semigroup.
    /// ‖S_t f - π‖ ≤ e^{-λ₁ t} ‖f - π‖_μ
    pub fn convergence_rate(&self, t: f64) -> f64 {
        (-self.lambda1 * t).exp()
    }

    /// Verify Poincaré inequality: Var_π(f) ≤ E(f,f) / λ₁.
    pub fn verify_poincare(
        &self,
        form: &DirichletForm,
        f: &DVector<f64>,
    ) -> bool {
        if self.lambda1 <= 0.0 {
            return false;
        }
        let n = form.n;
        let mu = &form.measure;
        let total_mu: f64 = mu.sum();

        // Mean under μ
        let mean = f.dot(mu) / total_mu;

        // Variance: Var_μ(f) = ⟨f-mean, f-mean⟩_μ
        let centered = f - DVector::from_element(n, mean);
        let variance = centered.component_mul(mu).dot(&centered);

        // Dirichlet energy
        let energy = form.energy(f);

        // Poincaré: variance ≤ energy / λ₁
        variance <= energy / self.lambda1 + 1e-8
    }

    /// Entropy decay rate: for the heat semigroup,
    /// d/dt H(μ_t) ≤ -2 λ₁ H(μ_t) (under log-Sobolev).
    /// Here we compute the spectral estimate of the decay.
    pub fn entropy_decay_rate(&self) -> f64 {
        2.0 * self.lambda1
    }

    /// Cheeger constant lower bound: h ≥ λ₁/2.
    pub fn cheeger_bound(&self) -> f64 {
        self.lambda1 / 2.0
    }

    /// Compute the relaxation time τ_rel = 1/λ₁.
    pub fn relaxation_time(&self) -> f64 {
        if self.lambda1 > 0.0 {
            1.0 / self.lambda1
        } else {
            f64::INFINITY
        }
    }
}

/// Ergodic convergence checker.
pub struct ErgodicConvergence {
    pub analysis: SpectralGapAnalysis,
    pub measure: DVector<f64>,
}

impl ErgodicConvergence {
    pub fn new(form: &DirichletForm) -> Self {
        let analysis = SpectralGapAnalysis::from_dirichlet_form(form);
        Self {
            analysis,
            measure: form.measure.clone(),
        }
    }

    /// Check if the semigroup converges to equilibrium.
    pub fn is_ergodic(&self) -> bool {
        self.analysis.is_connected
    }

    /// Bound on distance to equilibrium after time t.
    pub fn distance_to_equilibrium(&self, initial: &DVector<f64>, t: f64) -> f64 {
        let n = initial.len();
        let total_mu: f64 = self.measure.sum();
        let mean = initial.dot(&self.measure) / total_mu;
        let centered = initial - DVector::from_element(n, mean);
        let initial_distance = centered.component_mul(&self.measure).dot(&centered).sqrt();
        initial_distance * self.analysis.convergence_rate(t)
    }
}
