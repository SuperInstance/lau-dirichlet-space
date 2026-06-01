//! JKO (Jordan-Kinderlehrer-Otto) scheme: gradient flow of entropy in W₂.
//!
//! The heat flow is the gradient flow of the Boltzmann entropy H(μ) = ∫ μ log μ dvol
//! in the Wasserstein space W₂. The JKO scheme is the implicit Euler discretization:
//!
//! μ_{k+1} = argmin_ν { H(ν) + W₂²(ν, μ_k) / (2τ) }
//!
//! where τ is the time step.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::dirichlet_form::DirichletForm;
use crate::registers::MeasureRegister;
use crate::wasserstein::WassersteinSpace;

/// JKO gradient flow of entropy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JKOGradientFlow {
    /// Time step τ.
    pub tau: f64,
    /// Number of steps taken.
    pub steps: usize,
    /// The sequence of measures μ₀, μ₁, μ₂, ...
    pub trajectory: Vec<DVector<f64>>,
    /// The Wasserstein space.
    pub w2: WassersteinSpace,
    /// The Laplacian (for computing heat kernel steps).
    pub laplacian: DMatrix<f64>,
}

impl JKOGradientFlow {
    /// Create a JKO flow from a Dirichlet form.
    pub fn from_dirichlet_form(form: &DirichletForm, tau: f64) -> Self {
        let w2 = WassersteinSpace::from_dirichlet_form(form);
        Self {
            tau,
            steps: 0,
            trajectory: vec![],
            w2,
            laplacian: form.laplacian.clone(),
        }
    }

    /// Initialize with a starting measure.
    pub fn with_initial(mut self, mu0: &DVector<f64>) -> Self {
        self.trajectory.push(mu0.clone());
        self
    }

    /// Compute one JKO step via proximal optimization.
    /// In practice, for finite graphs, this is approximated by the heat kernel step.
    pub fn step(&mut self) -> DVector<f64> {
        let current = self.trajectory.last().unwrap().clone();
        let n = current.len();

        // Approximate JKO step using heat kernel: e^{-τΔ} μ
        // This is the exact solution for the continuous-time gradient flow
        let step_kernel = self.heat_kernel(self.tau);
        let mut next = &step_kernel * &current;

        // Ensure positivity and normalization
        for v in next.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let total = next.sum();
        if total > 1e-15 {
            next = next / total;
        }

        self.trajectory.push(next.clone());
        self.steps += 1;
        next
    }

    /// Run n steps of JKO.
    pub fn run(&mut self, n_steps: usize) -> Vec<DVector<f64>> {
        for _ in 0..n_steps {
            self.step();
        }
        self.trajectory[self.trajectory.len() - n_steps..].to_vec()
    }

    /// Compute the heat kernel e^{-tΔ}.
    fn heat_kernel(&self, t: f64) -> DMatrix<f64> {
        let n = self.laplacian.nrows();
        let eigen = self.laplacian.clone().symmetric_eigen();
        let mut kernel = DMatrix::zeros(n, n);

        for k in 0..n {
            let phi = eigen.eigenvectors.column(k);
            let lambda = eigen.eigenvalues[k];
            let decay = (-t * lambda).exp();
            for i in 0..n {
                for j in 0..n {
                    kernel[(i, j)] += decay * phi[i] * phi[j];
                }
            }
        }
        kernel
    }

    /// Compute entropy at each step.
    pub fn entropy_trajectory(&self) -> Vec<f64> {
        self.trajectory
            .iter()
            .map(|mu| {
                -mu.iter()
                    .filter(|&&v| v > 1e-15)
                    .map(|&v| v * v.ln())
                    .sum::<f64>()
            })
            .collect()
    }

    /// Verify entropy is non-increasing (JKO property).
    pub fn entropy_nonincreasing(&self) -> bool {
        let entropies = self.entropy_trajectory();
        for i in 1..entropies.len() {
            if entropies[i] > entropies[i - 1] + 1e-8 {
                return false;
            }
        }
        true
    }

    /// Compute W₂ distances between consecutive measures.
    pub fn w2_distances(&self) -> Vec<f64> {
        let mut distances = vec![];
        for i in 1..self.trajectory.len() {
            let d = self.w2.w2(&self.trajectory[i - 1], &self.trajectory[i]);
            distances.push(d);
        }
        distances
    }

    /// Get the current measure.
    pub fn current(&self) -> &DVector<f64> {
        self.trajectory.last().unwrap()
    }

    /// Check convergence to equilibrium.
    pub fn converged_to_equilibrium(&self, tol: f64) -> bool {
        if self.trajectory.len() < 2 {
            return false;
        }
        let n = self.trajectory[0].len();
        let equilibrium = MeasureRegister::uniform(n);
        let current = self.trajectory.last().unwrap();
        let diff = current - &equilibrium.measure;
        diff.norm() < tol
    }

    /// JKO objective: H(ν) + W₂²(ν, μ_k) / (2τ).
    pub fn jko_objective(&self, nu: &DVector<f64>) -> f64 {
        let entropy = -nu.iter()
            .filter(|&&v| v > 1e-15)
            .map(|&v| v * v.ln())
            .sum::<f64>();

        let mu_k = self.trajectory.last().unwrap();
        let w2_sq = self.w2.w2_squared(nu, mu_k);

        entropy + w2_sq / (2.0 * self.tau)
    }
}
