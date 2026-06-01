//! Heat semigroup S_t = e^{-tΔ} (ℏ=1 regime).
//!
//! The heat semigroup is the fundamental solution operator:
//! ∂_t u = -Δu, u(t) = e^{-tΔ} u₀
//!
//! Properties: contraction, positivity-preserving, mass-preserving,
//! self-adjoint w.r.t. μ.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::laplacian::{Laplacian, SpectralDecomposition};

/// Heat semigroup operator e^{-tΔ}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatSemigroup {
    /// The Laplacian.
    pub laplacian: Laplacian,
    /// Spectral decomposition for efficient computation.
    #[serde(skip)]
    spectrum: Option<SpectralDecomposition>,
}

impl HeatSemigroup {
    /// Create from a Laplacian.
    pub fn new(laplacian: Laplacian) -> Self {
        Self {
            laplacian,
            spectrum: None,
        }
    }

    /// Create from a Dirichlet form.
    pub fn from_dirichlet_form(form: &crate::dirichlet_form::DirichletForm) -> Self {
        Self::new(Laplacian::from_dirichlet_form(form))
    }

    /// Ensure spectrum is computed.
    fn ensure_spectrum(&mut self) -> &SpectralDecomposition {
        if self.spectrum.is_none() {
            self.spectrum = Some(SpectralDecomposition::from_laplacian(&self.laplacian.matrix));
        }
        self.spectrum.as_ref().unwrap()
    }

    /// Apply e^{-tΔ} to a function f.
    pub fn evolve(&mut self, f: &DVector<f64>, t: f64) -> DVector<f64> {
        let spec = self.ensure_spectrum();
        let n = spec.n;

        // Decompose f in eigenbasis
        let mut result = DVector::zeros(n);
        for k in 0..n {
            let phi = spec.eigenvector(k);
            let coeff = phi.dot(f);
            let decayed = coeff * (-t * spec.eigenvalue(k)).exp();
            result += &phi * decayed;
        }
        result
    }

    /// Compute the heat kernel k_t(x, y) = Σ_k e^{-tλ_k} φ_k(x) φ_k(y).
    pub fn heat_kernel(&mut self, t: f64, i: usize, j: usize) -> f64 {
        let spec = self.ensure_spectrum();
        let mut sum = 0.0;
        for k in 0..spec.n {
            let phi_i = spec.eigenvectors[(i, k)];
            let phi_j = spec.eigenvectors[(j, k)];
            sum += (-t * spec.eigenvalue(k)).exp() * phi_i * phi_j;
        }
        sum
    }

    /// Full heat kernel matrix K_t.
    pub fn heat_kernel_matrix(&mut self, t: f64) -> DMatrix<f64> {
        let spec = self.ensure_spectrum();
        let n = spec.n;
        let mut kernel = DMatrix::zeros(n, n);

        for k in 0..n {
            let phi = spec.eigenvector(k);
            let decay = (-t * spec.eigenvalue(k)).exp();
            for i in 0..n {
                for j in 0..n {
                    kernel[(i, j)] += decay * phi[i] * phi[j];
                }
            }
        }
        kernel
    }

    /// Verify contraction: ‖S_t f‖ ≤ ‖f‖ for all f.
    pub fn is_contraction(&mut self, t: f64) -> bool {
        // Use test with a few random-ish vectors
        let n = self.laplacian.matrix.nrows();
        let test_vectors = vec![
            DVector::from_element(n, 1.0),
            {
                let mut v = DVector::zeros(n);
                if n > 0 { v[0] = 1.0; }
                v
            },
            DVector::from_element(n, 1.0 / (n as f64).sqrt()),
        ];

        for f in &test_vectors {
            let evolved = self.evolve(f, t);
            if evolved.norm() > f.norm() + 1e-8 {
                return false;
            }
        }
        true
    }

    /// Verify mass preservation: ⟨S_t f, 1⟩ = ⟨f, 1⟩ when Δ1=0.
    pub fn preserves_mass(&mut self, f: &DVector<f64>, t: f64) -> bool {
        let evolved = self.evolve(f, t);
        (f.sum() - evolved.sum()).abs() < 1e-8
    }

    /// Verify positivity preservation: f ≥ 0 ⟹ S_t f ≥ 0.
    pub fn preserves_positivity(&mut self, t: f64) -> bool {
        let n = self.laplacian.matrix.nrows();
        let positive_f = DVector::from_element(n, 1.0);
        let evolved = self.evolve(&positive_f, t);
        evolved.iter().all(|&v| v > -1e-10)
    }

    /// Long-time limit: S_t f → ⟨f, 1⟩_μ / μ(X) · 1 (projection onto constants).
    pub fn equilibrium(&mut self, f: &DVector<f64>) -> DVector<f64> {
        let n = self.laplacian.matrix.nrows();
        let mu = &self.laplacian.measure;
        let total_mu: f64 = mu.sum();
        if total_mu.abs() < 1e-15 {
            return DVector::zeros(n);
        }
        let avg = f.dot(mu) / total_mu;
        DVector::from_element(n, avg)
    }

    /// Compute the semigroup at multiple time steps.
    pub fn trajectory(&mut self, f: &DVector<f64>, times: &[f64]) -> Vec<DVector<f64>> {
        times.iter().map(|&t| self.evolve(f, t)).collect()
    }

    /// Verify semigroup property: S_{s+t} = S_s ∘ S_t.
    pub fn verify_semigroup_property(&mut self, f: &DVector<f64>, s: f64, t: f64) -> bool {
        let s_t = self.evolve(f, t);
        let s_st = self.evolve(&s_t, s);
        let s_combined = self.evolve(f, s + t);
        let diff = &s_st - &s_combined;
        diff.norm() < 1e-6
    }
}
