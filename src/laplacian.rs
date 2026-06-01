//! Laplacian generator Δ and spectral decomposition.
//!
//! The Laplacian is the generator of the Dirichlet form: E(f,g) = ⟨Δf, g⟩_μ.
//! For a finite graph: Δf(i) = Σ_j w_{ij}(f(j) - f(i)).
//! Spectral decomposition: Δφ_k = λ_k φ_k.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::dirichlet_form::DirichletForm;

/// Spectral decomposition of a Laplacian.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDecomposition {
    /// Eigenvalues λ₀ ≤ λ₁ ≤ ... ≤ λ_{n-1}.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns of a matrix.
    pub eigenvectors: DMatrix<f64>,
    /// The dimension.
    pub n: usize,
}

impl SpectralDecomposition {
    /// Compute the spectral decomposition of a Laplacian.
    pub fn from_laplacian(laplacian: &DMatrix<f64>) -> Self {
        let n = laplacian.nrows();
        let eigen = laplacian.clone().symmetric_eigen();
        let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
        let mut eigenvectors = eigen.eigenvectors;

        // Sort by eigenvalue
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

        let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
        let sorted_vecs = DMatrix::from_columns(
            &indices.iter().map(|&i| eigenvectors.column(i).into_owned()).collect::<Vec<_>>(),
        );

        Self {
            eigenvalues: sorted_vals,
            eigenvectors: sorted_vecs,
            n,
        }
    }

    /// Get eigenvalue λ_k.
    pub fn eigenvalue(&self, k: usize) -> f64 {
        self.eigenvalues[k]
    }

    /// Get eigenvector φ_k.
    pub fn eigenvector(&self, k: usize) -> DVector<f64> {
        self.eigenvectors.column(k).into_owned()
    }

    /// The smallest eigenvalue (always 0 for graph Laplacians).
    pub fn lambda_0(&self) -> f64 {
        self.eigenvalues[0]
    }

    /// The spectral gap λ₁ (first nonzero eigenvalue).
    pub fn spectral_gap(&self) -> f64 {
        for &v in &self.eigenvalues {
            if v > 1e-10 {
                return v;
            }
        }
        0.0
    }

    /// Number of zero eigenvalues (kernel dimension).
    pub fn kernel_dim(&self) -> usize {
        self.eigenvalues.iter().filter(|&&v| v.abs() < 1e-10).count()
    }

    /// Reconstruct a function from spectral coefficients.
    pub fn reconstruct(&self, coefficients: &DVector<f64>) -> DVector<f64> {
        &self.eigenvectors * coefficients
    }

    /// Project f onto the first k eigenmodes.
    pub fn project(&self, f: &DVector<f64>, k: usize) -> DVector<f64> {
        let mut result = DVector::zeros(self.n);
        for i in 0..k.min(self.n) {
            let phi = self.eigenvector(i);
            let coeff = phi.dot(f);
            result += &phi * coeff;
        }
        result
    }
}

/// Laplacian operator on a finite space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Laplacian {
    /// The Laplacian matrix L.
    pub matrix: DMatrix<f64>,
    /// The measure μ (for weighted inner product).
    pub measure: DVector<f64>,
    /// Spectral decomposition (computed on demand).
    #[serde(skip)]
    pub spectrum: Option<SpectralDecomposition>,
}

impl Laplacian {
    /// Create from a Dirichlet form.
    pub fn from_dirichlet_form(form: &DirichletForm) -> Self {
        Self {
            matrix: form.laplacian.clone(),
            measure: form.measure.clone(),
            spectrum: None,
        }
    }

    /// Create from a raw matrix.
    pub fn from_matrix(matrix: DMatrix<f64>, measure: DVector<f64>) -> Self {
        Self {
            matrix,
            measure,
            spectrum: None,
        }
    }

    /// Apply the Laplacian: Δf.
    pub fn apply(&self, f: &DVector<f64>) -> DVector<f64> {
        &self.matrix * f
    }

    /// Compute spectral decomposition (cached).
    pub fn spectrum(&mut self) -> &SpectralDecomposition {
        if self.spectrum.is_none() {
            self.spectrum = Some(SpectralDecomposition::from_laplacian(&self.matrix));
        }
        self.spectrum.as_ref().unwrap()
    }

    /// Compute the trace Tr(Δ) = Σ λ_k.
    pub fn trace(&self) -> f64 {
        (0..self.matrix.nrows()).map(|i| self.matrix[(i, i)]).sum()
    }

    /// Inner product ⟨f, g⟩_μ = Σ f_i g_i μ_i.
    pub fn weighted_inner(&self, f: &DVector<f64>, g: &DVector<f64>) -> f64 {
        f.component_mul(&self.measure).dot(g)
    }

    /// Weighted norm ‖f‖_μ.
    pub fn weighted_norm(&self, f: &DVector<f64>) -> f64 {
        self.weighted_inner(f, f).sqrt()
    }

    /// Verify Δ1 = 0 (constant functions are harmonic).
    pub fn annihilates_constants(&self) -> bool {
        let ones = DVector::from_element(self.matrix.nrows(), 1.0);
        let result = self.apply(&ones);
        result.iter().all(|&v| v.abs() < 1e-10)
    }
}
