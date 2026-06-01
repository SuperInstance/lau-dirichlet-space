//! Dirichlet forms E(f,g) = ⟨Δf, g⟩_μ.
//!
//! A Dirichlet form is a bilinear, symmetric, positive, closed, Markovian form.
//! On a finite graph: E(f,g) = Σ_{i~j} w_{ij}(f(i) - f(j))(g(i) - g(j)).

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::MetricMeasureSpace;

/// A Dirichlet form on a finite space.
///
/// Represented by its associated Laplacian matrix L and measure μ,
/// so that E(f,g) = ⟨Lf, g⟩_μ = f^T L g.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirichletForm {
    /// The number of vertices.
    pub n: usize,
    /// Laplacian matrix L (symmetric positive semidefinite, row sums = 0).
    pub laplacian: DMatrix<f64>,
    /// The reference measure μ.
    pub measure: DVector<f64>,
}

impl DirichletForm {
    /// Construct from a graph Laplacian and measure.
    pub fn new(laplacian: DMatrix<f64>, measure: DVector<f64>) -> Self {
        let n = laplacian.nrows();
        assert_eq!(laplacian.ncols(), n);
        assert_eq!(measure.len(), n);
        Self {
            n,
            laplacian,
            measure,
        }
    }

    /// Construct from an adjacency/weight matrix W on a uniform space.
    /// L = D - W where D = diag(W·1).
    pub fn from_weight_matrix(weights: &DMatrix<f64>) -> Self {
        let n = weights.nrows();
        let row_sums: DVector<f64> = weights.row_sum().transpose();
        let degree = DMatrix::from_diagonal(&row_sums);
        let laplacian = &degree - weights;
        Self {
            n,
            laplacian,
            measure: DVector::from_element(n, 1.0),
        }
    }

    /// Evaluate E(f, g) = f^T L g.
    pub fn eval(&self, f: &DVector<f64>, g: &DVector<f64>) -> f64 {
        let lg = &self.laplacian * g;
        f.dot(&lg)
    }

    /// Energy E(f, f) = f^T L f ≥ 0.
    pub fn energy(&self, f: &DVector<f64>) -> f64 {
        self.eval(f, f)
    }

    /// Verify symmetry: E(f,g) = E(g,f) for random vectors.
    pub fn is_symmetric(&self) -> bool {
        // Check matrix symmetry directly
        for i in 0..self.n {
            for j in 0..self.n {
                if (self.laplacian[(i, j)] - self.laplacian[(j, i)]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Verify positive semidefiniteness: E(f,f) ≥ 0 for all f.
    pub fn is_positive_semidefinite(&self) -> bool {
        // Check via eigenvalue decomposition
        let eig = self.laplacian.symmetric_eigenvalues();
        eig.iter().all(|&v| v > -1e-10)
    }

    /// Verify the Markov property: E(Tf, Tf) ≤ E(f, f) for T(s) = max(0, min(1, s)).
    /// For a quadratic form, this is equivalent to the off-diagonal entries being ≤ 0
    /// and the form being positive.
    pub fn is_markovian(&self) -> bool {
        // Check Beurling-Deny conditions:
        // Off-diagonal entries of L are ≤ 0
        for i in 0..self.n {
            for j in 0..self.n {
                if i != j && self.laplacian[(i, j)] > 1e-10 {
                    return false;
                }
            }
        }
        // Row sums = 0
        for i in 0..self.n {
            let row_sum: f64 = (0..self.n).map(|j| self.laplacian[(i, j)]).sum();
            if row_sum.abs() > 1e-10 {
                return false;
            }
        }
        true
    }

    /// Verify closedness (automatic in finite dimensions).
    pub fn is_closed(&self) -> bool {
        // In finite dimensions, all bilinear forms are closed.
        true
    }

    /// Full Dirichlet form verification.
    pub fn is_dirichlet_form(&self) -> bool {
        self.is_symmetric() && self.is_positive_semidefinite() && self.is_markovian()
    }

    /// Compute the spectral gap λ₁ (smallest nonzero eigenvalue).
    pub fn spectral_gap(&self) -> f64 {
        let mut eigvals: Vec<f64> = self.laplacian.symmetric_eigenvalues().iter().copied().collect();
        eigvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Find first nonzero eigenvalue
        for &v in &eigvals {
            if v > 1e-10 {
                return v;
            }
        }
        0.0 // Complete graph on 1 vertex
    }

    /// Get the kernel (constant functions for connected graphs).
    pub fn kernel_dimension(&self) -> usize {
        let eigvals = self.laplacian.symmetric_eigenvalues();
        eigvals.iter().filter(|&&v| v.abs() < 1e-10).count()
    }
}

/// Builder for constructing Dirichlet forms from graph specifications.
pub struct DirichletFormBuilder {
    n: usize,
    weights: DMatrix<f64>,
    measure: Option<DVector<f64>>,
}

impl DirichletFormBuilder {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            weights: DMatrix::zeros(n, n),
            measure: None,
        }
    }

    /// Add an undirected edge with weight.
    pub fn edge(mut self, i: usize, j: usize, weight: f64) -> Self {
        assert!(i < self.n && j < self.n && i != j);
        self.weights[(i, j)] = weight;
        self.weights[(j, i)] = weight;
        self
    }

    /// Add a complete graph (all edges weight 1).
    pub fn complete(n: usize) -> Self {
        let mut builder = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                builder = builder.edge(i, j, 1.0);
            }
        }
        builder
    }

    /// Add a path graph.
    pub fn path(n: usize) -> Self {
        let mut builder = Self::new(n);
        for i in 0..n.saturating_sub(1) {
            builder = builder.edge(i, i + 1, 1.0);
        }
        builder
    }

    /// Add a cycle graph.
    pub fn cycle(n: usize) -> Self {
        let mut builder = Self::path(n);
        if n > 2 {
            builder = builder.edge(0, n - 1, 1.0);
        }
        builder
    }

    pub fn measure(mut self, measure: DVector<f64>) -> Self {
        assert_eq!(measure.len(), self.n);
        self.measure = Some(measure);
        self
    }

    pub fn build(self) -> DirichletForm {
        let measure = self.measure.unwrap_or_else(|| DVector::from_element(self.n, 1.0));
        let mut form = DirichletForm::from_weight_matrix(&self.weights);
        form.measure = measure;
        form
    }
}
