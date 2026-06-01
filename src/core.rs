//! Core types for the Dequantizable Dirichlet Space.
//!
//! The fundamental structure is 𝕊 = (X, d, μ, E, {S^ℏ_t}) where:
//! - (X, d, μ) is a metric measure space
//! - E is a Dirichlet form with Laplacian generator Δ
//! - S^ℏ_t interpolates from linear heat semigroup (ℏ=1) to Hopf-Lax (ℏ=0)

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A finite metric measure space (X, d, μ).
///
/// In the finite setting, X = {0, 1, ..., n-1}, d is a distance matrix,
/// and μ is a positive measure on vertices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricMeasureSpace {
    /// Number of points in the space.
    pub n: usize,
    /// Distance matrix d(i,j) — symmetric, zero on diagonal.
    pub distance: DMatrix<f64>,
    /// Measure μ on each vertex (positive).
    pub measure: DVector<f64>,
}

impl MetricMeasureSpace {
    /// Create a trivial single-point space.
    pub fn trivial() -> Self {
        Self {
            n: 1,
            distance: DMatrix::zeros(1, 1),
            measure: DVector::from_element(1, 1.0),
        }
    }

    /// Create a uniform space with n vertices and unit measure.
    pub fn uniform(n: usize) -> Self {
        Self {
            n,
            distance: DMatrix::zeros(n, n),
            measure: DVector::from_element(n, 1.0),
        }
    }

    /// Create a space from a weighted adjacency matrix.
    /// The distance is 1/weight (for positive weights) or ∞ for zero weights.
    pub fn from_adjacency(weights: &DMatrix<f64>) -> Self {
        let n = weights.nrows();
        let mut dist = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    dist[(i, j)] = 0.0;
                } else if weights[(i, j)] > 0.0 {
                    dist[(i, j)] = 1.0 / weights[(i, j)];
                } else {
                    dist[(i, j)] = f64::INFINITY;
                }
            }
        }
        Self {
            n,
            distance: dist,
            measure: DVector::from_element(n, 1.0),
        }
    }

    /// Total measure μ(X).
    pub fn total_measure(&self) -> f64 {
        self.measure.sum()
    }

    /// Normalize the measure to be a probability distribution.
    pub fn normalize_measure(&self) -> DVector<f64> {
        let total = self.total_measure();
        if total > 0.0 {
            &self.measure / total
        } else {
            DVector::from_element(self.n, 1.0 / self.n as f64)
        }
    }
}

/// The full Dequantizable Dirichlet Space 𝕊 = (X, d, μ, E, {S^ℏ_t}).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirichletSpace {
    /// The underlying metric measure space.
    pub space: MetricMeasureSpace,
    /// The Dirichlet form matrix (positive semidefinite).
    pub dirichlet_matrix: DMatrix<f64>,
    /// The Laplacian (generator of the Dirichlet form).
    pub laplacian: DMatrix<f64>,
}

/// The dequantization parameter ℏ ∈ [0, 1].
/// - ℏ = 1: classical linear heat semigroup
/// - ℏ = 0: tropical Hopf-Lax semigroup
/// - 0 < ℏ < 1: interpolating regime
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HBar {
    pub value: f64,
}

impl HBar {
    pub fn new(value: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&value),
            "ℏ must be in [0, 1], got {}",
            value
        );
        Self { value }
    }

    pub fn classical() -> Self {
        Self { value: 1.0 }
    }

    pub fn tropical() -> Self {
        Self { value: 0.0 }
    }

    pub fn is_classical(&self) -> bool {
        self.value == 1.0
    }

    pub fn is_tropical(&self) -> bool {
        self.value == 0.0
    }
}

impl Default for HBar {
    fn default() -> Self {
        Self::classical()
    }
}
