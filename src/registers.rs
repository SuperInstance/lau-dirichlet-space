//! Three register views of the Dirichlet space.
//!
//! Any function on the space can be viewed in three equivalent ways:
//! 1. **Observables** (function register): f: X → ℝ, the "what is" view
//! 2. **Beliefs** (measure register): probability measures on X, the "what we believe" view
//! 3. **Sheaf sections** (multi-modal register): consistent local data across a cover

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Function register — observables f: X → ℝ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservableRegister {
    /// The observable values f(x) for each x ∈ X.
    pub values: DVector<f64>,
    /// The dimension (number of points).
    pub n: usize,
}

impl ObservableRegister {
    pub fn new(values: DVector<f64>) -> Self {
        let n = values.len();
        Self { values, n }
    }

    /// Constant observable (all ones).
    pub fn ones(n: usize) -> Self {
        Self {
            values: DVector::from_element(n, 1.0),
            n,
        }
    }

    /// Delta function at vertex i.
    pub fn delta(n: usize, i: usize) -> Self {
        let mut values = DVector::zeros(n);
        values[i] = 1.0;
        Self { values, n }
    }

    /// Apply a linear operator (e.g., Laplacian).
    pub fn apply(&self, op: &DMatrix<f64>) -> Self {
        Self::new(op * &self.values)
    }

    /// L² norm.
    pub fn l2_norm(&self) -> f64 {
        self.values.norm()
    }

    /// L² inner product.
    pub fn dot(&self, other: &Self) -> f64 {
        self.values.dot(&other.values)
    }

    /// Convert to belief (measure) register via exponentiation:
    /// μ(x) ∝ exp(-β f(x)) for some inverse temperature β.
    pub fn to_belief(&self, beta: f64) -> MeasureRegister {
        let log_weights = self.values.map(|v| -beta * v);
        let max_lw = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights = log_weights.map(|v| (v - max_lw).exp());
        let total = weights.sum();
        MeasureRegister {
            measure: weights / total,
            n: self.n,
        }
    }
}

/// Measure register — beliefs (probability measures on X).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureRegister {
    /// The probability measure μ(x) ≥ 0, Σ μ(x) = 1.
    pub measure: DVector<f64>,
    /// The dimension.
    pub n: usize,
}

impl MeasureRegister {
    /// Create a uniform measure.
    pub fn uniform(n: usize) -> Self {
        Self {
            measure: DVector::from_element(n, 1.0 / n as f64),
            n,
        }
    }

    /// Create from unnormalized weights (softmax-like).
    pub fn from_weights(weights: &DVector<f64>) -> Self {
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_w = weights.map(|v| (v - max_w).exp());
        let total = exp_w.sum();
        Self {
            measure: exp_w / total,
            n: weights.len(),
        }
    }

    /// Create a Dirac measure at vertex i.
    pub fn dirac(n: usize, i: usize) -> Self {
        let mut measure = DVector::zeros(n);
        measure[i] = 1.0;
        Self { measure, n }
    }

    /// Verify this is a valid probability measure.
    pub fn is_valid(&self) -> bool {
        self.measure.iter().all(|&v| v >= -1e-10)
            && (self.measure.sum() - 1.0).abs() < 1e-8
    }

    /// Shannon entropy H(μ) = -Σ μ(x) log μ(x).
    pub fn entropy(&self) -> f64 {
        -self.measure.iter()
            .filter(|&&v| v > 1e-15)
            .map(|&v| v * v.ln())
            .sum::<f64>()
    }

    /// KL divergence D_KL(self || other).
    pub fn kl_divergence(&self, other: &Self) -> f64 {
        self.measure.iter()
            .zip(other.measure.iter())
            .filter(|(&p, _)| p > 1e-15)
            .map(|(p, q)| {
                let q_safe = q.max(1e-15);
                p * (p / q_safe).ln()
            })
            .sum()
    }

    /// Convert to observable via negative log: f(x) = -log μ(x).
    pub fn to_observable(&self) -> ObservableRegister {
        ObservableRegister::new(self.measure.map(|v| {
            if v > 1e-15 {
                -v.ln()
            } else {
                f64::INFINITY
            }
        }))
    }

    /// Push forward through a kernel (Markov transition).
    pub fn push_forward(&self, kernel: &DMatrix<f64>) -> Self {
        // kernel is column-stochastic: columns sum to 1
        let new_measure = kernel * &self.measure;
        let total = new_measure.sum();
        Self {
            measure: new_measure / total,
            n: self.n,
        }
    }

    /// Rényi entropy of order α.
    pub fn renyi_entropy(&self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < 1e-10 {
            return self.entropy();
        }
        let sum_p_alpha: f64 = self.measure.iter()
            .map(|&v| v.max(0.0).powf(alpha))
            .sum();
        if sum_p_alpha > 0.0 {
            (1.0 / (1.0 - alpha)) * sum_p_alpha.ln()
        } else {
            0.0
        }
    }
}

/// Sheaf section register — multi-modal data with consistency conditions.
///
/// A sheaf assigns data to opens of a cover {U_i} with restriction maps.
/// A section assigns compatible data to each open.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SheafSectionRegister {
    /// Number of patches in the cover.
    pub n_patches: usize,
    /// Data on each patch: section[i] is the data on patch i.
    pub sections: Vec<DVector<f64>>,
    /// Restriction maps: restriction[i][j] gives the matrix for U_i ∩ U_j.
    /// For simplicity, we track consistency violations.
    pub overlap_matrix: DMatrix<f64>,
    /// The base dimension (size of data on each patch).
    pub dim: usize,
}

impl SheafSectionRegister {
    /// Create with trivial cover (single patch covering all of X).
    pub fn trivial(data: DVector<f64>) -> Self {
        let dim = data.len();
        Self {
            n_patches: 1,
            sections: vec![data],
            overlap_matrix: DMatrix::zeros(1, 1),
            dim,
        }
    }

    /// Create with an open cover, each patch containing all vertices.
    pub fn from_cover(sections: Vec<DVector<f64>>) -> Self {
        let n_patches = sections.len();
        let dim = sections.first().map(|v| v.len()).unwrap_or(0);
        Self {
            n_patches,
            sections,
            overlap_matrix: DMatrix::zeros(n_patches, n_patches),
            dim,
        }
    }

    /// Compute consistency: the discrepancy between patches.
    /// Returns a matrix of L² differences between sections.
    pub fn consistency(&self) -> DMatrix<f64> {
        let n = self.n_patches;
        let mut consistency = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let diff = &self.sections[i] - &self.sections[j];
                consistency[(i, j)] = diff.norm();
            }
        }
        consistency
    }

    /// Check if sections are globally consistent (all equal).
    pub fn is_globally_consistent(&self, tol: f64) -> bool {
        for i in 1..self.n_patches {
            let diff = &self.sections[i] - &self.sections[0];
            if diff.norm() > tol {
                return false;
            }
        }
        true
    }

    /// Global section (average of all local sections).
    pub fn global_section(&self) -> DVector<f64> {
        let mut avg = DVector::zeros(self.dim);
        for section in &self.sections {
            avg += section;
        }
        avg / self.n_patches as f64
    }

    /// Sheaf cohomology dimension (informal):
    /// H⁰ = space of global sections (dimension of globally consistent subspace).
    /// We approximate: if all sections are consistent, H⁰ = dim, else smaller.
    pub fn h0_dimension(&self, tol: f64) -> usize {
        if self.is_globally_consistent(tol) {
            self.dim
        } else {
            // Rough estimate: count consistent pairs
            let consistent_pairs = self.consistency().iter()
                .filter(|&&v| v < tol)
                .count();
            consistent_pairs.max(1) / self.n_patches.max(1)
        }
    }
}

/// Convert between register views.
pub trait RegisterConversion {
    fn to_observable(&self) -> ObservableRegister;
    fn to_measure(&self) -> MeasureRegister;
}

impl RegisterConversion for ObservableRegister {
    fn to_observable(&self) -> ObservableRegister {
        self.clone()
    }

    fn to_measure(&self) -> MeasureRegister {
        self.to_belief(1.0)
    }
}

impl RegisterConversion for MeasureRegister {
    fn to_observable(&self) -> ObservableRegister {
        self.to_observable()
    }

    fn to_measure(&self) -> MeasureRegister {
        self.clone()
    }
}
