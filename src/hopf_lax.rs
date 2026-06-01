//! Hopf-Lax semigroup: tropical limit (ℏ=0).
//!
//! In the tropical limit, the heat equation becomes the Hopf-Lax formula:
//! S_t f(x) = inf_y { f(y) + t·L((y-x)/t) }
//! where L is the Lagrangian.
//!
//! In the (min, +) tropical semiring, multiplication = addition,
//! addition = min. The heat kernel becomes a tropical convolution.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::MetricMeasureSpace;

/// Tropical semiring operations (min, +).
pub mod tropical {
    use nalgebra::{DMatrix, DVector};

    /// Tropical addition = min.
    pub fn add(a: f64, b: f64) -> f64 {
        a.min(b)
    }

    /// Tropical multiplication = addition.
    pub fn mul(a: f64, b: f64) -> f64 {
        a + b
    }

    /// Tropical identity for multiplication = 0.
    pub fn one() -> f64 {
        0.0
    }

    /// Tropical identity for addition = +∞.
    pub fn zero() -> f64 {
        f64::INFINITY
    }

    /// Tropical exponentiation: a ⊗ n = n·a.
    pub fn pow(a: f64, n: usize) -> f64 {
        a * n as f64
    }

    /// Tropical matrix-vector product in (min, +).
    pub fn mat_vec(matrix: &DMatrix<f64>, vector: &DVector<f64>) -> DVector<f64> {
        let n = matrix.nrows();
        let mut result = DVector::from_element(n, f64::INFINITY);
        for i in 0..n {
            for j in 0..matrix.ncols() {
                result[i] = result[i].min(matrix[(i, j)] + vector[j]);
            }
        }
        result
    }

    /// Tropical matrix multiplication in (min, +).
    pub fn mat_mat(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
        let (m, k1) = a.shape();
        let (k2, n) = b.shape();
        assert_eq!(k1, k2);
        let mut result = DMatrix::from_element(m, n, f64::INFINITY);
        for i in 0..m {
            for j in 0..n {
                for l in 0..k1 {
                    result[(i, j)] = result[(i, j)].min(a[(i, l)] + b[(l, j)]);
                }
            }
        }
        result
    }
}

/// Hopf-Lax semigroup in the tropical limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopfLaxSemigroup {
    /// The underlying space.
    pub space: MetricMeasureSpace,
    /// Cost matrix c(i,j) for the inf-convolution.
    pub cost: DMatrix<f64>,
}

impl HopfLaxSemigroup {
    /// Create from a metric measure space with distance as cost.
    pub fn from_space(space: &MetricMeasureSpace) -> Self {
        Self {
            space: space.clone(),
            cost: space.distance.clone(),
        }
    }

    /// Create from a cost matrix directly.
    pub fn from_cost(cost: DMatrix<f64>) -> Self {
        let n = cost.nrows();
        Self {
            space: MetricMeasureSpace::uniform(n),
            cost,
        }
    }

    /// Apply one step of the Hopf-Lax semigroup:
    /// S_t f(x) = inf_y { f(y) + t·c(x,y) }
    /// This is tropical matrix-vector multiplication with scaled cost.
    pub fn evolve(&self, f: &DVector<f64>, t: f64) -> DVector<f64> {
        let scaled_cost = self.cost.scale(t);
        tropical::mat_vec(&scaled_cost, f)
    }

    /// Apply n steps of the Hopf-Lax semigroup.
    pub fn evolve_steps(&self, f: &DVector<f64>, t: f64, steps: usize) -> DVector<f64> {
        let step_size = t / steps as f64;
        let scaled_cost = self.cost.scale(step_size);
        let tropical_kernel = tropical::mat_mat(&scaled_cost, &scaled_cost);
        // For simplicity, apply step-by-step
        let mut current = f.clone();
        for _ in 0..steps {
            current = tropical::mat_vec(&scaled_cost, &current);
        }
        current
    }

    /// Compute the tropical (min,+) heat kernel: K_t(i,j) = t·c(i,j).
    pub fn tropical_kernel(&self, t: f64) -> DMatrix<f64> {
        self.cost.scale(t)
    }

    /// Verify the semigroup property: S_{s+t} = S_s ∘ S_t.
    pub fn verify_semigroup(&self, f: &DVector<f64>, s: f64, t: f64) -> bool {
        let s_t = self.evolve(f, t);
        let s_st = self.evolve(&s_t, s);
        let s_combined = self.evolve(f, s + t);

        // In tropical arithmetic, exact equality holds
        for i in 0..s_st.len() {
            if (s_st[i] - s_combined[i]).abs() > 1e-6 {
                return false;
            }
        }
        true
    }

    /// Verify contraction in sup-norm.
    pub fn is_contraction(&self, f: &DVector<f64>, g: &DVector<f64>, t: f64) -> bool {
        let sf = self.evolve(f, t);
        let sg = self.evolve(g, t);
        let input_diff = (f - g).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let output_diff = (&sf - &sg).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        output_diff <= input_diff + 1e-10
    }

    /// Tropical eigenvalue: the tropical Perron-Frobenius eigenvalue
    /// is the minimum cycle mean in the cost graph.
    pub fn tropical_eigenvalue(&self) -> f64 {
        let n = self.cost.nrows();
        let mut min_cycle_mean = f64::INFINITY;

        // Check cycles of length 1 (self-loops)
        for i in 0..n {
            min_cycle_mean = min_cycle_mean.min(self.cost[(i, i)]);
        }

        // Check cycles of length 2
        for i in 0..n {
            for j in 0..n {
                let cycle_cost = self.cost[(i, j)] + self.cost[(j, i)];
                min_cycle_mean = min_cycle_mean.min(cycle_cost / 2.0);
            }
        }

        if min_cycle_mean.is_finite() {
            min_cycle_mean
        } else {
            0.0
        }
    }
}
