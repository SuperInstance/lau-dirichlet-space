//! # Dequantizable Dirichlet Space
//!
//! The unified mathematical object underlying conservation spectral theory,
//! sheaf cohomology, ergodic theory, information geometry, optimal transport,
//! tropical geometry, and Kähler geometry.
//!
//! Core structure: `(X, d, μ, E, {S^ℏ_t})` where:
//! - `(X, d, μ)` is a metric measure space
//! - `E` is a Dirichlet form with generator Δ
//! - `S^ℏ_t` is a deformation from linear heat semigroup (ℏ=1) to
//!   Hopf-Lax tropical semigroup (ℏ=0)

pub mod core;
pub mod dirichlet_form;
pub mod laplacian;
pub mod heat_semigroup;
pub mod cole_hopf;
pub mod hopf_lax;
pub mod registers;
pub mod spectral;
pub mod wasserstein;
pub mod jko;
pub mod dequantization;
pub mod belief_dynamics;

pub use core::*;
pub use dirichlet_form::DirichletForm;
pub use laplacian::Laplacian;
pub use heat_semigroup::HeatSemigroup;
pub use cole_hopf::ColeHopfTransform;
pub use hopf_lax::HopfLaxSemigroup;
pub use registers::*;
pub use laplacian::SpectralDecomposition;
pub use wasserstein::WassersteinSpace;
pub use jko::JKOGradientFlow;
pub use dequantization::DequantizationAxis;
pub use belief_dynamics::BeliefDynamics;
