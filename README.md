# lau-dirichlet-space

**Dequantizable Dirichlet Space — the unified mathematical object underlying spectral theory, ergodic dynamics, information geometry, and tropical limits.**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/edition-2021-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2021/)

---

## What This Does

`lau-dirichlet-space` implements the full machinery of **Dirichlet spaces** — the analytic framework where probability measures, heat diffusion, and gradient flows live on a common geometric stage. The crate provides:

- **Dirichlet forms** — the bilinear energy functionals that encode how "rough" a function is relative to a measure
- **Laplacians** — the infinitesimal generator associated with a Dirichlet form, with spectral decomposition
- **Heat semigroups** — the time evolution operator $e^{-t\Lambda}$ that smooths functions via diffusion
- **Cole–Hopf transform** — the nonlinear substitution that linearizes Hamilton–Jacobi equations through the heat equation
- **Hopf–Lax formula** — the tropical ($\hbar \to 0$) limit of the Cole–Hopf solution, recovering viscosity solutions of Hamilton–Jacobi PDEs
- **Wasserstein space** — the metric geometry of probability measures equipped with the $W_2$ optimal-transport distance
- **JKO gradient flow** — the Jordan–Kinderlehrer–Otto minimizing-movement scheme that discretizes gradient flows in Wasserstein space
- **Belief dynamics** — applying Dirichlet-space operators to probability distributions that represent an agent's beliefs
- **Measure registers** — quantum-register-inspired containers for probability measures, supporting dequantization (classical limit $\hbar \to 0$)

This crate is the foundation of a four-crate mathematical stack:

```
lau-dirichlet-space  →  lau-gradient-ricci  →  lau-spectral-agent
                  ↘  lau-trace-monoid  ↗
```

---

## Key Idea

A **Dirichlet space** $(X, \mathcal{E}, m)$ consists of a state space $X$, a closed symmetric bilinear form $\mathcal{E}(f, g) = \int \nabla f \cdot \nabla g \, dm$, and a reference measure $m$. The form $\mathcal{E}$ determines everything:

1. Its generator is the **Laplacian** $\Delta$ (via $\mathcal{E}(f, g) = -\langle f, \Delta g \rangle$).
2. The operator $e^{-t\Delta}$ is the **heat semigroup** — diffusion for time $t$.
3. By the **Cole–Hopf** substitution $u = e^{-V/\hbar}$, nonlinear Hamilton–Jacobi equations become linear heat equations.
4. In the **tropical limit** $\hbar \to 0$, the Cole–Hopf solution converges to the **Hopf–Lax** formula — a min-plus convolution that solves the Hamilton–Jacobi equation directly.
5. On the space of probability measures, the heat semigroup is a **Wasserstein gradient flow** of the entropy, discretized by the **JKO scheme**:

$$\mu_{n+1} = \arg\min_{\mu} \left[ \frac{1}{2\tau} W_2^2(\mu, \mu_n) + \mathcal{F}(\mu) \right]$$

6. A **dequantization parameter** $\hbar$ interpolates between quantum-like superposition ($\hbar > 0$) and classical deterministic dynamics ($\hbar \to 0$).

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-dirichlet-space = "0.1"
```

Or clone directly:

```bash
git clone https://github.com/SuperInstance/lau-dirichlet-space.git
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| [`nalgebra`](https://crates.io/crates/nalgebra) `0.33` | Linear algebra (matrices, eigenvalues, `DMatrix`, `DVector`) |
| [`serde`](https://crates.io/crates/serde) `1` | Serialization of all mathematical objects |
| [`serde_json`](https://crates.io/crates/serde_json) `1` | JSON round-tripping |

### Dev Dependencies

- [`approx`](https://crates.io/crates/approx) `0.5` — floating-point assertions in tests

---

## Quick Start

```rust
use lau_dirichlet_space::{
    DirichletForm, Laplacian, HeatSemigroup,
    ColeHopfTransform, HopfLaxFormula,
    WassersteinSpace, JKOGradientFlow,
    BeliefDynamics, MeasureRegister,
};

// 1. Build a Laplacian on a 4-node chain graph
let laplacian = Laplacian::chain(4);

// 2. Construct the associated Dirichlet form
let dirichlet = DirichletForm::from_laplacian(&laplacian);

// 3. Evolve under the heat semigroup for time t=0.5
let semigroup = HeatSemigroup::new(&laplacian);
let initial = vec![1.0, 0.0, 0.0, 0.0];
let evolved = semigroup.evolve(&initial, 0.5);

// 4. Cole–Hopf transform: solve Hamilton–Jacobi via the heat equation
let ch = ColeHopfTransform::new(1.0); // ℏ = 1.0
let potential = ch.potential_from_temperature(&evolved);

// 5. Tropical (ℏ → 0) limit via Hopf–Lax
let hl = HopfLaxFormula::new(hamiltonian);
let viscosity_solution = hl.solve(x0, t);

// 6. Wasserstein gradient flow via JKO
let jko = JKOGradientFlow::new(dirichlet, 0.01); // step size τ
let trajectory = jko.flow(&initial_measure, 100);

// 7. Belief dynamics on a measure register
let mut belief = BeliefDynamics::new(MeasureRegister::dirac(0));
belief.diffuse(&laplacian, 1.0);
```

---

## API Reference

### `core` — Core Types

The foundational types shared across all modules. Defines the `Dequantizable` trait for objects that admit a classical ($\hbar \to 0$) limit.

### `dirichlet_form` — `DirichletForm`

| Method | Description |
|--------|-------------|
| `from_laplacian(L)` | Construct from a Laplacian operator |
| `energy(f, g)` | Compute $\mathcal{E}(f, g) = \langle f, -\Delta g \rangle$ |
| `capacity(set)` | Infimal energy of functions equal to 1 on the set |

### `laplacian` — `Laplacian`

| Method | Description |
|--------|-------------|
| `chain(n)` | Combinatorial Laplacian of an $n$-node path graph |
| `eigenvalues()` | Spectrum $\lambda_0 \leq \lambda_1 \leq \cdots \leq \lambda_{n-1}$ |
| `eigenvectors()` | Associated eigenbasis |
| `apply(f)` | Compute $\Delta f$ |

### `heat_semigroup` — `HeatSemigroup`

| Method | Description |
|--------|-------------|
| `new(laplacian)` | Build from a Laplacian via eigendecomposition |
| `evolve(f, t)` | Compute $e^{-t\Delta} f = \sum_k e^{-\lambda_k t} \langle f, \phi_k \rangle \phi_k$ |
| `kernel(x, y, t)` | Heat kernel $p_t(x, y)$ — the fundamental solution |

### `cole_hopf` — `ColeHopfTransform`

| Method | Description |
|--------|-------------|
| `new(hbar)` | Create with dequantization parameter $\hbar$ |
| `transform(psi)` | Apply $V = -\hbar \log \psi$ (wavefunction → potential) |
| `inverse(V)` | Apply $\psi = e^{-V/\hbar}$ (potential → wavefunction) |
| `potential_from_temperature(u)` | Recover the Hamilton–Jacobi potential from a heat solution |

### `hopf_lax` — `HopfLaxFormula`

| Method | Description |
|--------|-------------|
| `new(hamiltonian)` | Create from a Hamiltonian $H(p)$ |
| `solve(x, t)` | Compute $S(x, t) = \inf_y \left[ S_0(y) + t \cdot L\!\left(\frac{x-y}{t}\right) \right]$ |
| `lagrangian()` | The Legendre transform $L(v) = \sup_p [p \cdot v - H(p)]$ |

### `wasserstein` — `WassersteinSpace`

| Method | Description |
|--------|-------------|
| `w2_distance(mu, nu)` | $W_2$ optimal-transport distance between two discrete measures |
| `optimal_coupling(mu, nu)` | The transport plan $\pi$ achieving the infimum |
| `barycenter(measures, weights)` | Fréchet mean in $W_2$ space |

### `jko` — `JKOGradientFlow`

| Method | Description |
|--------|-------------|
| `new(dirichlet, tau)` | Create with Dirichlet form and time step $\tau$ |
| `step(mu)` | One JKO step: $\mu \mapsto \arg\min_\nu \left[\frac{1}{2\tau}W_2^2(\nu, \mu) + \mathcal{F}(\nu)\right]$ |
| `flow(mu, steps)` | Run $n$ JKO steps, returning the full trajectory |

### `belief_dynamics` — `BeliefDynamics`

| Method | Description |
|--------|-------------|
| `new(register)` | Wrap a `MeasureRegister` as a belief state |
| `diffuse(laplacian, t)` | Apply heat semigroup to the belief (Bayesian update as diffusion) |
| `observe(likelihood)` | Multiply belief by a likelihood (Bayes' rule) |
| `entropy()` | Shannon entropy of the current belief |

### `registers` — `MeasureRegister`

| Method | Description |
|--------|-------------|
| `dirac(x)` | Point mass at $x$ |
| `uniform(n)` | Uniform distribution on $\{0, \ldots, n-1\}$ |
| `from_weights(w)` | Categorical distribution from weights |
| `dequantize(hbar)` | Take the $\hbar \to 0$ classical limit |

### `dequantization` — Dequantization

Implements the interpolation between quantum and classical regimes. As $\hbar \to 0$, complex amplitudes collapse to classical probabilities and interference effects vanish.

### `spectral` — Spectral Analysis

Spectral decomposition tools for the Dirichlet form's generator, providing eigenvalue/eigenvector access and spectral projections.

---

## How It Works

The crate follows the mathematical chain of constructions:

```
Dirichlet form 𝓔
    │
    ├── generates ──→ Laplacian Δ
    │                     │
    │                     ├── exponentiates ──→ Heat semigroup e^{-tΔ}
    │                     │                          │
    │                     │                          ├── Cole–Hopf ──→ solves HJ equations
    │                     │                          │
    │                     │                          └── ℏ → 0 ──→ Hopf–Lax (tropical)
    │                     │
    │                     └── spectrum ──→ eigenvalues λ_k
    │
    └── on measures ──→ Wasserstein space (P(X), W₂)
                            │
                            └── gradient flow ──→ JKO scheme
                                                      │
                                                      └── belief dynamics
```

1. **Discrete Laplacian**: For a graph with adjacency $A$ and degree matrix $D$, the combinatorial Laplacian is $L = D - A$. The normalized Laplacian is $\mathcal{L} = D^{-1/2} L D^{-1/2}$.

2. **Heat semigroup**: Diagonalize $L = \Phi \Lambda \Phi^\top$, then $e^{-tL} = \Phi \, \mathrm{diag}(e^{-\lambda_k t}) \, \Phi^\top$.

3. **Cole–Hopf**: Given $\partial_t \psi = \Delta \psi$, substitute $V = -\hbar \log \psi$. Then $V$ satisfies a Hamilton–Jacobi–Bellman equation parameterized by $\hbar$.

4. **Hopf–Lax tropical limit**: As $\hbar \to 0$, $V(x,t) \to \inf_y [V_0(y) + t \cdot L((x-y)/t)]$, where $L = H^*$ is the Legendre dual of the Hamiltonian. This is a min-plus (tropical) convolution.

5. **JKO scheme**: On the metric space $(\mathcal{P}(X), W_2)$, the heat flow is the gradient flow of the relative entropy $\mathcal{F}(\mu) = \int \mu \log \mu \, dm$. The JKO discretization is the implicit Euler scheme on this metric space.

6. **Dequantization**: The parameter $\hbar$ controls the quantum-to-classical transition. At $\hbar > 0$, superposition and interference are present. At $\hbar = 0$, the system is purely classical — all probabilistic mixtures are commuting.

---

## The Math

### Dirichlet Form

A **Dirichlet form** on $L^2(X, m)$ is a closed, symmetric, non-negative bilinear form $\mathcal{E}$ satisfying the Markov property:

$$\mathcal{E}(f, f) \geq 0, \qquad \mathcal{E}(\bar{f}, \bar{f}) \leq \mathcal{E}(f, f)$$

where $\bar{f} = \max(0, \min(1, f))$ is the projection onto $[0,1]$.

### Spectral Theorem

The self-adjoint generator $\Delta$ has a complete orthonormal eigenbasis $\{\phi_k\}$ with eigenvalues $0 = \lambda_0 < \lambda_1 \leq \lambda_2 \leq \cdots$. The **spectral gap** $\lambda_1$ controls the convergence rate of the heat semigroup:

$$\|e^{-t\Delta} f - \bar{f}\|_2 \leq e^{-\lambda_1 t} \|f - \bar{f}\|_2$$

### Cole–Hopf Transform

Given the viscous Hamilton–Jacobi equation:

$$\partial_t V + H(\nabla V) = \hbar \, \Delta V$$

the substitution $\psi = e^{-V/\hbar}$ transforms it into the **heat equation**:

$$\partial_t \psi = \Delta \psi$$

### Hopf–Lax Formula

In the inviscid limit ($\hbar \to 0$), the solution converges to:

$$V(x, t) = \inf_{y \in X} \left[ V_0(y) + t \cdot L\!\left(\frac{x - y}{t}\right) \right]$$

This is a **tropical** (min-plus) convolution: $V = V_0 \oplus_{tL}$ where $\oplus$ denotes the min-plus product.

### JKO Gradient Flow

On $(\mathcal{P}_2(\mathbb{R}^d), W_2)$, the heat equation $\partial_t \mu = \Delta \mu$ is the gradient flow of the Boltzmann entropy $\mathcal{F}(\mu) = \int \mu \log \mu$. The JKO scheme discretizes:

$$\mu^{n+1} \in \arg\min_{\mu} \left\{ \frac{1}{2\tau} W_2^2(\mu, \mu^n) + \mathcal{F}(\mu) \right\}$$

---

## License

MIT License. See [LICENSE](LICENSE) for details.
