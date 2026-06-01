//! Comprehensive tests for the Dequantizable Dirichlet Space.

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    use lau_dirichlet_space::dirichlet_form::{DirichletForm, DirichletFormBuilder};
    use lau_dirichlet_space::laplacian::{Laplacian, SpectralDecomposition};
    use lau_dirichlet_space::heat_semigroup::HeatSemigroup;
    use lau_dirichlet_space::cole_hopf::{ColeHopfTransform, DequantizationPath};
    use lau_dirichlet_space::hopf_lax::{HopfLaxSemigroup, tropical};
    use lau_dirichlet_space::registers::{ObservableRegister, MeasureRegister, SheafSectionRegister, RegisterConversion};
    use lau_dirichlet_space::spectral::SpectralGapAnalysis;
    use lau_dirichlet_space::wasserstein::{WassersteinSpace, BakryEmeryCurvature};
    use lau_dirichlet_space::jko::JKOGradientFlow;
    use lau_dirichlet_space::dequantization::DequantizationAxis;
    use lau_dirichlet_space::belief_dynamics::BeliefDynamics;
    use lau_dirichlet_space::core::{MetricMeasureSpace, HBar, DirichletSpace};

    // ===== Helper: standard test graph (K₃, complete graph on 3 vertices) =====

    fn k3_form() -> DirichletForm {
        DirichletFormBuilder::complete(3).build()
    }

    fn path4_form() -> DirichletForm {
        DirichletFormBuilder::path(4).build()
    }

    fn cycle4_form() -> DirichletForm {
        DirichletFormBuilder::cycle(4).build()
    }

    // ===== Core Tests =====

    #[test]
    fn test_hbar_classical() {
        let h = HBar::classical();
        assert_eq!(h.value, 1.0);
        assert!(h.is_classical());
        assert!(!h.is_tropical());
    }

    #[test]
    fn test_hbar_tropical() {
        let h = HBar::tropical();
        assert_eq!(h.value, 0.0);
        assert!(h.is_tropical());
    }

    #[test]
    fn test_hbar_range() {
        let h = HBar::new(0.5);
        assert_eq!(h.value, 0.5);
        assert!(!h.is_classical());
        assert!(!h.is_tropical());
    }

    #[test]
    #[should_panic]
    fn test_hbar_out_of_range() {
        HBar::new(1.5);
    }

    #[test]
    fn test_metric_space_trivial() {
        let space = MetricMeasureSpace::trivial();
        assert_eq!(space.n, 1);
        assert_eq!(space.total_measure(), 1.0);
    }

    #[test]
    fn test_metric_space_uniform() {
        let space = MetricMeasureSpace::uniform(5);
        assert_eq!(space.n, 5);
        assert_eq!(space.total_measure(), 5.0);
    }

    // ===== Dirichlet Form Tests =====

    #[test]
    fn test_dirichlet_form_complete_graph() {
        let form = k3_form();
        assert_eq!(form.n, 3);
        assert!(form.is_symmetric());
        assert!(form.is_positive_semidefinite());
        assert!(form.is_markovian());
        assert!(form.is_dirichlet_form());
    }

    #[test]
    fn test_dirichlet_form_path_graph() {
        let form = path4_form();
        assert!(form.is_dirichlet_form());
    }

    #[test]
    fn test_dirichlet_form_cycle_graph() {
        let form = cycle4_form();
        assert!(form.is_dirichlet_form());
    }

    #[test]
    fn test_dirichlet_form_energy_nonnegative() {
        let form = k3_form();
        let f = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let energy = form.energy(&f);
        assert!(energy >= -1e-10, "Energy should be non-negative, got {}", energy);
    }

    #[test]
    fn test_dirichlet_form_bilinear() {
        let form = k3_form();
        let f = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let g = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        let ef_g = form.eval(&f, &g);
        let eg_f = form.eval(&g, &f);
        assert_relative_eq!(ef_g, eg_f, epsilon = 1e-10);
    }

    #[test]
    fn test_dirichlet_form_annihilates_constants() {
        let form = k3_form();
        let ones = DVector::from_element(3, 1.0);
        let energy = form.energy(&ones);
        assert_relative_eq!(energy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dirichlet_form_builder_edge() {
        let form = DirichletFormBuilder::new(3)
            .edge(0, 1, 2.0)
            .edge(1, 2, 1.0)
            .build();
        assert!(form.is_dirichlet_form());
    }

    #[test]
    fn test_dirichlet_form_closed() {
        let form = k3_form();
        assert!(form.is_closed()); // Always true in finite dim
    }

    #[test]
    fn test_kernel_dimension_connected() {
        let form = k3_form();
        assert_eq!(form.kernel_dimension(), 1); // Only constants in kernel
    }

    // ===== Laplacian Tests =====

    #[test]
    fn test_laplacian_annihilates_constants() {
        let form = k3_form();
        let lap = Laplacian::from_dirichlet_form(&form);
        assert!(lap.annihilates_constants());
    }

    #[test]
    fn test_laplacian_trace() {
        let form = k3_form();
        let lap = Laplacian::from_dirichlet_form(&form);
        // For K₃, L = [[2,-1,-1],[-1,2,-1],[-1,-1,2]], trace = 6
        assert_eq!(lap.trace(), 6.0);
    }

    #[test]
    fn test_spectral_decomposition_k3() {
        let form = k3_form();
        let mut lap = Laplacian::from_dirichlet_form(&form);
        let spec = lap.spectrum();

        // Eigenvalues of K₃: 0, 3, 3
        assert_relative_eq!(spec.eigenvalue(0), 0.0, epsilon = 1e-8);
        assert_relative_eq!(spec.eigenvalue(1), 3.0, epsilon = 1e-8);
        assert_relative_eq!(spec.eigenvalue(2), 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_spectral_gap_laplacian_k3() {
        let form = k3_form();
        let mut lap = Laplacian::from_dirichlet_form(&form);
        let spec = lap.spectrum();
        assert_relative_eq!(spec.spectral_gap(), 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_spectral_reconstruction() {
        let form = k3_form();
        let spec = SpectralDecomposition::from_laplacian(&form.laplacian);
        let f = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let coeffs = DVector::from_vec(
            (0..3).map(|k| spec.eigenvector(k).dot(&f)).collect()
        );
        let reconstructed = spec.reconstruct(&coeffs);
        assert_relative_eq!(reconstructed[0], f[0], epsilon = 1e-8);
        assert_relative_eq!(reconstructed[1], f[1], epsilon = 1e-8);
        assert_relative_eq!(reconstructed[2], f[2], epsilon = 1e-8);
    }

    // ===== Heat Semigroup Tests =====

    #[test]
    fn test_heat_semigroup_preserves_mass() {
        let form = k3_form();
        let mut heat = HeatSemigroup::from_dirichlet_form(&form);
        let f = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(heat.preserves_mass(&f, 0.5));
    }

    #[test]
    fn test_heat_semigroup_preserves_positivity() {
        let form = k3_form();
        let mut heat = HeatSemigroup::from_dirichlet_form(&form);
        assert!(heat.preserves_positivity(0.5));
    }

    #[test]
    fn test_heat_semigroup_is_contraction() {
        let form = k3_form();
        let mut heat = HeatSemigroup::from_dirichlet_form(&form);
        assert!(heat.is_contraction(0.5));
    }

    #[test]
    fn test_heat_semigroup_property() {
        let form = k3_form();
        let mut heat = HeatSemigroup::from_dirichlet_form(&form);
        let f = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(heat.verify_semigroup_property(&f, 0.3, 0.2));
    }

    #[test]
    fn test_heat_semigroup_equilibrium() {
        let form = k3_form();
        let mut heat = HeatSemigroup::from_dirichlet_form(&form);
        let f = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let eq = heat.equilibrium(&f);
        // Should be the average: [2, 2, 2]
        assert_relative_eq!(eq[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(eq[1], 2.0, epsilon = 1e-8);
        assert_relative_eq!(eq[2], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_heat_semigroup_convergence() {
        let form = k3_form();
        let mut heat = HeatSemigroup::from_dirichlet_form(&form);
        let f = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let at_large_t = heat.evolve(&f, 10.0);
        // Should converge to average = [1/3, 1/3, 1/3]
        assert_relative_eq!(at_large_t[0], 1.0 / 3.0, epsilon = 1e-4);
        assert_relative_eq!(at_large_t[1], 1.0 / 3.0, epsilon = 1e-4);
        assert_relative_eq!(at_large_t[2], 1.0 / 3.0, epsilon = 1e-4);
    }

    // ===== Cole-Hopf Tests =====

    #[test]
    fn test_cole_hopf_forward() {
        let ch = ColeHopfTransform::classical();
        let u = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v = ch.forward(&u).unwrap();
        assert_relative_eq!(v[0], 0.0, epsilon = 1e-10); // -1·log(1) = 0
        assert_relative_eq!(v[1], -2.0_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(v[2], -3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_cole_hopf_inverse() {
        let ch = ColeHopfTransform::classical();
        let u = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v = ch.forward(&u).unwrap();
        let u_back = ch.inverse(&v);
        assert_relative_eq!(u_back[0], u[0], epsilon = 1e-10);
        assert_relative_eq!(u_back[1], u[1], epsilon = 1e-10);
        assert_relative_eq!(u_back[2], u[2], epsilon = 1e-10);
    }

    #[test]
    fn test_cole_hopf_roundtrip() {
        let ch = ColeHopfTransform::new(0.5);
        let u = DVector::from_vec(vec![0.5, 1.0, 2.0]);
        assert!(ch.verify_roundtrip(&u));
    }

    #[test]
    fn test_cole_hopf_positive_required() {
        let ch = ColeHopfTransform::classical();
        let u = DVector::from_vec(vec![-1.0, 2.0, 3.0]);
        assert!(ch.forward(&u).is_err());
    }

    #[test]
    fn test_dequantization_path() {
        let path = DequantizationPath::uniform(5);
        assert_eq!(path.hbar_values.len(), 6);
        assert_eq!(path.hbar_values[0], 1.0);
        assert_eq!(path.hbar_values[5], 0.0);
    }

    // ===== Hopf-Lax / Tropical Tests =====

    #[test]
    fn test_tropical_arithmetic() {
        assert_eq!(tropical::add(3.0, 5.0), 3.0); // min
        assert_eq!(tropical::mul(3.0, 5.0), 8.0); // addition
        assert_eq!(tropical::one(), 0.0); // multiplicative identity
        assert_eq!(tropical::zero(), f64::INFINITY); // additive identity
    }

    #[test]
    fn test_tropical_mat_vec() {
        let mat = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let vec = DVector::from_vec(vec![0.0, 0.0]); // tropical one
        let result = tropical::mat_vec(&mat, &vec);
        // Row 0: min(1+0, 2+0) = 1
        // Row 1: min(3+0, 4+0) = 3
        assert_relative_eq!(result[0], 1.0);
        assert_relative_eq!(result[1], 3.0);
    }

    #[test]
    fn test_hopf_lax_evolve() {
        let cost = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 2.0,
            1.0, 0.0, 1.0,
            2.0, 1.0, 0.0,
        ]);
        let hl = HopfLaxSemigroup::from_cost(cost);
        let f = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let evolved = hl.evolve(&f, 1.0);
        // S₁f(0) = min(f(0)+1·c(0,0), f(1)+1·c(0,1), f(2)+1·c(0,2))
        //        = min(0+0, 1+1, 2+2) = 0
        assert_relative_eq!(evolved[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hopf_lax_contraction() {
        let cost = DMatrix::from_row_slice(3, 3, &[
            0.0, 1.0, 2.0,
            1.0, 0.0, 1.0,
            2.0, 1.0, 0.0,
        ]);
        let hl = HopfLaxSemigroup::from_cost(cost);
        let f = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let g = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(hl.is_contraction(&f, &g, 0.5));
    }

    #[test]
    fn test_hopf_lax_tropical_kernel() {
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let hl = HopfLaxSemigroup::from_cost(cost);
        let kernel = hl.tropical_kernel(2.0);
        assert_relative_eq!(kernel[(0, 1)], 2.0);
        assert_relative_eq!(kernel[(0, 0)], 0.0);
    }

    // ===== Register Tests =====

    #[test]
    fn test_observable_register() {
        let obs = ObservableRegister::new(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(obs.n, 3);
        assert_relative_eq!(obs.l2_norm(), 14.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_observable_delta() {
        let delta = ObservableRegister::delta(3, 1);
        assert_relative_eq!(delta.values[0], 0.0);
        assert_relative_eq!(delta.values[1], 1.0);
        assert_relative_eq!(delta.values[2], 0.0);
    }

    #[test]
    fn test_measure_uniform() {
        let mu = MeasureRegister::uniform(4);
        assert!(mu.is_valid());
        assert_relative_eq!(mu.measure[0], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_measure_dirac() {
        let mu = MeasureRegister::dirac(3, 0);
        assert!(mu.is_valid());
        assert_relative_eq!(mu.measure[0], 1.0);
        assert_relative_eq!(mu.measure[1], 0.0);
    }

    #[test]
    fn test_measure_entropy() {
        let mu = MeasureRegister::uniform(2);
        let entropy = mu.entropy();
        assert_relative_eq!(entropy, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_measure_kl_divergence() {
        let mu = MeasureRegister::uniform(2);
        let kl = mu.kl_divergence(&mu);
        assert_relative_eq!(kl, 0.0, epsilon = 1e-10); // D_KL(p||p) = 0
    }

    #[test]
    fn test_observable_to_belief() {
        let obs = ObservableRegister::new(DVector::from_vec(vec![0.0, 0.0, 0.0]));
        let belief = obs.to_belief(1.0);
        assert!(belief.is_valid());
        // All equal → uniform
        assert_relative_eq!(belief.measure[0], 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_measure_to_observable() {
        let mu = MeasureRegister::uniform(4);
        let obs = mu.to_observable();
        // -log(0.25) = log(4)
        assert_relative_eq!(obs.values[0], 4.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_sheaf_trivial() {
        let data = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let sheaf = SheafSectionRegister::trivial(data);
        assert!(sheaf.is_globally_consistent(1e-10));
        assert_eq!(sheaf.h0_dimension(1e-10), 3);
    }

    #[test]
    fn test_sheaf_consistency() {
        let sections = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![1.0, 2.0]),
        ];
        let sheaf = SheafSectionRegister::from_cover(sections);
        assert!(sheaf.is_globally_consistent(1e-10));
    }

    #[test]
    fn test_sheaf_inconsistent() {
        let sections = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        ];
        let sheaf = SheafSectionRegister::from_cover(sections);
        assert!(!sheaf.is_globally_consistent(0.1));
    }

    #[test]
    fn test_sheaf_global_section() {
        let sections = vec![
            DVector::from_vec(vec![1.0, 3.0]),
            DVector::from_vec(vec![3.0, 5.0]),
        ];
        let sheaf = SheafSectionRegister::from_cover(sections);
        let global = sheaf.global_section();
        assert_relative_eq!(global[0], 2.0);
        assert_relative_eq!(global[1], 4.0);
    }

    #[test]
    fn test_register_conversion() {
        let obs = ObservableRegister::new(DVector::from_vec(vec![0.0, 0.0, 0.0]));
        let measure = obs.to_measure();
        assert!(measure.is_valid());
    }

    // ===== Spectral Gap Tests =====

    #[test]
    fn test_spectral_gap_k3() {
        let form = k3_form();
        let analysis = SpectralGapAnalysis::from_dirichlet_form(&form);
        assert!(analysis.is_connected);
        assert_relative_eq!(analysis.lambda1, 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_spectral_gap_convergence_rate() {
        let form = k3_form();
        let analysis = SpectralGapAnalysis::from_dirichlet_form(&form);
        let rate = analysis.convergence_rate(1.0);
        assert_relative_eq!(rate, (-3.0_f64).exp(), epsilon = 1e-8);
    }

    #[test]
    fn test_spectral_gap_poincare() {
        let form = k3_form();
        let analysis = SpectralGapAnalysis::from_dirichlet_form(&form);
        let f = DVector::from_vec(vec![1.0, -1.0, 0.0]);
        assert!(analysis.verify_poincare(&form, &f));
    }

    #[test]
    fn test_cheeger_bound() {
        let form = k3_form();
        let analysis = SpectralGapAnalysis::from_dirichlet_form(&form);
        assert!(analysis.cheeger_bound() > 0.0);
    }

    #[test]
    fn test_relaxation_time() {
        let form = k3_form();
        let analysis = SpectralGapAnalysis::from_dirichlet_form(&form);
        assert_relative_eq!(analysis.relaxation_time(), 1.0 / 3.0, epsilon = 1e-8);
    }

    // ===== Bakry-Émery / Wasserstein Tests =====

    #[test]
    fn test_bakry_emery_curvature() {
        let form = k3_form();
        let be = BakryEmeryCurvature::from_dirichlet_form(&form);
        assert!(be.verify_cd_condition());
        assert!(be.curvature_bound > 0.0);
    }

    #[test]
    fn test_bakry_emery_contraction() {
        let be = BakryEmeryCurvature::with_bound(1.0, 3);
        let rate = be.contraction_rate(1.0);
        assert_relative_eq!(rate, 1.0 / std::f64::consts::E, epsilon = 1e-10);
    }

    #[test]
    fn test_log_sobolev_constant() {
        let form = k3_form();
        let be = BakryEmeryCurvature::from_dirichlet_form(&form);
        let lsi = be.log_sobolev_constant();
        assert!(lsi > 0.0);
        assert!(lsi.is_finite());
    }

    #[test]
    fn test_wasserstein_from_form() {
        let form = k3_form();
        let w2 = WassersteinSpace::from_dirichlet_form(&form);
        assert_eq!(w2.n, 3);
    }

    #[test]
    fn test_wasserstein_w2_same_measure() {
        let form = k3_form();
        let w2 = WassersteinSpace::from_dirichlet_form(&form);
        let mu = MeasureRegister::uniform(3);
        let d = w2.w2(&mu.measure, &mu.measure);
        assert_relative_eq!(d, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_wasserstein_symmetry() {
        let form = k3_form();
        let w2 = WassersteinSpace::from_dirichlet_form(&form);
        let mu = MeasureRegister::dirac(3, 0);
        let nu = MeasureRegister::dirac(3, 1);
        let d_forward = w2.w2(&mu.measure, &nu.measure);
        let d_backward = w2.w2(&nu.measure, &mu.measure);
        assert_relative_eq!(d_forward, d_backward, epsilon = 1e-4);
    }

    // ===== JKO Tests =====

    #[test]
    fn test_jko_single_step() {
        let form = k3_form();
        let mu0 = MeasureRegister::dirac(3, 0);
        let mut jko = JKOGradientFlow::from_dirichlet_form(&form, 0.1)
            .with_initial(&mu0.measure);
        let next = jko.step();
        assert_eq!(next.len(), 3);
        let total: f64 = next.sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_jko_convergence() {
        let form = k3_form();
        let mu0 = MeasureRegister::dirac(3, 0);
        let mut jko = JKOGradientFlow::from_dirichlet_form(&form, 0.5)
            .with_initial(&mu0.measure);
        jko.run(20);
        // Should be close to uniform
        let current = jko.current();
        assert_relative_eq!(current[0], 1.0 / 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_jko_preserves_total_mass() {
        let form = k3_form();
        let mu0 = MeasureRegister::dirac(3, 0);
        let mut jko = JKOGradientFlow::from_dirichlet_form(&form, 0.1)
            .with_initial(&mu0.measure);
        jko.run(10);
        for mu in &jko.trajectory {
            let total: f64 = mu.sum();
            assert_relative_eq!(total, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_jko_entropy_trajectory() {
        let form = k3_form();
        let mu0 = MeasureRegister::dirac(3, 0);
        let mut jko = JKOGradientFlow::from_dirichlet_form(&form, 0.1)
            .with_initial(&mu0.measure);
        jko.run(5);
        let entropies = jko.entropy_trajectory();
        assert_eq!(entropies.len(), 6); // initial + 5 steps
    }

    // ===== Dequantization Tests =====

    #[test]
    fn test_dequantization_axis_hbar_values() {
        let form = k3_form();
        let axis = DequantizationAxis::from_dirichlet_form(form, 10);
        let values = axis.hbar_values();
        assert_eq!(values.len(), 11);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[10], 0.0);
    }

    #[test]
    fn test_dequantization_spectral_degeneration() {
        let form = k3_form();
        let axis = DequantizationAxis::from_dirichlet_form(form, 5);
        let deg = axis.spectral_degeneration();
        assert_eq!(deg.n, 3);
        assert!(deg.classical_eigenvalues[0] >= -1e-10);
    }

    #[test]
    fn test_dequantization_regime_comparison() {
        let form = k3_form();
        let axis = DequantizationAxis::from_dirichlet_form(form, 5);
        let f = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let comp = axis.compare_regimes(&f, 0.5);
        assert_eq!(comp.classical.len(), 3);
        assert_eq!(comp.tropical.len(), 3);
        assert!(comp.l2_difference >= 0.0);
    }

    #[test]
    fn test_dequantization_evolve_axis() {
        let form = k3_form();
        let axis = DequantizationAxis::from_dirichlet_form(form, 3);
        let f = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let results = axis.evolve_axis(&f, 0.5);
        assert_eq!(results.len(), 4); // 0..=3
    }

    // ===== Belief Dynamics Tests =====

    #[test]
    fn test_belief_evolution() {
        let form = k3_form();
        let bd = BeliefDynamics::from_dirichlet_form(&form);
        let mu0 = MeasureRegister::dirac(3, 0);
        let mu_t = bd.evolve(&mu0.measure, 1.0);
        let total: f64 = mu_t.sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-6);
        assert!(mu_t.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_belief_convergence_to_stationary() {
        let form = k3_form();
        let bd = BeliefDynamics::from_dirichlet_form(&form);
        let mu0 = MeasureRegister::dirac(3, 0);
        let mu_t = bd.evolve(&mu0.measure, 100.0);
        // Should be close to uniform [1/3, 1/3, 1/3]
        assert_relative_eq!(mu_t[0], 1.0 / 3.0, epsilon = 1e-2);
        assert_relative_eq!(mu_t[1], 1.0 / 3.0, epsilon = 1e-2);
        assert_relative_eq!(mu_t[2], 1.0 / 3.0, epsilon = 1e-2);
    }

    #[test]
    fn test_belief_entropy_nonincreasing() {
        let form = k3_form();
        let bd = BeliefDynamics::from_dirichlet_form(&form);
        let mu0 = DVector::from_vec(vec![0.5, 0.3, 0.2]);
        let times: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();
        assert!(bd.entropy_nonincreasing(&mu0, &times));
    }

    #[test]
    fn test_belief_fisher_information() {
        let form = k3_form();
        let bd = BeliefDynamics::from_dirichlet_form(&form);
        let mu = MeasureRegister::uniform(3);
        let fisher = bd.fisher_information(&mu.measure);
        assert_relative_eq!(fisher, 0.0, epsilon = 1e-8); // At stationary
    }

    #[test]
    fn test_belief_stationary_is_fixed_point() {
        let form = k3_form();
        let bd = BeliefDynamics::from_dirichlet_form(&form);
        let mu_t = bd.evolve(&bd.stationary, 1.0);
        for i in 0..3 {
            assert_relative_eq!(mu_t[i], bd.stationary[i], epsilon = 1e-6);
        }
    }

    // ===== Integration Tests =====

    #[test]
    fn test_full_pipeline() {
        // Build a graph → Dirichlet form → Laplacian → heat semigroup → JKO → dequantization
        let form = DirichletFormBuilder::path(4).build();
        assert!(form.is_dirichlet_form());

        let mut lap = Laplacian::from_dirichlet_form(&form);
        assert!(lap.annihilates_constants());

        let spec = lap.spectrum();
        assert!(spec.spectral_gap() > 0.0);

        let mut heat = HeatSemigroup::from_dirichlet_form(&form);
        let f = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let evolved = heat.evolve(&f, 1.0);
        assert!(evolved.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_cole_hopf_with_heat_flow() {
        let form = k3_form();
        let mut heat = HeatSemigroup::from_dirichlet_form(&form);
        let f = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let u_t = heat.evolve(&f, 0.1);
        let ch = ColeHopfTransform::new(0.5);
        let v_t = ch.forward(&u_t).unwrap();
        assert!(v_t.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_measure_push_forward() {
        let form = k3_form();
        let spec = SpectralDecomposition::from_laplacian(&form.laplacian);
        let t = 0.1;
        let n = 3;
        let mut kernel = DMatrix::zeros(n, n);
        for k in 0..n {
            let phi = spec.eigenvector(k);
            let decay = ((-t * spec.eigenvalue(k)) as f64).exp();
            for i in 0..n {
                for j in 0..n {
                    kernel[(i, j)] += decay * phi[i] * phi[j];
                }
            }
        }
        let mu = MeasureRegister::dirac(3, 0);
        let pushed = mu.push_forward(&kernel);
        let total: f64 = pushed.measure.sum();
        assert_relative_eq!(total, 1.0, epsilon = 0.1); // Rough normalization
    }

    #[test]
    fn test_renyi_entropy() {
        let mu = MeasureRegister::uniform(4);
        // Rényi entropy of order 2 = -log(Σ p²) = -log(4 * 1/16) = log(4)
        let h2 = mu.renyi_entropy(2.0);
        assert_relative_eq!(h2, 4.0_f64.ln(), epsilon = 1e-8);

        // Shannon = Rényi with α→1
        let h_shannon = mu.entropy();
        let h_renyi1 = mu.renyi_entropy(1.0);
        assert_relative_eq!(h_shannon, h_renyi1, epsilon = 1e-8);
    }

    #[test]
    fn test_dequantization_path_apply() {
        let path = DequantizationPath::uniform(3);
        let u = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let results = path.apply(&u);
        assert_eq!(results.len(), 4); // 0..=3
    }
}
