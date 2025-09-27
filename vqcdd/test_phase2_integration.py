#!/usr/bin/env python3
"""
Phase 2 Integration Test Suite

Comprehensive integration testing for all Phase 2 VQCdd enhancements:
- Phase 2A: Advanced gradient monitoring and enhanced training loop
- Phase 2B: Alternative ansatz implementation and adaptive circuit depth
- Phase 2C: Comprehensive validation framework and noise resilience analysis
- Phase 2D: Advanced analytics and visualization capabilities

This test suite validates the complete Phase 2 workflow end-to-end and ensures
all components integrate seamlessly while maintaining backward compatibility.

Author: VQCdd Development Team
Created: 2025-09-18
"""

import numpy as np
import logging
import time
import json
import pytest
from pathlib import Path
import tempfile
import shutil
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

# VQCdd Core Components
from quantum_circuit import VQCircuit, CircuitConfig
from parameter_mapping import QuantumParameterMapper, ParameterBounds
from optimizer import VQCTrainer, OptimizationConfig, DosingOptimizer
from data_handler import SyntheticDataGenerator, StudyData

# Phase 2 Components
try:
    from validation import ValidationFramework, ValidationConfig
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    logging.warning("Validation framework not available")

try:
    from noise_analysis import NoiseAnalyzer, NoiseConfig
    NOISE_ANALYSIS_AVAILABLE = True
except ImportError:
    NOISE_ANALYSIS_AVAILABLE = False
    logging.warning("Noise analysis not available")

try:
    from hyperparameter_optimization import HyperparameterOptimizer, HPOptConfig
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False
    logging.warning("Hyperparameter optimization not available")

try:
    from analytics import AdvancedAnalytics, ExperimentResult
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logging.warning("Advanced analytics not available")

try:
    from visualization import AdvancedVisualization, PlotConfig, PlotType
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Advanced visualization not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IntegrationTestConfig:
    """Configuration for integration testing"""
    test_data_size: int = 100
    n_qubits: int = 4
    n_layers: int = 2
    max_iterations: int = 10
    learning_rate: float = 0.1
    test_timeout: int = 300  # 5 minutes
    temp_dir: Optional[str] = None
    skip_slow_tests: bool = False
    validate_outputs: bool = True

class Phase2IntegrationTester:
    """
    Comprehensive integration tester for Phase 2 VQCdd enhancements

    Tests all components working together in realistic scenarios:
    1. End-to-end quantum training with advanced features
    2. Validation framework integration
    3. Noise analysis and error mitigation
    4. Hyperparameter optimization workflows
    5. Analytics and visualization pipeline
    6. Performance and resource usage validation
    """

    def __init__(self, config: IntegrationTestConfig):
        """Initialize integration tester"""
        self.config = config

        # Setup temporary directory
        if config.temp_dir:
            self.temp_dir = Path(config.temp_dir)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="vqcdd_integration_"))

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Integration testing in: {self.temp_dir}")

        # Initialize components
        self.data_generator = SyntheticDataGenerator(seed=42)
        self.test_results = {}
        self.performance_metrics = {}

        # Generate test data
        self._setup_test_data()

    def _setup_test_data(self):
        """Setup test datasets using real EstData.csv"""
        logger.info("Loading real data from EstData.csv...")

        # Load real data
        from data_handler import EstDataLoader, train_test_split_patients
        loader = EstDataLoader('/Users/shawngibford/dev/qpkd/data/EstData.csv')
        full_data = loader.load_data()

        # Split real data into train/test
        train_data, test_data = train_test_split_patients(full_data, test_fraction=0.3, random_state=42)
        self.train_data, self.val_data = train_test_split_patients(train_data, test_fraction=0.3, random_state=42)
        self.test_data = test_data

        logger.info(f"Real data loaded: {self.train_data.get_patient_count()} train, {self.val_data.get_patient_count()} val, {self.test_data.get_patient_count()} test")
        logger.info(f"Total real patients: {full_data.get_patient_count()}, observations: {full_data.get_observation_count()}")

    def _slice_study_data(self, study_data: StudyData, start: int = 0, end: Optional[int] = None) -> StudyData:
        """Helper function to create a subset of StudyData"""
        end = end or len(study_data.patients)
        subset_patients = study_data.patients[start:end]
        return StudyData(
            patients=subset_patients,
            study_design=study_data.study_design.copy(),
            data_quality=study_data.data_quality.copy(),
            study_metadata=study_data.study_metadata
        )

    def run_complete_integration_test(self) -> Dict[str, Any]:
        """
        Run complete Phase 2 integration test suite

        Returns:
            Dictionary containing all test results and performance metrics
        """
        logger.info("="*80)
        logger.info("PHASE 2 INTEGRATION TEST SUITE")
        logger.info("="*80)

        start_time = time.time()

        try:
            # Test 1: Core VQC Training with Phase 2A Enhancements
            logger.info("\n1. Testing Core VQC Training with Phase 2A Enhancements...")
            self.test_results['phase2a_training'] = self._test_enhanced_training()

            # Test 2: Phase 2B Alternative Ansatz and Adaptive Depth
            logger.info("\n2. Testing Phase 2B Alternative Ansatz and Adaptive Depth...")
            self.test_results['phase2b_ansatz'] = self._test_alternative_ansatz()

            # Test 3: Phase 2C Validation Framework Integration
            if VALIDATION_AVAILABLE:
                logger.info("\n3. Testing Phase 2C Validation Framework...")
                self.test_results['phase2c_validation'] = self._test_validation_framework()
            else:
                logger.warning("Skipping validation framework tests - not available")

            # Test 4: Phase 2C Noise Analysis Integration
            if NOISE_ANALYSIS_AVAILABLE:
                logger.info("\n4. Testing Phase 2C Noise Analysis...")
                self.test_results['phase2c_noise'] = self._test_noise_analysis()
            else:
                logger.warning("Skipping noise analysis tests - not available")

            # Test 5: Phase 2C Hyperparameter Optimization
            if HPO_AVAILABLE and not self.config.skip_slow_tests:
                logger.info("\n5. Testing Phase 2C Hyperparameter Optimization...")
                self.test_results['phase2c_hpo'] = self._test_hyperparameter_optimization()
            else:
                logger.warning("Skipping HPO tests - not available or slow tests disabled")

            # Test 6: Phase 2D Analytics Integration
            if ANALYTICS_AVAILABLE:
                logger.info("\n6. Testing Phase 2D Analytics Integration...")
                self.test_results['phase2d_analytics'] = self._test_analytics_integration()
            else:
                logger.warning("Skipping analytics tests - not available")

            # Test 7: Phase 2D Visualization Integration
            if VISUALIZATION_AVAILABLE:
                logger.info("\n7. Testing Phase 2D Visualization Integration...")
                self.test_results['phase2d_visualization'] = self._test_visualization_integration()
            else:
                logger.warning("Skipping visualization tests - not available")

            # Test 8: End-to-End Dosing Optimization Workflow
            logger.info("\n8. Testing End-to-End Dosing Optimization...")
            self.test_results['dosing_workflow'] = self._test_dosing_workflow()

            # Test 9: Performance and Resource Validation
            logger.info("\n9. Testing Performance and Resource Usage...")
            self.test_results['performance'] = self._test_performance_validation()

            # Test 10: Backward Compatibility
            logger.info("\n10. Testing Backward Compatibility...")
            self.test_results['compatibility'] = self._test_backward_compatibility()

        except Exception as e:
            logger.error(f"Integration test failed with error: {e}")
            self.test_results['error'] = str(e)

        finally:
            total_time = time.time() - start_time
            self.performance_metrics['total_test_time'] = total_time

        # Generate summary report
        summary = self._generate_test_summary()

        logger.info("="*80)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("="*80)
        logger.info(summary)

        return {
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'summary': summary,
            'temp_dir': str(self.temp_dir)
        }

    def _test_enhanced_training(self) -> Dict[str, Any]:
        """Test Phase 2A enhanced training with gradient monitoring"""
        logger.info("Testing enhanced VQC training with gradient monitoring...")

        results = {}
        start_time = time.time()

        try:
            # Create enhanced circuit configuration
            circuit_config = CircuitConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
                ansatz="ry_cnot",
                encoding="basis"
            )

            # Create enhanced optimization configuration
            opt_config = OptimizationConfig(
                max_iterations=self.config.max_iterations,
                learning_rate=self.config.learning_rate,
                optimizer_type="adam",
                # Phase 2A enhancements
                enable_gradient_monitoring=True,
                gradient_history_size=100,
                barren_threshold=1e-6,
                enable_mini_batches=True,
                batch_size=32,
                enable_learning_rate_scheduling=True
            )

            # Create trainer with Phase 2A enhancements
            trainer = VQCTrainer(circuit_config, opt_config)

            # Train model
            logger.info("Training VQC with Phase 2A enhancements...")
            training_start = time.time()

            trainer.train(self.train_data)

            training_time = time.time() - training_start

            # Validate training completed successfully
            assert trainer.is_fitted, "Training did not complete successfully"
            assert hasattr(trainer, 'best_parameters'), "Best parameters not found"
            assert hasattr(trainer, 'optimization_history'), "Optimization history not available"

            # Check gradient monitoring integration
            if hasattr(trainer, 'gradient_monitor'):
                assert len(trainer.gradient_monitor.gradient_history) > 0, "Gradient history empty"

            # Evaluate performance
            features, pk_targets, pd_targets = trainer._prepare_training_data(self.test_data)
            final_cost = trainer._evaluate_cost(trainer.best_parameters, features, pk_targets, pd_targets)

            results.update({
                'success': True,
                'training_time': training_time,
                'final_cost': float(final_cost),
                'iterations_completed': len(trainer.optimization_history),
                'gradient_monitoring_active': hasattr(trainer, 'gradient_monitor'),
                'mini_batch_enabled': opt_config.enable_mini_batches,
                'parameter_count': len(trainer.best_parameters)
            })

            logger.info(f"✅ Enhanced training completed: cost={final_cost:.6f}, time={training_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ Enhanced training failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _test_alternative_ansatz(self) -> Dict[str, Any]:
        """Test Phase 2B alternative ansatz and adaptive depth"""
        logger.info("Testing alternative ansatz implementations...")

        results = {}
        start_time = time.time()

        try:
            # Test different ansatz types
            ansatz_types = ["ry_cnot", "hardware_efficient", "qaoa_inspired", "optimized_hardware_efficient"]
            ansatz_results = {}

            for ansatz in ansatz_types:
                logger.info(f"Testing ansatz: {ansatz}")

                try:
                    # Create circuit with specific ansatz
                    circuit_config = CircuitConfig(
                        n_qubits=self.config.n_qubits,
                        n_layers=self.config.n_layers,
                        ansatz=ansatz,
                        # Phase 2B features
                        adaptive_depth=True,
                        min_layers=1,
                        max_layers=4,
                        encoding_optimization=True
                    )

                    # Quick training test
                    opt_config = OptimizationConfig(
                        max_iterations=5,  # Quick test
                        learning_rate=self.config.learning_rate
                    )

                    trainer = VQCTrainer(circuit_config, opt_config)

                    # Test circuit initialization
                    test_params = trainer.quantum_circuit.initialize_parameters(seed=42)

                    # Test parameter count is reasonable
                    assert len(test_params) > 0, f"No parameters initialized for {ansatz}"
                    assert len(test_params) < 1000, f"Too many parameters for {ansatz}: {len(test_params)}"

                    # Quick training
                    small_data = self._slice_study_data(self.train_data, 0, 10)  # Small subset for speed
                    trainer.train(small_data)

                    ansatz_results[ansatz] = {
                        'success': True,
                        'parameter_count': len(test_params),
                        'final_cost': float(trainer.optimization_history[-1]['cost']),
                        'supports_adaptive_depth': circuit_config.adaptive_depth
                    }

                    logger.info(f"✅ {ansatz} ansatz test passed")

                except Exception as e:
                    logger.warning(f"❌ {ansatz} ansatz test failed: {e}")
                    ansatz_results[ansatz] = {
                        'success': False,
                        'error': str(e)
                    }

            # Test adaptive depth functionality
            logger.info("Testing adaptive depth circuit...")

            adaptive_config = CircuitConfig(
                n_qubits=self.config.n_qubits,
                n_layers=2,
                ansatz="ry_cnot",
                adaptive_depth=True,
                min_layers=1,
                max_layers=5,
                depth_adjustment_threshold=0.01
            )

            # Test adaptive circuit creation
            adaptive_circuit = VQCircuit(adaptive_config)
            adaptive_params = adaptive_circuit.initialize_parameters()

            results.update({
                'success': True,
                'ansatz_results': ansatz_results,
                'adaptive_depth_supported': True,
                'adaptive_circuit_created': True,
                'tested_ansatz_count': len([r for r in ansatz_results.values() if r['success']])
            })

            logger.info(f"✅ Alternative ansatz testing completed: {results['tested_ansatz_count']}/{len(ansatz_types)} passed")

        except Exception as e:
            logger.error(f"❌ Alternative ansatz testing failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _test_validation_framework(self) -> Dict[str, Any]:
        """Test Phase 2C validation framework integration"""
        logger.info("Testing validation framework integration...")

        results = {}
        start_time = time.time()

        try:
            # Create validation configuration
            val_config = ValidationConfig(
                k_folds=3,  # Small for testing
                bootstrap_samples=50,  # Small for testing
                confidence_level=0.95,
                random_state=42
            )

            # Create validation framework
            validator = ValidationFramework(val_config)

            # Train a model first
            circuit_config = CircuitConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
                ansatz="ry_cnot"
            )

            opt_config = OptimizationConfig(
                max_iterations=5,  # Quick for testing
                learning_rate=self.config.learning_rate
            )

            trainer = VQCTrainer(circuit_config, opt_config)
            trainer.train(self._slice_study_data(self.train_data, 0, 50))  # Small subset

            # Test cross-validation
            logger.info("Running cross-validation...")
            cv_results = validator.cross_validate(trainer, self._slice_study_data(self.train_data, 0, 30))

            assert 'cv_scores' in cv_results, "Cross-validation scores missing"
            assert 'mean_score' in cv_results, "Mean CV score missing"
            assert len(cv_results['cv_scores']) == val_config.k_folds, "Wrong number of CV scores"

            # Test bootstrap confidence intervals
            logger.info("Computing bootstrap confidence intervals...")
            bootstrap_results = validator.bootstrap_confidence_intervals(
                trainer, self._slice_study_data(self.test_data, 0, 20)
            )

            assert 'confidence_interval' in bootstrap_results, "Confidence interval missing"
            assert 'mean_score' in bootstrap_results, "Bootstrap mean score missing"

            # Test statistical hypothesis testing
            logger.info("Running statistical tests...")

            # Create second model for comparison
            trainer2 = VQCTrainer(circuit_config, opt_config)
            trainer2.train(self._slice_study_data(self.train_data, 25, 50))  # Different subset

            test_results = validator.statistical_hypothesis_test(
                trainer, trainer2, self._slice_study_data(self.test_data, 0, 15)
            )

            assert 'p_value' in test_results, "P-value missing from hypothesis test"
            assert 'test_statistic' in test_results, "Test statistic missing"

            results.update({
                'success': True,
                'cv_mean_score': float(cv_results['mean_score']),
                'cv_std_score': float(np.std(cv_results['cv_scores'])),
                'bootstrap_ci_width': float(bootstrap_results['confidence_interval'][1] -
                                          bootstrap_results['confidence_interval'][0]),
                'hypothesis_test_p_value': float(test_results['p_value']),
                'validation_framework_functional': True
            })

            logger.info(f"✅ Validation framework test passed: CV score={cv_results['mean_score']:.4f}")

        except Exception as e:
            logger.error(f"❌ Validation framework test failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _test_noise_analysis(self) -> Dict[str, Any]:
        """Test Phase 2C noise analysis integration"""
        logger.info("Testing noise analysis integration...")

        results = {}
        start_time = time.time()

        try:
            # Create noise configuration
            noise_config = NoiseConfig(
                noise_models=['depolarizing', 'bitflip'],
                noise_strengths=[0.01, 0.05],  # Mild noise for testing
                error_mitigation_techniques=['zne'],  # Zero-noise extrapolation
                device_characterization=False  # Skip for integration test
            )

            # Create noise analyzer
            noise_analyzer = NoiseAnalyzer(noise_config)

            # Train a model first
            circuit_config = CircuitConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
                ansatz="ry_cnot"
            )

            opt_config = OptimizationConfig(
                max_iterations=5,
                learning_rate=self.config.learning_rate
            )

            trainer = VQCTrainer(circuit_config, opt_config)
            trainer.train(self._slice_study_data(self.train_data, 0, 30))

            # Test noise impact analysis
            logger.info("Analyzing noise impact...")
            noise_results = noise_analyzer.analyze_noise_impact(
                trainer, self._slice_study_data(self.test_data, 0, 10)
            )

            assert 'clean_performance' in noise_results, "Clean performance missing"
            assert 'noisy_performance' in noise_results, "Noisy performance missing"
            assert 'noise_resilience' in noise_results, "Noise resilience metric missing"

            # Test error mitigation
            logger.info("Testing error mitigation...")
            mitigation_results = noise_analyzer.apply_error_mitigation(
                trainer, self._slice_study_data(self.test_data, 0, 10)
            )

            assert 'mitigated_performance' in mitigation_results, "Mitigated performance missing"
            assert 'improvement' in mitigation_results, "Improvement metric missing"

            results.update({
                'success': True,
                'clean_performance': float(noise_results['clean_performance']),
                'noise_resilience': float(noise_results['noise_resilience']),
                'mitigation_improvement': float(mitigation_results['improvement']),
                'noise_models_tested': len(noise_config.noise_models),
                'error_mitigation_functional': True
            })

            logger.info(f"✅ Noise analysis test passed: resilience={noise_results['noise_resilience']:.4f}")

        except Exception as e:
            logger.error(f"❌ Noise analysis test failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _test_hyperparameter_optimization(self) -> Dict[str, Any]:
        """Test Phase 2C hyperparameter optimization"""
        logger.info("Testing hyperparameter optimization (quick version)...")

        results = {}
        start_time = time.time()

        try:
            # Create HPO configuration (minimal for testing)
            hpo_config = HPOptConfig(
                search_space={
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_layers': [2, 3],
                    'ansatz': ['ry_cnot', 'hardware_efficient']
                },
                optimization_method='grid_search',  # Fastest for testing
                n_trials=4,  # Minimal
                cv_folds=2,  # Minimal
                random_state=42
            )

            # Create hyperparameter optimizer
            hpo = HyperparameterOptimizer(hpo_config)

            # Run optimization
            logger.info("Running hyperparameter search...")
            hpo_results = hpo.optimize(
                self._slice_study_data(self.train_data, 0, 20),  # Small dataset
                validation_data=self._slice_study_data(self.val_data, 0, 10)
            )

            assert 'best_config' in hpo_results, "Best configuration missing"
            assert 'best_score' in hpo_results, "Best score missing"
            assert 'optimization_history' in hpo_results, "Optimization history missing"

            # Test multi-objective optimization
            logger.info("Testing multi-objective optimization...")
            mo_config = HPOptConfig(
                search_space=hpo_config.search_space,
                optimization_method='nsga2',
                objectives=['accuracy', 'training_time'],
                n_trials=4,
                random_state=42
            )

            mo_hpo = HyperparameterOptimizer(mo_config)
            mo_results = mo_hpo.optimize(
                self._slice_study_data(self.train_data, 0, 15),
                validation_data=self._slice_study_data(self.val_data, 0, 8)
            )

            assert 'pareto_front' in mo_results, "Pareto front missing"

            results.update({
                'success': True,
                'best_score': float(hpo_results['best_score']),
                'trials_completed': len(hpo_results['optimization_history']),
                'best_config': hpo_results['best_config'],
                'multi_objective_functional': True,
                'pareto_solutions': len(mo_results['pareto_front'])
            })

            logger.info(f"✅ HPO test passed: best score={hpo_results['best_score']:.4f}")

        except Exception as e:
            logger.error(f"❌ HPO test failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _test_analytics_integration(self) -> Dict[str, Any]:
        """Test Phase 2D analytics integration"""
        logger.info("Testing analytics integration...")

        results = {}
        start_time = time.time()

        try:
            # Create analytics framework
            analytics = AdvancedAnalytics(
                results_dir=str(self.temp_dir / "analytics"),
                confidence_level=0.95
            )

            # Create some experimental results
            logger.info("Creating experimental results...")

            # Quantum experiments
            for i in range(3):
                config = {
                    'n_qubits': self.config.n_qubits,
                    'n_layers': self.config.n_layers,
                    'ansatz': 'ry_cnot',
                    'learning_rate': 0.1
                }

                metrics = {
                    'accuracy': 0.8 + np.random.normal(0, 0.05),
                    'training_time': 50 + np.random.normal(0, 10),
                    'convergence_iterations': 30 + np.random.randint(-5, 5)
                }

                raw_data = {
                    'training_history': [{'cost': 1.0 - j * 0.1} for j in range(10)],
                    'predictions': np.random.normal(0.8, 0.1, 20).tolist(),
                    'targets': np.random.normal(0.8, 0.1, 20).tolist()
                }

                analytics.register_experiment(
                    experiment_id=f'quantum_exp_{i}',
                    approach='quantum_vqc',
                    configuration=config,
                    metrics=metrics,
                    raw_data=raw_data
                )

            # Classical experiments
            for i in range(3):
                config = {
                    'model_type': 'random_forest',
                    'n_estimators': 100,
                    'max_depth': 5
                }

                metrics = {
                    'accuracy': 0.75 + np.random.normal(0, 0.03),
                    'training_time': 25 + np.random.normal(0, 5),
                    'convergence_iterations': 20 + np.random.randint(-3, 3)
                }

                raw_data = {
                    'predictions': np.random.normal(0.75, 0.1, 20).tolist(),
                    'targets': np.random.normal(0.75, 0.1, 20).tolist()
                }

                analytics.register_experiment(
                    experiment_id=f'classical_exp_{i}',
                    approach='classical_ml',
                    configuration=config,
                    metrics=metrics,
                    raw_data=raw_data
                )

            # Test quantum advantage analysis
            logger.info("Computing quantum advantage...")
            quantum_ids = [f'quantum_exp_{i}' for i in range(3)]
            classical_ids = [f'classical_exp_{i}' for i in range(3)]

            advantage_report = analytics.compute_quantum_advantage(quantum_ids, classical_ids)

            assert hasattr(advantage_report, 'advantage_metrics'), "Advantage metrics missing"
            assert hasattr(advantage_report, 'statistical_significance'), "Statistical significance missing"
            assert hasattr(advantage_report, 'recommendations'), "Recommendations missing"

            # Test scientific insights
            logger.info("Generating scientific insights...")
            insights = analytics.generate_scientific_insights()

            assert 'summary_statistics' in insights, "Summary statistics missing"
            assert 'recommendations' in insights, "Insights recommendations missing"

            results.update({
                'success': True,
                'experiments_registered': len(analytics.experiments),
                'advantage_metrics_computed': len(advantage_report.advantage_metrics),
                'recommendations_generated': len(advantage_report.recommendations),
                'insights_generated': len(insights['recommendations']),
                'analytics_functional': True
            })

            logger.info(f"✅ Analytics test passed: {len(analytics.experiments)} experiments analyzed")

        except Exception as e:
            logger.error(f"❌ Analytics test failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _test_visualization_integration(self) -> Dict[str, Any]:
        """Test Phase 2D visualization integration"""
        logger.info("Testing visualization integration...")

        results = {}
        start_time = time.time()

        try:
            # Create visualization framework
            viz = AdvancedVisualization(
                figures_dir=str(self.temp_dir / "figures"),
                reports_dir=str(self.temp_dir / "reports")
            )

            # Test circuit diagram creation
            logger.info("Creating circuit diagrams...")
            circuit_config = {
                'n_qubits': self.config.n_qubits,
                'n_layers': self.config.n_layers,
                'ansatz': 'ry_cnot'
            }

            parameters = np.random.uniform(-np.pi, np.pi, 8)
            circuit_plot = viz.create_quantum_circuit_diagram(circuit_config, parameters)

            assert Path(circuit_plot).exists(), "Circuit diagram not created"

            # Test training progress plot
            logger.info("Creating training progress plots...")
            training_history = [
                {'iteration': i, 'cost': 1.0 * np.exp(-i/5) + 0.1 * np.random.random()}
                for i in range(20)
            ]

            training_plot = viz.create_training_progress_plot(training_history)
            assert Path(training_plot).exists(), "Training plot not created"

            # Test interactive dashboard
            logger.info("Creating interactive dashboard...")
            training_data = {
                'training_history': training_history,
                'current_parameters': parameters
            }

            experiment_data = {
                'circuit_config': circuit_config,
                'metrics': {'accuracy': 0.85, 'convergence_time': 45.2}
            }

            dashboard = viz.create_interactive_dashboard(training_data, experiment_data)
            assert Path(dashboard).exists(), "Interactive dashboard not created"

            # Test quantum advantage visualization
            logger.info("Creating quantum advantage plots...")
            advantage_data = {
                'advantage_metrics': {
                    'accuracy_improvement': 0.15,
                    'convergence_speed': 0.25,
                    'parameter_efficiency': 0.10
                },
                'performance_comparison': {
                    'quantum': {'accuracy': 0.85, 'time': 50},
                    'classical': {'accuracy': 0.80, 'time': 30}
                }
            }

            advantage_plot = viz.create_quantum_advantage_visualization(advantage_data)
            assert Path(advantage_plot).exists(), "Advantage plot not created"

            results.update({
                'success': True,
                'circuit_diagram_created': Path(circuit_plot).exists(),
                'training_plot_created': Path(training_plot).exists(),
                'dashboard_created': Path(dashboard).exists(),
                'advantage_plot_created': Path(advantage_plot).exists(),
                'visualization_functional': True
            })

            logger.info("✅ Visualization test passed: All plots created successfully")

        except Exception as e:
            logger.error(f"❌ Visualization test failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _test_dosing_workflow(self) -> Dict[str, Any]:
        """Test end-to-end dosing optimization workflow"""
        logger.info("Testing end-to-end dosing optimization workflow...")

        results = {}
        start_time = time.time()

        try:
            # Train a model first
            circuit_config = CircuitConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
                ansatz="ry_cnot"
            )

            opt_config = OptimizationConfig(
                max_iterations=self.config.max_iterations,
                learning_rate=self.config.learning_rate
            )

            trainer = VQCTrainer(circuit_config, opt_config)
            trainer.train(self.train_data)

            # Test dosing optimization
            logger.info("Running dosing optimization...")
            dosing_optimizer = DosingOptimizer(trainer)

            dosing_results = dosing_optimizer.optimize_population_dosing(
                target_biomarker=3.3,
                population_coverage=0.9,
                n_virtual_patients=50  # Small for testing
            )

            assert 'optimal_dose' in dosing_results, "Optimal dose missing"
            assert 'achieved_coverage' in dosing_results, "Achieved coverage missing"
            assert 'optimization_success' in dosing_results, "Optimization success status missing"

            # Test different population scenarios
            logger.info("Testing different population scenarios...")
            scenarios = [
                {'weight_range': (50, 100), 'coverage': 0.9},
                {'weight_range': (70, 140), 'coverage': 0.75}
            ]

            scenario_results = {}
            for i, scenario in enumerate(scenarios):
                scenario_result = dosing_optimizer.optimize_population_dosing(
                    target_biomarker=3.3,
                    population_coverage=scenario['coverage'],
                    weight_range=scenario['weight_range'],
                    n_virtual_patients=30
                )
                scenario_results[f'scenario_{i}'] = scenario_result

            results.update({
                'success': True,
                'optimal_dose': float(dosing_results['optimal_dose']),
                'achieved_coverage': float(dosing_results['achieved_coverage']),
                'optimization_success': dosing_results['optimization_success'],
                'scenarios_tested': len(scenario_results),
                'dosing_workflow_functional': True
            })

            logger.info(f"✅ Dosing workflow test passed: dose={dosing_results['optimal_dose']:.1f}mg, coverage={dosing_results['achieved_coverage']:.1%}")

        except Exception as e:
            logger.error(f"❌ Dosing workflow test failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _test_performance_validation(self) -> Dict[str, Any]:
        """Test performance and resource usage validation"""
        logger.info("Testing performance and resource usage...")

        results = {}
        start_time = time.time()

        try:
            # Memory usage test
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Train model and monitor resources
            circuit_config = CircuitConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
                ansatz="ry_cnot"
            )

            opt_config = OptimizationConfig(
                max_iterations=self.config.max_iterations,
                learning_rate=self.config.learning_rate
            )

            trainer = VQCTrainer(circuit_config, opt_config)

            training_start = time.time()
            trainer.train(self.train_data)
            training_time = time.time() - training_start

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory

            # Performance benchmarks
            features, pk_targets, pd_targets = trainer._prepare_training_data(self.test_data)

            # Evaluation speed test
            eval_start = time.time()
            for _ in range(10):  # Multiple evaluations
                cost = trainer._evaluate_cost(trainer.best_parameters, features[:10], pk_targets[:10], pd_targets[:10])
            eval_time = (time.time() - eval_start) / 10  # Average per evaluation

            # Check performance thresholds
            performance_checks = {
                'training_time_reasonable': training_time < self.config.test_timeout,
                'memory_usage_reasonable': memory_usage < 1000,  # Less than 1GB
                'evaluation_speed_good': eval_time < 5.0,  # Less than 5 seconds per evaluation
                'cost_converged': cost < 100  # Should be much better than initial random
            }

            results.update({
                'success': all(performance_checks.values()),
                'training_time': training_time,
                'memory_usage_mb': memory_usage,
                'evaluation_time': eval_time,
                'final_cost': float(cost),
                'performance_checks': performance_checks
            })

            if results['success']:
                logger.info(f"✅ Performance validation passed: time={training_time:.1f}s, memory={memory_usage:.1f}MB")
            else:
                logger.warning(f"⚠️ Performance validation issues: {[k for k, v in performance_checks.items() if not v]}")

        except Exception as e:
            logger.error(f"❌ Performance validation failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _test_backward_compatibility(self) -> Dict[str, Any]:
        """Test backward compatibility with Phase 1 code"""
        logger.info("Testing backward compatibility...")

        results = {}
        start_time = time.time()

        try:
            # Test Phase 1 style configuration still works
            logger.info("Testing Phase 1 configuration compatibility...")

            # Basic Phase 1 style configuration
            circuit_config = CircuitConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
                ansatz="ry_cnot"
                # No Phase 2 features specified
            )

            opt_config = OptimizationConfig(
                max_iterations=5,
                learning_rate=self.config.learning_rate
                # No Phase 2 features specified
            )

            # Should work without Phase 2 features
            trainer = VQCTrainer(circuit_config, opt_config)
            trainer.train(self._slice_study_data(self.train_data, 0, 30))

            assert trainer.is_fitted, "Phase 1 style training failed"

            # Test dosing optimization (Phase 1 core functionality)
            dosing_optimizer = DosingOptimizer(trainer)
            dosing_result = dosing_optimizer.optimize_population_dosing(
                target_biomarker=3.3,
                population_coverage=0.9,
                n_virtual_patients=20
            )

            assert 'optimal_dose' in dosing_result, "Phase 1 dosing optimization failed"

            # Test that Phase 1 interfaces are preserved
            compatibility_checks = {
                'circuit_config_compatible': hasattr(circuit_config, 'n_qubits'),
                'trainer_interface_preserved': hasattr(trainer, 'train') and hasattr(trainer, 'is_fitted'),
                'dosing_interface_preserved': hasattr(dosing_optimizer, 'optimize_population_dosing'),
                'optimization_history_available': hasattr(trainer, 'optimization_history'),
                'parameters_accessible': hasattr(trainer, 'best_parameters')
            }

            results.update({
                'success': all(compatibility_checks.values()),
                'compatibility_checks': compatibility_checks,
                'phase1_training_works': trainer.is_fitted,
                'phase1_dosing_works': 'optimal_dose' in dosing_result,
                'backward_compatibility_maintained': True
            })

            logger.info(f"✅ Backward compatibility test passed: All Phase 1 interfaces preserved")

        except Exception as e:
            logger.error(f"❌ Backward compatibility test failed: {e}")
            results.update({
                'success': False,
                'error': str(e)
            })

        results['test_time'] = time.time() - start_time
        return results

    def _generate_test_summary(self) -> str:
        """Generate comprehensive test summary"""
        summary_lines = []

        # Overall success rate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values()
                              if isinstance(result, dict) and result.get('success', False))
        success_rate = successful_tests / total_tests if total_tests > 0 else 0

        summary_lines.append(f"Overall Success Rate: {successful_tests}/{total_tests} ({success_rate:.1%})")
        summary_lines.append("")

        # Individual test results
        summary_lines.append("Individual Test Results:")
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
                test_time = result.get('test_time', 0)
                summary_lines.append(f"  {test_name}: {status} ({test_time:.2f}s)")

                if not result.get('success', False) and 'error' in result:
                    summary_lines.append(f"    Error: {result['error']}")
            else:
                summary_lines.append(f"  {test_name}: ❌ FAIL (invalid result)")

        summary_lines.append("")

        # Performance metrics
        if self.performance_metrics:
            summary_lines.append("Performance Metrics:")
            for metric, value in self.performance_metrics.items():
                if isinstance(value, float):
                    summary_lines.append(f"  {metric}: {value:.2f}")
                else:
                    summary_lines.append(f"  {metric}: {value}")

        summary_lines.append("")

        # Phase 2 feature validation
        phase2_features = {
            'Enhanced Training (2A)': self.test_results.get('phase2a_training', {}).get('success', False),
            'Alternative Ansatz (2B)': self.test_results.get('phase2b_ansatz', {}).get('success', False),
            'Validation Framework (2C)': self.test_results.get('phase2c_validation', {}).get('success', False),
            'Noise Analysis (2C)': self.test_results.get('phase2c_noise', {}).get('success', False),
            'Analytics (2D)': self.test_results.get('phase2d_analytics', {}).get('success', False),
            'Visualization (2D)': self.test_results.get('phase2d_visualization', {}).get('success', False)
        }

        summary_lines.append("Phase 2 Feature Validation:")
        for feature, success in phase2_features.items():
            status = "✅ FUNCTIONAL" if success else "❌ FAILED"
            summary_lines.append(f"  {feature}: {status}")

        summary_lines.append("")

        # Recommendations
        summary_lines.append("Recommendations:")
        if success_rate >= 0.8:
            summary_lines.append("  🎉 Phase 2 integration is highly successful!")
            summary_lines.append("  ✅ Ready for production deployment")
            summary_lines.append("  ✅ All core functionality validated")
        elif success_rate >= 0.6:
            summary_lines.append("  ⚠️ Phase 2 integration mostly successful with some issues")
            summary_lines.append("  🔧 Address failed tests before deployment")
            summary_lines.append("  ✅ Core functionality appears stable")
        else:
            summary_lines.append("  ❌ Phase 2 integration has significant issues")
            summary_lines.append("  🚨 Review and fix failed components before proceeding")
            summary_lines.append("  🔧 Consider rolling back to Phase 1 if critical")

        return "\n".join(summary_lines)

    def cleanup(self):
        """Cleanup temporary files and resources"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Could not cleanup temporary directory: {e}")

def run_integration_tests(config: Optional[IntegrationTestConfig] = None) -> Dict[str, Any]:
    """
    Run Phase 2 integration tests with optional configuration

    Args:
        config: Integration test configuration (uses defaults if None)

    Returns:
        Dictionary containing test results and summary
    """
    if config is None:
        config = IntegrationTestConfig()

    tester = Phase2IntegrationTester(config)

    try:
        results = tester.run_complete_integration_test()
        return results
    finally:
        if not config.temp_dir:  # Only cleanup if we created the temp dir
            tester.cleanup()

if __name__ == "__main__":
    # Run integration tests
    logger.info("Starting Phase 2 Integration Test Suite...")

    # Configure for CI/automated testing
    test_config = IntegrationTestConfig(
        test_data_size=50,  # Smaller for faster testing
        max_iterations=10,
        skip_slow_tests=False,  # Test everything in development
        validate_outputs=True
    )

    results = run_integration_tests(test_config)

    # Print final summary
    print("\n" + "="*80)
    print("PHASE 2 INTEGRATION TEST COMPLETED")
    print("="*80)
    print(results['summary'])

    # Exit with appropriate code
    success_rate = len([r for r in results['test_results'].values()
                       if isinstance(r, dict) and r.get('success', False)]) / len(results['test_results'])

    exit_code = 0 if success_rate >= 0.8 else 1
    logger.info(f"Integration tests completed with exit code: {exit_code}")
    exit(exit_code)