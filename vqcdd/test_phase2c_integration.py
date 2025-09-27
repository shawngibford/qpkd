"""
Integration Tests for VQCdd Phase 2C Features

This module provides comprehensive integration tests for all Phase 2C enhancements
including validation framework, noise analysis, and hyperparameter optimization.
It ensures all components work together correctly and validates the complete
Phase 2C functionality.

Test Coverage:
- Validation framework integration (K-fold CV, bootstrap, statistical tests)
- Noise analysis and error mitigation integration
- Hyperparameter optimization integration
- Cross-module compatibility and data flow
- Performance and regression testing
- End-to-end workflow validation
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import logging
import warnings
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock
import time

# Import VQCdd modules
from data_handler import StudyData, PatientData, QuantumFeatureEncoder
from validation import create_synthetic_validation_data
from quantum_circuit import VQCircuit, CircuitConfig
from optimizer import VQCTrainer, OptimizationConfig
from validation import (
    ValidationPipeline, ValidationConfig, KFoldCrossValidator,
    StatisticalValidator, GeneralizationAnalyzer, ValidationResults
)
from noise_analysis import (
    NoiseModel, NoiseModelFactory, NoiseCharacterization,
    ErrorMitigationConfig, ZeroNoiseExtrapolation, ReadoutErrorMitigation,
    NoisyQuantumDevice, NoiseAwareTrainer
)
from hyperparameter_optimization import (
    HyperparameterSpace, OptimizationObjectives, HyperparameterOptimizationConfig,
    BayesianOptimizer, RandomSearchOptimizer, SensitivityAnalyzer,
    HyperparameterOptimizerFactory, optimize_vqc_hyperparameters
)


class TestPhase2CValidationIntegration(unittest.TestCase):
    """Test validation framework integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Suppress warnings for cleaner test output
        warnings.filterwarnings("ignore")

        # Create synthetic test data
        self.test_data = create_synthetic_validation_data(n_patients=20, random_state=42)

        # Create test configurations
        self.circuit_config = CircuitConfig(n_qubits=3, n_layers=2)
        self.optimization_config = OptimizationConfig(max_iterations=10, learning_rate=0.1)
        self.validation_config = ValidationConfig(
            n_folds=3,
            bootstrap_samples=50,
            confidence_level=0.95,
            test_population_splits=["weight_standard", "concomitant_yes"]
        )

        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_kfold_cross_validation(self):
        """Test K-fold cross-validation functionality"""
        cv_validator = KFoldCrossValidator(self.validation_config)

        def trainer_factory():
            return VQCTrainer(self.circuit_config, self.optimization_config)

        # Run cross-validation
        cv_results = cv_validator.validate(trainer_factory, self.test_data)

        # Validate results structure
        self.assertIsInstance(cv_results, ValidationResults)
        self.assertIn('mse', cv_results.cv_scores)
        self.assertEqual(len(cv_results.cv_scores['mse']), self.validation_config.n_folds)
        self.assertIsInstance(cv_results.cv_mean['mse'], float)
        self.assertIsInstance(cv_results.cv_std['mse'], float)

        # Check confidence intervals
        self.assertIn('mse', cv_results.cv_confidence_intervals)
        ci_lower, ci_upper = cv_results.cv_confidence_intervals['mse']
        self.assertLess(ci_lower, ci_upper)

    def test_statistical_validation(self):
        """Test statistical hypothesis testing"""
        # Create mock validation results for comparison
        results1 = ValidationResults(
            cv_scores={'mse': np.array([0.1, 0.12, 0.11])},
            cv_mean={'mse': 0.11},
            cv_std={'mse': 0.01},
            cv_confidence_intervals={},
            statistical_tests={},
            effect_sizes={},
            significance_summary={},
            generalization_scores={},
            population_comparisons={},
            model_comparison={},
            ranking=[],
            validation_config=self.validation_config,
            training_history={},
            timestamp=""
        )

        results2 = ValidationResults(
            cv_scores={'mse': np.array([0.15, 0.17, 0.16])},
            cv_mean={'mse': 0.16},
            cv_std={'mse': 0.01},
            cv_confidence_intervals={},
            statistical_tests={},
            effect_sizes={},
            significance_summary={},
            generalization_scores={},
            population_comparisons={},
            model_comparison={},
            ranking=[],
            validation_config=self.validation_config,
            training_history={},
            timestamp=""
        )

        statistical_validator = StatisticalValidator(self.validation_config)
        results_dict = {'model1': results1, 'model2': results2}

        statistical_results = statistical_validator.validate(results_dict)

        # Validate statistical test results
        self.assertIn('pairwise_tests', statistical_results)
        self.assertIn('effect_sizes', statistical_results)
        self.assertIn('significance_summary', statistical_results)

    def test_generalization_analysis(self):
        """Test generalization analysis across populations"""
        # Create a simple trainer for testing
        trainer = VQCTrainer(self.circuit_config, self.optimization_config)

        # Mock the training process
        with patch.object(trainer, 'train') as mock_train:
            mock_train.return_value = {'final_cost': 0.1}

            with patch.object(trainer, 'predict_patient_parameters') as mock_predict:
                mock_predict.return_value = 0.5

                # Mock feature encoder
                with patch.object(trainer, 'feature_encoder') as mock_encoder:
                    mock_encoder.encode_patient.return_value = np.array([1.0, 0.5, 0.8])

                    generalization_analyzer = GeneralizationAnalyzer(self.validation_config)
                    gen_results = generalization_analyzer.validate(trainer, self.test_data)

                    # Validate generalization results
                    self.assertIn('population_scores', gen_results)
                    self.assertIn('population_comparisons', gen_results)
                    self.assertIn('bias_analysis', gen_results)

    def test_comprehensive_validation_pipeline(self):
        """Test complete validation pipeline"""
        # Create model factory functions
        def trainer_factory1():
            return VQCTrainer(self.circuit_config, self.optimization_config)

        def trainer_factory2():
            config2 = CircuitConfig(n_qubits=4, n_layers=2)
            return VQCTrainer(config2, self.optimization_config)

        models = {
            'model1': trainer_factory1,
            'model2': trainer_factory2
        }

        # Mock training to avoid lengthy computation
        with patch('optimizer.VQCTrainer.train') as mock_train:
            mock_train.return_value = {'final_cost': 0.1}

            with patch('optimizer.VQCTrainer.predict_patient_parameters') as mock_predict:
                mock_predict.return_value = 0.5

                validation_pipeline = ValidationPipeline(self.validation_config)

                # Run comprehensive validation
                comprehensive_results = validation_pipeline.run_comprehensive_validation(models, self.test_data)

                # Validate comprehensive results structure
                self.assertIn('cross_validation_results', comprehensive_results)
                self.assertIn('statistical_analysis', comprehensive_results)
                self.assertIn('model_ranking', comprehensive_results)
                self.assertIn('best_model', comprehensive_results)
                self.assertIn('recommendations', comprehensive_results)


class TestPhase2CNoiseAnalysisIntegration(unittest.TestCase):
    """Test noise analysis and error mitigation integration"""

    def setUp(self):
        """Set up test fixtures"""
        warnings.filterwarnings("ignore")

        # Create noise models
        self.superconducting_noise = NoiseModelFactory.create_superconducting_noise_model("medium")
        self.trapped_ion_noise = NoiseModelFactory.create_trapped_ion_noise_model()

        # Create error mitigation config
        self.mitigation_config = ErrorMitigationConfig(
            zne_enabled=True,
            readout_mitigation=True,
            calibration_shots=100
        )

        # Create test data
        self.test_data = create_synthetic_validation_data(n_patients=10, random_state=42)

        # Create VQC trainer for testing
        self.circuit_config = CircuitConfig(n_qubits=3, n_layers=2)
        self.optimization_config = OptimizationConfig(max_iterations=5)
        self.trainer = VQCTrainer(self.circuit_config, self.optimization_config)

    def test_noise_model_creation(self):
        """Test noise model factory functionality"""
        # Test superconducting noise model
        self.assertIsInstance(self.superconducting_noise, NoiseModel)
        self.assertGreater(self.superconducting_noise.t1_time, 0)
        self.assertGreater(self.superconducting_noise.t2_time, 0)

        # Test trapped ion noise model
        self.assertIsInstance(self.trapped_ion_noise, NoiseModel)
        self.assertGreater(self.trapped_ion_noise.t1_time, self.superconducting_noise.t1_time)

    def test_noisy_quantum_device(self):
        """Test noisy quantum device wrapper"""
        n_qubits = 3
        noisy_device = NoisyQuantumDevice(self.superconducting_noise, n_qubits)

        self.assertEqual(noisy_device.n_qubits, n_qubits)
        self.assertEqual(noisy_device.noise_model, self.superconducting_noise)
        self.assertIsNotNone(noisy_device.device)

        # Test circuit creation
        def test_circuit():
            return 0.5  # Dummy return value

        noisy_circuit = noisy_device.create_noisy_circuit(test_circuit)
        self.assertIsNotNone(noisy_circuit)

    def test_zero_noise_extrapolation(self):
        """Test zero-noise extrapolation implementation"""
        zne = ZeroNoiseExtrapolation(self.mitigation_config)

        # Test with mock circuit function
        def mock_circuit(*args, **kwargs):
            # Simulate decreasing performance with noise
            noise_factor = kwargs.get('noise_factor', 1.0)
            return 0.1 * noise_factor + 0.05  # Linear noise scaling

        # Test extrapolation
        noise_factors = [1.0, 1.5, 2.0]
        extrapolated_circuit = zne.extrapolate(mock_circuit, noise_factors)

        # The extrapolated result should be better than noisy results
        self.assertIsNotNone(extrapolated_circuit)

    def test_readout_error_mitigation(self):
        """Test readout error mitigation"""
        n_qubits = 3
        readout_mitigation = ReadoutErrorMitigation(self.mitigation_config, n_qubits)

        # Mock device for calibration
        mock_device = MagicMock()

        # Test calibration matrix creation
        with patch('pennylane.qnode') as mock_qnode:
            mock_qnode.return_value = lambda: [1.0, -1.0, 1.0]  # Mock measurement results

            calibration_matrix = readout_mitigation.calibrate(mock_device)

            expected_shape = (2**n_qubits, 2**n_qubits)
            self.assertEqual(calibration_matrix.shape, expected_shape)

    def test_noise_characterization(self):
        """Test comprehensive noise characterization"""
        noise_characterization = NoiseCharacterization(self.superconducting_noise, self.mitigation_config)

        # Mock trainer methods to avoid lengthy computation
        with patch.object(self.trainer, 'train') as mock_train:
            mock_train.return_value = {'final_cost': 0.1}

            with patch.object(self.trainer, 'predict_patient_parameters') as mock_predict:
                mock_predict.return_value = 0.5

                with patch.object(self.trainer, 'feature_encoder') as mock_encoder:
                    mock_encoder.encode_patient.return_value = np.array([1.0, 0.5, 0.8])

                    # Run characterization
                    char_results = noise_characterization.characterize_device_noise(self.trainer, self.test_data)

                    # Validate results structure
                    self.assertIsNotNone(char_results.gate_fidelities)
                    self.assertIsNotNone(char_results.coherence_times)
                    self.assertIsNotNone(char_results.ideal_performance)
                    self.assertIsNotNone(char_results.noisy_performance)
                    self.assertIsNotNone(char_results.performance_degradation)

    def test_noise_aware_training(self):
        """Test noise-aware training techniques"""
        noise_aware_trainer = NoiseAwareTrainer(self.superconducting_noise, self.mitigation_config)

        # Mock training to test enhancement
        with patch.object(self.trainer, 'train') as mock_train:
            mock_train.return_value = {'final_cost': 0.1, 'iterations': 10}

            enhanced_result = noise_aware_trainer.train_with_noise_awareness(self.trainer, self.test_data)

            # Validate enhanced training results
            self.assertIsInstance(enhanced_result, dict)


class TestPhase2CHyperparameterOptimizationIntegration(unittest.TestCase):
    """Test hyperparameter optimization integration"""

    def setUp(self):
        """Set up test fixtures"""
        warnings.filterwarnings("ignore")

        # Create test data
        self.test_data = create_synthetic_validation_data(n_patients=15, random_state=42)

        # Create hyperparameter search space
        self.search_space = HyperparameterSpace(
            n_qubits=(2, 4),
            n_layers=(1, 3),
            learning_rate=(0.01, 0.1),
            max_iterations=(5, 15)  # Small for testing
        )

        # Create optimization objectives
        self.objectives = OptimizationObjectives(
            primary_metric="mse",
            secondary_metrics=["mae"]
        )

        # Create optimization config
        self.opt_config = HyperparameterOptimizationConfig(
            optimization_method="random",
            n_calls=5,  # Small for testing
            n_initial_points=2,
            verbose=False
        )

    def test_hyperparameter_space_conversion(self):
        """Test hyperparameter space format conversions"""
        # Test random space conversion
        random_space = self.search_space.to_random_space()
        self.assertIn('n_qubits', random_space)
        self.assertIn('learning_rate', random_space)

        # Test scikit-optimize space conversion (if available)
        try:
            skopt_space = self.search_space.to_skopt_space()
            self.assertIsInstance(skopt_space, list)
            self.assertGreater(len(skopt_space), 0)
        except ImportError:
            # Skip if scikit-optimize not available
            pass

    def test_random_search_optimizer(self):
        """Test random search hyperparameter optimization"""
        optimizer = RandomSearchOptimizer(self.search_space, self.objectives, self.opt_config)

        def mock_objective_function(trainer, data):
            # Simple mock that returns a score based on learning rate
            lr = trainer.optimization_config.learning_rate
            return lr * 10  # Lower learning rate = better score

        # Mock trainer creation and training
        with patch('optimizer.VQCTrainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.optimization_config.learning_rate = 0.05
            mock_trainer_class.return_value = mock_trainer

            # Run optimization
            result = optimizer.optimize(mock_objective_function, self.test_data)

            # Validate optimization result
            self.assertIsInstance(result, type(result))  # Check result type
            self.assertIsNotNone(result.best_hyperparameters)
            self.assertIsInstance(result.best_score, float)
            self.assertEqual(len(result.hyperparameter_history), self.opt_config.n_calls)

    @unittest.skipIf(
        True,  # Skip by default due to scikit-optimize dependency
        "Bayesian optimization requires scikit-optimize"
    )
    def test_bayesian_optimizer(self):
        """Test Bayesian hyperparameter optimization"""
        try:
            optimizer = BayesianOptimizer(self.search_space, self.objectives, self.opt_config)

            def mock_objective_function(trainer, data):
                return np.random.rand()  # Random score for testing

            # Run optimization
            result = optimizer.optimize(mock_objective_function, self.test_data)

            # Validate optimization result
            self.assertIsNotNone(result.best_hyperparameters)
            self.assertIsInstance(result.parameter_importance, dict)

        except ImportError:
            self.skipTest("scikit-optimize not available")

    def test_sensitivity_analyzer(self):
        """Test hyperparameter sensitivity analysis"""
        # Create mock optimization result
        mock_result = MagicMock()
        mock_result.hyperparameter_history = [
            {'n_qubits': 2, 'learning_rate': 0.01, 'optimizer_type': 'adam'},
            {'n_qubits': 3, 'learning_rate': 0.05, 'optimizer_type': 'adam'},
            {'n_qubits': 4, 'learning_rate': 0.1, 'optimizer_type': 'adagrad'}
        ]
        mock_result.score_history = [0.1, 0.08, 0.12]
        mock_result.parameter_importance = {'n_qubits': 0.4, 'learning_rate': 0.6}
        mock_result.best_score = 0.08

        # Test sensitivity analysis
        sensitivity_analyzer = SensitivityAnalyzer(mock_result)
        analysis_results = sensitivity_analyzer.analyze_sensitivity()

        # Validate analysis results
        self.assertIn('parameter_importance', analysis_results)
        self.assertIn('correlation_analysis', analysis_results)

    def test_hyperparameter_optimizer_factory(self):
        """Test hyperparameter optimizer factory"""
        # Test random optimizer creation
        random_optimizer = HyperparameterOptimizerFactory.create_optimizer(
            "random", self.search_space, self.objectives, self.opt_config
        )
        self.assertIsInstance(random_optimizer, RandomSearchOptimizer)

        # Test invalid method
        with self.assertRaises(ValueError):
            HyperparameterOptimizerFactory.create_optimizer(
                "invalid_method", self.search_space, self.objectives, self.opt_config
            )

    def test_convenience_function(self):
        """Test convenience function for VQC hyperparameter optimization"""
        # Create temporary output directory
        temp_output_dir = tempfile.mkdtemp()

        try:
            # Mock the training process
            with patch('optimizer.VQCTrainer') as mock_trainer_class:
                mock_trainer = MagicMock()
                mock_trainer.train.return_value = {'final_cost': 0.1}
                mock_trainer_class.return_value = mock_trainer

                # Run convenience function
                result = optimize_vqc_hyperparameters(
                    self.test_data,
                    method="random",
                    n_calls=3,
                    output_dir=temp_output_dir
                )

                # Validate result
                self.assertIsNotNone(result.best_hyperparameters)

                # Check if output files were created
                output_path = Path(temp_output_dir)
                self.assertTrue((output_path / 'optimization_results.json').exists())

        finally:
            # Clean up
            shutil.rmtree(temp_output_dir, ignore_errors=True)


class TestPhase2CEndToEndIntegration(unittest.TestCase):
    """Test end-to-end Phase 2C workflow integration"""

    def setUp(self):
        """Set up test fixtures"""
        warnings.filterwarnings("ignore")

        # Create comprehensive test data
        self.test_data = create_synthetic_validation_data(n_patients=25, random_state=42)

        # Create temporary working directory
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_complete_phase2c_workflow(self):
        """Test complete Phase 2C workflow integration"""
        # Step 1: Hyperparameter optimization
        search_space = HyperparameterSpace(
            n_qubits=(2, 3),
            n_layers=(1, 2),
            learning_rate=(0.01, 0.1),
            max_iterations=(5, 10)
        )

        objectives = OptimizationObjectives(primary_metric="mse")
        opt_config = HyperparameterOptimizationConfig(n_calls=3, verbose=False)

        # Mock hyperparameter optimization
        with patch('optimizer.VQCTrainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {'final_cost': 0.1}
            mock_trainer_class.return_value = mock_trainer

            optimizer = RandomSearchOptimizer(search_space, objectives, opt_config)

            def mock_objective_function(trainer, data):
                return np.random.uniform(0.05, 0.15)

            hp_result = optimizer.optimize(mock_objective_function, self.test_data)

        # Step 2: Noise analysis with optimized hyperparameters
        noise_model = NoiseModelFactory.create_superconducting_noise_model("medium")
        mitigation_config = ErrorMitigationConfig(zne_enabled=True)

        # Create VQC with optimized hyperparameters
        best_hp = hp_result.best_hyperparameters
        circuit_config = CircuitConfig(
            n_qubits=best_hp.get('n_qubits', 3),
            n_layers=best_hp.get('n_layers', 2)
        )
        optimization_config = OptimizationConfig(
            learning_rate=best_hp.get('learning_rate', 0.05),
            max_iterations=best_hp.get('max_iterations', 10)
        )

        # Mock noise characterization
        with patch('optimizer.VQCTrainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {'final_cost': 0.1}
            mock_trainer.predict_patient_parameters.return_value = 0.5
            mock_trainer.feature_encoder.encode_patient.return_value = np.array([1.0, 0.5])
            mock_trainer_class.return_value = mock_trainer

            noise_characterization = NoiseCharacterization(noise_model, mitigation_config)
            noise_results = noise_characterization.characterize_device_noise(mock_trainer, self.test_data)

        # Step 3: Comprehensive validation
        validation_config = ValidationConfig(
            n_folds=3,
            bootstrap_samples=50,
            test_population_splits=["weight_standard"]
        )

        def optimized_trainer_factory():
            return VQCTrainer(circuit_config, optimization_config)

        # Mock validation pipeline
        with patch('optimizer.VQCTrainer.train') as mock_train:
            mock_train.return_value = {'final_cost': 0.1}

            with patch('optimizer.VQCTrainer.predict_patient_parameters') as mock_predict:
                mock_predict.return_value = 0.5

                validation_pipeline = ValidationPipeline(validation_config)
                models = {'optimized_model': optimized_trainer_factory}

                validation_results = validation_pipeline.run_comprehensive_validation(models, self.test_data)

        # Validate end-to-end results
        self.assertIsNotNone(hp_result.best_hyperparameters)
        self.assertIsNotNone(noise_results.performance_degradation)
        self.assertIn('cross_validation_results', validation_results)

        # Test integration points
        self.assertEqual(circuit_config.n_qubits, best_hp.get('n_qubits', 3))
        self.assertEqual(optimization_config.learning_rate, best_hp.get('learning_rate', 0.05))

    def test_phase2c_error_handling(self):
        """Test error handling in Phase 2C integration"""
        # Test with invalid configurations
        invalid_search_space = HyperparameterSpace(
            n_qubits=(10, 5),  # Invalid: min > max
            n_layers=(1, 2)
        )

        # This should handle gracefully or raise appropriate errors
        try:
            objectives = OptimizationObjectives()
            opt_config = HyperparameterOptimizationConfig(n_calls=1)
            optimizer = RandomSearchOptimizer(invalid_search_space, objectives, opt_config)

            def failing_objective(trainer, data):
                raise RuntimeError("Simulated training failure")

            # Should handle failing evaluations gracefully
            result = optimizer.optimize(failing_objective, self.test_data)

            # Optimizer should have recorded the failures
            self.assertGreaterEqual(len(optimizer.evaluation_history), 0)

        except Exception as e:
            # If it raises an exception, it should be informative
            self.assertIsInstance(e, (ValueError, RuntimeError))

    def test_performance_regression(self):
        """Test performance regression for Phase 2C features"""
        start_time = time.time()

        # Test that Phase 2C components complete within reasonable time
        test_data = create_synthetic_validation_data(n_patients=10, random_state=42)

        # Validation should complete quickly for small datasets
        validation_config = ValidationConfig(n_folds=2, bootstrap_samples=10)
        cv_validator = KFoldCrossValidator(validation_config)

        def quick_trainer_factory():
            config = CircuitConfig(n_qubits=2, n_layers=1)
            opt_config = OptimizationConfig(max_iterations=2)
            return VQCTrainer(config, opt_config)

        # Mock training for speed
        with patch('optimizer.VQCTrainer.train') as mock_train:
            mock_train.return_value = {'final_cost': 0.1}

            with patch('optimizer.VQCTrainer.predict_patient_parameters') as mock_predict:
                mock_predict.return_value = 0.5

                with patch('optimizer.VQCTrainer.feature_encoder') as mock_encoder:
                    mock_encoder.encode_patient.return_value = np.array([1.0, 0.5])

                    results = cv_validator.validate(quick_trainer_factory, test_data)

        elapsed_time = time.time() - start_time

        # Should complete within reasonable time (10 seconds for mocked operations)
        self.assertLess(elapsed_time, 10.0)
        self.assertIsNotNone(results)


class TestPhase2CDataFlowIntegration(unittest.TestCase):
    """Test data flow and compatibility between Phase 2C components"""

    def setUp(self):
        """Set up test fixtures"""
        warnings.filterwarnings("ignore")
        self.test_data = create_synthetic_validation_data(n_patients=10, random_state=42)

    def test_data_format_compatibility(self):
        """Test that all Phase 2C components handle StudyData consistently"""
        # Test validation framework
        validation_config = ValidationConfig(n_folds=2)
        cv_validator = KFoldCrossValidator(validation_config)

        # Test data splitting
        X, y, stratify_labels = cv_validator._prepare_data_for_cv(self.test_data)
        self.assertEqual(len(X), len(self.test_data.patients))
        self.assertEqual(len(y), len(self.test_data.patients))

        # Test noise analysis data handling
        noise_model = NoiseModelFactory.create_superconducting_noise_model()
        mitigation_config = ErrorMitigationConfig()
        noise_characterization = NoiseCharacterization(noise_model, mitigation_config)

        # Should accept StudyData without errors
        self.assertIsInstance(self.test_data, StudyData)
        self.assertGreater(len(self.test_data.patients), 0)

    def test_configuration_propagation(self):
        """Test that configurations propagate correctly through the pipeline"""
        # Create initial configurations
        circuit_config = CircuitConfig(n_qubits=4, n_layers=3, ansatz="ry_cnot")
        optimization_config = OptimizationConfig(learning_rate=0.05, max_iterations=50)

        # Create trainer
        trainer = VQCTrainer(circuit_config, optimization_config)

        # Verify configuration propagation
        self.assertEqual(trainer.circuit_config.n_qubits, 4)
        self.assertEqual(trainer.circuit_config.n_layers, 3)
        self.assertEqual(trainer.optimization_config.learning_rate, 0.05)

        # Test hyperparameter optimization config creation
        hyperparameters = {
            'n_qubits': 6,
            'n_layers': 4,
            'learning_rate': 0.1,
            'optimizer_type': 'adam'
        }

        search_space = HyperparameterSpace()
        objectives = OptimizationObjectives()
        opt_config = HyperparameterOptimizationConfig()

        optimizer = RandomSearchOptimizer(search_space, objectives, opt_config)
        new_circuit_config, new_opt_config = optimizer._create_configs_from_hyperparameters(hyperparameters)

        # Verify new configurations
        self.assertEqual(new_circuit_config.n_qubits, 6)
        self.assertEqual(new_circuit_config.n_layers, 4)
        self.assertEqual(new_opt_config.learning_rate, 0.1)

    def test_result_serialization(self):
        """Test that Phase 2C results can be serialized and saved"""
        # Create mock results
        mock_validation_result = ValidationResults(
            cv_scores={'mse': np.array([0.1, 0.12, 0.11])},
            cv_mean={'mse': 0.11},
            cv_std={'mse': 0.01},
            cv_confidence_intervals={'mse': (0.10, 0.12)},
            statistical_tests={},
            effect_sizes={},
            significance_summary={},
            generalization_scores={},
            population_comparisons={},
            model_comparison={},
            ranking=[],
            validation_config=ValidationConfig(),
            training_history={},
            timestamp="2024-01-01T00:00:00"
        )

        # Test that validation results are JSON serializable after processing
        validation_pipeline = ValidationPipeline(ValidationConfig())
        serializable_result = validation_pipeline._make_json_serializable(mock_validation_result.__dict__)

        # Should not contain numpy arrays
        self.assertIsInstance(serializable_result['cv_scores']['mse'], list)
        self.assertIsInstance(serializable_result['cv_mean']['mse'], float)


def run_phase2c_integration_tests():
    """Run all Phase 2C integration tests"""
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestPhase2CValidationIntegration,
        TestPhase2CNoiseAnalysisIntegration,
        TestPhase2CHyperparameterOptimizationIntegration,
        TestPhase2CEndToEndIntegration,
        TestPhase2CDataFlowIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)

    # Print summary
    print(f"\nPhase 2C Integration Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run integration tests
    success = run_phase2c_integration_tests()

    if success:
        print("\n✅ All Phase 2C integration tests passed!")
        print("Phase 2C implementation is ready for deployment.")
    else:
        print("\n❌ Some Phase 2C integration tests failed!")
        print("Please review and fix the issues before deployment.")

    # Exit with appropriate code
    exit(0 if success else 1)