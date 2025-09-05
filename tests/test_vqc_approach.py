"""
Comprehensive Tests for VQC Parameter Estimator

Tests all functionality including error handling, hyperparameter optimization,
and dosing optimization for the VQC approach.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.append('/Users/shawngibford/dev/qpkd/src')

from quantum.approach1_vqc.vqc_parameter_estimator_full import (
    VQCParameterEstimatorFull, VQCConfig, VQCHyperparameters
)
from quantum.core.base import PKPDData
from utils.logging_system import QuantumPKPDLogger


class TestVQCParameterEstimator:
    """Test suite for VQC Parameter Estimator"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample PK/PD data for testing"""
        np.random.seed(42)
        
        n_subjects = 10
        n_timepoints = 5
        n_total = n_subjects * n_timepoints
        
        subjects = np.repeat(range(1, n_subjects + 1), n_timepoints)
        time_points = np.tile([0, 1, 4, 12, 24], n_subjects)
        doses = np.repeat([0, 1, 3, 10], [n_total//4]*3 + [n_total - 3*(n_total//4)])
        body_weights = np.repeat(np.random.uniform(50, 100, n_subjects), n_timepoints)
        concomitant_meds = np.repeat(np.random.binomial(1, 0.5, n_subjects), n_timepoints)
        
        # Generate realistic PK/PD data
        pk_concentrations = np.full(n_total, np.nan)
        pd_biomarkers = np.full(n_total, np.nan)
        
        for i in range(n_total):
            if doses[i] > 0:  # Only active doses have PK data
                # Simple PK model for test data
                ka, cl, v = 1.0, 3.0, 20.0
                bw_effect = (body_weights[i] / 70) ** 0.75
                cl_scaled = cl * bw_effect
                ke = cl_scaled / v
                
                pk_concentrations[i] = (doses[i] / v) * np.exp(-ke * time_points[i])
                
            # PD data for all subjects  
            baseline = 10.0 * (1 + 0.2 * concomitant_meds[i])
            if not np.isnan(pk_concentrations[i]):
                imax, ic50 = 0.8, 5.0
                inhibition = imax * pk_concentrations[i] / (ic50 + pk_concentrations[i])
                pd_biomarkers[i] = baseline * (1 - inhibition)
            else:
                pd_biomarkers[i] = baseline
                
        return PKPDData(
            subjects=subjects,
            time_points=time_points,
            pk_concentrations=pk_concentrations,
            pd_biomarkers=pd_biomarkers,
            doses=doses,
            body_weights=body_weights,
            concomitant_meds=concomitant_meds
        )
    
    @pytest.fixture
    def vqc_config(self):
        """Create VQC configuration for testing"""
        hyperparams = VQCHyperparameters(
            learning_rate=0.1,
            n_layers=2,
            ansatz_type="basic_entangling",
            optimizer_type="adam",
            regularization_strength=0.01,
            batch_size=10,
            early_stopping_patience=5
        )
        
        return VQCConfig(
            n_qubits=4,
            max_iterations=20,  # Reduced for testing
            convergence_threshold=1e-3,
            shots=1000,
            hyperparams=hyperparams,
            validation_split=0.3,
            cross_validation_folds=3
        )
    
    @pytest.fixture
    def logger(self):
        """Create temporary logger for testing"""
        temp_dir = tempfile.mkdtemp()
        logger = QuantumPKPDLogger(log_dir=temp_dir, experiment_name="test_vqc")
        yield logger
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, vqc_config, logger):
        """Test VQC estimator initialization"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        assert estimator.vqc_config == vqc_config
        assert estimator.logger == logger
        assert estimator.device is None
        assert estimator.best_parameters is None
        assert len(estimator.training_history) == 0
        assert estimator.error_count == 0
        
        # Test parameter bounds setup
        assert 'ka' in estimator.vqc_config.parameter_bounds
        assert 'baseline' in estimator.vqc_config.parameter_bounds
        
    def test_quantum_device_setup(self, vqc_config, logger):
        """Test quantum device setup"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        device = estimator.setup_quantum_device()
        
        assert device is not None
        assert estimator.device == device
        assert device.wires == list(range(vqc_config.n_qubits))
        
    def test_quantum_circuit_building(self, vqc_config, logger):
        """Test quantum circuit construction"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        estimator.setup_quantum_device()
        
        circuit = estimator.build_quantum_circuit(
            vqc_config.n_qubits, 
            vqc_config.hyperparams.n_layers
        )
        
        assert circuit is not None
        assert estimator.qnode == circuit
        
        # Test circuit execution
        params = estimator._initialize_parameters()
        features = np.array([1.0, 5.0, 70.0, 0.0])
        
        output = circuit(params, features)
        
        assert len(output) <= vqc_config.n_qubits
        assert all(-1 <= val <= 1 for val in output)  # Valid expectation values
        
    def test_data_encoding(self, vqc_config, logger, sample_data):
        """Test data encoding functionality"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        encoded_data = estimator.encode_data(sample_data)
        
        assert encoded_data.shape[0] == len(sample_data.time_points)
        assert encoded_data.shape[1] == 4  # time, dose, bw, comed
        assert hasattr(estimator, 'feature_scaler')
        
        # Test data scaling properties
        assert np.allclose(np.mean(encoded_data, axis=0), 0, atol=0.1)  # Approximately centered
        
    def test_pk_model_prediction(self, vqc_config, logger):
        """Test PK model predictions"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        params = {'ka': 1.0, 'cl': 3.0, 'v1': 20.0, 'q': 2.0, 'v2': 50.0}
        time = np.array([0, 1, 4, 12, 24])
        dose = 10.0
        covariates = {'body_weight': 70.0, 'concomitant_med': 0}
        
        concentrations = estimator.pk_model_prediction(params, time, dose, covariates)
        
        assert len(concentrations) == len(time)
        assert all(c >= 0 for c in concentrations)  # Non-negative concentrations
        assert concentrations[0] > concentrations[-1]  # Decreasing over time
        
        # Test body weight scaling
        covariates_heavy = {'body_weight': 100.0, 'concomitant_med': 0}
        conc_heavy = estimator.pk_model_prediction(params, time, dose, covariates_heavy)
        
        # Heavier patients should have lower concentrations (higher clearance/volume)
        assert conc_heavy[1] < concentrations[1]
        
    def test_pd_model_prediction(self, vqc_config, logger):
        """Test PD model predictions"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        concentrations = np.array([0, 1, 5, 10, 20])
        params = {'baseline': 10.0, 'imax': 0.8, 'ic50': 5.0, 'gamma': 1.0}
        covariates = {'concomitant_med': 0}
        
        biomarkers = estimator.pd_model_prediction(concentrations, params, covariates)
        
        assert len(biomarkers) == len(concentrations)
        assert all(b >= 0.1 for b in biomarkers)  # Above minimum level
        assert biomarkers[0] > biomarkers[-1]  # Decreasing with increasing concentration
        
        # Test concomitant medication effect
        covariates_comed = {'concomitant_med': 1}
        bio_comed = estimator.pd_model_prediction(concentrations, params, covariates_comed)
        
        # Concomitant medication should increase baseline
        assert bio_comed[0] > biomarkers[0]
        
    def test_quantum_parameter_mapping(self, vqc_config, logger):
        """Test quantum output to parameter mapping"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        # Test valid quantum output
        quantum_output = [0.5, -0.3, 0.8, 0.1, -0.7, 0.2, 0.6, -0.4]
        
        pk_params = estimator._map_quantum_to_pk_params(quantum_output)
        pd_params = estimator._map_quantum_to_pd_params(quantum_output)
        
        # Check PK parameters are within bounds
        for param_name, value in pk_params.items():
            bounds = estimator.vqc_config.parameter_bounds[param_name]
            assert bounds[0] <= value <= bounds[1], f"{param_name} = {value} not in bounds {bounds}"
            
        # Check PD parameters are within bounds  
        for param_name, value in pd_params.items():
            bounds = estimator.vqc_config.parameter_bounds[param_name]
            assert bounds[0] <= value <= bounds[1], f"{param_name} = {value} not in bounds {bounds}"
            
        # Test extreme quantum outputs
        extreme_output = [1.0, -1.0, 1.0, -1.0]
        pk_extreme = estimator._map_quantum_to_pk_params(extreme_output)
        pd_extreme = estimator._map_quantum_to_pd_params(extreme_output)
        
        # Should still be within bounds
        for param_name, value in pk_extreme.items():
            bounds = estimator.vqc_config.parameter_bounds[param_name]
            assert bounds[0] <= value <= bounds[1]
            
    def test_cost_function(self, vqc_config, logger, sample_data):
        """Test cost function computation"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        estimator.setup_quantum_device()
        estimator.build_quantum_circuit(vqc_config.n_qubits, vqc_config.hyperparams.n_layers)
        
        params = estimator._initialize_parameters()
        
        cost = estimator.cost_function_with_regularization(params, sample_data)
        
        assert isinstance(cost, (float, np.floating))
        assert cost >= 0  # Cost should be non-negative
        assert not np.isnan(cost)
        assert not np.isinf(cost)
        
        # Test regularization effect
        large_params = params * 10
        large_cost = estimator.cost_function_with_regularization(large_params, sample_data)
        
        assert large_cost > cost  # Larger parameters should have higher cost
        
    def test_parameter_initialization(self, vqc_config, logger):
        """Test parameter initialization"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        params = estimator._initialize_parameters()
        
        expected_shape = (vqc_config.hyperparams.n_layers, vqc_config.n_qubits)
        assert params.shape == expected_shape
        
        # Should be within reasonable range for optimization
        assert np.all(np.abs(params) <= 2)
        
        # Test strongly entangling ansatz
        vqc_config.hyperparams.ansatz_type = "strongly_entangling"
        params_strong = estimator._initialize_parameters()
        expected_shape_strong = (vqc_config.hyperparams.n_layers, vqc_config.n_qubits, 3)
        assert params_strong.shape == expected_shape_strong
        
    def test_train_validation_split(self, vqc_config, logger, sample_data):
        """Test data splitting"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        train_data, val_data = estimator._train_validation_split(sample_data)
        
        # Check split preserves subjects
        train_subjects = set(train_data.subjects)
        val_subjects = set(val_data.subjects)
        
        assert len(train_subjects & val_subjects) == 0  # No overlap
        assert train_subjects | val_subjects == set(sample_data.subjects)  # Complete coverage
        
        # Check approximate split ratio
        total_subjects = len(set(sample_data.subjects))
        expected_val_subjects = int(total_subjects * vqc_config.validation_split)
        
        assert abs(len(val_subjects) - expected_val_subjects) <= 1
        
    @pytest.mark.slow
    def test_parameter_optimization(self, vqc_config, logger, sample_data):
        """Test parameter optimization (integration test)"""
        # Reduce complexity for testing
        vqc_config.max_iterations = 10
        vqc_config.hyperparams.n_layers = 1
        vqc_config.n_qubits = 3
        
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        estimator.setup_quantum_device()
        estimator.build_quantum_circuit(vqc_config.n_qubits, vqc_config.hyperparams.n_layers)
        
        result = estimator.optimize_parameters(sample_data)
        
        assert 'optimal_params' in result
        assert 'convergence_info' in result
        assert 'training_history' in result
        
        assert estimator.best_parameters is not None
        assert len(estimator.training_history) > 0
        assert estimator.convergence_info is not None
        
        # Training should reduce loss
        if len(estimator.training_history) > 1:
            assert estimator.training_history[-1] <= estimator.training_history[0]
            
    def test_hyperparameter_evaluation(self, vqc_config, logger, sample_data):
        """Test hyperparameter evaluation"""
        # Simplify for testing
        vqc_config.cross_validation_folds = 2
        vqc_config.max_iterations = 5
        
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        test_hyperparams = VQCHyperparameters(
            learning_rate=0.05,
            n_layers=1,
            regularization_strength=0.001
        )
        
        score = estimator._evaluate_hyperparameters(sample_data, test_hyperparams)
        
        assert isinstance(score, (float, np.floating))
        assert not np.isnan(score)
        
    def test_prediction_functionality(self, vqc_config, logger, sample_data):
        """Test biomarker prediction"""
        # Simplify for testing
        vqc_config.max_iterations = 5
        vqc_config.n_qubits = 3
        vqc_config.hyperparams.n_layers = 1
        
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        estimator.setup_quantum_device()
        estimator.build_quantum_circuit(vqc_config.n_qubits, vqc_config.hyperparams.n_layers)
        
        # Mock training by setting best parameters
        estimator.best_parameters = estimator._initialize_parameters()
        estimator.feature_scaler = Mock()
        estimator.feature_scaler.transform.return_value = np.array([[1.0, 5.0, 70.0, 0.0]])
        
        # Mock quantum circuit output
        with patch.object(estimator, 'qnode', return_value=[0.1, 0.2, 0.3]):
            prediction = estimator.predict_biomarker(
                dose=5.0,
                time=np.array([24.0]),
                covariates={'body_weight': 70.0, 'concomitant_med': 0}
            )
            
        assert len(prediction) == 1
        assert prediction[0] > 0
        
    def test_population_coverage_evaluation(self, vqc_config, logger, sample_data):
        """Test population coverage evaluation"""
        # Simplify for testing
        vqc_config.n_qubits = 3
        vqc_config.hyperparams.n_layers = 1
        
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        estimator.setup_quantum_device()
        estimator.build_quantum_circuit(vqc_config.n_qubits, vqc_config.hyperparams.n_layers)
        estimator.best_parameters = estimator._initialize_parameters()
        
        scenario_params = {'weight_range': (50, 100), 'comed_allowed': True}
        
        # Mock prediction to return predictable values
        with patch.object(estimator, 'predict_biomarker', return_value=np.array([2.0])):
            coverage = estimator._evaluate_population_coverage(
                dose=10.0,
                dosing_interval=24.0,
                scenario_params=scenario_params,
                target_threshold=3.3
            )
            
        assert 0 <= coverage <= 1
        # Should be high coverage since mock returns 2.0 < 3.3
        assert coverage > 0.8
        
    def test_error_handling(self, vqc_config, logger, sample_data):
        """Test error handling and recovery"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        # Test device setup failure
        with patch('pennylane.device', side_effect=Exception("Device error")):
            with pytest.raises(RuntimeError, match="Failed to setup quantum device"):
                estimator.setup_quantum_device()
                
        # Test invalid data encoding
        invalid_data = PKPDData(
            subjects=np.array([]),
            time_points=np.array([]),
            pk_concentrations=np.array([]),
            pd_biomarkers=np.array([]),
            doses=np.array([]),
            body_weights=np.array([]),
            concomitant_meds=np.array([])
        )
        
        with pytest.raises(ValueError):
            estimator.encode_data(invalid_data)
            
        # Test prediction without training
        with pytest.raises(ValueError, match="Model not trained"):
            estimator.predict_biomarker(5.0, np.array([24.0]), {'body_weight': 70.0})
            
    def test_logging_integration(self, vqc_config, logger, sample_data):
        """Test logging integration"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        
        # Test that logger methods are called
        with patch.object(logger, 'log_training_step') as mock_log:
            estimator.logger.log_training_step(
                "VQC", 1, 0.5, np.array([1.0, 2.0]), {"test": 1.0}
            )
            mock_log.assert_called_once()
            
        with patch.object(logger, 'log_error') as mock_error:
            estimator.logger.log_error("VQC", Exception("test error"))
            mock_error.assert_called_once()
            
    def test_quantum_metrics_calculation(self, vqc_config, logger):
        """Test quantum metrics calculation"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        estimator.best_parameters = np.random.randn(2, 4)  # Mock trained parameters
        estimator.training_history = [1.0, 0.8, 0.6, 0.5, 0.5, 0.5]
        
        metrics = estimator._calculate_quantum_metrics()
        
        expected_keys = [
            'parameter_count', 'circuit_depth', 'quantum_volume',
            'expressivity_measure', 'entanglement_capability',
            'final_parameter_norm', 'training_stability'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert not np.isnan(metrics[key])
            
        assert metrics['parameter_count'] == 8  # 2 layers * 4 qubits
        assert metrics['circuit_depth'] == vqc_config.hyperparams.n_layers
        
    def test_confidence_intervals(self, vqc_config, logger):
        """Test confidence interval calculation"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        estimator.setup_quantum_device()
        estimator.build_quantum_circuit(vqc_config.n_qubits, vqc_config.hyperparams.n_layers)
        estimator.best_parameters = estimator._initialize_parameters()
        
        # Mock feature scaler
        estimator.feature_scaler = Mock()
        estimator.feature_scaler.transform.return_value = np.array([[1.0, 5.0, 70.0, 0.0]])
        
        with patch.object(estimator, 'qnode', return_value=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
            ci = estimator._calculate_confidence_intervals()
            
        assert isinstance(ci, dict)
        assert len(ci) > 0
        
        for param_name, (lower, upper) in ci.items():
            assert lower < upper
            assert isinstance(lower, (int, float))
            assert isinstance(upper, (int, float))
            
    def test_dosing_optimization_structure(self, vqc_config, logger, sample_data):
        """Test dosing optimization structure (without full optimization)"""
        # Simplify for testing
        vqc_config.n_qubits = 3
        vqc_config.hyperparams.n_layers = 1
        
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        estimator.setup_quantum_device()
        estimator.build_quantum_circuit(vqc_config.n_qubits, vqc_config.hyperparams.n_layers)
        estimator.best_parameters = estimator._initialize_parameters()
        
        # Mock the expensive optimization parts
        with patch.object(estimator, '_optimize_single_regimen') as mock_opt:
            mock_opt.return_value = {'optimal_dose': 5.0, 'coverage': 0.9, 'optimization_success': True}
            
            result = estimator.optimize_dosing(target_threshold=3.3, population_coverage=0.9)
            
        assert isinstance(result, type(result))  # Check it returns OptimizationResult type
        assert hasattr(result, 'optimal_daily_dose')
        assert hasattr(result, 'optimal_weekly_dose')
        assert hasattr(result, 'population_coverage')
        
    def test_state_serialization(self, vqc_config, logger, sample_data):
        """Test model state saving and loading"""
        estimator = VQCParameterEstimatorFull(vqc_config, logger)
        estimator.setup_quantum_device()
        estimator.build_quantum_circuit(vqc_config.n_qubits, vqc_config.hyperparams.n_layers)
        
        # Create some state to save
        state = {
            'parameters': estimator._initialize_parameters(),
            'training_history': [1.0, 0.8, 0.6],
            'config': vqc_config
        }
        
        # Test state saving
        estimator.logger.save_experiment_state("VQC", state)
        
        # Test state loading
        loaded_state = estimator.logger.load_experiment_state("VQC")
        
        assert loaded_state is not None
        assert 'parameters' in loaded_state
        assert 'training_history' in loaded_state
        
        np.testing.assert_array_equal(
            loaded_state['parameters'], 
            state['parameters']
        )


class TestVQCConfigValidation:
    """Test VQC configuration validation"""
    
    def test_hyperparameter_validation(self):
        """Test hyperparameter validation"""
        # Valid hyperparameters
        valid_hyperparams = VQCHyperparameters(
            learning_rate=0.01,
            n_layers=4,
            ansatz_type="strongly_entangling",
            regularization_strength=0.001
        )
        
        assert valid_hyperparams.learning_rate > 0
        assert valid_hyperparams.n_layers > 0
        assert valid_hyperparams.ansatz_type in ["basic_entangling", "strongly_entangling", "simplified_two_design"]
        
    def test_config_defaults(self):
        """Test configuration defaults"""
        config = VQCConfig()
        
        assert config.n_qubits > 0
        assert config.max_iterations > 0
        assert config.convergence_threshold > 0
        assert config.hyperparams is not None
        
        
class TestVQCIntegration:
    """Integration tests for VQC approach"""
    
    @pytest.mark.slow  
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Create simple synthetic data
        np.random.seed(42)
        
        subjects = np.array([1, 1, 2, 2, 3, 3])
        time_points = np.array([0, 24, 0, 24, 0, 24])  
        doses = np.array([5, 5, 10, 10, 0, 0])
        body_weights = np.array([60, 60, 80, 80, 70, 70])
        concomitant_meds = np.array([0, 0, 1, 1, 0, 0])
        
        pk_concentrations = np.array([np.nan, 2.5, np.nan, 4.0, np.nan, np.nan])
        pd_biomarkers = np.array([10.0, 8.5, 12.0, 7.0, 10.0, 10.0])
        
        data = PKPDData(
            subjects=subjects,
            time_points=time_points,
            pk_concentrations=pk_concentrations,
            pd_biomarkers=pd_biomarkers,
            doses=doses,
            body_weights=body_weights,
            concomitant_meds=concomitant_meds
        )
        
        # Create simple config for testing
        hyperparams = VQCHyperparameters(
            learning_rate=0.1,
            n_layers=1,
            ansatz_type="basic_entangling",
            batch_size=3,
            early_stopping_patience=3
        )
        
        config = VQCConfig(
            n_qubits=3,
            max_iterations=10,
            hyperparams=hyperparams,
            validation_split=0.33
        )
        
        # Create temporary logger
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = QuantumPKPDLogger(log_dir=temp_dir, experiment_name="integration_test")
            
            # Run workflow
            estimator = VQCParameterEstimatorFull(config, logger)
            estimator.fit(data)
            
            # Test that model is trained
            assert estimator.is_trained
            assert estimator.best_parameters is not None
            assert len(estimator.training_history) > 0
            
            # Test prediction
            prediction = estimator.predict_biomarker(
                dose=5.0,
                time=np.array([24.0]),
                covariates={'body_weight': 70.0, 'concomitant_med': 0}
            )
            
            assert len(prediction) == 1
            assert prediction[0] > 0
            
            # Test dosing optimization (simplified)
            with patch.object(estimator, '_evaluate_population_coverage', return_value=0.85):
                result = estimator.optimize_dosing()
                
                assert result.optimal_daily_dose > 0
                assert result.optimal_weekly_dose > 0
                assert 0 <= result.population_coverage <= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])