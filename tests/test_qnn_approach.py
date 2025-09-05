"""
Comprehensive Tests for Quantum Neural Network Approach

Tests QNN ensemble methods, data reuploading, feature engineering,
and hierarchical population modeling.
"""

import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch

import sys
sys.path.append('/Users/shawngibford/dev/qpkd/src')

from quantum.approach2_qml.quantum_neural_network_full import (
    QuantumNeuralNetworkFull, QNNConfig, QNNHyperparameters
)
from quantum.core.base import PKPDData
from utils.logging_system import QuantumPKPDLogger


class TestQuantumNeuralNetwork:
    """Test suite for QNN approach"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        
        n_subjects = 8
        n_timepoints = 4
        n_total = n_subjects * n_timepoints
        
        subjects = np.repeat(range(1, n_subjects + 1), n_timepoints)
        time_points = np.tile([0, 6, 24, 48], n_subjects)
        doses = np.repeat([0, 3, 10], [n_total//3, n_total//3, n_total - 2*(n_total//3)])
        body_weights = np.repeat(np.random.uniform(55, 95, n_subjects), n_timepoints)
        concomitant_meds = np.repeat(np.random.binomial(1, 0.4, n_subjects), n_timepoints)
        
        # Generate PK/PD data
        pk_concentrations = np.full(n_total, np.nan)
        pd_biomarkers = np.full(n_total, np.nan)
        
        for i in range(n_total):
            if doses[i] > 0:
                # PK data
                pk_concentrations[i] = (doses[i] / 25.0) * np.exp(-0.1 * time_points[i])
            
            # PD data
            baseline = 9.0 * (1 + 0.25 * concomitant_meds[i])
            if not np.isnan(pk_concentrations[i]):
                inhibition = 0.7 * pk_concentrations[i] / (4.0 + pk_concentrations[i])
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
    def qnn_config(self):
        """Create QNN configuration"""
        hyperparams = QNNHyperparameters(
            learning_rate=0.1,
            encoding_layers=2,
            variational_layers=2,
            data_reuploading_layers=2,
            batch_size=8,
            early_stopping_patience=5
        )
        
        return QNNConfig(
            n_qubits=4,
            max_iterations=20,
            hyperparams=hyperparams,
            ensemble_size=3,
            feature_engineering=True,
            data_augmentation=True
        )
    
    @pytest.fixture  
    def logger(self):
        """Create temporary logger"""
        temp_dir = tempfile.mkdtemp()
        logger = QuantumPKPDLogger(log_dir=temp_dir, experiment_name="test_qnn")
        yield logger
        shutil.rmtree(temp_dir)
    
    def test_qnn_initialization(self, qnn_config, logger):
        """Test QNN initialization"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        assert qnn.qnn_config == qnn_config
        assert qnn.logger == logger
        assert qnn.device is None
        assert len(qnn.qnn_ensemble) == 0
        assert qnn.feature_scaler is None
        
    def test_device_setup(self, qnn_config, logger):
        """Test quantum device setup"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        device = qnn.setup_quantum_device()
        
        assert device is not None
        assert qnn.device == device
        
    def test_feature_engineering(self, qnn_config, logger):
        """Test feature engineering"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        engineered = qnn._engineer_features(
            time=24.0, dose=10.0, body_weight=70.0, comed=1.0
        )
        
        assert len(engineered) == 8  # Expected number of engineered features
        assert all(isinstance(f, (int, float)) for f in engineered)
        
        # Test specific features
        assert engineered[0] == np.log(24.0 + 1)  # Log-time
        assert engineered[1] == 10.0 / 70.0  # Dose per body weight
        assert engineered[2] == 24.0 * 10.0  # Time-dose interaction
        
    def test_data_encoding(self, qnn_config, logger, sample_data):
        """Test data encoding with feature engineering"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        encoded_data = qnn.encode_data(sample_data)
        
        assert encoded_data.shape[0] >= len(sample_data.time_points)  # May be augmented
        
        # With feature engineering, should have 4 base + 8 engineered = 12 features
        expected_features = 12 if qnn_config.feature_engineering else 4
        assert encoded_data.shape[1] == expected_features
        
        # Check that scaler was created
        assert qnn.feature_scaler is not None
        
    def test_data_augmentation(self, qnn_config, logger):
        """Test data augmentation for small datasets"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        # Small dataset that should trigger augmentation
        small_features = np.random.randn(15, 4)  # Less than 100 samples
        
        # Mock small PKPDData
        mock_data = Mock()
        mock_data.time_points = np.arange(15)
        
        augmented = qnn._augment_data(small_features, mock_data)
        
        # Should have more samples than original
        assert len(augmented) > len(small_features)
        assert augmented.shape[1] == small_features.shape[1]  # Same feature count
        
    def test_qnn_ensemble_creation(self, qnn_config, logger):
        """Test QNN ensemble creation"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        
        ensemble = qnn.create_qnn_ensemble()
        
        assert len(ensemble) == qnn_config.ensemble_size
        assert len(qnn.qnn_ensemble) == qnn_config.ensemble_size
        
        # Test that ensemble members are callable
        for qnn_member in ensemble:
            assert callable(qnn_member)
    
    def test_qnn_architectures(self, qnn_config, logger):
        """Test different QNN architectures"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        
        circuit = qnn.build_quantum_circuit(qnn_config.n_qubits, qnn_config.hyperparams.variational_layers)
        params = qnn._initialize_qnn_parameters()
        features = np.array([1.0, 5.0, 70.0, 0.0, 1.0, 2.0, 3.0, 4.0])  # With engineered features
        
        # Test layered architecture
        output_layered = circuit(params, features, "layered")
        assert len(output_layered) > 0
        assert all(-1 <= val <= 1 for val in output_layered)
        
        # Test tree architecture  
        output_tree = circuit(params, features, "tree")
        assert len(output_tree) > 0
        
        # Test alternating architecture
        output_alt = circuit(params, features, "alternating")
        assert len(output_alt) > 0
        
    def test_parameter_initialization(self, qnn_config, logger):
        """Test QNN parameter initialization"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        # Test different initialization methods
        for init_method in ["xavier", "he", "uniform"]:
            qnn.qnn_config.hyperparams.weight_initialization = init_method
            params = qnn._initialize_qnn_parameters()
            
            assert len(params) > 0
            assert np.all(np.isfinite(params))
            
            # Check parameter ranges are reasonable
            if init_method == "uniform":
                assert np.all(np.abs(params) <= np.pi)
            else:
                assert np.std(params) > 0  # Parameters should have some variance
    
    def test_qnn_to_pkpd_mapping(self, qnn_config, logger):
        """Test QNN output to PK/PD mapping"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        qnn_output = [0.2, -0.5, 0.8, -0.1, 0.6, 0.3, -0.7, 0.4]
        features = np.array([24.0, 10.0, 70.0, 1.0])  # time, dose, bw, comed
        
        pk_pred, pd_pred = qnn._map_qnn_to_pkpd(qnn_output, features)
        
        assert pk_pred > 0  # Concentration should be positive
        assert pd_pred > 0  # Biomarker should be positive
        assert isinstance(pk_pred, (int, float))
        assert isinstance(pd_pred, (int, float))
        
        # Test with minimal output
        minimal_output = [0.1]
        pk_min, pd_min = qnn._map_qnn_to_pkpd(minimal_output, features)
        assert pk_min > 0
        assert pd_min > 0
        
    def test_population_modeling_modes(self, qnn_config, logger, sample_data):
        """Test different population modeling approaches"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        qnn.create_qnn_ensemble()
        
        # Create dummy ensemble parameters
        ensemble_params = [qnn._initialize_qnn_parameters() for _ in range(3)]
        
        # Test hierarchical modeling
        qnn.qnn_config.population_modeling = "hierarchical"
        cost_hier = qnn.cost_function_population(ensemble_params, sample_data)
        assert isinstance(cost_hier, (int, float))
        assert not np.isnan(cost_hier)
        assert not np.isinf(cost_hier)
        
        # Test pooled modeling
        qnn.qnn_config.population_modeling = "pooled"
        cost_pooled = qnn.cost_function_population(ensemble_params, sample_data)
        assert isinstance(cost_pooled, (int, float))
        
        # Test mixed effects modeling
        qnn.qnn_config.population_modeling = "mixed_effects"
        cost_mixed = qnn.cost_function_population(ensemble_params, sample_data)
        assert isinstance(cost_mixed, (int, float))
        
    def test_data_splitting_strategies(self, qnn_config, logger, sample_data):
        """Test different data splitting strategies"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        # Test subject split
        qnn.qnn_config.validation_strategy = "subject_split"
        train_data, val_data = qnn._split_data_for_validation(sample_data)
        
        train_subjects = set(train_data.subjects)
        val_subjects = set(val_data.subjects)
        
        assert len(train_subjects & val_subjects) == 0  # No subject overlap
        assert len(train_data.subjects) > 0
        assert len(val_data.subjects) > 0
        
        # Test that data is properly subset
        assert len(train_data.time_points) == len(train_data.subjects)
        assert len(val_data.time_points) == len(val_data.subjects)
        
    def test_ensemble_weights_calculation(self, qnn_config, logger, sample_data):
        """Test ensemble weight calculation"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        qnn.create_qnn_ensemble()
        
        # Create dummy parameters
        ensemble_params = [qnn._initialize_qnn_parameters() for _ in range(3)]
        
        # Mock cost function to return different values for each member
        with patch.object(qnn, 'cost_function_population', side_effect=[2.0, 1.0, 3.0]):
            weights = qnn._calculate_ensemble_weights(ensemble_params, sample_data)
        
        assert len(weights) == len(ensemble_params)
        assert np.allclose(np.sum(weights), 1.0)  # Weights should sum to 1
        assert np.all(weights >= 0)  # All weights should be non-negative
        
        # Member with lowest cost should have highest weight
        assert weights[1] == np.max(weights)  # Second member had cost 1.0 (lowest)
        
    @pytest.mark.slow
    def test_parameter_optimization(self, qnn_config, logger, sample_data):
        """Test QNN parameter optimization"""
        # Reduce complexity for testing
        qnn_config.ensemble_size = 2
        qnn_config.max_iterations = 10
        qnn_config.n_qubits = 3
        qnn_config.hyperparams.variational_layers = 1
        qnn_config.hyperparams.data_reuploading_layers = 1
        
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        
        result = qnn.optimize_parameters(sample_data)
        
        assert 'ensemble_params' in result
        assert 'ensemble_weights' in result
        assert 'convergence_info' in result
        
        assert len(result['ensemble_params']) == qnn_config.ensemble_size
        assert len(result['ensemble_weights']) == qnn_config.ensemble_size
        assert qnn.is_trained
        
    def test_biomarker_prediction(self, qnn_config, logger, sample_data):
        """Test biomarker prediction with ensemble"""
        # Simplify for testing
        qnn_config.ensemble_size = 2
        qnn_config.n_qubits = 3
        
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        qnn.create_qnn_ensemble()
        
        # Mock training state
        qnn.best_ensemble_weights = [qnn._initialize_qnn_parameters() for _ in range(2)]
        qnn.ensemble_weights = np.array([0.6, 0.4])
        qnn.is_trained = True
        
        # Mock feature scaler
        qnn.feature_scaler = Mock()
        qnn.feature_scaler.transform.return_value = np.array([[1.0, 5.0, 70.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        
        # Mock ensemble QNN outputs
        with patch.object(qnn.qnn_ensemble[0], '__call__', return_value=[0.1, 0.2, 0.3]):
            with patch.object(qnn.qnn_ensemble[1], '__call__', return_value=[0.2, 0.3, 0.4]):
                prediction = qnn.predict_biomarker(
                    dose=5.0,
                    time=np.array([24.0]),
                    covariates={'body_weight': 70.0, 'concomitant_med': 0}
                )
        
        assert len(prediction) == 1
        assert prediction[0] > 0
        
    def test_population_coverage_evaluation(self, qnn_config, logger):
        """Test population coverage evaluation"""
        qnn_config.ensemble_size = 2
        qnn_config.n_qubits = 3
        
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        qnn.create_qnn_ensemble()
        
        # Mock training state
        qnn.is_trained = True
        qnn.best_ensemble_weights = [qnn._initialize_qnn_parameters() for _ in range(2)]
        qnn.ensemble_weights = np.array([0.5, 0.5])
        
        scenario_params = {'weight_range': (60, 90), 'comed_allowed': True}
        
        # Mock prediction to return values below threshold
        with patch.object(qnn, 'predict_biomarker', return_value=np.array([2.5])):
            coverage = qnn._evaluate_qnn_population_coverage(
                dose=8.0,
                dosing_interval=24.0,
                scenario_params=scenario_params,
                target_threshold=3.3
            )
        
        assert 0 <= coverage <= 1
        assert coverage > 0.8  # Should be high since mock returns 2.5 < 3.3
        
    def test_qnn_metrics_calculation(self, qnn_config, logger):
        """Test QNN-specific metrics"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        qnn.create_qnn_ensemble()
        
        # Mock training state
        qnn.is_trained = True
        qnn.best_ensemble_weights = [np.random.randn(20) for _ in range(qnn_config.ensemble_size)]
        qnn.ensemble_weights = np.ones(qnn_config.ensemble_size) / qnn_config.ensemble_size
        
        metrics = qnn._calculate_qnn_metrics()
        
        expected_keys = [
            'ensemble_size', 'total_parameters', 'avg_parameters_per_member',
            'data_reuploading_layers', 'variational_layers', 'ensemble_weight_entropy',
            'expressivity_measure', 'architecture_diversity', 'feature_engineering'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert not np.isnan(metrics[key])
        
        assert metrics['ensemble_size'] == qnn_config.ensemble_size
        assert metrics['feature_engineering'] == 1.0  # Enabled in config
        
    def test_error_handling(self, qnn_config, logger, sample_data):
        """Test error handling"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        # Test prediction without training
        with pytest.raises(ValueError, match="Model not trained"):
            qnn.predict_biomarker(5.0, np.array([24.0]), {'body_weight': 70.0})
        
        # Test invalid architecture
        qnn.setup_quantum_device()
        circuit = qnn.build_quantum_circuit(qnn_config.n_qubits, qnn_config.hyperparams.variational_layers)
        params = qnn._initialize_qnn_parameters()
        features = np.array([1.0, 5.0, 70.0, 0.0])
        
        # This should raise an error for unknown architecture
        with pytest.raises(ValueError, match="Unknown architecture"):
            circuit(params, features, "unknown_arch")
    
    def test_measurement_strategies(self, qnn_config, logger):
        """Test different measurement strategies"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        
        n_qubits = 4
        
        # Test single qubit measurement
        qnn.qnn_config.hyperparams.measurement_strategy = "single_qubit"
        measurements_single = qnn._measure_qnn_output(n_qubits)
        assert len(measurements_single) == 1
        
        # Test multi qubit measurement
        qnn.qnn_config.hyperparams.measurement_strategy = "multi_qubit"
        measurements_multi = qnn._measure_qnn_output(n_qubits)
        assert len(measurements_multi) == min(8, n_qubits)
        
        # Test ensemble measurement
        qnn.qnn_config.hyperparams.measurement_strategy = "ensemble"
        measurements_ensemble = qnn._measure_qnn_output(n_qubits)
        assert len(measurements_ensemble) >= 1
        
    def test_dosing_optimization_structure(self, qnn_config, logger):
        """Test dosing optimization structure"""
        qnn_config.ensemble_size = 2
        qnn_config.n_qubits = 3
        
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        qnn.setup_quantum_device()
        qnn.create_qnn_ensemble()
        
        # Mock training state
        qnn.is_trained = True
        qnn.best_ensemble_weights = [qnn._initialize_qnn_parameters() for _ in range(2)]
        qnn.ensemble_weights = np.array([0.5, 0.5])
        
        # Mock the expensive optimization parts
        with patch.object(qnn, '_optimize_qnn_dosing_regimen') as mock_opt:
            mock_opt.return_value = {'optimal_dose': 6.0, 'coverage': 0.88, 'optimization_success': True}
            
            result = qnn.optimize_dosing(target_threshold=3.3, population_coverage=0.9)
        
        assert hasattr(result, 'optimal_daily_dose')
        assert hasattr(result, 'optimal_weekly_dose')
        assert hasattr(result, 'population_coverage')
        assert hasattr(result, 'quantum_metrics')
        
    def test_data_subset_functionality(self, qnn_config, logger, sample_data):
        """Test data subsetting utility"""
        qnn = QuantumNeuralNetworkFull(qnn_config, logger)
        
        # Create a mask for first half of data
        mask = np.arange(len(sample_data.subjects)) < len(sample_data.subjects) // 2
        
        subset_data = qnn._subset_data_by_mask(sample_data, mask)
        
        assert len(subset_data.subjects) == np.sum(mask)
        assert len(subset_data.time_points) == len(subset_data.subjects)
        assert len(subset_data.doses) == len(subset_data.subjects)
        
        # Check that subset contains correct data
        expected_subjects = sample_data.subjects[mask]
        np.testing.assert_array_equal(subset_data.subjects, expected_subjects)


class TestQNNIntegration:
    """Integration tests for QNN approach"""
    
    @pytest.mark.slow
    def test_end_to_end_qnn_workflow(self):
        """Test complete QNN workflow"""
        np.random.seed(42)
        
        # Create minimal synthetic data
        subjects = np.array([1, 1, 2, 2, 3, 3])
        time_points = np.array([0, 24, 0, 24, 0, 24])
        doses = np.array([5, 5, 10, 10, 0, 0])
        body_weights = np.array([65, 65, 75, 75, 70, 70])
        concomitant_meds = np.array([0, 0, 1, 1, 0, 0])
        
        pk_concentrations = np.array([np.nan, 3.0, np.nan, 5.0, np.nan, np.nan])
        pd_biomarkers = np.array([9.5, 7.0, 11.0, 6.5, 9.0, 9.0])
        
        data = PKPDData(
            subjects=subjects,
            time_points=time_points,
            pk_concentrations=pk_concentrations,
            pd_biomarkers=pd_biomarkers,
            doses=doses,
            body_weights=body_weights,
            concomitant_meds=concomitant_meds
        )
        
        # Create simple QNN config
        hyperparams = QNNHyperparameters(
            learning_rate=0.1,
            encoding_layers=1,
            variational_layers=1,
            data_reuploading_layers=1,
            batch_size=3
        )
        
        config = QNNConfig(
            n_qubits=3,
            max_iterations=8,
            hyperparams=hyperparams,
            ensemble_size=2,
            feature_engineering=False,  # Disable to simplify
            data_augmentation=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = QuantumPKPDLogger(log_dir=temp_dir, experiment_name="qnn_integration")
            
            qnn = QuantumNeuralNetworkFull(config, logger)
            qnn.fit(data)
            
            # Verify training completed
            assert qnn.is_trained
            assert len(qnn.best_ensemble_weights) == config.ensemble_size
            assert len(qnn.ensemble_weights) == config.ensemble_size
            
            # Test prediction
            prediction = qnn.predict_biomarker(
                dose=7.0,
                time=np.array([24.0]),
                covariates={'body_weight': 70.0, 'concomitant_med': 0}
            )
            
            assert len(prediction) == 1
            assert prediction[0] > 0
            
            # Test dosing optimization (simplified)
            with patch.object(qnn, '_evaluate_qnn_population_coverage', return_value=0.82):
                result = qnn.optimize_dosing()
                
                assert result.optimal_daily_dose > 0
                assert result.optimal_weekly_dose > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])