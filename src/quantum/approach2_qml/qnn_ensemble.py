"""
Quantum Neural Network Ensemble for Robust PK/PD Predictions

Implements ensemble methods using multiple quantum neural networks to improve
prediction reliability and uncertainty quantification.
"""

import numpy as np
import pennylane as qml
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import concurrent.futures
from collections import defaultdict

from ..core.base import QuantumPKPDBase, ModelConfig, PKPDData, OptimizationResult
from .quantum_neural_network import QuantumNeuralNetwork, QNNConfig


@dataclass
class QNNEnsembleConfig(ModelConfig):
    """Configuration for QNN Ensemble"""
    n_ensemble_members: int = 5
    ensemble_strategy: str = "bootstrap"  # "bootstrap", "bagging", "diverse_init", "architecture_mix"
    voting_method: str = "weighted_average"  # "simple_average", "weighted_average", "median", "uncertainty_weighted"
    diversity_regularization: float = 0.1
    parallel_training: bool = True
    uncertainty_quantification: bool = True
    member_configs: List[QNNConfig] = None


class QNNEnsemble(QuantumPKPDBase):
    """
    Ensemble of Quantum Neural Networks for Robust PK/PD Modeling

    Combines multiple QNNs to improve prediction accuracy and provide
    uncertainty quantification through ensemble variance.
    """

    def __init__(self, config: QNNEnsembleConfig):
        super().__init__(config)
        self.ensemble_config = config
        self.ensemble_members = []
        self.member_weights = None
        self.ensemble_predictions = None
        self.prediction_uncertainties = None

        # Create diverse ensemble members
        self._create_ensemble_members()

    def _create_ensemble_members(self):
        """Create diverse ensemble members with different configurations"""
        self.ensemble_members = []

        if self.ensemble_config.member_configs is not None:
            # Use provided configurations
            for i, member_config in enumerate(self.ensemble_config.member_configs):
                qnn = QuantumNeuralNetwork(member_config)
                self.ensemble_members.append(qnn)
        else:
            # Generate diverse configurations automatically
            for i in range(self.ensemble_config.n_ensemble_members):
                member_config = self._generate_diverse_config(i)
                qnn = QuantumNeuralNetwork(member_config)
                self.ensemble_members.append(qnn)

        # Initialize equal weights
        self.member_weights = np.ones(len(self.ensemble_members)) / len(self.ensemble_members)

    def _generate_diverse_config(self, member_index: int) -> QNNConfig:
        """Generate diverse QNN configuration for ensemble member"""
        base_config = QNNConfig(
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
            learning_rate=self.config.learning_rate,
            max_iterations=self.config.max_iterations,
            convergence_threshold=self.config.convergence_threshold,
            shots=self.config.shots
        )

        # Introduce diversity based on strategy
        if self.ensemble_config.ensemble_strategy == "architecture_mix":
            # Vary architecture parameters
            architectures = ["layered", "tree", "mps"]
            base_config.architecture = architectures[member_index % len(architectures)]

            encoding_layers_options = [1, 2, 3]
            base_config.encoding_layers = encoding_layers_options[member_index % len(encoding_layers_options)]

            variational_layers_options = [2, 3, 4, 5]
            base_config.variational_layers = variational_layers_options[member_index % len(variational_layers_options)]

        elif self.ensemble_config.ensemble_strategy == "diverse_init":
            # Vary hyperparameters
            learning_rates = [0.01, 0.02, 0.005, 0.03, 0.015]
            base_config.learning_rate = learning_rates[member_index % len(learning_rates)]

            dropout_probs = [0.0, 0.1, 0.2, 0.05, 0.15]
            base_config.dropout_probability = dropout_probs[member_index % len(dropout_probs)]

        # Vary measurement strategies
        measurement_strategies = ["single_qubit", "multi_qubit", "ensemble"]
        base_config.measurement_strategy = measurement_strategies[member_index % len(measurement_strategies)]

        # Enable/disable data reuploading randomly
        base_config.data_reuploading = (member_index % 2 == 0)

        return base_config

    def setup_quantum_device(self) -> qml.device:
        """Setup quantum device for ensemble (delegates to members)"""
        # Each ensemble member will setup its own device
        for member in self.ensemble_members:
            member.setup_quantum_device()

        # Create a representative device for the ensemble
        device = qml.device(
            "lightning.qubit",
            wires=self.config.n_qubits,
            shots=self.config.shots
        )
        self.device = device
        return device

    def build_quantum_circuit(self, n_qubits: int, n_layers: int) -> callable:
        """Build quantum circuits for all ensemble members"""
        for member in self.ensemble_members:
            member.build_quantum_circuit(n_qubits, n_layers)

        # Create a representative circuit for the ensemble
        @qml.qnode(self.device)
        def ensemble_representative_circuit(params, x=None):
            # This is just a representative - actual ensemble uses member circuits
            for i in range(n_qubits):
                qml.RY(x[i % len(x)] if x is not None else 0, wires=i)

            param_idx = 0
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(params[param_idx], wires=qubit)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=qubit)
                    param_idx += 1

                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(min(4, n_qubits))]

        self.circuit = ensemble_representative_circuit
        return ensemble_representative_circuit

    def encode_data(self, data: PKPDData) -> np.ndarray:
        """Encode data using first ensemble member's encoding (they should be similar)"""
        if len(self.ensemble_members) == 0:
            raise ValueError("No ensemble members available")

        return self.ensemble_members[0].encode_data(data)

    def _train_single_member(self, member_index: int, data: PKPDData, bootstrap_indices: Optional[np.ndarray] = None) -> Tuple[int, Dict[str, Any]]:
        """Train a single ensemble member (for parallel execution)"""
        member = self.ensemble_members[member_index]

        # Prepare training data
        if bootstrap_indices is not None and self.ensemble_config.ensemble_strategy == "bootstrap":
            # Create bootstrap sample
            bootstrap_data = self._create_bootstrap_sample(data, bootstrap_indices)
            training_data = bootstrap_data
        else:
            training_data = data

        # Train the member
        try:
            optimization_result = member.optimize_parameters(training_data)
            member.parameters = optimization_result['optimal_params']
            member.is_trained = True

            return member_index, {
                'success': True,
                'optimization_result': optimization_result,
                'member_id': member_index
            }
        except Exception as e:
            return member_index, {
                'success': False,
                'error': str(e),
                'member_id': member_index
            }

    def _create_bootstrap_sample(self, data: PKPDData, indices: np.ndarray) -> PKPDData:
        """Create bootstrap sample of the data"""
        # Sample with replacement
        bootstrap_subjects = data.subjects[indices]
        bootstrap_doses = data.doses[indices]
        bootstrap_body_weights = data.body_weights[indices]
        bootstrap_concomitant_meds = data.concomitant_meds[indices]

        # For concentration data, we need to handle the 2D structure
        if hasattr(data, 'pk_concentrations') and data.pk_concentrations.size > 0:
            bootstrap_pk_concentrations = data.pk_concentrations[indices, :]
        else:
            bootstrap_pk_concentrations = np.array([])

        if hasattr(data, 'pd_biomarkers') and data.pd_biomarkers.size > 0:
            bootstrap_pd_biomarkers = data.pd_biomarkers[indices, :]
        else:
            bootstrap_pd_biomarkers = np.array([])

        return PKPDData(
            subjects=bootstrap_subjects,
            time_points=data.time_points,  # Keep same time grid
            pk_concentrations=bootstrap_pk_concentrations,
            pd_biomarkers=bootstrap_pd_biomarkers,
            doses=bootstrap_doses,
            body_weights=bootstrap_body_weights,
            concomitant_meds=bootstrap_concomitant_meds
        )

    def cost_function(self, params: np.ndarray, data: PKPDData) -> float:
        """
        Ensemble cost function (not directly used - members have their own cost functions)
        """
        # This is a placeholder - ensemble training uses member cost functions
        if len(self.ensemble_members) == 0:
            return 1000.0

        # Average cost across trained members
        total_cost = 0.0
        n_trained = 0

        for member in self.ensemble_members:
            if member.is_trained:
                try:
                    member_cost = member.cost_function(member.parameters, data)
                    total_cost += member_cost
                    n_trained += 1
                except Exception:
                    continue

        if n_trained > 0:
            return total_cost / n_trained
        else:
            return 1000.0

    def optimize_parameters(self, data: PKPDData) -> Dict[str, Any]:
        """Train all ensemble members"""
        if not self.ensemble_members:
            self._create_ensemble_members()

        n_subjects = len(data.subjects)
        training_results = []

        # Generate bootstrap samples for each member
        bootstrap_samples = []
        for i in range(len(self.ensemble_members)):
            if self.ensemble_config.ensemble_strategy == "bootstrap":
                # Create bootstrap indices
                bootstrap_indices = np.random.choice(n_subjects, n_subjects, replace=True)
                bootstrap_samples.append(bootstrap_indices)
            else:
                bootstrap_samples.append(None)

        # Train ensemble members
        if self.ensemble_config.parallel_training and len(self.ensemble_members) > 1:
            # Parallel training
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(self.ensemble_members))) as executor:
                future_to_member = {
                    executor.submit(self._train_single_member, i, data, bootstrap_samples[i]): i
                    for i in range(len(self.ensemble_members))
                }

                for future in concurrent.futures.as_completed(future_to_member):
                    member_index, result = future.result()
                    training_results.append(result)
        else:
            # Sequential training
            for i in range(len(self.ensemble_members)):
                member_index, result = self._train_single_member(i, data, bootstrap_samples[i])
                training_results.append(result)

        # Calculate member weights based on performance
        self._calculate_member_weights(training_results)

        # Summary statistics
        successful_members = sum(1 for result in training_results if result['success'])
        total_members = len(self.ensemble_members)

        return {
            'successful_members': successful_members,
            'total_members': total_members,
            'member_weights': self.member_weights.tolist(),
            'training_results': training_results,
            'ensemble_ready': successful_members > 0
        }

    def _calculate_member_weights(self, training_results: List[Dict[str, Any]]):
        """Calculate weights for ensemble members based on training performance"""
        weights = np.zeros(len(self.ensemble_members))

        if self.ensemble_config.voting_method == "simple_average":
            # Equal weights for successful members
            for result in training_results:
                if result['success']:
                    weights[result['member_id']] = 1.0
        elif self.ensemble_config.voting_method == "weighted_average":
            # Weight based on training performance (inverse of final cost)
            for result in training_results:
                if result['success']:
                    final_cost = result['optimization_result'].get('final_loss', 1.0)
                    # Use inverse of cost as weight (add small constant to avoid division by zero)
                    weights[result['member_id']] = 1.0 / (final_cost + 1e-6)

        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all members failed, use equal weights
            weights = np.ones(len(self.ensemble_members)) / len(self.ensemble_members)

        self.member_weights = weights

    def predict_biomarker(self, dose: float, time: np.ndarray,
                         covariates: Dict[str, float]) -> np.ndarray:
        """Predict biomarker using ensemble of QNNs"""
        if not any(member.is_trained for member in self.ensemble_members):
            raise ValueError("Ensemble must be trained before making predictions")

        member_predictions = []
        member_weights_valid = []

        # Collect predictions from all trained members
        for i, member in enumerate(self.ensemble_members):
            if member.is_trained and self.member_weights[i] > 0:
                try:
                    prediction = member.predict_biomarker(dose, time, covariates)
                    member_predictions.append(prediction)
                    member_weights_valid.append(self.member_weights[i])
                except Exception:
                    # Skip members that fail to predict
                    continue

        if not member_predictions:
            raise ValueError("No ensemble members could make predictions")

        # Combine predictions
        member_predictions = np.array(member_predictions)
        member_weights_valid = np.array(member_weights_valid)

        # Normalize weights
        if np.sum(member_weights_valid) > 0:
            member_weights_valid = member_weights_valid / np.sum(member_weights_valid)

        if self.ensemble_config.voting_method == "median":
            # Use median prediction
            ensemble_prediction = np.median(member_predictions, axis=0)
        elif self.ensemble_config.voting_method in ["simple_average", "weighted_average", "uncertainty_weighted"]:
            # Weighted average
            ensemble_prediction = np.average(member_predictions, axis=0, weights=member_weights_valid)
        else:
            # Default: simple average
            ensemble_prediction = np.mean(member_predictions, axis=0)

        # Store predictions for uncertainty quantification
        self.ensemble_predictions = member_predictions
        if self.ensemble_config.uncertainty_quantification:
            self.prediction_uncertainties = np.std(member_predictions, axis=0)

        return ensemble_prediction

    def get_prediction_uncertainty(self) -> Optional[np.ndarray]:
        """Get uncertainty estimates from ensemble predictions"""
        if self.ensemble_config.uncertainty_quantification and self.prediction_uncertainties is not None:
            return self.prediction_uncertainties
        else:
            return None

    def get_confidence_intervals(self, confidence_level: float = 0.95) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get confidence intervals from ensemble predictions"""
        if self.ensemble_predictions is None:
            return None

        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        lower_bounds = np.percentile(self.ensemble_predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(self.ensemble_predictions, upper_percentile, axis=0)

        return lower_bounds, upper_bounds

    def optimize_dosing(self, target_threshold: float = 3.3,
                       population_coverage: float = 0.9) -> OptimizationResult:
        """Optimize dosing using ensemble predictions"""
        if not any(member.is_trained for member in self.ensemble_members):
            raise ValueError("Ensemble must be trained before optimizing dosing")

        # Generate population for simulation
        n_population = 1000
        body_weights = np.random.normal(75, 12, n_population)
        body_weights = np.clip(body_weights, 50, 100)
        concomitant_meds = np.random.binomial(1, 0.3, n_population)

        population_params = {
            'body_weight': body_weights,
            'concomitant_med': concomitant_meds
        }

        # Optimize daily dose
        dose_range = np.linspace(10, 200, 20)
        best_daily_dose = None
        best_coverage = 0.0

        for dose in dose_range:
            coverage = self.evaluate_population_coverage(
                dose=dose,
                dosing_interval=24.0,
                population_params=population_params,
                threshold=target_threshold
            )

            if coverage >= population_coverage and coverage > best_coverage:
                best_daily_dose = dose
                best_coverage = coverage

        # Optimize weekly dose
        weekly_dose_range = np.linspace(50, 1000, 15)
        best_weekly_dose = None
        best_weekly_coverage = 0.0

        for weekly_dose in weekly_dose_range:
            coverage = self.evaluate_population_coverage(
                dose=weekly_dose,
                dosing_interval=168.0,
                population_params=population_params,
                threshold=target_threshold
            )

            if coverage >= population_coverage and coverage > best_weekly_coverage:
                best_weekly_dose = weekly_dose
                best_weekly_coverage = coverage

        # Aggregate parameter estimates from ensemble members
        parameter_estimates = {}
        trained_members = [member for member in self.ensemble_members if member.is_trained]

        if trained_members:
            # Use first trained member's parameter structure as template
            if hasattr(trained_members[0], 'parameters') and trained_members[0].parameters is not None:
                parameter_estimates['ensemble_size'] = len(trained_members)
                parameter_estimates['member_weights'] = self.member_weights.tolist()

        return OptimizationResult(
            optimal_daily_dose=best_daily_dose or 0.0,
            optimal_weekly_dose=best_weekly_dose or 0.0,
            population_coverage=max(best_coverage, best_weekly_coverage),
            parameter_estimates=parameter_estimates,
            confidence_intervals={
                'daily_dose': (best_daily_dose * 0.8, best_daily_dose * 1.2) if best_daily_dose else (0, 0),
                'weekly_dose': (best_weekly_dose * 0.8, best_weekly_dose * 1.2) if best_weekly_dose else (0, 0)
            },
            convergence_info={
                'approach': 'QNN Ensemble',
                'ensemble_size': len([m for m in self.ensemble_members if m.is_trained]),
                'voting_method': self.ensemble_config.voting_method,
                'uncertainty_quantification': self.ensemble_config.uncertainty_quantification
            },
            quantum_metrics={
                'total_ensemble_members': len(self.ensemble_members),
                'successful_members': len([m for m in self.ensemble_members if m.is_trained]),
                'n_qubits_per_member': self.config.n_qubits,
                'n_layers_per_member': self.config.n_layers
            }
        )