"""
Population Pharmacokinetic Model using Quantum Machine Learning

Implements quantum-enhanced population PK modeling with inter-individual variability
using quantum neural networks for parameter inference.
"""

import numpy as np
import pennylane as qml
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from ..core.base import QuantumPKPDBase, ModelConfig, PKPDData, OptimizationResult


@dataclass
class PopulationPKConfig(ModelConfig):
    """Configuration for Population PK modeling with QML"""
    population_size: int = 100
    n_random_effects: int = 4  # Number of random effects parameters
    hierarchical_encoding: str = "nested"  # "nested", "parallel", "sequential"
    variability_modeling: str = "quantum_covariance"  # "quantum_covariance", "classical_mixed", "hybrid"
    covariate_effects: List[str] = None  # ["body_weight", "age", "sex", "concomitant_med"]
    shrinkage_regularization: float = 0.1


class PopulationPKModel(QuantumPKPDBase):
    """
    Quantum-enhanced Population Pharmacokinetic Model

    Models inter-individual variability using quantum circuits to capture
    complex correlations in population parameters and covariate effects.
    """

    def __init__(self, config: PopulationPKConfig):
        super().__init__(config)
        self.pop_config = config
        if self.pop_config.covariate_effects is None:
            self.pop_config.covariate_effects = ["body_weight", "concomitant_med"]

        self.population_parameters = None
        self.fixed_effects = None
        self.random_effects_distribution = None
        self.covariate_model = None

    def setup_quantum_device(self) -> qml.device:
        """Setup quantum device for population modeling"""
        device = qml.device(
            "lightning.qubit",
            wires=self.config.n_qubits,
            shots=self.config.shots
        )
        self.device = device
        return device

    def build_quantum_circuit(self, n_qubits: int, n_layers: int) -> callable:
        """
        Build quantum circuit for population parameter inference

        The circuit models:
        - Fixed effects (population mean parameters)
        - Random effects (inter-individual variability)
        - Covariate relationships
        """

        @qml.qnode(self.device)
        def population_qnn(params, individual_features=None, parameter_type="pk"):
            """
            Quantum neural network for population PK parameter prediction

            Args:
                params: Circuit parameters
                individual_features: [dose, body_weight, age, concomitant_med, ...]
                parameter_type: "pk" or "random_effects"
            """
            n_features = len(individual_features) if individual_features is not None else n_qubits
            param_idx = 0

            # Feature encoding layer
            if individual_features is not None:
                for i in range(min(n_qubits, n_features)):
                    # Angle encoding for continuous features
                    feature_val = individual_features[i] if i < len(individual_features) else 0
                    qml.RY(feature_val, wires=i)
                    qml.RZ(feature_val * params[param_idx], wires=i)
                    param_idx += 1

            # Variational layers for parameter inference
            for layer in range(n_layers):
                # Population-level parameter encoding
                for qubit in range(n_qubits):
                    qml.RY(params[param_idx], wires=qubit)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=qubit)
                    param_idx += 1

                # Inter-individual correlation modeling
                if self.pop_config.hierarchical_encoding == "nested":
                    # Nested correlations: nearest neighbor + long-range
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if n_qubits > 2:
                        qml.CNOT(wires=[0, n_qubits - 1])
                elif self.pop_config.hierarchical_encoding == "parallel":
                    # Parallel correlations: star topology
                    for i in range(1, n_qubits):
                        qml.CNOT(wires=[0, i])
                else:  # sequential
                    # Sequential correlations: chain
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])

                # Additional parameterized gates for expressivity
                for qubit in range(n_qubits):
                    qml.RX(params[param_idx], wires=qubit)
                    param_idx += 1

            # Measurements based on parameter type
            if parameter_type == "pk":
                # Measure PK parameters: CL, V, Ka, F
                if n_qubits >= 4:
                    return [qml.expval(qml.PauliZ(i)) for i in range(4)]
                else:
                    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            elif parameter_type == "random_effects":
                # Measure random effects correlations
                measurements = []
                for i in range(min(self.pop_config.n_random_effects, n_qubits)):
                    measurements.append(qml.expval(qml.PauliZ(i)))
                return measurements
            else:
                # Default: measure all qubits
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = population_qnn
        return population_qnn

    def encode_data(self, data: PKPDData) -> np.ndarray:
        """
        Encode population data for quantum processing

        Creates individual-specific feature vectors for each subject
        """
        n_subjects = len(data.subjects)
        n_features = len(self.pop_config.covariate_effects) + 2  # dose + time + covariates

        encoded_population = np.zeros((n_subjects, n_features))

        for i, subject_id in enumerate(data.subjects):
            feature_vector = []

            # Dose information (normalized)
            dose_norm = data.doses[i] / 100.0 if np.max(data.doses) > 0 else 0
            feature_vector.append(dose_norm * 2 * np.pi)

            # Representative time (use median or key time point)
            if len(data.time_points) > 0:
                median_time = np.median(data.time_points)
                time_norm = median_time / np.max(data.time_points) if np.max(data.time_points) > 0 else 0
                feature_vector.append(time_norm * 2 * np.pi)

            # Covariate effects
            for cov_name in self.pop_config.covariate_effects:
                if cov_name == "body_weight":
                    # Normalize body weight (typical range 50-100 kg)
                    bw_norm = (data.body_weights[i] - 50) / 50
                    feature_vector.append(bw_norm * np.pi)
                elif cov_name == "concomitant_med":
                    # Binary covariate
                    feature_vector.append(data.concomitant_meds[i] * np.pi)
                else:
                    # Placeholder for additional covariates
                    feature_vector.append(0.0)

            # Pad or truncate to desired length
            while len(feature_vector) < n_features:
                feature_vector.append(0.0)
            feature_vector = feature_vector[:n_features]

            encoded_population[i] = feature_vector

        return encoded_population

    def predict_individual_pk_parameters(self, individual_features: np.ndarray,
                                       quantum_params: np.ndarray) -> Dict[str, float]:
        """
        Predict PK parameters for an individual using quantum circuit
        """
        if self.circuit is None:
            raise ValueError("Circuit not built")

        # Get quantum predictions
        quantum_output = self.circuit(quantum_params, individual_features, parameter_type="pk")

        # Map quantum outputs to PK parameter ranges
        if len(quantum_output) >= 4:
            cl_raw, v_raw, ka_raw, f_raw = quantum_output[:4]
        else:
            # Handle smaller circuits
            cl_raw = quantum_output[0] if len(quantum_output) > 0 else 0
            v_raw = quantum_output[1] if len(quantum_output) > 1 else 0
            ka_raw = quantum_output[2] if len(quantum_output) > 2 else 0
            f_raw = quantum_output[3] if len(quantum_output) > 3 else 0

        # Transform to physiological ranges
        cl = np.exp(1.0 + 1.5 * cl_raw)  # Clearance: ~1-12 L/h
        v = np.exp(2.5 + 1.0 * v_raw)   # Volume: ~6-33 L
        ka = np.exp(0.5 + 1.0 * ka_raw)  # Absorption rate: ~0.6-4.5 h^-1
        f = 0.7 + 0.25 * (f_raw + 1) / 2  # Bioavailability: 0.7-0.95

        return {
            'cl': cl,
            'v': v,
            'ka': ka,
            'bioavailability': f
        }

    def predict_random_effects(self, individual_features: np.ndarray,
                             quantum_params: np.ndarray) -> Dict[str, float]:
        """
        Predict individual random effects (deviations from population mean)
        """
        random_effects_output = self.circuit(
            quantum_params, individual_features, parameter_type="random_effects"
        )

        # Map to eta (random effects) values
        etas = {}
        param_names = ['eta_cl', 'eta_v', 'eta_ka', 'eta_f']

        for i, param_name in enumerate(param_names):
            if i < len(random_effects_output):
                # Scale to typical random effect range (-2 to +2)
                eta_value = 2.0 * random_effects_output[i]
                etas[param_name] = eta_value
            else:
                etas[param_name] = 0.0

        return etas

    def cost_function(self, params: np.ndarray, data: PKPDData) -> float:
        """
        Population PK likelihood-based cost function
        """
        if self.circuit is None:
            raise ValueError("Circuit not built")

        # Encode population data
        population_features = self.encode_data(data)

        total_likelihood = 0.0
        n_observations = 0

        # Iterate through individuals
        for i, subject_id in enumerate(data.subjects):
            try:
                # Predict individual PK parameters
                pk_params = self.predict_individual_pk_parameters(
                    population_features[i], params
                )

                # Get individual-specific data
                subject_mask = (data.subjects == subject_id)
                if not np.any(subject_mask):
                    continue

                individual_doses = data.doses[i] if i < len(data.doses) else 100
                individual_bw = data.body_weights[i] if i < len(data.body_weights) else 70

                # Apply covariate effects
                bw_effect = (individual_bw / 70) ** 0.75
                pk_params['cl'] *= bw_effect

                # Simple PK prediction (one-compartment model)
                ke = pk_params['cl'] / pk_params['v']

                # Calculate likelihood for observed concentrations
                for j, time in enumerate(data.time_points[:min(10, len(data.time_points))]):
                    if i < data.pk_concentrations.shape[0] and j < data.pk_concentrations.shape[1]:
                        observed_conc = data.pk_concentrations[i, j]

                        if not np.isnan(observed_conc) and observed_conc > 0:
                            # Predicted concentration
                            pred_conc = (individual_doses * pk_params['bioavailability'] / pk_params['v']) * np.exp(-ke * time)

                            # Log-likelihood contribution (assuming log-normal distribution)
                            log_pred = np.log(pred_conc + 1e-8)
                            log_obs = np.log(observed_conc + 1e-8)

                            likelihood_contrib = (log_pred - log_obs) ** 2
                            total_likelihood += likelihood_contrib
                            n_observations += 1

            except Exception:
                # Skip problematic individuals
                continue

        # Add regularization for population parameters
        regularization = self.pop_config.shrinkage_regularization * np.sum(params**2)

        # Return negative log-likelihood (to minimize)
        if n_observations > 0:
            return total_likelihood / n_observations + regularization
        else:
            return 1000.0 + regularization  # High cost if no valid observations

    def optimize_parameters(self, data: PKPDData) -> Dict[str, Any]:
        """
        Optimize population PK parameters using quantum circuits
        """
        if self.device is None:
            self.setup_quantum_device()
        if self.circuit is None:
            self.circuit = self.build_quantum_circuit(self.config.n_qubits, self.config.n_layers)

        # Calculate parameter count
        n_feature_params = min(self.config.n_qubits, len(self.pop_config.covariate_effects) + 2)
        n_layer_params = 3 * self.config.n_qubits  # RY, RZ, RX per qubit per layer
        total_params = n_feature_params + n_layer_params * self.config.n_layers

        # Initialize parameters
        params = np.random.normal(0, 0.1, total_params)

        # Optimization
        optimizer = qml.AdamOptimizer(stepsize=self.config.learning_rate)

        cost_history = []
        best_cost = float('inf')
        best_params = params.copy()

        for iteration in range(self.config.max_iterations):
            params, cost = optimizer.step_and_cost(
                lambda p: self.cost_function(p, data), params
            )

            cost_history.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()

            # Check convergence
            if iteration > 15 and np.abs(np.mean(cost_history[-5:]) - np.mean(cost_history[-10:-5])) < self.config.convergence_threshold:
                break

        return {
            'optimal_params': best_params,
            'final_cost': best_cost,
            'cost_history': cost_history,
            'n_iterations': iteration + 1,
            'converged': iteration < self.config.max_iterations - 1
        }

    def predict_biomarker(self, dose: float, time: np.ndarray,
                         covariates: Dict[str, float]) -> np.ndarray:
        """
        Predict biomarker using population PK model
        """
        if not self.is_trained or self.parameters is None:
            raise ValueError("Model must be trained before making predictions")

        # Create individual feature vector
        individual_features = []

        # Dose (normalized)
        dose_norm = dose / 100.0 * 2 * np.pi
        individual_features.append(dose_norm)

        # Time (use median time for feature vector)
        median_time = np.median(time)
        time_norm = median_time / np.max(time) if np.max(time) > 0 else 0
        individual_features.append(time_norm * 2 * np.pi)

        # Covariates
        for cov_name in self.pop_config.covariate_effects:
            if cov_name == "body_weight":
                bw_norm = (covariates.get('body_weight', 70) - 50) / 50
                individual_features.append(bw_norm * np.pi)
            elif cov_name == "concomitant_med":
                individual_features.append(covariates.get('concomitant_med', 0) * np.pi)
            else:
                individual_features.append(0.0)

        # Pad to required length
        while len(individual_features) < self.config.n_qubits:
            individual_features.append(0.0)
        individual_features = np.array(individual_features[:self.config.n_qubits])

        # Predict PK parameters
        pk_params = self.predict_individual_pk_parameters(individual_features, self.parameters)

        # Apply covariate effects
        bw_effect = (covariates.get('body_weight', 70) / 70) ** 0.75
        pk_params['cl'] *= bw_effect

        # PK model simulation
        ke = pk_params['cl'] / pk_params['v']
        concentrations = (dose * pk_params['bioavailability'] / pk_params['v']) * np.exp(-ke * time)

        # Simple PD model (placeholder - should be integrated with PD approach)
        baseline = 10.0
        imax = 0.8
        ic50 = 5.0
        gamma = 1.0

        inhibition = imax * concentrations**gamma / (ic50**gamma + concentrations**gamma)
        biomarker = baseline * (1 - inhibition)

        return biomarker

    def optimize_dosing(self, target_threshold: float = 3.3,
                       population_coverage: float = 0.9) -> OptimizationResult:
        """
        Optimize dosing using population PK model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before optimizing dosing")

        # Generate synthetic population
        n_population = self.pop_config.population_size
        body_weights = np.random.normal(75, 12, n_population)
        body_weights = np.clip(body_weights, 50, 100)
        concomitant_meds = np.random.binomial(1, 0.3, n_population)

        population_params = {
            'body_weight': body_weights,
            'concomitant_med': concomitant_meds
        }

        # Dose optimization
        dose_range = np.linspace(10, 200, 25)
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

        # Weekly dose optimization
        weekly_dose_range = np.linspace(50, 1000, 20)
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

        # Extract representative population parameters
        median_features = np.array([
            50.0,  # median dose (normalized)
            0.5,   # median time
            0.5,   # median body weight effect
            0.3    # median concomitant med effect
        ])

        pop_pk_params = self.predict_individual_pk_parameters(median_features, self.parameters)

        return OptimizationResult(
            optimal_daily_dose=best_daily_dose or 0.0,
            optimal_weekly_dose=best_weekly_dose or 0.0,
            population_coverage=max(best_coverage, best_weekly_coverage),
            parameter_estimates=pop_pk_params,
            confidence_intervals={
                param: (val * 0.7, val * 1.3) for param, val in pop_pk_params.items()
            },
            convergence_info={
                'approach': 'Population PK QML',
                'population_size': n_population,
                'n_covariates': len(self.pop_config.covariate_effects)
            },
            quantum_metrics={
                'n_qubits': self.config.n_qubits,
                'n_layers': self.config.n_layers,
                'parameter_count': len(self.parameters) if self.parameters is not None else 0
            }
        )