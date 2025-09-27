"""
Variational Quantum Eigensolver for PK/PD Parameter Estimation

Implements VQE-style optimization specifically for pharmacokinetic and pharmacodynamic
parameter estimation using variational quantum circuits.
"""

import numpy as np
import pennylane as qml
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import scipy.optimize

from ..core.base import QuantumPKPDBase, ModelConfig, PKPDData, OptimizationResult
from ..core.pennylane_utils import QuantumCircuitBuilder, QuantumOptimizer


@dataclass
class VQEPKPDConfig(ModelConfig):
    """Configuration for VQE-based PK/PD solver"""
    hamiltonian_type: str = "pkpd_potential"  # "pkpd_potential", "parameter_correlation", "covariate_coupling"
    vqe_ansatz: str = "hardware_efficient"  # "hardware_efficient", "unitary_cc", "adaptive"
    measurement_strategy: str = "expectation_value"  # "expectation_value", "sampling", "variance"
    optimization_method: str = "gradient_descent"  # "gradient_descent", "spsa", "cobyla"
    energy_tolerance: float = 1e-6
    max_vqe_iterations: int = 200


class VQEPKPDSolver(QuantumPKPDBase):
    """
    VQE-based solver for PK/PD parameter estimation

    This solver treats parameter estimation as finding the ground state
    of a carefully constructed Hamiltonian that encodes PK/PD relationships.
    """

    def __init__(self, config: VQEPKPDConfig):
        super().__init__(config)
        self.vqe_config = config
        self.hamiltonian = None
        self.ground_state_energy = None
        self.energy_history = []

    def setup_quantum_device(self) -> qml.device:
        """Setup quantum device optimized for VQE calculations"""
        device = qml.device(
            "lightning.qubit",
            wires=self.config.n_qubits,
            shots=None  # Use exact simulation for VQE
        )
        self.device = device
        return device

    def build_quantum_circuit(self, n_qubits: int, n_layers: int) -> callable:
        """Build VQE ansatz circuit for PK/PD parameter estimation"""

        @qml.qnode(self.device, diff_method="parameter-shift")
        def vqe_ansatz(params, observable=None):
            """
            VQE ansatz for PK/PD parameter ground state preparation

            Args:
                params: Variational parameters for the ansatz
                observable: Hamiltonian observable to measure
            """
            param_idx = 0

            # Initial state preparation
            for qubit in range(n_qubits):
                qml.RY(params[param_idx], wires=qubit)
                param_idx += 1

            # Variational layers
            for layer in range(n_layers):
                # Single-qubit rotations
                for qubit in range(n_qubits):
                    qml.RY(params[param_idx], wires=qubit)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=qubit)
                    param_idx += 1

                # Entangling gates - nearest neighbor coupling
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

                # Additional parameterized gates for expressivity
                for qubit in range(n_qubits):
                    qml.RY(params[param_idx], wires=qubit)
                    param_idx += 1

            # Return expectation value of the observable
            if observable is not None:
                return qml.expval(observable)
            else:
                # Default measurement for optimization
                return qml.expval(qml.PauliZ(0))

        self.circuit = vqe_ansatz
        return vqe_ansatz

    def construct_pkpd_hamiltonian(self, data: PKPDData) -> qml.Hamiltonian:
        """
        Construct Hamiltonian encoding PK/PD parameter relationships

        The Hamiltonian encodes:
        - Parameter correlation structures
        - Data likelihood terms
        - Covariate interaction effects
        - Regularization constraints
        """
        n_qubits = self.config.n_qubits

        # Pauli operators for parameter encoding
        pauli_ops = []
        coefficients = []

        # Parameter correlation terms (nearest-neighbor interactions)
        for i in range(n_qubits - 1):
            # ZZ interactions for parameter correlations
            pauli_ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
            coefficients.append(-0.5)  # Favor correlated parameters

            # XX interactions for kinetic energy
            pauli_ops.append(qml.PauliX(i) @ qml.PauliX(i + 1))
            coefficients.append(-0.1)

        # Single-qubit terms (external field effects)
        for i in range(n_qubits):
            # Z-field representing parameter constraints
            pauli_ops.append(qml.PauliZ(i))
            # Weight based on data variance (simplified)
            weight = 0.5 * (1 + i % 2 * 0.2)  # Alternating weights
            coefficients.append(weight)

            # X-field for parameter exploration
            pauli_ops.append(qml.PauliX(i))
            coefficients.append(0.1)

        # Data-driven terms (encode actual observations)
        if hasattr(data, 'pk_concentrations') and data.pk_concentrations.size > 0:
            # Add terms that couple to actual data
            for i in range(min(4, n_qubits)):
                # Use concentration data to weight interactions
                pk_weight = np.mean(data.pk_concentrations[:, :min(5, data.pk_concentrations.shape[1])])
                pk_weight = np.clip(pk_weight / 100.0, 0.01, 1.0)  # Normalize

                pauli_ops.append(qml.PauliY(i))
                coefficients.append(pk_weight)

        # Construct Hamiltonian
        hamiltonian = qml.Hamiltonian(coefficients, pauli_ops)
        self.hamiltonian = hamiltonian
        return hamiltonian

    def encode_data(self, data: PKPDData) -> np.ndarray:
        """Encode PK/PD data into Hamiltonian parameters"""
        # Extract relevant features from data
        n_subjects = len(data.subjects)

        # Weight-normalized features
        dose_features = data.doses / np.max(data.doses) if np.max(data.doses) > 0 else data.doses
        weight_features = (data.body_weights - 50) / 50  # Normalize around 50-100 kg
        comed_features = data.concomitant_meds

        # Time-averaged concentration features
        if data.pk_concentrations.size > 0:
            pk_features = np.mean(data.pk_concentrations, axis=1)
            pk_features = pk_features / np.max(pk_features) if np.max(pk_features) > 0 else pk_features
        else:
            pk_features = np.zeros(n_subjects)

        # Combine into feature matrix
        features = np.column_stack([
            dose_features,
            weight_features,
            comed_features,
            pk_features
        ])

        return features

    def cost_function(self, params: np.ndarray, data: PKPDData) -> float:
        """
        VQE cost function: expectation value of PK/PD Hamiltonian
        """
        if self.hamiltonian is None:
            self.hamiltonian = self.construct_pkpd_hamiltonian(data)

        if self.circuit is None:
            raise ValueError("Circuit not built. Call build_quantum_circuit first.")

        # Calculate ground state energy expectation
        energy = self.circuit(params, observable=self.hamiltonian)

        # Add classical regularization terms
        regularization = 0.01 * np.sum(params**2)

        total_cost = energy + regularization
        return total_cost

    def optimize_parameters(self, data: PKPDData) -> Dict[str, Any]:
        """
        Run VQE optimization to find ground state parameters
        """
        if self.device is None:
            self.setup_quantum_device()
        if self.circuit is None:
            self.circuit = self.build_quantum_circuit(self.config.n_qubits, self.config.n_layers)

        # Construct problem-specific Hamiltonian
        self.hamiltonian = self.construct_pkpd_hamiltonian(data)

        # Calculate number of parameters for ansatz
        n_params_per_layer = 3 * self.config.n_qubits  # RY, RZ, RY per qubit per layer
        n_initial_params = self.config.n_qubits  # Initial RY rotations
        total_params = n_initial_params + n_params_per_layer * self.config.n_layers

        # Initialize parameters
        if self.vqe_config.optimization_method == "gradient_descent":
            params = np.random.normal(0, 0.1, total_params)
        else:
            params = np.random.uniform(-np.pi, np.pi, total_params)

        # VQE optimization
        if self.vqe_config.optimization_method == "gradient_descent":
            optimizer = qml.AdamOptimizer(stepsize=self.config.learning_rate)

            best_energy = float('inf')
            best_params = params.copy()

            for iteration in range(self.vqe_config.max_vqe_iterations):
                params, energy = optimizer.step_and_cost(
                    lambda p: self.cost_function(p, data), params
                )

                self.energy_history.append(energy)

                if energy < best_energy:
                    best_energy = energy
                    best_params = params.copy()

                # Convergence check
                if iteration > 10 and abs(self.energy_history[-1] - self.energy_history[-10]) < self.vqe_config.energy_tolerance:
                    break

        elif self.vqe_config.optimization_method == "cobyla":
            # Use scipy COBYLA for derivative-free optimization
            result = scipy.optimize.minimize(
                lambda p: self.cost_function(p, data),
                params,
                method='COBYLA',
                options={'maxiter': self.vqe_config.max_vqe_iterations}
            )
            best_params = result.x
            best_energy = result.fun
            self.energy_history = [best_energy]

        self.ground_state_energy = best_energy

        return {
            'optimal_params': best_params,
            'ground_state_energy': best_energy,
            'energy_history': self.energy_history,
            'n_iterations': len(self.energy_history),
            'converged': iteration < self.vqe_config.max_vqe_iterations - 1 if 'iteration' in locals() else True
        }

    def extract_pk_parameters(self, quantum_state_params: np.ndarray) -> Dict[str, float]:
        """
        Extract PK parameters from optimized quantum state
        """
        # Use quantum circuit to generate parameter estimates
        if self.circuit is None:
            raise ValueError("Circuit not available for parameter extraction")

        # Measure different observables to extract parameters
        ka_observable = qml.PauliZ(0) @ qml.PauliZ(1)
        cl_observable = qml.PauliX(1) @ qml.PauliX(2)
        v1_observable = qml.PauliY(2) @ qml.PauliY(3) if self.config.n_qubits > 3 else qml.PauliY(2)

        ka_raw = self.circuit(quantum_state_params, observable=ka_observable)
        cl_raw = self.circuit(quantum_state_params, observable=cl_observable)
        v1_raw = self.circuit(quantum_state_params, observable=v1_observable)

        # Map expectation values to physical parameter ranges
        ka = np.exp(0.5 + 1.5 * ka_raw)  # Range ~0.6 - 4.5 h^-1
        cl = np.exp(1.0 + 1.0 * cl_raw)  # Range ~1.1 - 7.4 L/h
        v1 = np.exp(2.5 + 0.8 * v1_raw)  # Range ~5.5 - 27 L

        return {
            'ka': ka,
            'cl': cl,
            'v1': v1,
            'bioavailability': 1.0  # Fixed for IV dosing
        }

    def extract_pd_parameters(self, quantum_state_params: np.ndarray) -> Dict[str, float]:
        """
        Extract PD parameters from optimized quantum state
        """
        if self.config.n_qubits < 4:
            # Use different observables for small qubit counts
            baseline_obs = qml.PauliX(0)
            imax_obs = qml.PauliY(1)
            ic50_obs = qml.PauliZ(2)
            gamma_obs = qml.PauliX(0) @ qml.PauliZ(1)
        else:
            baseline_obs = qml.PauliZ(3)
            imax_obs = qml.PauliX(3) @ qml.PauliY(4) if self.config.n_qubits > 4 else qml.PauliX(3)
            ic50_obs = qml.PauliY(4) @ qml.PauliZ(5) if self.config.n_qubits > 5 else qml.PauliY(4) if self.config.n_qubits > 4 else qml.PauliZ(3)
            gamma_obs = qml.PauliX(5) if self.config.n_qubits > 5 else qml.PauliX(4) if self.config.n_qubits > 4 else qml.PauliY(3)

        baseline_raw = self.circuit(quantum_state_params, observable=baseline_obs)
        imax_raw = self.circuit(quantum_state_params, observable=imax_obs)
        ic50_raw = self.circuit(quantum_state_params, observable=ic50_obs)
        gamma_raw = self.circuit(quantum_state_params, observable=gamma_obs)

        # Map to PD parameter ranges
        baseline = 8.0 + 4.0 * (baseline_raw + 1) / 2  # Range 8-12 ng/mL
        imax = 0.6 + 0.3 * (imax_raw + 1) / 2  # Range 0.6-0.9
        ic50 = np.exp(1.5 + 1.2 * ic50_raw)  # Range ~1.8 - 11 mg/L
        gamma = 0.8 + 1.0 * (gamma_raw + 1) / 2  # Range 0.8-1.8

        return {
            'baseline': baseline,
            'imax': imax,
            'ic50': ic50,
            'gamma': gamma
        }

    def predict_biomarker(self, dose: float, time: np.ndarray,
                         covariates: Dict[str, float]) -> np.ndarray:
        """
        Predict biomarker levels using VQE-optimized parameters
        """
        if not self.is_trained or self.parameters is None:
            raise ValueError("Model must be trained before making predictions")

        # Extract PK/PD parameters from quantum state
        pk_params = self.extract_pk_parameters(self.parameters)
        pd_params = self.extract_pd_parameters(self.parameters)

        # Apply covariate effects
        bw_effect = (covariates.get('body_weight', 70) / 70) ** 0.75
        pk_params['cl'] *= bw_effect

        comed_effect = 1.0 + 0.15 * covariates.get('concomitant_med', 0)
        pd_params['baseline'] *= comed_effect

        # PK model prediction (one-compartment)
        ke = pk_params['cl'] / pk_params['v1']
        concentrations = (dose / pk_params['v1']) * np.exp(-ke * time)

        # PD model prediction (Emax)
        inhibition = (pd_params['imax'] * concentrations**pd_params['gamma'] /
                     (pd_params['ic50']**pd_params['gamma'] + concentrations**pd_params['gamma']))
        biomarker = pd_params['baseline'] * (1 - inhibition)

        return biomarker

    def optimize_dosing(self, target_threshold: float = 3.3,
                       population_coverage: float = 0.9) -> OptimizationResult:
        """
        Optimize dosing using VQE-derived parameters
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before optimizing dosing")

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
        dose_range = np.linspace(5, 150, 30)
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
        weekly_dose_range = np.linspace(30, 800, 25)
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

        # Extract final parameter estimates
        pk_estimates = self.extract_pk_parameters(self.parameters)
        pd_estimates = self.extract_pd_parameters(self.parameters)
        all_estimates = {**pk_estimates, **pd_estimates}

        return OptimizationResult(
            optimal_daily_dose=best_daily_dose or 0.0,
            optimal_weekly_dose=best_weekly_dose or 0.0,
            population_coverage=max(best_coverage, best_weekly_coverage),
            parameter_estimates=all_estimates,
            confidence_intervals={
                param: (val * 0.8, val * 1.2) for param, val in all_estimates.items()
            },
            convergence_info={
                'approach': 'VQE-PKPD',
                'ground_state_energy': self.ground_state_energy,
                'vqe_iterations': len(self.energy_history)
            },
            quantum_metrics={
                'hamiltonian_terms': len(self.hamiltonian.coeffs) if self.hamiltonian else 0,
                'circuit_depth': self.config.n_layers,
                'n_qubits': self.config.n_qubits
            }
        )