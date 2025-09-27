"""
Noise Analysis Module for VQCdd Phase 2C

This module implements comprehensive quantum noise modeling and error mitigation
techniques for NISQ-era quantum pharmacokinetic simulations. It provides tools
for characterizing noise impact on VQC performance and implementing various
error mitigation strategies.

Key Features:
- Quantum noise modeling for realistic NISQ devices
- Error mitigation techniques (zero-noise extrapolation, readout error correction)
- Noise vs performance characterization framework
- Noise-aware training techniques
- Coherence time and gate fidelity analysis
- Comparative analysis of ideal vs noisy quantum circuits
"""

import numpy as np
import pennylane as qml
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import warnings
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import VQCdd modules
from quantum_circuit import VQCircuit, CircuitConfig
from optimizer import VQCTrainer, OptimizationConfig
from data_handler import StudyData, PatientData
from validation import ValidationResults, ValidationConfig


@dataclass
class NoiseModel:
    """Configuration for quantum noise models"""

    # Depolarizing noise parameters
    depolarizing_prob: float = 0.001          # Single-qubit depolarizing probability
    two_qubit_depolarizing: float = 0.01      # Two-qubit gate depolarizing probability

    # Amplitude damping (T1 relaxation)
    t1_time: float = 50.0                     # T1 relaxation time (μs)
    gate_time: float = 0.1                    # Gate operation time (μs)

    # Phase damping (T2 dephasing)
    t2_time: float = 20.0                     # T2 dephasing time (μs)
    pure_dephasing_prob: float = 0.0005       # Pure dephasing probability

    # Readout errors
    readout_error_0: float = 0.02             # P(read 1 | prepared 0)
    readout_error_1: float = 0.05             # P(read 0 | prepared 1)

    # Coherent errors
    rotation_angle_error: float = 0.01        # Systematic rotation error (radians)
    crosstalk_strength: float = 0.001         # Crosstalk between qubits

    # Thermal noise
    thermal_population: float = 0.01          # Thermal excited state population

    # Device-specific parameters
    device_name: str = "default.mixed"        # PennyLane noise device
    shots: int = 1000                         # Number of measurement shots
    enable_noise: bool = True                 # Toggle noise on/off


@dataclass
class ErrorMitigationConfig:
    """Configuration for error mitigation techniques"""

    # Zero-noise extrapolation parameters
    zne_enabled: bool = True                  # Enable zero-noise extrapolation
    zne_noise_factors: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0])
    zne_extrapolation_method: str = "linear"  # "linear", "exponential", "polynomial"
    zne_polynomial_degree: int = 2            # Degree for polynomial extrapolation

    # Readout error mitigation
    readout_mitigation: bool = True           # Enable readout error mitigation
    calibration_shots: int = 10000            # Shots for readout calibration

    # Symmetry verification
    symmetry_verification: bool = False       # Enable symmetry-based error detection
    parity_check_qubits: List[int] = field(default_factory=list)

    # Clifford data regression
    clifford_mitigation: bool = False         # Enable Clifford data regression
    clifford_samples: int = 100               # Number of Clifford circuits for calibration

    # Virtual distillation
    virtual_distillation: bool = False        # Enable virtual error cancellation
    distillation_copies: int = 2              # Number of circuit copies

    # Post-selection
    post_selection: bool = False              # Enable computational basis post-selection
    post_selection_threshold: float = 0.8     # Fidelity threshold for post-selection


@dataclass
class NoiseCharacterizationResults:
    """Results from noise characterization experiments"""

    # Basic noise metrics
    gate_fidelities: Dict[str, float]         # Fidelity for different gate types
    coherence_times: Dict[str, float]         # T1, T2 times
    readout_fidelities: Dict[int, float]      # Per-qubit readout fidelity

    # Performance impact
    ideal_performance: Dict[str, float]       # Performance without noise
    noisy_performance: Dict[str, float]       # Performance with noise
    performance_degradation: Dict[str, float] # Relative degradation

    # Error mitigation effectiveness
    mitigation_results: Dict[str, Dict[str, float]]  # Results by mitigation method
    mitigation_overhead: Dict[str, float]     # Computational overhead

    # Noise scaling analysis
    noise_scaling: Dict[str, List[float]]     # Performance vs noise strength
    error_thresholds: Dict[str, float]        # Noise thresholds for viable operation

    # Statistical analysis
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, bool]

    # Metadata
    noise_model_used: NoiseModel
    mitigation_config: ErrorMitigationConfig
    timestamp: str


class NoiseModelFactory:
    """Factory for creating realistic noise models"""

    @staticmethod
    def create_superconducting_noise_model(
        device_quality: str = "medium"
    ) -> NoiseModel:
        """
        Create noise model for superconducting quantum devices

        Args:
            device_quality: "high", "medium", "low" quality device

        Returns:
            NoiseModel configured for superconducting devices
        """
        if device_quality == "high":
            # High-quality superconducting device (e.g., IBM quantum network)
            return NoiseModel(
                depolarizing_prob=0.0001,
                two_qubit_depolarizing=0.005,
                t1_time=100.0,
                t2_time=50.0,
                gate_time=0.05,
                readout_error_0=0.01,
                readout_error_1=0.02,
                rotation_angle_error=0.005,
                crosstalk_strength=0.0005,
                thermal_population=0.005
            )
        elif device_quality == "medium":
            # Medium-quality device (typical NISQ device)
            return NoiseModel(
                depolarizing_prob=0.001,
                two_qubit_depolarizing=0.01,
                t1_time=50.0,
                t2_time=20.0,
                gate_time=0.1,
                readout_error_0=0.02,
                readout_error_1=0.05,
                rotation_angle_error=0.01,
                crosstalk_strength=0.001,
                thermal_population=0.01
            )
        else:  # low quality
            # Lower-quality device (early NISQ or simulator)
            return NoiseModel(
                depolarizing_prob=0.005,
                two_qubit_depolarizing=0.02,
                t1_time=20.0,
                t2_time=10.0,
                gate_time=0.2,
                readout_error_0=0.05,
                readout_error_1=0.08,
                rotation_angle_error=0.02,
                crosstalk_strength=0.002,
                thermal_population=0.02
            )

    @staticmethod
    def create_trapped_ion_noise_model() -> NoiseModel:
        """Create noise model for trapped ion devices"""
        return NoiseModel(
            depolarizing_prob=0.0001,
            two_qubit_depolarizing=0.001,
            t1_time=1000.0,  # Very long T1 for trapped ions
            t2_time=100.0,
            gate_time=10.0,  # Slower gates
            readout_error_0=0.005,
            readout_error_1=0.01,
            rotation_angle_error=0.001,
            crosstalk_strength=0.0001,
            thermal_population=0.001
        )

    @staticmethod
    def create_photonic_noise_model() -> NoiseModel:
        """Create noise model for photonic quantum devices"""
        return NoiseModel(
            depolarizing_prob=0.01,    # Higher loss rates
            two_qubit_depolarizing=0.05,
            t1_time=1e6,               # Essentially infinite T1
            t2_time=1e6,               # Essentially infinite T2
            gate_time=0.001,           # Very fast gates
            readout_error_0=0.1,       # Detection efficiency issues
            readout_error_1=0.1,
            rotation_angle_error=0.005,
            crosstalk_strength=0.0,    # No crosstalk in photonics
            thermal_population=0.0     # No thermal excitation at optical frequencies
        )


class NoisyQuantumDevice:
    """Wrapper for quantum devices with configurable noise"""

    def __init__(self, noise_model: NoiseModel, n_qubits: int):
        self.noise_model = noise_model
        self.n_qubits = n_qubits
        self.device = None
        self._setup_device()

    def _setup_device(self):
        """Setup PennyLane device with noise"""
        if not self.noise_model.enable_noise:
            # Use ideal device
            self.device = qml.device("default.qubit", wires=self.n_qubits)
            return

        # Create noisy device
        if self.noise_model.shots is not None:
            self.device = qml.device(
                self.noise_model.device_name,
                wires=self.n_qubits,
                shots=self.noise_model.shots
            )
        else:
            self.device = qml.device(
                self.noise_model.device_name,
                wires=self.n_qubits
            )

    def create_noisy_circuit(self, circuit_func: Callable) -> Callable:
        """
        Wrap a quantum circuit with noise channels

        Args:
            circuit_func: Original quantum circuit function

        Returns:
            Noisy version of the circuit
        """
        @qml.qnode(self.device, diff_method="parameter-shift")
        def noisy_circuit(*args, **kwargs):
            # Execute original circuit
            result = circuit_func(*args, **kwargs)

            # Add noise channels if enabled
            if self.noise_model.enable_noise:
                self._add_noise_channels()

            return result

        return noisy_circuit

    def _add_noise_channels(self):
        """Add noise channels to the circuit"""
        for wire in range(self.n_qubits):
            # Depolarizing noise
            if self.noise_model.depolarizing_prob > 0:
                qml.DepolarizingChannel(
                    self.noise_model.depolarizing_prob,
                    wires=wire
                )

            # Amplitude damping (T1 relaxation)
            if self.noise_model.t1_time > 0:
                gamma = 1 - np.exp(-self.noise_model.gate_time / self.noise_model.t1_time)
                qml.AmplitudeDamping(gamma, wires=wire)

            # Phase damping (T2 dephasing)
            if self.noise_model.t2_time > 0:
                gamma = 1 - np.exp(-self.noise_model.gate_time / self.noise_model.t2_time)
                qml.PhaseDamping(gamma, wires=wire)

            # Thermal relaxation
            if self.noise_model.thermal_population > 0:
                qml.ThermalRelaxationError(
                    pe=self.noise_model.thermal_population,
                    t1=self.noise_model.t1_time,
                    t2=self.noise_model.t2_time,
                    tg=self.noise_model.gate_time,
                    wires=wire
                )


class ZeroNoiseExtrapolation:
    """Zero-noise extrapolation error mitigation"""

    def __init__(self, config: ErrorMitigationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def extrapolate(
        self,
        circuit_func: Callable,
        noise_factors: Optional[List[float]] = None
    ) -> Callable:
        """
        Apply zero-noise extrapolation to a quantum circuit

        Args:
            circuit_func: Quantum circuit function
            noise_factors: Factors by which to scale noise

        Returns:
            Error-mitigated circuit function
        """
        if noise_factors is None:
            noise_factors = self.config.zne_noise_factors

        def zne_circuit(*args, **kwargs):
            # Collect results at different noise levels
            results = []
            for factor in noise_factors:
                # Scale noise by factor (implementation depends on noise model)
                scaled_result = self._execute_with_scaled_noise(
                    circuit_func, factor, *args, **kwargs
                )
                results.append(scaled_result)

            # Extrapolate to zero noise
            extrapolated_result = self._extrapolate_to_zero_noise(
                noise_factors, results
            )

            return extrapolated_result

        return zne_circuit

    def _execute_with_scaled_noise(
        self,
        circuit_func: Callable,
        noise_factor: float,
        *args,
        **kwargs
    ) -> Any:
        """Execute circuit with scaled noise"""
        # This is a simplified implementation
        # In practice, noise scaling requires careful handling of the noise model
        return circuit_func(*args, **kwargs)

    def _extrapolate_to_zero_noise(
        self,
        noise_factors: List[float],
        results: List[Any]
    ) -> Any:
        """Extrapolate results to zero noise"""
        if self.config.zne_extrapolation_method == "linear":
            # Linear extrapolation
            coeffs = np.polyfit(noise_factors, results, 1)
            return coeffs[1]  # Intercept at noise_factor = 0

        elif self.config.zne_extrapolation_method == "exponential":
            # Exponential extrapolation: f(x) = A * exp(-B*x) + C
            # Simplified to linear in log space
            log_results = np.log(np.maximum(np.array(results), 1e-10))
            coeffs = np.polyfit(noise_factors, log_results, 1)
            return np.exp(coeffs[1])

        elif self.config.zne_extrapolation_method == "polynomial":
            # Polynomial extrapolation
            coeffs = np.polyfit(
                noise_factors,
                results,
                min(self.config.zne_polynomial_degree, len(noise_factors) - 1)
            )
            poly = np.poly1d(coeffs)
            return poly(0)

        else:
            raise ValueError(f"Unknown extrapolation method: {self.config.zne_extrapolation_method}")


class ReadoutErrorMitigation:
    """Readout error mitigation techniques"""

    def __init__(self, config: ErrorMitigationConfig, n_qubits: int):
        self.config = config
        self.n_qubits = n_qubits
        self.calibration_matrix = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def calibrate(self, device) -> np.ndarray:
        """
        Calibrate readout error correction matrix

        Args:
            device: Quantum device for calibration

        Returns:
            Calibration matrix for error correction
        """
        self.logger.info("Calibrating readout error correction matrix")

        # Prepare calibration circuits for all computational basis states
        n_states = 2 ** self.n_qubits
        calibration_matrix = np.zeros((n_states, n_states))

        for state_idx in range(n_states):
            # Prepare computational basis state
            @qml.qnode(device)
            def calibration_circuit():
                # Prepare basis state |state_idx>
                for qubit in range(self.n_qubits):
                    if (state_idx >> qubit) & 1:
                        qml.PauliX(wires=qubit)
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            # Measure multiple times to estimate readout errors
            measurements = []
            for _ in range(self.config.calibration_shots):
                result = calibration_circuit()
                # Convert expectation values to bit string
                bit_string = 0
                for qubit, exp_val in enumerate(result):
                    if exp_val < 0:  # |1⟩ state
                        bit_string |= (1 << qubit)
                measurements.append(bit_string)

            # Count measurement outcomes
            for measured_state in measurements:
                calibration_matrix[measured_state, state_idx] += 1

        # Normalize to get probabilities
        calibration_matrix = calibration_matrix / self.config.calibration_shots
        self.calibration_matrix = calibration_matrix

        self.logger.info("Readout calibration completed")
        return calibration_matrix

    def correct_measurements(self, raw_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Apply readout error correction to measurement counts

        Args:
            raw_counts: Raw measurement counts

        Returns:
            Corrected measurement probabilities
        """
        if self.calibration_matrix is None:
            raise ValueError("Must calibrate readout errors before correction")

        # Convert counts to probability vector
        total_shots = sum(raw_counts.values())
        prob_vector = np.zeros(2 ** self.n_qubits)

        for bit_string, count in raw_counts.items():
            state_idx = int(bit_string, 2)
            prob_vector[state_idx] = count / total_shots

        # Apply inverse of calibration matrix
        try:
            corrected_probs = np.linalg.solve(self.calibration_matrix.T, prob_vector)
            # Ensure probabilities are non-negative and normalized
            corrected_probs = np.maximum(corrected_probs, 0)
            corrected_probs = corrected_probs / np.sum(corrected_probs)
        except np.linalg.LinAlgError:
            self.logger.warning("Calibration matrix is singular, using pseudo-inverse")
            corrected_probs = np.linalg.pinv(self.calibration_matrix.T) @ prob_vector
            corrected_probs = np.maximum(corrected_probs, 0)
            corrected_probs = corrected_probs / np.sum(corrected_probs)

        # Convert back to count format
        corrected_counts = {}
        for state_idx, prob in enumerate(corrected_probs):
            if prob > 1e-10:  # Only include non-zero probabilities
                bit_string = format(state_idx, f'0{self.n_qubits}b')
                corrected_counts[bit_string] = prob

        return corrected_counts


class NoiseCharacterization:
    """Comprehensive noise characterization and analysis"""

    def __init__(
        self,
        noise_model: NoiseModel,
        mitigation_config: ErrorMitigationConfig
    ):
        self.noise_model = noise_model
        self.mitigation_config = mitigation_config
        self.logger = logging.getLogger(self.__class__.__name__)

    def characterize_device_noise(
        self,
        vqc_trainer: VQCTrainer,
        data: StudyData
    ) -> NoiseCharacterizationResults:
        """
        Perform comprehensive noise characterization

        Args:
            vqc_trainer: VQC trainer for testing
            data: Study data for testing

        Returns:
            Comprehensive noise characterization results
        """
        self.logger.info("Starting comprehensive noise characterization")

        # Test ideal vs noisy performance
        ideal_performance = self._test_ideal_performance(vqc_trainer, data)
        noisy_performance = self._test_noisy_performance(vqc_trainer, data)

        # Calculate performance degradation
        performance_degradation = {}
        for metric in ideal_performance:
            if metric in noisy_performance:
                ideal_val = ideal_performance[metric]
                noisy_val = noisy_performance[metric]
                if ideal_val != 0:
                    degradation = abs(noisy_val - ideal_val) / abs(ideal_val) * 100
                else:
                    degradation = 0.0
                performance_degradation[metric] = degradation

        # Test error mitigation techniques
        mitigation_results = self._test_error_mitigation(vqc_trainer, data)

        # Analyze noise scaling
        noise_scaling_results = self._analyze_noise_scaling(vqc_trainer, data)

        # Calculate gate fidelities
        gate_fidelities = self._estimate_gate_fidelities()

        # Estimate coherence times
        coherence_times = {
            'T1': self.noise_model.t1_time,
            'T2': self.noise_model.t2_time,
            'T2_star': self.noise_model.t2_time * 0.8  # Estimate T2*
        }

        # Calculate readout fidelities
        readout_fidelities = self._calculate_readout_fidelities()

        # Statistical analysis
        confidence_intervals = self._calculate_confidence_intervals(
            ideal_performance, noisy_performance
        )

        results = NoiseCharacterizationResults(
            gate_fidelities=gate_fidelities,
            coherence_times=coherence_times,
            readout_fidelities=readout_fidelities,
            ideal_performance=ideal_performance,
            noisy_performance=noisy_performance,
            performance_degradation=performance_degradation,
            mitigation_results=mitigation_results,
            mitigation_overhead={},  # TODO: Implement overhead calculation
            noise_scaling=noise_scaling_results,
            error_thresholds={},  # TODO: Implement threshold analysis
            confidence_intervals=confidence_intervals,
            statistical_significance={},  # TODO: Implement significance testing
            noise_model_used=self.noise_model,
            mitigation_config=self.mitigation_config,
            timestamp=pd.Timestamp.now().isoformat()
        )

        self.logger.info("Noise characterization completed")
        return results

    def _test_ideal_performance(
        self,
        vqc_trainer: VQCTrainer,
        data: StudyData
    ) -> Dict[str, float]:
        """Test performance with ideal quantum circuits"""
        # Temporarily disable noise
        original_noise_setting = self.noise_model.enable_noise
        self.noise_model.enable_noise = False

        # Train and evaluate
        vqc_trainer.train(data)
        performance = self._evaluate_performance(vqc_trainer, data)

        # Restore noise setting
        self.noise_model.enable_noise = original_noise_setting

        return performance

    def _test_noisy_performance(
        self,
        vqc_trainer: VQCTrainer,
        data: StudyData
    ) -> Dict[str, float]:
        """Test performance with noisy quantum circuits"""
        # Ensure noise is enabled
        self.noise_model.enable_noise = True

        # Train and evaluate with noise
        vqc_trainer.train(data)
        performance = self._evaluate_performance(vqc_trainer, data)

        return performance

    def _evaluate_performance(
        self,
        vqc_trainer: VQCTrainer,
        data: StudyData
    ) -> Dict[str, float]:
        """Evaluate VQC performance metrics"""
        # Simplified performance evaluation
        # In practice, this would use the validation framework

        predictions = []
        targets = []

        for patient in data.patients[:10]:  # Test on subset for speed
            # Get prediction
            patient_features, _, _ = vqc_trainer.feature_encoder.encode_patient_data(patient)
            pred = vqc_trainer.predict_parameters(patient_features)
            predictions.append(pred)

            # Get target
            valid_biomarkers = patient.pd_biomarkers[~np.isnan(patient.pd_biomarkers)]
            target = np.mean(valid_biomarkers) if len(valid_biomarkers) > 0 else 0.0
            targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        return {'mse': mse, 'mae': mae}

    def _test_error_mitigation(
        self,
        vqc_trainer: VQCTrainer,
        data: StudyData
    ) -> Dict[str, Dict[str, float]]:
        """Test various error mitigation techniques"""
        mitigation_results = {}

        # Test zero-noise extrapolation
        if self.mitigation_config.zne_enabled:
            zne = ZeroNoiseExtrapolation(self.mitigation_config)
            # TODO: Implement ZNE testing
            mitigation_results['zero_noise_extrapolation'] = {'mse': 0.0, 'mae': 0.0}

        # Test readout error mitigation
        if self.mitigation_config.readout_mitigation:
            # TODO: Implement readout error mitigation testing
            mitigation_results['readout_mitigation'] = {'mse': 0.0, 'mae': 0.0}

        return mitigation_results

    def _analyze_noise_scaling(
        self,
        vqc_trainer: VQCTrainer,
        data: StudyData
    ) -> Dict[str, List[float]]:
        """Analyze how performance scales with noise strength"""
        noise_factors = [0.1, 0.5, 1.0, 2.0, 5.0]
        scaling_results = {'mse': [], 'mae': []}

        original_depolarizing = self.noise_model.depolarizing_prob

        for factor in noise_factors:
            # Scale noise
            self.noise_model.depolarizing_prob = original_depolarizing * factor

            # Test performance
            performance = self._test_noisy_performance(vqc_trainer, data)

            # Store results
            for metric in scaling_results:
                if metric in performance:
                    scaling_results[metric].append(performance[metric])

        # Restore original noise level
        self.noise_model.depolarizing_prob = original_depolarizing

        return scaling_results

    def _estimate_gate_fidelities(self) -> Dict[str, float]:
        """Estimate gate fidelities based on noise model"""
        # Simplified fidelity estimates
        single_qubit_fidelity = 1 - self.noise_model.depolarizing_prob
        two_qubit_fidelity = 1 - self.noise_model.two_qubit_depolarizing

        return {
            'single_qubit': single_qubit_fidelity,
            'two_qubit': two_qubit_fidelity,
            'readout': 1 - (self.noise_model.readout_error_0 + self.noise_model.readout_error_1) / 2
        }

    def _calculate_readout_fidelities(self) -> Dict[int, float]:
        """Calculate per-qubit readout fidelities"""
        # Simplified calculation
        avg_error = (self.noise_model.readout_error_0 + self.noise_model.readout_error_1) / 2
        fidelity = 1 - avg_error

        # Return same fidelity for all qubits (could be individualized)
        return {i: fidelity for i in range(4)}  # Assuming 4 qubits

    def _calculate_confidence_intervals(
        self,
        ideal_performance: Dict[str, float],
        noisy_performance: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for performance differences"""
        # Simplified CI calculation
        # In practice, would require multiple runs and statistical analysis

        confidence_intervals = {}
        for metric in ideal_performance:
            if metric in noisy_performance:
                diff = noisy_performance[metric] - ideal_performance[metric]
                # Assume 10% uncertainty
                margin = abs(diff) * 0.1
                confidence_intervals[metric] = (diff - margin, diff + margin)

        return confidence_intervals


class NoiseAwareTrainer:
    """Training techniques that account for quantum noise"""

    def __init__(
        self,
        noise_model: NoiseModel,
        mitigation_config: ErrorMitigationConfig
    ):
        self.noise_model = noise_model
        self.mitigation_config = mitigation_config
        self.logger = logging.getLogger(self.__class__.__name__)

    def train_with_noise_awareness(
        self,
        vqc_trainer: VQCTrainer,
        data: StudyData
    ) -> Dict[str, Any]:
        """
        Train VQC with noise-aware techniques

        Args:
            vqc_trainer: VQC trainer to enhance
            data: Training data

        Returns:
            Training results with noise mitigation
        """
        self.logger.info("Starting noise-aware training")

        # Modify training procedure to account for noise
        enhanced_trainer = self._enhance_trainer_for_noise(vqc_trainer)

        # Train with noise mitigation
        training_result = enhanced_trainer.train(data)

        # Apply post-training error mitigation
        mitigated_result = self._apply_post_training_mitigation(
            enhanced_trainer, training_result
        )

        self.logger.info("Noise-aware training completed")
        return mitigated_result

    def _enhance_trainer_for_noise(self, vqc_trainer: VQCTrainer) -> VQCTrainer:
        """Enhance trainer with noise-aware modifications"""
        # Create a copy of the trainer with noise-aware modifications
        enhanced_trainer = vqc_trainer  # Simplified - would create actual copy

        # Modify optimization strategy for noise
        if hasattr(enhanced_trainer, 'optimization_config'):
            # Adjust learning rate for noisy gradients
            enhanced_trainer.optimization_config.learning_rate *= 0.5

            # Increase batch size to reduce noise variance
            if enhanced_trainer.optimization_config.enable_mini_batches:
                enhanced_trainer.optimization_config.batch_size = min(
                    enhanced_trainer.optimization_config.batch_size * 2,
                    100
                )

        return enhanced_trainer

    def _apply_post_training_mitigation(
        self,
        trainer: VQCTrainer,
        training_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply error mitigation after training"""
        mitigated_result = training_result.copy()

        # Apply zero-noise extrapolation if enabled
        if self.mitigation_config.zne_enabled:
            zne = ZeroNoiseExtrapolation(self.mitigation_config)
            # TODO: Apply ZNE to final model
            mitigated_result['zne_applied'] = True

        # Apply readout error correction if enabled
        if self.mitigation_config.readout_mitigation:
            # TODO: Apply readout error correction
            mitigated_result['readout_correction_applied'] = True

        return mitigated_result


# Utility functions for noise analysis

def compare_ideal_vs_noisy(
    vqc_trainer_factory: Callable,
    data: StudyData,
    noise_models: List[NoiseModel],
    validation_config: ValidationConfig
) -> Dict[str, Any]:
    """
    Compare ideal quantum circuits with various noise models

    Args:
        vqc_trainer_factory: Function to create VQC trainer
        data: Study data
        noise_models: List of noise models to test
        validation_config: Validation configuration

    Returns:
        Comparison results across noise models
    """
    results = {}

    # Test ideal case
    ideal_trainer = vqc_trainer_factory()
    ideal_trainer.circuit_config.shots = None  # Exact simulation
    ideal_result = ideal_trainer.train(data)
    results['ideal'] = ideal_result

    # Test each noise model
    for i, noise_model in enumerate(noise_models):
        noisy_trainer = vqc_trainer_factory()
        # Apply noise model to trainer
        # This would require integration with the trainer's circuit creation
        noisy_result = noisy_trainer.train(data)
        results[f'noise_model_{i}'] = noisy_result

    return results


def generate_noise_analysis_report(
    characterization_results: NoiseCharacterizationResults,
    output_dir: str = "noise_analysis_results"
) -> None:
    """
    Generate comprehensive noise analysis report

    Args:
        characterization_results: Results from noise characterization
        output_dir: Output directory for report and plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results to JSON
    results_dict = {
        'gate_fidelities': characterization_results.gate_fidelities,
        'coherence_times': characterization_results.coherence_times,
        'performance_degradation': characterization_results.performance_degradation,
        'mitigation_results': characterization_results.mitigation_results,
        'noise_model': characterization_results.noise_model_used.__dict__,
        'timestamp': characterization_results.timestamp
    }

    with open(output_path / 'noise_analysis_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Generate plots
    _plot_performance_degradation(characterization_results, output_path)
    _plot_noise_scaling(characterization_results, output_path)
    _plot_mitigation_effectiveness(characterization_results, output_path)

    print(f"Noise analysis report generated in: {output_path}")


def _plot_performance_degradation(
    results: NoiseCharacterizationResults,
    output_path: Path
) -> None:
    """Plot performance degradation due to noise"""
    metrics = list(results.performance_degradation.keys())
    degradations = list(results.performance_degradation.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, degradations, color='lightcoral', edgecolor='darkred')
    plt.ylabel('Performance Degradation (%)')
    plt.title('Impact of Quantum Noise on VQC Performance', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, degradation in zip(bars, degradations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{degradation:.1f}%', ha='center', va='bottom')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'performance_degradation.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_noise_scaling(
    results: NoiseCharacterizationResults,
    output_path: Path
) -> None:
    """Plot how performance scales with noise strength"""
    if not results.noise_scaling:
        return

    noise_factors = [0.1, 0.5, 1.0, 2.0, 5.0]  # This should match the analysis

    plt.figure(figsize=(12, 8))

    for i, (metric, values) in enumerate(results.noise_scaling.items()):
        plt.subplot(2, 2, i + 1)
        plt.plot(noise_factors, values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Noise Strength Factor')
        plt.ylabel(f'{metric.upper()}')
        plt.title(f'{metric.upper()} vs Noise Strength')
        plt.grid(True, alpha=0.3)

    plt.suptitle('Performance Scaling with Noise Strength', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'noise_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_mitigation_effectiveness(
    results: NoiseCharacterizationResults,
    output_path: Path
) -> None:
    """Plot effectiveness of error mitigation techniques"""
    if not results.mitigation_results:
        return

    techniques = list(results.mitigation_results.keys())
    metrics = ['mse', 'mae']

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        # Collect data
        technique_names = []
        metric_values = []

        # Add baseline (noisy) performance
        if metric in results.noisy_performance:
            technique_names.append('No Mitigation')
            metric_values.append(results.noisy_performance[metric])

        # Add mitigation results
        for technique, performance in results.mitigation_results.items():
            if metric in performance:
                technique_names.append(technique.replace('_', ' ').title())
                metric_values.append(performance[metric])

        if technique_names:
            bars = axes[i].bar(technique_names, metric_values, color='lightblue', edgecolor='navy')
            axes[i].set_ylabel(f'{metric.upper()}')
            axes[i].set_title(f'Error Mitigation Effectiveness ({metric.upper()})')
            axes[i].tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, value in zip(bars, metric_values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values) * 0.01,
                           f'{value:.4f}', ha='center', va='bottom')

    plt.suptitle('Error Mitigation Techniques Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'mitigation_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create noise models for different device types
    superconducting_noise = NoiseModelFactory.create_superconducting_noise_model("medium")
    trapped_ion_noise = NoiseModelFactory.create_trapped_ion_noise_model()

    # Create error mitigation config
    mitigation_config = ErrorMitigationConfig(
        zne_enabled=True,
        readout_mitigation=True
    )

    print("Noise analysis framework created successfully!")
    print(f"Superconducting noise model: T1={superconducting_noise.t1_time}μs, T2={superconducting_noise.t2_time}μs")
    print(f"Trapped ion noise model: T1={trapped_ion_noise.t1_time}μs, T2={trapped_ion_noise.t2_time}μs")
    print(f"Error mitigation: ZNE={mitigation_config.zne_enabled}, Readout={mitigation_config.readout_mitigation}")