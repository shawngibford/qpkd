"""
Parameter Mapping Module for VQCdd

This module handles the transformation between quantum circuit outputs and
pharmacokinetic/pharmacodynamic parameters. The mapping ensures that quantum
outputs (expectation values in range [-1, 1]) are converted to physiologically
meaningful parameter values with appropriate bounds.

Key Features:
- Sigmoid transformations for bounded parameter spaces
- Separate PK and PD parameter mapping
- Physiological constraints and validation
- Inverse mapping for parameter initialization
- Uncertainty quantification support
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging


@dataclass
class ParameterBounds:
    """Physiological bounds for PK/PD parameters"""

    # Pharmacokinetic parameters
    ka: Tuple[float, float] = (0.1, 10.0)      # Absorption rate (1/h)
    cl: Tuple[float, float] = (1.0, 50.0)      # Clearance (L/h)
    v1: Tuple[float, float] = (10.0, 100.0)    # Central volume (L)
    q: Tuple[float, float] = (0.5, 20.0)       # Inter-compartmental clearance (L/h)
    v2: Tuple[float, float] = (20.0, 200.0)    # Peripheral volume (L)

    # Pharmacodynamic parameters
    baseline: Tuple[float, float] = (2.0, 25.0)    # Baseline biomarker (ng/mL)
    imax: Tuple[float, float] = (0.1, 0.9)         # Maximum inhibition (capped at 90% to avoid numerical issues)
    ic50: Tuple[float, float] = (0.5, 50.0)        # IC50 (mg/L)
    gamma: Tuple[float, float] = (0.5, 4.0)        # Hill coefficient

    # Warning thresholds for parameter validation
    imax_warning_threshold: float = 0.95           # Warn only above 95% inhibition (extreme values)

    def get_bounds_dict(self) -> Dict[str, Tuple[float, float]]:
        """Return all bounds as dictionary"""
        return {
            'ka': self.ka, 'cl': self.cl, 'v1': self.v1, 'q': self.q, 'v2': self.v2,
            'baseline': self.baseline, 'imax': self.imax, 'ic50': self.ic50, 'gamma': self.gamma
        }

    def validate_parameter(self, param_name: str, value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if parameter value is within physiological bounds"""
        bounds_dict = self.get_bounds_dict()
        if param_name not in bounds_dict:
            return False

        min_val, max_val = bounds_dict[param_name]

        # Handle both scalar and array inputs
        if isinstance(value, np.ndarray):
            return (min_val <= value) & (value <= max_val)
        else:
            return min_val <= value <= max_val


class TransformationFunction(ABC):
    """Abstract base class for parameter transformation functions"""

    @abstractmethod
    def forward(self, quantum_output: float, min_val: float, max_val: float) -> float:
        """Transform quantum output to parameter value"""
        pass

    @abstractmethod
    def inverse(self, param_value: float, min_val: float, max_val: float) -> float:
        """Transform parameter value back to quantum output"""
        pass


class SigmoidTransform(TransformationFunction):
    """Sigmoid transformation for bounded parameters"""

    def __init__(self, steepness: float = 5.0):
        """
        Initialize sigmoid transformation

        Args:
            steepness: Controls the steepness of the sigmoid curve
        """
        self.steepness = steepness

    def forward(self, quantum_output: float, min_val: float, max_val: float) -> float:
        """
        Transform quantum output [-1, 1] to parameter range [min_val, max_val]

        Args:
            quantum_output: Quantum expectation value in [-1, 1]
            min_val: Minimum parameter value
            max_val: Maximum parameter value

        Returns:
            Transformed parameter value
        """
        # Clamp quantum output to valid range
        quantum_output = np.clip(quantum_output, -1.0, 1.0)

        # Sigmoid transformation
        sigmoid_val = 1.0 / (1.0 + np.exp(-self.steepness * quantum_output))

        # Scale to parameter range
        return min_val + (max_val - min_val) * sigmoid_val

    def inverse(self, param_value: float, min_val: float, max_val: float) -> float:
        """
        Transform parameter value back to quantum output range

        Args:
            param_value: Parameter value in [min_val, max_val]
            min_val: Minimum parameter value
            max_val: Maximum parameter value

        Returns:
            Quantum output in [-1, 1]
        """
        # Normalize to [0, 1]
        normalized = (param_value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0.001, 0.999)  # Avoid infinite values

        # Inverse sigmoid
        return np.log(normalized / (1 - normalized)) / self.steepness


class LinearTransform(TransformationFunction):
    """Linear transformation for unbounded or roughly linear parameters"""

    def forward(self, quantum_output: float, min_val: float, max_val: float) -> float:
        """Linear scaling from [-1, 1] to [min_val, max_val]"""
        quantum_output = np.clip(quantum_output, -1.0, 1.0)
        return min_val + (max_val - min_val) * (quantum_output + 1.0) / 2.0

    def inverse(self, param_value: float, min_val: float, max_val: float) -> float:
        """Linear scaling from [min_val, max_val] to [-1, 1]"""
        normalized = (param_value - min_val) / (max_val - min_val)
        return 2.0 * normalized - 1.0


class QuantumParameterMapper:
    """
    Maps quantum circuit outputs to pharmacokinetic and pharmacodynamic parameters

    This class handles the critical transformation between quantum expectation values
    and meaningful PK/PD parameters, ensuring physiological validity and providing
    clear traceability for scientific analysis.
    """

    def __init__(self, bounds: Optional[ParameterBounds] = None,
                 transform_type: str = "sigmoid",
                 n_qubits: int = 4):
        """
        Initialize parameter mapper

        Args:
            bounds: Parameter bounds (uses defaults if None)
            transform_type: "sigmoid" or "linear" transformation
            n_qubits: Number of qubits in quantum circuit
        """
        self.bounds = bounds or ParameterBounds()
        self.n_qubits = n_qubits

        # Select transformation function
        if transform_type == "sigmoid":
            self.transform = SigmoidTransform()
        elif transform_type == "linear":
            self.transform = LinearTransform()
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

        # Define parameter mapping strategy
        self._setup_parameter_mapping()

        # Logging for debugging
        self.logger = logging.getLogger(__name__)

    def _setup_parameter_mapping(self):
        """
        Define how quantum outputs map to specific parameters

        This mapping is crucial for interpretability and should be
        based on domain knowledge or empirical analysis.
        """
        # Define parameter order and qubit assignments
        self.pk_params = ['ka', 'cl', 'v1', 'q', 'v2']
        self.pd_params = ['baseline', 'imax', 'ic50', 'gamma']

        # Simple mapping strategy: distribute parameters across available qubits
        all_params = self.pk_params + self.pd_params
        self.param_to_qubit = {}

        # If we have enough qubits, assign one per parameter
        if self.n_qubits >= len(all_params):
            for i, param in enumerate(all_params):
                self.param_to_qubit[param] = i
        else:
            # Use qubit combinations or reuse qubits with different processing
            for i, param in enumerate(all_params):
                self.param_to_qubit[param] = i % self.n_qubits

    def quantum_to_pk_parameters(self, quantum_outputs: np.ndarray) -> Dict[str, float]:
        """
        Convert quantum outputs to pharmacokinetic parameters

        Args:
            quantum_outputs: Array of quantum expectation values (one per qubit)

        Returns:
            Dictionary of PK parameters
        """
        quantum_outputs = np.atleast_1d(quantum_outputs)

        # Ensure we have enough outputs
        if len(quantum_outputs) < self.n_qubits:
            # Pad with zeros or repeat last value
            padded = np.zeros(self.n_qubits)
            padded[:len(quantum_outputs)] = quantum_outputs
            quantum_outputs = padded

        pk_parameters = {}
        bounds_dict = self.bounds.get_bounds_dict()

        for param in self.pk_params:
            qubit_idx = self.param_to_qubit[param]
            quantum_val = quantum_outputs[qubit_idx]

            # Apply small perturbation if using same qubit for multiple parameters
            if sum(1 for p in self.pk_params if self.param_to_qubit[p] == qubit_idx) > 1:
                perturbation = 0.1 * hash(param) / 2**31  # Small deterministic perturbation
                quantum_val += perturbation
                quantum_val = np.clip(quantum_val, -1.0, 1.0)

            min_val, max_val = bounds_dict[param]
            pk_parameters[param] = self.transform.forward(quantum_val, min_val, max_val)

        # Validate and log parameters
        self._validate_pk_parameters(pk_parameters)

        return pk_parameters

    def quantum_to_pd_parameters(self, quantum_outputs: np.ndarray) -> Dict[str, float]:
        """
        Convert quantum outputs to pharmacodynamic parameters

        Args:
            quantum_outputs: Array of quantum expectation values

        Returns:
            Dictionary of PD parameters
        """
        quantum_outputs = np.atleast_1d(quantum_outputs)

        # Ensure we have enough outputs
        if len(quantum_outputs) < self.n_qubits:
            padded = np.zeros(self.n_qubits)
            padded[:len(quantum_outputs)] = quantum_outputs
            quantum_outputs = padded

        pd_parameters = {}
        bounds_dict = self.bounds.get_bounds_dict()

        for param in self.pd_params:
            qubit_idx = self.param_to_qubit[param]
            quantum_val = quantum_outputs[qubit_idx]

            # Apply perturbation for parameter differentiation
            if sum(1 for p in self.pd_params if self.param_to_qubit[p] == qubit_idx) > 1:
                perturbation = 0.15 * hash(param) / 2**31  # Different perturbation for PD
                quantum_val += perturbation
                quantum_val = np.clip(quantum_val, -1.0, 1.0)

            min_val, max_val = bounds_dict[param]
            pd_parameters[param] = self.transform.forward(quantum_val, min_val, max_val)

        # Validate and log parameters
        self._validate_pd_parameters(pd_parameters)

        return pd_parameters

    def parameters_to_quantum(self, pk_params: Dict[str, float],
                             pd_params: Dict[str, float]) -> np.ndarray:
        """
        Convert PK/PD parameters back to quantum outputs (inverse mapping)

        Useful for parameter initialization and validation

        Args:
            pk_params: PK parameter dictionary
            pd_params: PD parameter dictionary

        Returns:
            Array of quantum outputs
        """
        quantum_outputs = np.zeros(self.n_qubits)
        bounds_dict = self.bounds.get_bounds_dict()

        # Process all parameters
        all_params = {**pk_params, **pd_params}

        for param, value in all_params.items():
            if param in self.param_to_qubit:
                qubit_idx = self.param_to_qubit[param]
                min_val, max_val = bounds_dict[param]
                quantum_val = self.transform.inverse(value, min_val, max_val)

                # Average if multiple parameters map to same qubit
                quantum_outputs[qubit_idx] = (quantum_outputs[qubit_idx] + quantum_val) / 2.0

        return np.clip(quantum_outputs, -1.0, 1.0)

    def _validate_pk_parameters(self, pk_params: Dict[str, float]) -> None:
        """Validate PK parameters and log warnings if needed"""
        for param, value in pk_params.items():
            validation_result = self.bounds.validate_parameter(param, value)
            # Handle both boolean and array results
            if isinstance(validation_result, np.ndarray):
                if not validation_result.all():
                    self.logger.warning(f"PK parameter {param}={value} contains values outside physiological bounds")
            elif not validation_result:
                self.logger.warning(f"PK parameter {param}={value:.3f} outside physiological bounds")

        # Additional PK-specific validations
        if 'cl' in pk_params and 'v1' in pk_params:
            ke = pk_params['cl'] / pk_params['v1']  # Elimination rate constant
            # Handle both scalar and array cases for batch processing
            if hasattr(ke, '__iter__') and not isinstance(ke, str):
                # Array case - check if any values are too high
                if np.any(ke > 5.0):
                    high_ke_indices = np.where(ke > 5.0)[0]
                    self.logger.warning(f"High elimination rates detected at indices {high_ke_indices}: ke values {ke[high_ke_indices]}")
            else:
                # Scalar case
                if ke > 5.0:  # Very rapid elimination
                    self.logger.warning(f"High elimination rate: ke={ke:.3f} h⁻¹")

    def _validate_pd_parameters(self, pd_params: Dict[str, float]) -> None:
        """Validate PD parameters and log warnings if needed"""
        for param, value in pd_params.items():
            validation_result = self.bounds.validate_parameter(param, value)
            # Handle both boolean and array results
            if isinstance(validation_result, np.ndarray):
                if not validation_result.all():
                    self.logger.warning(f"PD parameter {param}={value} contains values outside physiological bounds")
            elif not validation_result:
                self.logger.warning(f"PD parameter {param}={value:.3f} outside physiological bounds")

        # Additional PD-specific validations
        if 'imax' in pd_params and np.any(pd_params['imax'] > self.bounds.imax_warning_threshold):
            imax_val = pd_params['imax']
            if isinstance(imax_val, np.ndarray):
                # For arrays, report the maximum value
                max_imax = np.max(imax_val)
                self.logger.warning(f"Extreme Imax detected: max={max_imax:.3f} (>{max_imax*100:.0f}% inhibition) - above {self.bounds.imax_warning_threshold*100:.0f}% threshold")
            else:
                # For scalars, use original formatting
                self.logger.warning(f"Extreme Imax: {imax_val:.3f} (>{imax_val*100:.0f}% inhibition) - above {self.bounds.imax_warning_threshold*100:.0f}% threshold")

    def get_parameter_info(self) -> Dict:
        """Get detailed information about parameter mapping"""
        return {
            "n_qubits": self.n_qubits,
            "pk_parameters": self.pk_params,
            "pd_parameters": self.pd_params,
            "parameter_to_qubit_mapping": self.param_to_qubit,
            "transform_type": type(self.transform).__name__,
            "parameter_bounds": self.bounds.get_bounds_dict()
        }

    def generate_random_parameters(self, seed: Optional[int] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Generate random PK/PD parameters within physiological bounds

        Useful for testing and Monte Carlo simulations

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (pk_parameters, pd_parameters)
        """
        if seed is not None:
            np.random.seed(seed)

        bounds_dict = self.bounds.get_bounds_dict()

        pk_params = {}
        for param in self.pk_params:
            min_val, max_val = bounds_dict[param]
            pk_params[param] = np.random.uniform(min_val, max_val)

        pd_params = {}
        for param in self.pd_params:
            min_val, max_val = bounds_dict[param]
            pd_params[param] = np.random.uniform(min_val, max_val)

        return pk_params, pd_params


class ParameterUncertaintyEstimator:
    """
    Estimates uncertainty in parameter mapping from quantum measurements

    This class provides tools for understanding how quantum measurement
    noise propagates to parameter uncertainty, which is crucial for
    dosing decisions.
    """

    def __init__(self, mapper: QuantumParameterMapper, n_samples: int = 1000):
        """
        Initialize uncertainty estimator

        Args:
            mapper: Parameter mapper instance
            n_samples: Number of samples for Monte Carlo estimation
        """
        self.mapper = mapper
        self.n_samples = n_samples

    def estimate_parameter_uncertainty(self, mean_quantum_outputs: np.ndarray,
                                     quantum_std: Optional[np.ndarray] = None) -> Dict:
        """
        Estimate parameter uncertainty using Monte Carlo sampling

        Args:
            mean_quantum_outputs: Mean quantum circuit outputs
            quantum_std: Standard deviation of quantum outputs (if None, assumes shot noise)

        Returns:
            Dictionary with parameter means, std, and confidence intervals
        """
        if quantum_std is None:
            # Default assumption: shot noise scales as 1/sqrt(n_shots)
            # For exact simulation, use small default uncertainty
            quantum_std = np.full_like(mean_quantum_outputs, 0.1)

        # Monte Carlo sampling
        pk_samples = []
        pd_samples = []

        for _ in range(self.n_samples):
            # Sample quantum outputs
            noisy_outputs = np.random.normal(mean_quantum_outputs, quantum_std)
            noisy_outputs = np.clip(noisy_outputs, -1.0, 1.0)

            # Convert to parameters
            pk_params = self.mapper.quantum_to_pk_parameters(noisy_outputs)
            pd_params = self.mapper.quantum_to_pd_parameters(noisy_outputs)

            pk_samples.append(pk_params)
            pd_samples.append(pd_params)

        # Calculate statistics
        results = {
            "pk_parameters": self._calculate_parameter_stats(pk_samples),
            "pd_parameters": self._calculate_parameter_stats(pd_samples)
        }

        return results

    def _calculate_parameter_stats(self, param_samples: List[Dict[str, float]]) -> Dict:
        """Calculate mean, std, and confidence intervals for parameters"""
        if not param_samples:
            return {}

        param_names = param_samples[0].keys()
        stats = {}

        for param in param_names:
            values = [sample[param] for sample in param_samples]
            values = np.array(values)

            stats[param] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "ci_95": [np.percentile(values, 2.5), np.percentile(values, 97.5)],
                "cv": np.std(values) / np.mean(values) if np.mean(values) > 0 else np.inf
            }

        return stats


if __name__ == "__main__":
    # Example usage and testing
    print("VQCdd Parameter Mapping Module")
    print("=" * 40)

    # Create parameter mapper
    mapper = QuantumParameterMapper(n_qubits=4, transform_type="sigmoid")

    # Test with example quantum outputs
    quantum_outputs = np.array([0.5, -0.3, 0.8, -0.1])

    # Convert to parameters
    pk_params = mapper.quantum_to_pk_parameters(quantum_outputs)
    pd_params = mapper.quantum_to_pd_parameters(quantum_outputs)

    print("Example Quantum Outputs:", quantum_outputs)
    print("PK Parameters:", pk_params)
    print("PD Parameters:", pd_params)

    # Test inverse mapping
    reconstructed = mapper.parameters_to_quantum(pk_params, pd_params)
    print("Reconstructed Quantum Outputs:", reconstructed)
    print("Reconstruction Error:", np.abs(quantum_outputs - reconstructed).mean())

    # Test uncertainty estimation
    uncertainty_estimator = ParameterUncertaintyEstimator(mapper, n_samples=100)
    uncertainty = uncertainty_estimator.estimate_parameter_uncertainty(quantum_outputs)

    print("\nParameter Uncertainty Analysis:")
    for param_type in ["pk_parameters", "pd_parameters"]:
        print(f"\n{param_type.upper()}:")
        for param, stats in uncertainty[param_type].items():
            print(f"  {param}: {stats['mean']:.3f} ± {stats['std']:.3f} (CV: {stats['cv']:.1%})")

    # Show mapping info
    print("\nMapping Information:")
    info = mapper.get_parameter_info()
    print(f"Qubits: {info['n_qubits']}")
    print(f"PK params: {info['pk_parameters']}")
    print(f"PD params: {info['pd_parameters']}")
    print(f"Mapping: {info['parameter_to_qubit_mapping']}")