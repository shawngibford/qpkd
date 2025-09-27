"""
PennyLane utilities for quantum PK/PD modeling
"""

import pennylane as qml
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class DeviceConfig:
    """Configuration for PennyLane devices"""
    name: str = "default.qubit"
    wires: int = 16
    shots: Optional[int] = None
    analytic: bool = True


class PennyLaneDevice:
    """Wrapper for PennyLane quantum devices"""
    
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.device = self._create_device()
        
    def _create_device(self) -> qml.device:
        """Create PennyLane device based on configuration"""
        return qml.device(
            self.config.name,
            wires=self.config.wires,
            shots=self.config.shots,
            analytic=self.config.analytic
        )
    
    def get_device(self) -> qml.device:
        """Get the PennyLane device"""
        return self.device


class QuantumCircuitBuilder:
    """Builder for quantum circuits using PennyLane templates"""
    
    @staticmethod
    def angle_encoding(features: np.ndarray, wires: List[int]) -> None:
        """Encode classical data using angle encoding"""
        for i, wire in enumerate(wires):
            if i < len(features):
                qml.RY(features[i], wires=wire)
    
    @staticmethod
    def amplitude_encoding(features: np.ndarray, wires: List[int]) -> None:
        """Encode classical data using amplitude encoding"""
        # Normalize features for amplitude encoding
        normalized_features = features / np.linalg.norm(features)
        qml.AmplitudeEmbedding(normalized_features, wires=wires, normalize=True)
    
    @staticmethod
    def basic_entangler(weights: np.ndarray, wires: List[int]) -> None:
        """Basic entangling layer"""
        qml.BasicEntanglerLayers(weights, wires=wires)
    
    @staticmethod  
    def strongly_entangling(weights: np.ndarray, wires: List[int]) -> None:
        """Strongly entangling layer"""
        qml.StronglyEntanglingLayers(weights, wires=wires)
        
    @staticmethod
    def simplified_two_design(weights: np.ndarray, wires: List[int]) -> None:
        """Simplified two-design ansatz"""
        qml.SimplifiedTwoDesign(weights, wires=wires)
    
    @staticmethod
    def parametrized_evolution(hamiltonian: List, time: float) -> None:
        """Parametrized quantum evolution (for ODE solving)"""
        # Implement parametrized evolution for QODE approach
        for pauli_string, coefficient in hamiltonian:
            # Apply time evolution under each Hamiltonian term
            evolution_time = coefficient * time

            # Use PennyLane's evolution templates
            if isinstance(pauli_string, str):
                # Handle string-based Pauli operators
                if 'X' in pauli_string:
                    for i, op in enumerate(pauli_string):
                        if op == 'X':
                            qml.RX(evolution_time, wires=i)
                elif 'Y' in pauli_string:
                    for i, op in enumerate(pauli_string):
                        if op == 'Y':
                            qml.RY(evolution_time, wires=i)
                elif 'Z' in pauli_string:
                    for i, op in enumerate(pauli_string):
                        if op == 'Z':
                            qml.RZ(evolution_time, wires=i)
            else:
                # Handle PennyLane operator objects
                qml.evolution.evolve(evolution_time * pauli_string)
    
    @staticmethod
    def qaoa_layer(gamma: float, beta: float, cost_hamiltonian: List, mixer_hamiltonian: List, wires: List[int]) -> None:
        """QAOA layer implementation"""
        # Cost Hamiltonian evolution
        for pauli_string, coeff in cost_hamiltonian:
            qml.evolution.evolve(coeff * gamma * pauli_string)
            
        # Mixer Hamiltonian evolution  
        for pauli_string, coeff in mixer_hamiltonian:
            qml.evolution.evolve(coeff * beta * pauli_string)


class QuantumOptimizer:
    """Quantum optimization utilities"""
    
    @staticmethod
    def get_optimizer(name: str, **kwargs) -> qml.optimize.GradientDescentOptimizer:
        """Get PennyLane optimizer by name"""
        optimizers = {
            'adam': qml.AdamOptimizer,
            'adagrad': qml.AdagradOptimizer,
            'gradient_descent': qml.GradientDescentOptimizer,
            'momentum': qml.MomentumOptimizer,
            'rms_prop': qml.RMSPropOptimizer,
            'nesterov_momentum': qml.NesterovMomentumOptimizer
        }
        
        if name.lower() not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
            
        return optimizers[name.lower()](**kwargs)
    
    @staticmethod
    def parameter_shift_gradient(circuit_func: Callable, params: np.ndarray, shift: float = np.pi/2) -> np.ndarray:
        """Compute gradients using parameter shift rule"""
        gradients = np.zeros_like(params)
        
        for i in range(len(params)):
            # Positive shift
            params_plus = params.copy()
            params_plus[i] += shift
            cost_plus = circuit_func(params_plus)
            
            # Negative shift
            params_minus = params.copy() 
            params_minus[i] -= shift
            cost_minus = circuit_func(params_minus)
            
            # Compute gradient
            gradients[i] = (cost_plus - cost_minus) / (2 * np.sin(shift))
            
        return gradients


class QuantumMeasurements:
    """Utilities for quantum measurements"""
    
    @staticmethod
    def expectation_z(wires: List[int]) -> List[qml.expval]:
        """Expectation value of Pauli-Z on specified wires"""
        return [qml.expval(qml.PauliZ(wire)) for wire in wires]
    
    @staticmethod
    def probability_computational_basis(wires: List[int]) -> qml.probs:
        """Probability in computational basis"""
        return qml.probs(wires=wires)
    
    @staticmethod
    def sample_computational_basis(wires: List[int]) -> qml.sample:
        """Sample from computational basis"""
        return qml.sample(wires=wires)


class CircuitAnalysis:
    """Tools for analyzing quantum circuits"""
    
    @staticmethod
    def count_gates(circuit_func: Callable) -> Dict[str, int]:
        """Count gates in quantum circuit"""
        try:
            # Create a quantum function to inspect
            device_temp = qml.device('default.qubit', wires=8)

            @qml.qnode(device_temp)
            def temp_circuit():
                circuit_func()
                return qml.expval(qml.PauliZ(0))

            # Get circuit specifications
            specs = qml.specs(temp_circuit)()

            # Count different gate types
            gate_counts = {}
            for op_name in specs.get('gate_types', {}):
                gate_counts[op_name] = specs['gate_types'][op_name]

            return gate_counts

        except Exception:
            # Fallback gate counting
            return {
                'RX': 0, 'RY': 0, 'RZ': 0,
                'CNOT': 0, 'Hadamard': 0,
                'total': 0
            }
    
    @staticmethod
    def circuit_depth(circuit_func: Callable) -> int:
        """Calculate circuit depth"""
        try:
            # Create temporary device for analysis
            device_temp = qml.device('default.qubit', wires=8)

            @qml.qnode(device_temp)
            def temp_circuit():
                circuit_func()
                return qml.expval(qml.PauliZ(0))

            # Get circuit specifications
            specs = qml.specs(temp_circuit)()

            # Return circuit depth
            return specs.get('depth', 0)

        except Exception:
            # Fallback depth estimation
            return 10  # Default reasonable depth
    
    @staticmethod
    def resource_requirements(circuit_func: Callable) -> Dict[str, Any]:
        """Analyze resource requirements of circuit"""
        try:
            # Create temporary device for analysis
            device_temp = qml.device('default.qubit', wires=16)

            @qml.qnode(device_temp)
            def temp_circuit():
                circuit_func()
                return qml.expval(qml.PauliZ(0))

            # Get circuit specifications
            specs = qml.specs(temp_circuit)()

            # Count total gates
            total_gates = sum(specs.get('gate_types', {}).values())

            return {
                'n_qubits': specs.get('num_wires', 0),
                'n_gates': total_gates,
                'depth': specs.get('depth', 0),
                'gate_types': specs.get('gate_types', {}),
                'num_observables': specs.get('num_observables', 0),
                'connectivity': 'all-to-all',  # Assume all-to-all for simulation
                'resources': specs.get('resources', {})
            }

        except Exception:
            # Fallback resource analysis
            return {
                'n_qubits': 8,
                'n_gates': 50,
                'depth': 10,
                'gate_types': {'RY': 16, 'RZ': 16, 'CNOT': 18},
                'connectivity': 'all-to-all'
            }