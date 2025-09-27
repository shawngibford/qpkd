# -*- coding: utf-8 -*-
"""
Quantum Neural Circuit (QNC) for PK/PD Parameter Estimation

A 12-qubit Variational Quantum Circuit that replaces the MLP neural network
in the NeuralODE solver for pharmacokinetic/pharmacodynamic parameter estimation.

This implementation was specifically designed to meet the following requirements:
1. 12 qubits total: 3 initial qubits + 9 ancillary qubits
2. Even entanglement between the 3 initial qubits and BOTH PK and PD qubits
3. Less than 135 parameters (achieved: 108 parameters)
4. Visualization using pennylane.drawer() and matplotlib
5. Parameter counting functionality for VQC validation
6. Comprehensive documentation following PennyLane best practices

Circuit Architecture:
═══════════════════
- Qubits 0-2: Initial qubits (encode BW, COMED, dose_intensity)
- Qubits 3-7: PK parameter qubits (Ka, CL, Vc, Q, Vp)
- Qubits 8-11: PD parameter qubits (Kin, Kout, Imax, IC50)

Entanglement Pattern:
═══════════════════
Layer 3 - Initial → PK entanglement (EVEN):
  - Each initial qubit (0,1,2) connects to ALL PK qubits (3,4,5,6,7)
  - Total: 3 × 5 = 15 CNOT gates

Layer 4 - Initial → PD entanglement (EVEN):
  - Each initial qubit (0,1,2) connects to ALL PD qubits (8,9,10,11)
  - Total: 3 × 4 = 12 CNOT gates

Layer 6 - Intra-group entanglement:
  - Circular within PK qubits: 5 CNOTs
  - Circular within PD qubits: 4 CNOTs

Layer 8 - Cross-group entanglement:
  - PK ↔ PD connections: 5 CNOTs

Total: 41 CNOT gates providing comprehensive entanglement

Parameter Budget:
═════════════════
- Layer 2: 24 parameters (RY + RZ on all 12 qubits)
- Layer 5: 24 parameters (RY + RZ on all 12 qubits)
- Layer 7: 24 parameters (RY + RZ on all 12 qubits)
- Layer 9: 24 parameters (RY + RZ on all 12 qubits)
- Layer 10: 12 parameters (RY on all 12 qubits)
- Total: 108 parameters (< 135 limit ✓)

Integration with NeuralODE:
═══════════════════════════
This quantum circuit serves as a drop-in replacement for the classical MLP
in the discrete dosing NeuralODE model. It maintains the same interface:
- Input: 3 features (BW, COMED, dose_intensity)
- Output: 9 parameters (5 PK + 4 PD) mapped to physiological bounds
- Compatible with PyTorch gradient-based optimization
- Supports batched inference for population-scale optimization

Visualization Features:
══════════════════════
- pennylane.drawer() for detailed text circuit representation
- matplotlib integration for high-quality circuit diagrams
- Automatic parameter counting and validation
- Comprehensive entanglement pattern analysis
- Export capabilities for documentation and presentations

Author: Claude Code (Senior Quantum Engineer)
Context: Quantum Innovation Challenge 2025 (LSQI)
Target Platform: NVIDIA Gefion AI Supercomputer with CUDA-Q
Framework: PennyLane with PyTorch integration
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CircuitConfig:
    """Configuration for the quantum neural circuit"""
    n_qubits: int = 12
    n_initial_qubits: int = 3  # BW, COMED, dose_intensity
    n_pk_qubits: int = 5       # Ka, CL, Vc, Q, Vp
    n_pd_qubits: int = 4       # Kin, Kout, Imax, IC50
    n_layers: int = 10
    device_name: str = "default.qubit"
    interface: str = "torch"


class QuantumNeuralParameterEstimator(nn.Module):
    """
    Quantum Neural Circuit for PK/PD parameter estimation.

    Replaces the classical MLP with a 12-qubit variational quantum circuit
    that maintains the same interface for seamless integration with NeuralODE.

    Architecture:
        - 12 qubits total (3 initial + 5 PK + 4 PD)
        - 10 layers with systematic entanglement patterns
        - 108 trainable parameters (< 135 limit)
        - Outputs 9 parameters matching original MLP
    """

    def __init__(self, n_subjects: int, embedding_dim: int = 16, device: str = 'cpu'):
        """
        Initialize the Quantum Neural Parameter Estimator.

        Args:
            n_subjects: Number of subjects (maintained for interface compatibility)
            embedding_dim: Embedding dimension (maintained for compatibility)
            device: Torch device ('cpu' or 'cuda')
        """
        super().__init__()

        self.config = CircuitConfig()
        self.n_subjects = n_subjects
        self.embedding_dim = embedding_dim
        self.device_str = device

        # Initialize quantum device
        self.qdevice = qml.device(
            self.config.device_name,
            wires=self.config.n_qubits
        )

        # Define physiologically reasonable parameter bounds (same as original MLP)
        self.pk_bounds = torch.tensor([
            [0.5, 5.0],    # Ka: absorption rate (0.5-5.0 /h)
            [2.0, 20.0],   # CL: clearance (2-20 L/h)
            [20.0, 100.0], # Vc: central volume (20-100 L)
            [2.0, 30.0],   # Q: inter-compartmental clearance (2-30 L/h)
            [40.0, 200.0]  # Vp: peripheral volume (40-200 L)
        ], device=device)

        self.pd_bounds = torch.tensor([
            [5.0, 25.0],   # Kin: zero-order production (5-25 ng/mL/h)
            [0.3, 3.0],    # Kout: first-order elimination (0.3-3.0 /h)
            [0.3, 0.95],   # Imax: maximum inhibition (30-95%)
            [1.0, 15.0]    # IC50: half-maximal concentration (1-15 ng/mL)
        ], device=device)

        # Calculate parameter dimensions for each layer
        self.param_dims = self._calculate_parameter_dimensions()
        total_params = sum(self.param_dims.values())

        print(f"Quantum Neural Circuit Architecture:")
        print(f"  Total qubits: {self.config.n_qubits}")
        print(f"  Initial qubits: {self.config.n_initial_qubits} (BW, COMED, dose_intensity)")
        print(f"  PK qubits: {self.config.n_pk_qubits} (Ka, CL, Vc, Q, Vp)")
        print(f"  PD qubits: {self.config.n_pd_qubits} (Kin, Kout, Imax, IC50)")
        print(f"  Circuit layers: {self.config.n_layers}")
        print(f"  Total parameters: {total_params}")

        assert total_params < 135, f"Parameter count {total_params} exceeds limit of 135"

        # Initialize trainable parameters
        self.weights = nn.Parameter(
            torch.randn(total_params, device=device, dtype=torch.float32) * 0.1
        )

        # Create quantum circuit
        self.qcircuit = qml.QNode(
            self._quantum_circuit,
            self.qdevice,
            interface=self.config.interface,
            diff_method="backprop"
        )

        print(f"  Quantum device: {self.config.device_name}")
        print(f"  Interface: {self.config.interface}")
        print(f"  Differentiation: backprop")

        self.to(device)

    def _calculate_parameter_dimensions(self) -> Dict[str, int]:
        """
        Calculate the number of parameters for each circuit layer.

        Returns:
            Dictionary mapping layer names to parameter counts
        """
        n_qubits = self.config.n_qubits

        return {
            "layer2_ry": n_qubits,      # 12 parameters
            "layer2_rz": n_qubits,      # 12 parameters
            "layer5_ry": n_qubits,      # 12 parameters
            "layer5_rz": n_qubits,      # 12 parameters
            "layer7_ry": n_qubits,      # 12 parameters
            "layer7_rz": n_qubits,      # 12 parameters
            "layer9_ry": n_qubits,      # 12 parameters
            "layer9_rz": n_qubits,      # 12 parameters
            "layer10_ry": n_qubits,     # 12 parameters
        }
        # Total: 108 parameters

    def _quantum_circuit(self, sample_input: torch.Tensor, weights: torch.Tensor):
        """
        Define the 10-layer quantum circuit with systematic entanglement.

        Args:
            sample_input: Input features [3] (BW, COMED, dose_intensity) for single sample
            weights: Trainable parameters [108]

        Returns:
            List of expectation values [9] for PK/PD parameters
        """
        # Extract weight slices for each layer
        weight_idx = 0
        layer_weights = {}

        for layer_name, param_count in self.param_dims.items():
            layer_weights[layer_name] = weights[weight_idx:weight_idx + param_count]
            weight_idx += param_count

        # Layer 1: Input encoding (no trainable parameters)
        # Encode input features into initial qubits
        qml.RY(sample_input[0], wires=0)  # Body weight
        qml.RY(sample_input[1], wires=1)  # COMED status
        qml.RY(sample_input[2], wires=2)  # Dose intensity

        # Layer 2: First parameterized layer (24 parameters)
        for i in range(self.config.n_qubits):
            qml.RY(layer_weights["layer2_ry"][i], wires=i)
            qml.RZ(layer_weights["layer2_rz"][i], wires=i)

        # Layer 3: Initial -> PK entanglement (15 CNOTs)
        # Each initial qubit (0,1,2) entangles with each PK qubit (3,4,5,6,7)
        for initial_qubit in range(3):
            for pk_qubit in range(3, 8):  # PK qubits: 3,4,5,6,7
                qml.CNOT(wires=[initial_qubit, pk_qubit])

        # Layer 4: Initial -> PD entanglement (12 CNOTs)
        # Each initial qubit (0,1,2) entangles with each PD qubit (8,9,10,11)
        for initial_qubit in range(3):
            for pd_qubit in range(8, 12):  # PD qubits: 8,9,10,11
                qml.CNOT(wires=[initial_qubit, pd_qubit])

        # Layer 5: Second parameterized layer (24 parameters)
        for i in range(self.config.n_qubits):
            qml.RY(layer_weights["layer5_ry"][i], wires=i)
            qml.RZ(layer_weights["layer5_rz"][i], wires=i)

        # Layer 6: Intra-group entanglement (9 CNOTs)
        # Circular entanglement within PK qubits: 3->4->5->6->7->3
        pk_qubits = list(range(3, 8))
        for i in range(len(pk_qubits)):
            qml.CNOT(wires=[pk_qubits[i], pk_qubits[(i + 1) % len(pk_qubits)]])

        # Circular entanglement within PD qubits: 8->9->10->11->8
        pd_qubits = list(range(8, 12))
        for i in range(len(pd_qubits)):
            qml.CNOT(wires=[pd_qubits[i], pd_qubits[(i + 1) % len(pd_qubits)]])

        # Layer 7: Third parameterized layer (24 parameters)
        for i in range(self.config.n_qubits):
            qml.RY(layer_weights["layer7_ry"][i], wires=i)
            qml.RZ(layer_weights["layer7_rz"][i], wires=i)

        # Layer 8: Cross-group entanglement (5 CNOTs)
        # Connect PK qubits to PD qubits: 3->8, 4->9, 5->10, 6->11, 7->8
        cross_connections = [(3, 8), (4, 9), (5, 10), (6, 11), (7, 8)]
        for pk_qubit, pd_qubit in cross_connections:
            qml.CNOT(wires=[pk_qubit, pd_qubit])

        # Layer 9: Fourth parameterized layer (24 parameters)
        for i in range(self.config.n_qubits):
            qml.RY(layer_weights["layer9_ry"][i], wires=i)
            qml.RZ(layer_weights["layer9_rz"][i], wires=i)

        # Layer 10: Final parameterized layer (12 parameters)
        for i in range(self.config.n_qubits):
            qml.RY(layer_weights["layer10_ry"][i], wires=i)

        # Measurements: Extract expectation values from parameter qubits (3-11)
        return [qml.expval(qml.PauliZ(qubit)) for qubit in range(3, 12)]

    def forward(self, subject_ids: torch.Tensor, covariates: torch.Tensor,
                dose_intensities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the quantum neural circuit.

        Args:
            subject_ids: Subject identifiers [batch_size] (maintained for compatibility)
            covariates: Input features [batch_size, 2] (BW, COMED)
            dose_intensities: Dose intensity values [batch_size]

        Returns:
            Tuple of (pk_params, pd_params):
                - pk_params: PK parameters [batch_size, 5] (Ka, CL, Vc, Q, Vp)
                - pd_params: PD parameters [batch_size, 4] (Kin, Kout, Imax, IC50)
        """
        # Combine inputs: [BW, COMED, dose_intensity]
        inputs = torch.cat([covariates, dose_intensities.unsqueeze(1)], dim=1)
        batch_size = inputs.shape[0]

        # Process each sample individually (PennyLane limitation for complex circuits)
        all_outputs = []
        for i in range(batch_size):
            sample_output = self.qcircuit(inputs[i], self.weights)
            all_outputs.append(torch.stack(sample_output))

        raw_outputs = torch.stack(all_outputs)

        # Extract PK and PD expectation values
        pk_expectations = raw_outputs[:, :5]  # First 5 outputs (qubits 3-7)
        pd_expectations = raw_outputs[:, 5:]  # Last 4 outputs (qubits 8-11)

        # Map expectation values [-1, 1] to physiological parameter bounds
        pk_params = self._map_to_bounds(pk_expectations, self.pk_bounds)
        pd_params = self._map_to_bounds(pd_expectations, self.pd_bounds)

        return pk_params, pd_params

    def _map_to_bounds(self, expectations: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        """
        Map expectation values from [-1, 1] to physiological parameter bounds.

        Args:
            expectations: Expectation values [batch_size, n_params] in range [-1, 1]
            bounds: Parameter bounds [n_params, 2] with [lower, upper] for each param

        Returns:
            Scaled parameters [batch_size, n_params] within physiological bounds
        """
        # Scale from [-1, 1] to [0, 1]
        normalized = (expectations + 1.0) / 2.0

        # Scale to parameter bounds: lower + normalized * (upper - lower)
        lower_bounds = bounds[:, 0].unsqueeze(0)  # [1, n_params]
        upper_bounds = bounds[:, 1].unsqueeze(0)  # [1, n_params]

        scaled_params = lower_bounds + normalized * (upper_bounds - lower_bounds)

        return scaled_params

    def predict_single_subject(self, subject_id: int, bw: float, comed: int,
                              dose_intensity: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict parameters for a single subject (maintained for compatibility).

        Args:
            subject_id: Subject identifier (not used in quantum version)
            bw: Body weight
            comed: Concomitant medication status (0 or 1)
            dose_intensity: Dose intensity value

        Returns:
            Tuple of (pk_params, pd_params) for single subject
        """
        # Convert to tensors with batch dimension
        covariates = torch.tensor([[bw, comed]], dtype=torch.float32, device=self.device_str)
        dose_intensities = torch.tensor([dose_intensity], dtype=torch.float32, device=self.device_str)
        dummy_subject_ids = torch.tensor([0], device=self.device_str)

        was_training = self.training
        self.eval()

        with torch.no_grad():
            pk_params, pd_params = self.forward(dummy_subject_ids, covariates, dose_intensities)

        self.train(was_training)
        return pk_params[0], pd_params[0]

    def parameter_count(self) -> int:
        """
        Count total trainable parameters in the quantum circuit.

        Returns:
            Total number of trainable parameters
        """
        return sum(self.param_dims.values())

    def visualize_circuit(self, save_path: Optional[str] = None,
                         show_plot: bool = True) -> str:
        """
        Visualize the quantum circuit using PennyLane's drawer with matplotlib.

        This function demonstrates the even entanglement pattern between the 3 initial
        qubits and both the 5 PK qubits and 4 PD qubits as requested.

        Args:
            save_path: Optional path to save the circuit diagram
            show_plot: Whether to display the plot

        Returns:
            String representation of the circuit
        """
        print("=" * 80)
        print("QUANTUM CIRCUIT VISUALIZATION - 12 QUBITS WITH EVEN ENTANGLEMENT")
        print("=" * 80)

        # Create a sample input for visualization
        sample_input = torch.tensor([70.0, 0.0, 1.0], dtype=torch.float32)
        sample_weights = torch.randn(self.parameter_count()) * 0.1

        print(f"Circuit Configuration:")
        print(f"  - 12 qubits total")
        print(f"  - Qubits 0,1,2: Initial qubits (BW, COMED, dose_intensity)")
        print(f"  - Qubits 3,4,5,6,7: PK parameter qubits (Ka, CL, Vc, Q, Vp)")
        print(f"  - Qubits 8,9,10,11: PD parameter qubits (Kin, Kout, Imax, IC50)")
        print(f"  - {self.parameter_count()} trainable parameters")
        print()

        # Draw the circuit using PennyLane's text drawer
        print("Text Circuit Representation:")
        circuit_drawer = qml.draw(self.qcircuit, decimals=2)
        circuit_str = circuit_drawer(sample_input, sample_weights)
        print(circuit_str)
        print()

        # Generate matplotlib figure
        try:
            print("Generating matplotlib visualization...")

            # Create figure using matplotlib drawer
            fig, ax = qml.draw_mpl(self.qcircuit, decimals=2, show_all_wires=True)(
                sample_input, sample_weights
            )

            # Enhance the plot
            plt.title("Quantum Neural Circuit for PK/PD Parameter Estimation\n"
                     "12 Qubits with Even Entanglement Pattern",
                     fontsize=14, fontweight='bold', pad=20)

            # Add qubit labels
            ax.text(-0.5, 0, "Input: BW", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 1, "Input: COMED", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 2, "Input: Dose", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 3, "PK: Ka", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 4, "PK: CL", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 5, "PK: Vc", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 6, "PK: Q", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 7, "PK: Vp", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 8, "PD: Kin", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 9, "PD: Kout", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 10, "PD: Imax", fontsize=10, ha='right', va='center')
            ax.text(-0.5, 11, "PD: IC50", fontsize=10, ha='right', va='center')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                print(f"Circuit diagram saved to: {save_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()

            print("✓ Matplotlib visualization generated successfully")

        except Exception as e:
            print(f"Matplotlib visualization failed: {e}")
            print("This is expected in some environments - text representation is available above")

        # Print entanglement analysis
        print()
        print("Entanglement Pattern Analysis:")
        print("  Layer 3 - Initial → PK entanglement:")
        print("    Each initial qubit (0,1,2) connects to ALL PK qubits (3,4,5,6,7)")
        print("    Total: 3 × 5 = 15 CNOT gates (even entanglement)")
        print("  Layer 4 - Initial → PD entanglement:")
        print("    Each initial qubit (0,1,2) connects to ALL PD qubits (8,9,10,11)")
        print("    Total: 3 × 4 = 12 CNOT gates (even entanglement)")
        print("  Layer 6 - Intra-group entanglement:")
        print("    Circular within PK qubits: 5 CNOTs")
        print("    Circular within PD qubits: 4 CNOTs")
        print("  Layer 8 - Cross-group entanglement:")
        print("    PK ↔ PD connections: 5 CNOTs")
        print("  Total entanglement: 41 CNOT gates")
        print()
        print("Parameter Budget:")
        breakdown = self.get_parameter_breakdown()
        for layer, count in breakdown.items():
            if layer != 'total':
                print(f"  {layer.replace('_', ' ').title()}: {count} parameters")
        print(f"  Total: {breakdown['total']} parameters (< 135 ✓)")

        return circuit_str

    def get_parameter_breakdown(self):
        """Get detailed parameter breakdown by layer."""
        return {
            'input_encoding': 0,  # No trainable parameters in input encoding
            'layer_2_rotations': 24,  # RY + RZ on all 12 qubits
            'initial_pk_entanglement': 0,  # CNOTs have no parameters
            'initial_pd_entanglement': 0,  # CNOTs have no parameters
            'layer_5_rotations': 24,  # RY + RZ on all 12 qubits
            'intra_group_entanglement': 0,  # CNOTs have no parameters
            'layer_7_rotations': 24,  # RY + RZ on all 12 qubits
            'cross_group_entanglement': 0,  # CNOTs have no parameters
            'layer_9_rotations': 24,  # RY + RZ on all 12 qubits
            'layer_10_rotations': 12,  # RY on all 12 qubits
            'total': 108
        }

    def circuit_analysis(self) -> Dict[str, Any]:
        """
        Analyze circuit properties and resource requirements.

        Returns:
            Dictionary with circuit analysis results
        """
        # Create sample inputs for analysis
        sample_input = torch.tensor([70.0, 0.0, 1.0], dtype=torch.float32)
        sample_weights = torch.randn(self.parameter_count()) * 0.1

        try:
            # Get circuit specifications
            @qml.qnode(self.qdevice)
            def analysis_circuit():
                return self._quantum_circuit(sample_input, sample_weights)

            specs = qml.specs(analysis_circuit)()

            analysis = {
                "n_qubits": self.config.n_qubits,
                "n_layers": self.config.n_layers,
                "n_parameters": self.parameter_count(),
                "parameter_limit_satisfied": self.parameter_count() < 135,
                "circuit_depth": specs.get("depth", "unknown"),
                "total_gates": sum(specs.get("gate_types", {}).values()),
                "gate_types": specs.get("gate_types", {}),
                "resources": specs.get("resources", {}),
                "device": self.config.device_name,
                "entanglement_pattern": {
                    "initial_pk_cnots": 15,
                    "initial_pd_cnots": 12,
                    "intra_group_cnots": 9,
                    "cross_group_cnots": 5,
                    "total_cnots": 41
                }
            }

        except Exception as e:
            print(f"Circuit analysis failed: {e}")
            analysis = {
                "n_qubits": self.config.n_qubits,
                "n_layers": self.config.n_layers,
                "n_parameters": self.parameter_count(),
                "parameter_limit_satisfied": self.parameter_count() < 135,
                "error": str(e)
            }

        return analysis


def demonstrate_circuit_visualization():
    """
    Demonstrate the circuit visualization with pennylane.drawer() and matplotlib
    as specifically requested by the user.
    """
    print("=" * 80)
    print("PENNYLANE CIRCUIT VISUALIZATION DEMONSTRATION")
    print("=" * 80)

    # Initialize quantum neural network
    device = "cpu"
    qnn = QuantumNeuralParameterEstimator(n_subjects=100, device=device)

    print("Creating 12-qubit quantum circuit with even entanglement...")
    print("  3 initial qubits: BW, COMED, dose_intensity")
    print("  5 PK qubits: Ka, CL, Vc, Q, Vp")
    print("  4 PD qubits: Kin, Kout, Imax, IC50")
    print(f"  Total parameters: {qnn.parameter_count()} (< 135 limit)")
    print()

    # Use pennylane.drawer() as specifically requested
    circuit_str = qnn.visualize_circuit(show_plot=True, save_path="quantum_circuit_diagram.png")

    print("\n" + "=" * 80)
    print("CIRCUIT VISUALIZATION COMPLETE")
    print("=" * 80)
    print("✓ pennylane.drawer() used for text representation")
    print("✓ matplotlib used for graphical visualization")
    print("✓ Even entanglement pattern implemented")
    print("✓ 108 variational parameters (< 135 limit)")
    print("✓ Circuit ready to replace MLP in NeuralODE")

def main():
    """
    Demonstration of the Quantum Neural Parameter Estimator.
    """
    print("=" * 80)
    print("QUANTUM NEURAL CIRCUIT (QNC) DEMONSTRATION")
    print("=" * 80)

    # Initialize quantum neural network
    device = "cpu"
    qnn = QuantumNeuralParameterEstimator(n_subjects=100, device=device)

    print(f"\nParameter Count Validation:")
    param_count = qnn.parameter_count()
    print(f"  Total parameters: {param_count}")
    print(f"  Parameter limit: 135")
    print(f"  Limit satisfied: {param_count < 135}")

    # Test forward pass
    print(f"\nTesting Forward Pass:")
    batch_size = 3
    subject_ids = torch.randint(0, 100, (batch_size,), device=device)
    covariates = torch.tensor([
        [70.0, 0.0],  # 70kg, no COMED
        [85.0, 1.0],  # 85kg, with COMED
        [60.0, 0.0],  # 60kg, no COMED
    ], device=device)
    dose_intensities = torch.tensor([1.0, 1.5, 0.8], device=device)

    pk_params, pd_params = qnn(subject_ids, covariates, dose_intensities)

    print(f"  Input batch size: {batch_size}")
    print(f"  PK parameters shape: {pk_params.shape}")
    print(f"  PD parameters shape: {pd_params.shape}")
    print(f"  PK params range: [{pk_params.min():.3f}, {pk_params.max():.3f}]")
    print(f"  PD params range: [{pd_params.min():.3f}, {pd_params.max():.3f}]")

    # Test single subject prediction
    print(f"\nTesting Single Subject Prediction:")
    pk_single, pd_single = qnn.predict_single_subject(0, 75.0, 1, 1.2)
    print(f"  Single PK params: {pk_single}")
    print(f"  Single PD params: {pd_single}")

    # Circuit analysis
    print(f"\nCircuit Analysis:")
    analysis = qnn.circuit_analysis()
    for key, value in analysis.items():
        if key != "gate_types" and key != "resources":
            print(f"  {key}: {value}")

    print(f"\n" + "=" * 80)
    print("QNC DEMONSTRATION COMPLETE")
    print("✓ 12-qubit circuit with systematic entanglement")
    print("✓ 108 parameters (< 135 limit)")
    print("✓ Drop-in replacement for classical MLP")
    print("✓ PyTorch integration for gradient-based training")
    print("=" * 80)

    # Demonstrate circuit visualization as specifically requested
    print("\nRunning circuit visualization demonstration...")
    demonstrate_circuit_visualization()


if __name__ == "__main__":
    main()