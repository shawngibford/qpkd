"""
Quantum-Enhanced PK/PD Modeling Package

This package implements five different quantum computing approaches for 
pharmacokinetics-pharmacodynamics modeling using PennyLane and ZX calculus.

Approaches:
1. Variational Quantum Circuit Parameter Estimation (approach1_vqc)
2. Quantum Machine Learning for Population PK (approach2_qml) 
3. Quantum-Enhanced Differential Equation Solver (approach3_qode)
4. Quantum Annealing Multi-Objective Optimization (approach4_qaoa)
5. Tensor Network Population Modeling with ZX Calculus (approach5_tensor_zx)
"""

from .core.base import QuantumPKPDBase
from .core.pennylane_utils import PennyLaneDevice, QuantumCircuitBuilder
from .core.zx_utils import ZXTensorNetwork, ZXCircuitOptimizer

__all__ = [
    'QuantumPKPDBase',
    'PennyLaneDevice', 
    'QuantumCircuitBuilder',
    'ZXTensorNetwork',
    'ZXCircuitOptimizer'
]