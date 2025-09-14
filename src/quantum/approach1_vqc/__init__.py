"""
Approach 1: Variational Quantum Circuit Parameter Estimation

This module implements VQE/QAOA-based optimization for PK/PD parameter estimation
using PennyLane variational quantum circuits.

Key Features:
- Variational quantum eigensolver for parameter optimization
- QAOA-style optimization for combinatorial dosing problems
- Quantum feature maps for high-dimensional parameter spaces
- Hybrid classical-quantum optimization loops
"""

from .vqc_parameter_estimator import VQCParameterEstimator
from .vqc_parameter_estimator_full import VQCParameterEstimatorFull, VQCConfig
from .qaoa_dosing_optimizer import QAOADosingOptimizer  
# from .vqe_pkpd_solver import VQEPKPDSolver  # Missing file

__all__ = [
    'VQCParameterEstimator',
    'VQCParameterEstimatorFull',
    'VQCConfig',
    'QAOADosingOptimizer', 
    # 'VQEPKPDSolver'
]