"""
Approach 3: Quantum-Enhanced Differential Equation Solver

Variational quantum algorithms for solving PK/PD differential equations with enhanced
precision and stability for steady-state calculations.

Key Features:
- Variational Quantum Evolution Equation Solver for coupled PK/PD ODEs
- Enhanced precision for stiff differential equations
- Efficient parameter sensitivity analysis through quantum gradients  
- Uncertainty propagation in dose recommendations
"""

from .quantum_ode_solver import QuantumODESolver
from .pkpd_system_solver import PKPDSystemSolver
from .quantum_sensitivity_analysis import QuantumSensitivityAnalyzer

__all__ = [
    'QuantumODESolver',
    'PKPDSystemSolver',
    'QuantumSensitivityAnalyzer'
]