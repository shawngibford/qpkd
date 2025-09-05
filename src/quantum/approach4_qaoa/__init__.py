"""
Approach 4: Quantum Annealing Multi-Objective Optimization

QUBO formulation for simultaneous optimization of efficacy, safety, and population coverage
using quantum annealing or QAOA on classical simulators.

Key Features:
- Multi-objective dosing optimization as QUBO problem
- Global optimization avoiding local minima
- Simultaneous consideration of multiple population scenarios
- Trade-off analysis between efficacy, safety, and coverage
"""

from .qubo_formulator import QUBOFormulator
from .multi_objective_optimizer import MultiObjectiveOptimizer
from .population_scenario_analyzer import PopulationScenarioAnalyzer

__all__ = [
    'QUBOFormulator',
    'MultiObjectiveOptimizer',
    'PopulationScenarioAnalyzer'
]