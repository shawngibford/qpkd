"""
Approach 5: Tensor Network Population Modeling with ZX Calculus

Tensor network decomposition using ZX calculus for efficient representation of
high-dimensional population parameter distributions.

Key Features:
- Matrix Product State (MPS) representation of population tensors
- ZX calculus for quantum circuit optimization and simplification
- Efficient population simulation from limited data
- Interpretable parameter relationships and covariate effects
"""

from .tensor_population_model import TensorPopulationModel
from .zx_circuit_optimizer import ZXCircuitOptimizer  
from .mps_parameter_sampler import MPSParameterSampler

__all__ = [
    'TensorPopulationModel',
    'ZXCircuitOptimizer',
    'MPSParameterSampler'
]