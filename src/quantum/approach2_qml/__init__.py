"""
Approach 2: Quantum Machine Learning for Population PK

Quantum neural networks with enhanced expressivity for population pharmacokinetic modeling.
Focuses on capturing complex nonlinear relationships with limited clinical trial data.

Key Features:
- Quantum neural networks (QNNs) with parameterized quantum circuits
- Enhanced generalization capability for small datasets  
- Population-level parameter inference with uncertainty quantification
- Ensemble methods for robust predictions
"""

from .quantum_neural_network import QuantumNeuralNetwork
# from .population_pk_model import PopulationPKModel  # Missing file
# from .qnn_ensemble import QNNEnsemble  # Missing file

__all__ = [
    'QuantumNeuralNetwork',
    # 'PopulationPKModel', 
    # 'QNNEnsemble'
]