"""
QAOA Dosing Optimizer

Implements QAOA for multi-objective dosing optimization in PK/PD modeling.
Treats dose selection as a combinatorial optimization problem.
"""

import numpy as np
import pennylane as qml
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..core.base import ModelConfig, OptimizationResult
from ..core.pennylane_utils import QuantumCircuitBuilder


@dataclass
class QAOAConfig(ModelConfig):
    """Configuration for QAOA dosing optimization"""
    n_qaoa_layers: int = 3
    dose_resolution: float = 0.5  # mg increments
    max_dose: float = 20.0  # mg maximum dose
    population_size: int = 1000
    objective_weights: Dict[str, float] = None  # weights for multi-objective optimization
    

class QAOADosingOptimizer:
    """
    QAOA-based dosing optimization for PK/PD models
    
    Formulates dosing selection as Quadratic Unconstrained Binary Optimization (QUBO)
    problem and solves using Quantum Approximate Optimization Algorithm.
    
    Objectives:
    - Maximize population coverage (≥90% achieving biomarker < 3.3 ng/mL)
    - Minimize dose level (safety consideration)
    - Minimize inter-individual variability
    """
    
    def __init__(self, config: QAOAConfig):
        self.config = config
        self.device = None
        self.qaoa_circuit = None
        self.dose_options = None
        self.qubo_matrix = None
        self.optimal_params = None
        
        # Default objective weights
        if config.objective_weights is None:
            self.config.objective_weights = {
                'efficacy': 1.0,      # Population coverage
                'safety': 0.5,        # Minimize dose
                'variability': 0.3    # Minimize variability
            }
            
    def setup_quantum_device(self, n_qubits: int) -> qml.device:
        """Setup quantum device for QAOA"""
        self.device = qml.device("default.qubit", wires=n_qubits)
        return self.device
        
    def create_dose_encoding(self) -> List[float]:
        """Create binary encoding for dose options"""
        # Generate dose options with specified resolution
        doses = np.arange(self.config.dose_resolution, 
                         self.config.max_dose + self.config.dose_resolution,
                         self.config.dose_resolution)
        
        self.dose_options = doses
        return doses.tolist()
    
    def formulate_qubo_problem(self, 
                              population_predictions: Dict[float, np.ndarray],
                              target_threshold: float = 3.3) -> np.ndarray:
        """
        Formulate dosing optimization as QUBO problem
        
        Args:
            population_predictions: Dict mapping doses to biomarker predictions for population
            target_threshold: Biomarker threshold for efficacy
            
        Returns:
            QUBO matrix Q where objective = x^T Q x for binary variables x
        """
        n_doses = len(self.dose_options)
        Q = np.zeros((n_doses, n_doses))
        
        for i, dose in enumerate(self.dose_options):
            biomarker_values = population_predictions.get(dose, np.array([]))
            
            if len(biomarker_values) > 0:
                # Efficacy term: maximize population coverage
                coverage = np.mean(biomarker_values < target_threshold)
                efficacy_score = -self.config.objective_weights['efficacy'] * coverage
                
                # Safety term: minimize dose level
                safety_penalty = self.config.objective_weights['safety'] * (dose / self.config.max_dose)
                
                # Variability term: minimize inter-individual variability  
                variability_penalty = self.config.objective_weights['variability'] * np.std(biomarker_values)
                
                # Diagonal terms (single dose selection)
                Q[i, i] = efficacy_score + safety_penalty + variability_penalty
                
                # Off-diagonal terms (interactions between dose selections)
                # Penalize selecting multiple doses (want single optimal dose)
                for j in range(i + 1, n_doses):
                    Q[i, j] = 100.0  # Large penalty for selecting multiple doses
                    Q[j, i] = 100.0
        
        self.qubo_matrix = Q
        return Q
    
    def create_qaoa_circuit(self, n_qubits: int) -> callable:
        """Create QAOA circuit for dose optimization"""
        
        @qml.qnode(self.device)
        def qaoa_circuit(params):
            """
            QAOA circuit for dosing optimization
            
            Args:
                params: [gamma_1, beta_1, gamma_2, beta_2, ..., gamma_p, beta_p]
                       where p is number of QAOA layers
            """
            # Initialize in superposition
            for qubit in range(n_qubits):
                qml.Hadamard(wires=qubit)
            
            # QAOA layers
            for layer in range(self.config.n_qaoa_layers):
                gamma = params[2 * layer]
                beta = params[2 * layer + 1]
                
                # Cost Hamiltonian evolution (from QUBO matrix)
                self._apply_cost_hamiltonian(gamma, n_qubits)
                
                # Mixer Hamiltonian evolution
                self._apply_mixer_hamiltonian(beta, n_qubits)
            
            # Measure in computational basis
            return qml.probs(wires=range(n_qubits))
        
        self.qaoa_circuit = qaoa_circuit
        return qaoa_circuit
    
    def _apply_cost_hamiltonian(self, gamma: float, n_qubits: int):
        """Apply cost Hamiltonian evolution based on QUBO matrix"""
        # Diagonal terms (Z rotations)
        for i in range(n_qubits):
            if i < len(self.qubo_matrix):
                qml.RZ(2 * gamma * self.qubo_matrix[i, i], wires=i)
        
        # Off-diagonal terms (ZZ interactions)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if i < len(self.qubo_matrix) and j < len(self.qubo_matrix):
                    if abs(self.qubo_matrix[i, j]) > 1e-6:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma * self.qubo_matrix[i, j], wires=j)
                        qml.CNOT(wires=[i, j])
    
    def _apply_mixer_hamiltonian(self, beta: float, n_qubits: int):
        """Apply mixer Hamiltonian (X rotations)"""
        for qubit in range(n_qubits):
            qml.RX(2 * beta, wires=qubit)
    
    def evaluate_qaoa_cost(self, bit_string: str) -> float:
        """Evaluate QUBO cost for a given bit string"""
        x = np.array([int(bit) for bit in bit_string])
        cost = x.T @ self.qubo_matrix @ x
        return cost
    
    def optimize_qaoa_parameters(self, initial_params: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize QAOA parameters using classical optimizer"""
        n_qubits = len(self.dose_options)
        self.setup_quantum_device(n_qubits)
        self.create_qaoa_circuit(n_qubits)
        
        # Initialize QAOA parameters
        if initial_params is None:
            params = np.random.uniform(0, 2*np.pi, 2 * self.config.n_qaoa_layers)
        else:
            params = initial_params
            
        def qaoa_objective(qaoa_params):
            """Objective function for QAOA parameter optimization"""
            probabilities = self.qaoa_circuit(qaoa_params)
            
            expectation = 0.0
            for i, prob in enumerate(probabilities):
                bit_string = format(i, f'0{n_qubits}b')
                cost = self.evaluate_qaoa_cost(bit_string)
                expectation += prob * cost
                
            return expectation
        
        # Classical optimization of QAOA parameters
        optimizer = qml.AdamOptimizer(stepsize=0.1)
        
        cost_history = []
        for iteration in range(self.config.max_iterations):
            params, cost = optimizer.step_and_cost(qaoa_objective, params)
            cost_history.append(cost)
            
            if iteration > 10 and abs(cost_history[-1] - cost_history[-10]) < self.config.convergence_threshold:
                break
        
        self.optimal_params = params
        
        return {
            'optimal_params': params,
            'final_cost': cost_history[-1], 
            'cost_history': cost_history,
            'iterations': len(cost_history)
        }
    
    def extract_optimal_dose(self) -> Tuple[float, Dict[str, float]]:
        """Extract optimal dose from QAOA solution"""
        if self.optimal_params is None:
            raise ValueError("QAOA parameters not optimized yet")
            
        n_qubits = len(self.dose_options)
        probabilities = self.qaoa_circuit(self.optimal_params)
        
        # Find most probable bit string
        max_prob_idx = np.argmax(probabilities)
        optimal_bit_string = format(max_prob_idx, f'0{n_qubits}b')
        
        # Decode bit string to dose
        selected_doses = []
        for i, bit in enumerate(optimal_bit_string):
            if bit == '1' and i < len(self.dose_options):
                selected_doses.append(self.dose_options[i])
        
        if len(selected_doses) == 0:
            # Fallback: select dose with highest individual probability
            individual_probs = np.zeros(len(self.dose_options))
            for i, prob in enumerate(probabilities):
                bit_string = format(i, f'0{n_qubits}b')
                for j, bit in enumerate(bit_string):
                    if bit == '1' and j < len(self.dose_options):
                        individual_probs[j] += prob
            
            optimal_dose = self.dose_options[np.argmax(individual_probs)]
        else:
            # If multiple doses selected, take the first one (should be penalized by QUBO)
            optimal_dose = selected_doses[0]
        
        # Calculate solution quality metrics
        solution_metrics = {
            'probability': float(probabilities[max_prob_idx]),
            'n_selected_doses': len(selected_doses),
            'bit_string': optimal_bit_string,
            'qaoa_cost': self.evaluate_qaoa_cost(optimal_bit_string)
        }
        
        return optimal_dose, solution_metrics
    
    def optimize_daily_dosing(self, 
                            pk_model: callable,
                            pd_model: callable,
                            population_params: Dict[str, np.ndarray],
                            target_threshold: float = 3.3) -> OptimizationResult:
        """Optimize daily dosing regimen using QAOA"""
        
        # Create dose encoding
        self.create_dose_encoding()
        
        # Generate population predictions for each dose option
        population_predictions = {}
        
        for dose in self.dose_options:
            biomarker_values = []
            
            # Simulate population response
            for i in range(self.config.population_size):
                # Sample individual parameters from population distributions
                individual_params = {}
                for param_name, param_dist in population_params.items():
                    individual_params[param_name] = np.random.choice(param_dist)
                
                # Predict steady-state biomarker (simplified)
                # In practice, this would use the full PK/PD models
                time_ss = np.array([24.0 * 7])  # Steady state approximation
                
                # Placeholder prediction (will be replaced with actual models)
                baseline = individual_params.get('baseline', 10.0)
                imax = individual_params.get('imax', 0.8)
                ic50 = individual_params.get('ic50', 5.0)
                
                # Simple Emax model for demonstration
                inhibition = imax * dose / (ic50 + dose)
                biomarker = baseline * (1 - inhibition)
                biomarker_values.append(biomarker)
            
            population_predictions[dose] = np.array(biomarker_values)
        
        # Formulate and solve QUBO problem
        self.formulate_qubo_problem(population_predictions, target_threshold)
        optimization_result = self.optimize_qaoa_parameters()
        optimal_dose, solution_metrics = self.extract_optimal_dose()
        
        # Calculate final population coverage
        final_biomarkers = population_predictions[optimal_dose]
        coverage = np.mean(final_biomarkers < target_threshold)
        
        result = OptimizationResult(
            optimal_daily_dose=optimal_dose,
            optimal_weekly_dose=optimal_dose * 7,  # Simple conversion
            population_coverage=coverage,
            parameter_estimates=optimization_result,
            confidence_intervals={},
            convergence_info={
                'method': 'QAOA',
                'qaoa_layers': self.config.n_qaoa_layers,
                'solution_metrics': solution_metrics
            },
            quantum_metrics={
                'n_qubits': len(self.dose_options),
                'qaoa_cost': solution_metrics['qaoa_cost'],
                'solution_probability': solution_metrics['probability']
            }
        )
        
        return result