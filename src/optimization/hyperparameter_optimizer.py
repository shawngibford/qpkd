"""
Hyperparameter optimization utilities for quantum PK/PD models.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any, Callable, Optional
import logging
from dataclasses import dataclass, field
import optuna
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import time

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


@dataclass 
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    learning_rate: Tuple[float, float] = (0.001, 0.1)
    n_layers: Tuple[int, int] = (2, 8)
    n_qubits: Tuple[int, int] = (4, 12)
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    regularization: Tuple[float, float] = (1e-6, 1e-2)
    quantum_circuit_depth: Tuple[int, int] = (2, 10)
    ansatz_type: List[str] = field(default_factory=lambda: ['hardware_efficient', 'alternating', 'real_amplitudes'])
    optimization_method: List[str] = field(default_factory=lambda: ['adam', 'sgd', 'rmsprop'])


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization for quantum PK/PD models."""
    
    def __init__(self, 
                 optimization_method: str = 'bayesian',
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 timeout_minutes: int = 30,
                 n_jobs: int = 1):
        """Initialize hyperparameter optimizer.
        
        Args:
            optimization_method: 'grid', 'random', 'bayesian', 'optuna', 'evolutionary'
            n_trials: Number of optimization trials
            cv_folds: Cross-validation folds
            timeout_minutes: Timeout for individual trials
            n_jobs: Number of parallel jobs
        """
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.timeout_minutes = timeout_minutes
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization components
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = []
        
    def optimize(self,
                 model_class: Any,
                 data: Any,
                 param_space: HyperparameterSpace = None,
                 scoring_function: Callable = None,
                 **kwargs) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Args:
            model_class: Model class to optimize
            data: Training data
            param_space: Hyperparameter search space
            scoring_function: Custom scoring function
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimization results
        """
        if param_space is None:
            param_space = HyperparameterSpace()
            
        self.logger.info(f"Starting {self.optimization_method} hyperparameter optimization")
        
        # Select optimization method
        if self.optimization_method == 'grid':
            return self._grid_search_optimization(model_class, data, param_space, scoring_function)
        elif self.optimization_method == 'random':
            return self._random_search_optimization(model_class, data, param_space, scoring_function)
        elif self.optimization_method == 'bayesian':
            return self._bayesian_optimization(model_class, data, param_space, scoring_function)
        elif self.optimization_method == 'optuna':
            return self._optuna_optimization(model_class, data, param_space, scoring_function)
        elif self.optimization_method == 'evolutionary':
            return self._evolutionary_optimization(model_class, data, param_space, scoring_function)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
    def _grid_search_optimization(self,
                                 model_class: Any,
                                 data: Any,
                                 param_space: HyperparameterSpace,
                                 scoring_function: Callable) -> Dict[str, Any]:
        """Grid search hyperparameter optimization."""
        
        # Create grid of parameters
        param_grid = {
            'learning_rate': np.logspace(np.log10(param_space.learning_rate[0]), 
                                       np.log10(param_space.learning_rate[1]), 5),
            'n_layers': range(param_space.n_layers[0], param_space.n_layers[1] + 1, 2),
            'n_qubits': range(param_space.n_qubits[0], param_space.n_qubits[1] + 1, 2),
            'batch_size': param_space.batch_size,
            'ansatz_type': param_space.ansatz_type
        }
        
        best_score = -np.inf
        best_params = None
        results = []
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            try:
                score = self._evaluate_parameters(model_class, data, params, scoring_function)
                
                results.append({
                    'params': params,
                    'score': score,
                    'trial_number': len(results)
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                self.logger.info(f"Trial {len(results)}: Score={score:.4f}, Params={params}")
                
            except Exception as e:
                self.logger.warning(f"Failed parameter evaluation: {e}")
                continue
                
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': results,
            'method': 'grid_search'
        }
        
    def _random_search_optimization(self,
                                   model_class: Any,
                                   data: Any,
                                   param_space: HyperparameterSpace,
                                   scoring_function: Callable) -> Dict[str, Any]:
        """Random search hyperparameter optimization."""
        
        best_score = -np.inf
        best_params = None
        results = []
        
        for trial in range(self.n_trials):
            # Sample random parameters
            params = {
                'learning_rate': np.random.uniform(*param_space.learning_rate),
                'n_layers': np.random.randint(*param_space.n_layers),
                'n_qubits': np.random.randint(*param_space.n_qubits),
                'batch_size': np.random.choice(param_space.batch_size),
                'regularization': np.random.uniform(*param_space.regularization),
                'ansatz_type': np.random.choice(param_space.ansatz_type),
                'optimization_method': np.random.choice(param_space.optimization_method)
            }
            
            try:
                score = self._evaluate_parameters(model_class, data, params, scoring_function)
                
                results.append({
                    'params': params,
                    'score': score,
                    'trial_number': trial
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                self.logger.info(f"Trial {trial}: Score={score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Trial {trial} failed: {e}")
                continue
                
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': results,
            'method': 'random_search'
        }
        
    def _bayesian_optimization(self,
                              model_class: Any,
                              data: Any,
                              param_space: HyperparameterSpace,
                              scoring_function: Callable) -> Dict[str, Any]:
        """Bayesian optimization using Gaussian processes."""
        
        if not SKOPT_AVAILABLE:
            self.logger.warning("scikit-optimize not available, falling back to random search")
            return self._random_search_optimization(model_class, data, param_space, scoring_function)
            
        # Define search space for skopt
        dimensions = [
            Real(*param_space.learning_rate, name='learning_rate', prior='log-uniform'),
            Integer(*param_space.n_layers, name='n_layers'),
            Integer(*param_space.n_qubits, name='n_qubits'),
            Categorical(param_space.batch_size, name='batch_size'),
            Real(*param_space.regularization, name='regularization', prior='log-uniform'),
            Integer(*param_space.quantum_circuit_depth, name='quantum_circuit_depth'),
            Categorical(param_space.ansatz_type, name='ansatz_type'),
            Categorical(param_space.optimization_method, name='optimization_method')
        ]
        
        def objective(params):
            """Objective function for Bayesian optimization."""
            param_dict = {
                'learning_rate': params[0],
                'n_layers': params[1],
                'n_qubits': params[2],
                'batch_size': params[3],
                'regularization': params[4],
                'quantum_circuit_depth': params[5],
                'ansatz_type': params[6],
                'optimization_method': params[7]
            }
            
            try:
                score = self._evaluate_parameters(model_class, data, param_dict, scoring_function)
                return -score  # Minimize negative score
            except Exception as e:
                self.logger.warning(f"Bayesian optimization trial failed: {e}")
                return 1e6  # Large penalty
                
        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=self.n_trials,
            n_initial_points=10,
            random_state=42
        )
        
        # Extract best parameters
        best_params = {
            'learning_rate': result.x[0],
            'n_layers': result.x[1],
            'n_qubits': result.x[2],
            'batch_size': result.x[3],
            'regularization': result.x[4],
            'quantum_circuit_depth': result.x[5],
            'ansatz_type': result.x[6],
            'optimization_method': result.x[7]
        }
        
        return {
            'best_params': best_params,
            'best_score': -result.fun,
            'optimization_history': [{'score': -f, 'trial_number': i} for i, f in enumerate(result.func_vals)],
            'method': 'bayesian_optimization',
            'convergence': result.func_vals
        }
        
    def _optuna_optimization(self,
                            model_class: Any,
                            data: Any,
                            param_space: HyperparameterSpace,
                            scoring_function: Callable) -> Dict[str, Any]:
        """Optimization using Optuna framework."""
        
        try:
            import optuna
        except ImportError:
            self.logger.warning("Optuna not available, falling back to random search")
            return self._random_search_optimization(model_class, data, param_space, scoring_function)
            
        def objective(trial):
            """Optuna objective function."""
            params = {
                'learning_rate': trial.suggest_float('learning_rate', *param_space.learning_rate, log=True),
                'n_layers': trial.suggest_int('n_layers', *param_space.n_layers),
                'n_qubits': trial.suggest_int('n_qubits', *param_space.n_qubits),
                'batch_size': trial.suggest_categorical('batch_size', param_space.batch_size),
                'regularization': trial.suggest_float('regularization', *param_space.regularization, log=True),
                'quantum_circuit_depth': trial.suggest_int('quantum_circuit_depth', *param_space.quantum_circuit_depth),
                'ansatz_type': trial.suggest_categorical('ansatz_type', param_space.ansatz_type),
                'optimization_method': trial.suggest_categorical('optimization_method', param_space.optimization_method)
            }
            
            try:
                score = self._evaluate_parameters(model_class, data, params, scoring_function)
                return score
            except Exception as e:
                self.logger.warning(f"Optuna trial failed: {e}")
                raise optuna.TrialPruned()
                
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_minutes * 60)
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'params': trial.params,
                    'score': trial.value,
                    'trial_number': trial.number
                })
                
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': history,
            'method': 'optuna',
            'study': study
        }
        
    def _evolutionary_optimization(self,
                                  model_class: Any,
                                  data: Any,
                                  param_space: HyperparameterSpace,
                                  scoring_function: Callable) -> Dict[str, Any]:
        """Evolutionary optimization using differential evolution."""
        
        from scipy.optimize import differential_evolution
        
        # Define parameter bounds
        bounds = [
            param_space.learning_rate,
            param_space.n_layers,
            param_space.n_qubits,
            (0, len(param_space.batch_size) - 1),  # Index into batch_size list
            param_space.regularization,
            param_space.quantum_circuit_depth,
            (0, len(param_space.ansatz_type) - 1),  # Index into ansatz_type list
            (0, len(param_space.optimization_method) - 1)  # Index into optimization_method list
        ]
        
        def objective(params):
            """Objective function for evolutionary optimization."""
            param_dict = {
                'learning_rate': params[0],
                'n_layers': int(params[1]),
                'n_qubits': int(params[2]),
                'batch_size': param_space.batch_size[int(params[3])],
                'regularization': params[4],
                'quantum_circuit_depth': int(params[5]),
                'ansatz_type': param_space.ansatz_type[int(params[6])],
                'optimization_method': param_space.optimization_method[int(params[7])]
            }
            
            try:
                score = self._evaluate_parameters(model_class, data, param_dict, scoring_function)
                return -score  # Minimize negative score
            except Exception as e:
                self.logger.warning(f"Evolutionary optimization trial failed: {e}")
                return 1e6
                
        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            maxiter=self.n_trials // 10,
            popsize=15,
            seed=42
        )
        
        # Extract best parameters
        best_params = {
            'learning_rate': result.x[0],
            'n_layers': int(result.x[1]),
            'n_qubits': int(result.x[2]),
            'batch_size': param_space.batch_size[int(result.x[3])],
            'regularization': result.x[4],
            'quantum_circuit_depth': int(result.x[5]),
            'ansatz_type': param_space.ansatz_type[int(result.x[6])],
            'optimization_method': param_space.optimization_method[int(result.x[7])]
        }
        
        return {
            'best_params': best_params,
            'best_score': -result.fun,
            'optimization_history': [{'score': -result.fun, 'trial_number': 0}],  # Simplified history
            'method': 'evolutionary',
            'iterations': result.nit,
            'function_evaluations': result.nfev
        }
        
    def _evaluate_parameters(self,
                            model_class: Any,
                            data: Any,
                            params: Dict[str, Any],
                            scoring_function: Callable = None) -> float:
        """Evaluate model performance with given parameters.
        
        Args:
            model_class: Model class to instantiate
            data: Training data
            params: Parameters to evaluate
            scoring_function: Custom scoring function
            
        Returns:
            Model performance score
        """
        try:
            # Create model with parameters
            model = model_class(**params)
            
            if scoring_function:
                # Use custom scoring function
                score = scoring_function(model, data)
            else:
                # Default cross-validation scoring
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, data.features, data.biomarkers, 
                                       cv=self.cv_folds, scoring='neg_mean_squared_error')
                score = -np.mean(scores)  # Convert to positive MSE, then negate for maximization
                
            return float(score)
            
        except Exception as e:
            self.logger.warning(f"Parameter evaluation failed: {e}")
            raise
            
    def adaptive_optimization(self,
                             model_class: Any,
                             data: Any,
                             param_space: HyperparameterSpace = None,
                             early_stopping_patience: int = 10,
                             **kwargs) -> Dict[str, Any]:
        """Adaptive hyperparameter optimization with early stopping.
        
        Args:
            model_class: Model class to optimize
            data: Training data
            param_space: Hyperparameter search space
            early_stopping_patience: Number of trials without improvement before stopping
            **kwargs: Additional parameters
            
        Returns:
            Optimization results with early stopping
        """
        if param_space is None:
            param_space = HyperparameterSpace()
            
        best_score = -np.inf
        best_params = None
        no_improvement_count = 0
        results = []
        
        for trial in range(self.n_trials):
            # Sample parameters with adaptive strategy
            if trial < 10:
                # Random exploration phase
                params = self._sample_random_parameters(param_space)
            else:
                # Exploitation phase - sample near best parameters
                params = self._sample_near_best(param_space, best_params)
                
            try:
                score = self._evaluate_parameters(model_class, data, params)
                
                results.append({
                    'params': params,
                    'score': score,
                    'trial_number': trial
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    no_improvement_count = 0
                    self.logger.info(f"New best score: {score:.4f} at trial {trial}")
                else:
                    no_improvement_count += 1
                    
                # Early stopping check
                if no_improvement_count >= early_stopping_patience:
                    self.logger.info(f"Early stopping after {trial + 1} trials")
                    break
                    
            except Exception as e:
                self.logger.warning(f"Trial {trial} failed: {e}")
                no_improvement_count += 1
                continue
                
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': results,
            'method': 'adaptive',
            'early_stopped': no_improvement_count >= early_stopping_patience,
            'total_trials': len(results)
        }
        
    def _sample_random_parameters(self, param_space: HyperparameterSpace) -> Dict[str, Any]:
        """Sample random parameters from the space."""
        return {
            'learning_rate': np.random.uniform(*param_space.learning_rate),
            'n_layers': np.random.randint(*param_space.n_layers),
            'n_qubits': np.random.randint(*param_space.n_qubits),
            'batch_size': np.random.choice(param_space.batch_size),
            'regularization': np.random.uniform(*param_space.regularization),
            'quantum_circuit_depth': np.random.randint(*param_space.quantum_circuit_depth),
            'ansatz_type': np.random.choice(param_space.ansatz_type),
            'optimization_method': np.random.choice(param_space.optimization_method)
        }
        
    def _sample_near_best(self, param_space: HyperparameterSpace, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters near the current best."""
        if best_params is None:
            return self._sample_random_parameters(param_space)
            
        params = best_params.copy()
        
        # Add small perturbations
        params['learning_rate'] = np.clip(
            best_params['learning_rate'] * np.random.normal(1.0, 0.1),
            *param_space.learning_rate
        )
        
        params['n_layers'] = np.clip(
            best_params['n_layers'] + np.random.randint(-1, 2),
            *param_space.n_layers
        )
        
        params['regularization'] = np.clip(
            best_params['regularization'] * np.random.normal(1.0, 0.1),
            *param_space.regularization
        )
        
        # Sometimes sample completely different categorical parameters
        if np.random.random() < 0.3:
            params['ansatz_type'] = np.random.choice(param_space.ansatz_type)
            params['optimization_method'] = np.random.choice(param_space.optimization_method)
            
        return params