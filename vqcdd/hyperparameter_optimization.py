"""
Hyperparameter Optimization Module for VQCdd Phase 2C

This module implements advanced hyperparameter optimization capabilities including
Bayesian optimization, multi-objective optimization, and sensitivity analysis.
It provides intelligent hyperparameter tuning for quantum circuits and training
configurations to maximize VQC performance.

Key Features:
- Bayesian optimization with Gaussian Process surrogates
- Multi-objective optimization (accuracy, time, quantum resources)
- Hyperparameter sensitivity analysis and importance ranking
- Automated hyperparameter search space definition
- Integration with validation framework for robust evaluation
- Parallel optimization with multiple acquisition functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import warnings
from pathlib import Path
import json
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# Scientific computing and optimization libraries
from scipy.optimize import minimize
from scipy.stats import uniform, randint, loguniform
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import matplotlib.pyplot as plt
import seaborn as sns

# Import dependency management
from utils.dependencies import dependency_manager, has_bayesian_optimization, has_multi_objective_optimization

# Bayesian optimization libraries (managed gracefully)
SKOPT_AVAILABLE = has_bayesian_optimization()
if SKOPT_AVAILABLE:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective, plot_evaluations
    from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi

# Multi-objective optimization libraries (managed gracefully)
PYMOO_AVAILABLE = has_multi_objective_optimization()
if PYMOO_AVAILABLE:
    import pymoo
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as moo_minimize

# Import VQCdd modules
from optimizer import VQCTrainer, OptimizationConfig
from quantum_circuit import VQCircuit, CircuitConfig
from data_handler import StudyData, PatientData
from validation import ValidationPipeline, ValidationConfig, ValidationResults


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space"""

    # Circuit architecture parameters
    n_qubits: Tuple[int, int] = (2, 8)                    # (min, max) qubits
    n_layers: Tuple[int, int] = (1, 10)                   # (min, max) layers
    ansatz_options: List[str] = field(default_factory=lambda: ["ry_cnot", "strongly_entangling"])
    encoding_options: List[str] = field(default_factory=lambda: ["angle", "amplitude"])

    # Optimization parameters
    learning_rate: Tuple[float, float] = (1e-4, 1e-1)     # (min, max) learning rate
    optimizer_options: List[str] = field(default_factory=lambda: ["adam", "adagrad", "qng"])
    batch_size: Tuple[int, int] = (8, 128)                # (min, max) batch size
    max_iterations: Tuple[int, int] = (50, 500)           # (min, max) iterations

    # Regularization parameters
    regularization_weight: Tuple[float, float] = (1e-5, 1e-1)

    # Learning rate scheduling
    lr_scheduler_options: List[str] = field(default_factory=lambda: ["constant", "exponential", "adaptive"])
    lr_decay_rate: Tuple[float, float] = (0.8, 0.99)

    # Advanced parameters
    momentum: Tuple[float, float] = (0.7, 0.99)
    gradient_clipping: Tuple[float, float] = (0.1, 10.0)

    # Boolean parameters (represented as choices)
    enable_mini_batches: List[bool] = field(default_factory=lambda: [True, False])
    track_gradient_variance: List[bool] = field(default_factory=lambda: [True, False])

    def to_skopt_space(self) -> List:
        """Convert to scikit-optimize space format"""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")

        space = [
            Integer(self.n_qubits[0], self.n_qubits[1], name='n_qubits'),
            Integer(self.n_layers[0], self.n_layers[1], name='n_layers'),
            Categorical(self.ansatz_options, name='ansatz'),
            Categorical(self.encoding_options, name='encoding'),
            Real(self.learning_rate[0], self.learning_rate[1], prior='log-uniform', name='learning_rate'),
            Categorical(self.optimizer_options, name='optimizer_type'),
            Integer(self.batch_size[0], self.batch_size[1], name='batch_size'),
            Integer(self.max_iterations[0], self.max_iterations[1], name='max_iterations'),
            Real(self.regularization_weight[0], self.regularization_weight[1], prior='log-uniform', name='regularization_weight'),
            Categorical(self.lr_scheduler_options, name='lr_scheduler'),
            Real(self.lr_decay_rate[0], self.lr_decay_rate[1], name='lr_decay_rate'),
            Real(self.momentum[0], self.momentum[1], name='momentum'),
            Real(self.gradient_clipping[0], self.gradient_clipping[1], name='gradient_clipping'),
            Categorical(self.enable_mini_batches, name='enable_mini_batches'),
            Categorical(self.track_gradient_variance, name='track_gradient_variance')
        ]
        return space

    def to_random_space(self) -> Dict[str, Any]:
        """Convert to format for random search"""
        space = {
            'n_qubits': randint(self.n_qubits[0], self.n_qubits[1] + 1),
            'n_layers': randint(self.n_layers[0], self.n_layers[1] + 1),
            'ansatz': self.ansatz_options,
            'encoding': self.encoding_options,
            'learning_rate': loguniform(self.learning_rate[0], self.learning_rate[1]),
            'optimizer_type': self.optimizer_options,
            'batch_size': randint(self.batch_size[0], self.batch_size[1] + 1),
            'max_iterations': randint(self.max_iterations[0], self.max_iterations[1] + 1),
            'regularization_weight': loguniform(self.regularization_weight[0], self.regularization_weight[1]),
            'lr_scheduler': self.lr_scheduler_options,
            'lr_decay_rate': uniform(self.lr_decay_rate[0], self.lr_decay_rate[1] - self.lr_decay_rate[0]),
            'momentum': uniform(self.momentum[0], self.momentum[1] - self.momentum[0]),
            'gradient_clipping': uniform(self.gradient_clipping[0], self.gradient_clipping[1] - self.gradient_clipping[0]),
            'enable_mini_batches': self.enable_mini_batches,
            'track_gradient_variance': self.track_gradient_variance
        }
        return space


@dataclass
class OptimizationObjectives:
    """Definition of optimization objectives and constraints"""

    # Primary objectives (to minimize)
    primary_metric: str = "mse"                           # Main performance metric
    secondary_metrics: List[str] = field(default_factory=lambda: ["mae", "training_time"])

    # Multi-objective weights (if using weighted scalarization)
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "mse": 1.0,
        "mae": 0.5,
        "training_time": 0.1,
        "quantum_resources": 0.05
    })

    # Constraints
    max_training_time: float = 3600.0                     # Maximum training time (seconds)
    max_quantum_resources: float = 1000.0                # Maximum quantum resource usage
    min_performance_threshold: float = 0.8               # Minimum acceptable R² score

    # Multi-objective optimization settings
    population_size: int = 50                             # Population size for MOO
    n_generations: int = 100                              # Number of generations for MOO
    pareto_front_size: int = 10                          # Number of solutions in Pareto front


@dataclass
class HyperparameterOptimizationConfig:
    """Configuration for hyperparameter optimization"""

    # Optimization strategy
    optimization_method: str = "bayesian"                # "bayesian", "random", "grid", "evolutionary"
    n_calls: int = 100                                   # Number of optimization iterations
    n_initial_points: int = 10                           # Number of random initial points

    # Bayesian optimization settings
    acquisition_function: str = "EI"                     # "EI", "LCB", "PI"
    acquisition_kappa: float = 1.96                     # Exploration parameter for LCB
    acquisition_xi: float = 0.01                        # Exploration parameter for EI/PI

    # Cross-validation settings
    cv_folds: int = 3                                    # Number of CV folds for evaluation
    cv_scoring: str = "neg_mean_squared_error"          # Scoring metric for CV

    # Parallel processing
    n_jobs: int = 1                                      # Number of parallel jobs (-1 for all cores)
    timeout_per_evaluation: float = 1800.0              # Timeout per hyperparameter evaluation

    # Early stopping
    early_stopping_rounds: int = 20                     # Stop if no improvement
    early_stopping_delta: float = 1e-4                  # Minimum improvement threshold

    # Output settings
    save_intermediate_results: bool = True               # Save results during optimization
    plot_convergence: bool = True                       # Generate convergence plots
    verbose: bool = True                                # Verbose output

    # Reproducibility
    random_state: int = 42                              # Random seed


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization"""

    # Best configuration found
    best_hyperparameters: Dict[str, Any]
    best_score: float
    best_validation_results: ValidationResults

    # Optimization history
    hyperparameter_history: List[Dict[str, Any]]
    score_history: List[float]
    evaluation_times: List[float]

    # Sensitivity analysis
    parameter_importance: Dict[str, float]
    correlation_matrix: np.ndarray
    feature_names: List[str]

    # Statistics
    total_optimization_time: float
    n_evaluations: int

    # Configuration used
    optimization_config: HyperparameterOptimizationConfig
    search_space: HyperparameterSpace
    objectives: OptimizationObjectives

    # Multi-objective results (if applicable)
    pareto_front: Optional[List[Dict[str, Any]]] = None
    pareto_scores: Optional[List[List[float]]] = None
    convergence_iteration: Optional[int] = None


class BaseHyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimizers"""

    def __init__(
        self,
        search_space: HyperparameterSpace,
        objectives: OptimizationObjectives,
        config: HyperparameterOptimizationConfig
    ):
        self.search_space = search_space
        self.objectives = objectives
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Optimization state (memory-limited)
        from collections import deque
        self.evaluation_history = deque(maxlen=500)  # Limit hyperparameter evaluation history
        self.best_score = float('inf')
        self.best_hyperparameters = None
        self.start_time = None

    @abstractmethod
    def optimize(
        self,
        objective_function: Callable,
        data: StudyData
    ) -> OptimizationResult:
        """Perform hyperparameter optimization"""
        pass

    def _evaluate_hyperparameters(
        self,
        hyperparameters: Dict[str, Any],
        objective_function: Callable,
        data: StudyData
    ) -> float:
        """Evaluate a set of hyperparameters"""
        start_time = time.time()

        try:
            # Create configurations from hyperparameters
            circuit_config, optimization_config = self._create_configs_from_hyperparameters(hyperparameters)

            # Create trainer and train
            trainer = VQCTrainer(circuit_config, optimization_config)
            score = objective_function(trainer, data)

            # Record evaluation
            evaluation_time = time.time() - start_time
            self.evaluation_history.append({
                'hyperparameters': hyperparameters.copy(),
                'score': score,
                'evaluation_time': evaluation_time,
                'timestamp': time.time()
            })

            # Update best if improved
            if score < self.best_score:
                self.best_score = score
                self.best_hyperparameters = hyperparameters.copy()

            self.logger.info(f"Evaluated hyperparameters: score={score:.6f}, time={evaluation_time:.1f}s")

            return score

        except Exception as e:
            self.logger.warning(f"Evaluation failed: {e}")
            # Return penalty score for failed evaluations
            return float('inf')

    def _create_configs_from_hyperparameters(
        self,
        hyperparameters: Dict[str, Any]
    ) -> Tuple[CircuitConfig, OptimizationConfig]:
        """Create circuit and optimization configs from hyperparameters"""

        # Create circuit config
        circuit_config = CircuitConfig(
            n_qubits=int(hyperparameters.get('n_qubits', 4)),
            n_layers=int(hyperparameters.get('n_layers', 3)),
            ansatz=hyperparameters.get('ansatz', 'ry_cnot'),
            encoding=hyperparameters.get('encoding', 'angle')
        )

        # Create optimization config
        optimization_config = OptimizationConfig(
            max_iterations=int(hyperparameters.get('max_iterations', 100)),
            learning_rate=float(hyperparameters.get('learning_rate', 0.01)),
            optimizer_type=hyperparameters.get('optimizer_type', 'adam'),
            batch_size=int(hyperparameters.get('batch_size', 32)) if hyperparameters.get('enable_mini_batches', False) else None,
            regularization_weight=float(hyperparameters.get('regularization_weight', 0.01)),
            lr_scheduler=hyperparameters.get('lr_scheduler', 'constant'),
            lr_decay_rate=float(hyperparameters.get('lr_decay_rate', 0.95)),
            momentum=float(hyperparameters.get('momentum', 0.9)),
            gradient_clipping=float(hyperparameters.get('gradient_clipping', 1.0)),
            enable_mini_batches=bool(hyperparameters.get('enable_mini_batches', False)),
            track_gradient_variance=bool(hyperparameters.get('track_gradient_variance', True))
        )

        return circuit_config, optimization_config


class BayesianOptimizer(BaseHyperparameterOptimizer):
    """Bayesian optimization with Gaussian Process surrogate"""

    def __init__(
        self,
        search_space: HyperparameterSpace,
        objectives: OptimizationObjectives,
        config: HyperparameterOptimizationConfig
    ):
        super().__init__(search_space, objectives, config)

        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")

    def optimize(
        self,
        objective_function: Callable,
        data: StudyData
    ) -> OptimizationResult:
        """Perform Bayesian optimization"""
        self.logger.info("Starting Bayesian hyperparameter optimization")
        self.start_time = time.time()

        # Get search space
        space = self.search_space.to_skopt_space()
        space_keys = [dim.name for dim in space]

        # Define objective function for skopt
        @use_named_args(space)
        def skopt_objective(**hyperparameters):
            return self._evaluate_hyperparameters(hyperparameters, objective_function, data)

        # Choose acquisition function
        if self.config.acquisition_function == "EI":
            acq_func = "EI"
            acq_func_kwargs = {"xi": self.config.acquisition_xi}
        elif self.config.acquisition_function == "LCB":
            acq_func = "LCB"
            acq_func_kwargs = {"kappa": self.config.acquisition_kappa}
        elif self.config.acquisition_function == "PI":
            acq_func = "PI"
            acq_func_kwargs = {"xi": self.config.acquisition_xi}
        else:
            acq_func = "EI"
            acq_func_kwargs = {"xi": 0.01}

        # Perform Bayesian optimization
        try:
            result = gp_minimize(
                func=skopt_objective,
                dimensions=space,
                n_calls=self.config.n_calls,
                n_initial_points=self.config.n_initial_points,
                acq_func=acq_func,
                acq_func_kwargs=acq_func_kwargs,
                random_state=self.config.random_state,
                verbose=self.config.verbose
            )

            # Extract best hyperparameters
            best_hyperparameters = dict(zip(space_keys, result.x))

            # Perform final validation with best hyperparameters
            best_validation_results = self._validate_best_hyperparameters(
                best_hyperparameters, data
            )

            # Perform sensitivity analysis
            parameter_importance = self._analyze_parameter_importance(result)

            optimization_result = OptimizationResult(
                best_hyperparameters=best_hyperparameters,
                best_score=result.fun,
                best_validation_results=best_validation_results,
                hyperparameter_history=[eval_data['hyperparameters'] for eval_data in self.evaluation_history],
                score_history=[eval_data['score'] for eval_data in self.evaluation_history],
                evaluation_times=[eval_data['evaluation_time'] for eval_data in self.evaluation_history],
                parameter_importance=parameter_importance,
                correlation_matrix=np.corrcoef([list(h.values()) for h in [eval_data['hyperparameters'] for eval_data in self.evaluation_history]], rowvar=False),
                feature_names=space_keys,
                total_optimization_time=time.time() - self.start_time,
                n_evaluations=len(self.evaluation_history),
                convergence_iteration=self._find_convergence_iteration(),
                optimization_config=self.config,
                search_space=self.search_space,
                objectives=self.objectives
            )

            # Generate plots if requested
            if self.config.plot_convergence:
                self._plot_optimization_results(result, optimization_result)

            self.logger.info(f"Bayesian optimization completed in {optimization_result.total_optimization_time:.1f}s")
            return optimization_result

        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {e}")
            raise

    def _validate_best_hyperparameters(
        self,
        hyperparameters: Dict[str, Any],
        data: StudyData
    ) -> ValidationResults:
        """Perform comprehensive validation of best hyperparameters"""
        # Create configurations
        circuit_config, optimization_config = self._create_configs_from_hyperparameters(hyperparameters)

        # Create validation configuration
        validation_config = ValidationConfig(
            n_folds=self.config.cv_folds,
            bootstrap_samples=100,
            confidence_level=0.95
        )

        # Create trainer factory
        def trainer_factory():
            return VQCTrainer(circuit_config, optimization_config)

        # Perform validation
        validation_pipeline = ValidationPipeline(validation_config)
        models = {'best_model': trainer_factory}
        validation_results = validation_pipeline.run_comprehensive_validation(models, data)

        return validation_results['cross_validation_results']['best_model']

    def _analyze_parameter_importance(self, skopt_result) -> Dict[str, float]:
        """Analyze parameter importance from optimization history"""
        # Use feature importance from random forest if available
        if hasattr(skopt_result, 'models') and len(skopt_result.models) > 0:
            last_model = skopt_result.models[-1]
            if hasattr(last_model, 'feature_importances_'):
                importances = last_model.feature_importances_
                space_keys = [dim.name for dim in skopt_result.space]
                return dict(zip(space_keys, importances))

        # Fallback: compute variance-based importance
        space_keys = [dim.name for dim in skopt_result.space]
        X = np.array([list(eval_data['hyperparameters'].values()) for eval_data in self.evaluation_history])
        y = np.array([eval_data['score'] for eval_data in self.evaluation_history])

        importances = {}
        for i, key in enumerate(space_keys):
            # Calculate correlation between parameter and objective
            if X.shape[0] > 1:
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                importances[key] = corr if not np.isnan(corr) else 0.0
            else:
                importances[key] = 0.0

        # Normalize
        total_importance = sum(importances.values())
        if total_importance > 0:
            importances = {k: v / total_importance for k, v in importances.items()}

        return importances

    def _find_convergence_iteration(self) -> Optional[int]:
        """Find iteration where optimization converged"""
        if len(self.evaluation_history) < self.config.early_stopping_rounds:
            return None

        scores = [eval_data['score'] for eval_data in self.evaluation_history]
        best_score = min(scores)

        # Find first iteration where we reached within delta of best score
        for i, score in enumerate(scores):
            if score <= best_score + self.config.early_stopping_delta:
                return i

        return None

    def _plot_optimization_results(self, skopt_result, optimization_result: OptimizationResult):
        """Generate optimization visualization plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Plot 1: Convergence
            plot_convergence(skopt_result, ax=axes[0, 0])
            axes[0, 0].set_title('Optimization Convergence')

            # Plot 2: Parameter importance
            importance = optimization_result.parameter_importance
            if importance:
                params = list(importance.keys())
                values = list(importance.values())
                axes[0, 1].barh(params, values)
                axes[0, 1].set_title('Parameter Importance')
                axes[0, 1].set_xlabel('Importance')

            # Plot 3: Evaluation times
            axes[1, 0].plot(optimization_result.evaluation_times)
            axes[1, 0].set_title('Evaluation Time per Iteration')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Time (seconds)')

            # Plot 4: Score history
            axes[1, 1].plot(optimization_result.score_history, 'o-', alpha=0.7)
            axes[1, 1].axhline(y=optimization_result.best_score, color='r', linestyle='--', label='Best Score')
            axes[1, 1].set_title('Score History')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig('bayesian_optimization_results.png', dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info("Optimization plots saved to bayesian_optimization_results.png")

        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {e}")


class RandomSearchOptimizer(BaseHyperparameterOptimizer):
    """Random search hyperparameter optimization"""

    def optimize(
        self,
        objective_function: Callable,
        data: StudyData
    ) -> OptimizationResult:
        """Perform random search optimization"""
        self.logger.info("Starting random search hyperparameter optimization")
        self.start_time = time.time()

        # Get search space
        space = self.search_space.to_random_space()

        for iteration in range(self.config.n_calls):
            # Sample random hyperparameters
            hyperparameters = self._sample_random_hyperparameters(space)

            # Evaluate
            score = self._evaluate_hyperparameters(hyperparameters, objective_function, data)

            if self.config.verbose and iteration % 10 == 0:
                self.logger.info(f"Iteration {iteration}/{self.config.n_calls}, Best score: {self.best_score:.6f}")

        # Create result
        best_validation_results = self._validate_best_hyperparameters(self.best_hyperparameters, data)

        optimization_result = OptimizationResult(
            best_hyperparameters=self.best_hyperparameters,
            best_score=self.best_score,
            best_validation_results=best_validation_results,
            hyperparameter_history=[eval_data['hyperparameters'] for eval_data in self.evaluation_history],
            score_history=[eval_data['score'] for eval_data in self.evaluation_history],
            evaluation_times=[eval_data['evaluation_time'] for eval_data in self.evaluation_history],
            parameter_importance=self._compute_random_search_importance(),
            correlation_matrix=np.corrcoef([list(h.values()) for h in [eval_data['hyperparameters'] for eval_data in self.evaluation_history]], rowvar=False),
            feature_names=list(self.best_hyperparameters.keys()),
            total_optimization_time=time.time() - self.start_time,
            n_evaluations=len(self.evaluation_history),
            convergence_iteration=self._find_convergence_iteration(),
            optimization_config=self.config,
            search_space=self.search_space,
            objectives=self.objectives
        )

        self.logger.info(f"Random search completed in {optimization_result.total_optimization_time:.1f}s")
        return optimization_result

    def _sample_random_hyperparameters(self, space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random hyperparameters from space"""
        hyperparameters = {}

        for param, distribution in space.items():
            if isinstance(distribution, list):
                # Categorical parameter
                hyperparameters[param] = np.random.choice(distribution)
            elif hasattr(distribution, 'rvs'):
                # Scipy distribution
                hyperparameters[param] = distribution.rvs()
            else:
                # Default: assume it's a list for categorical
                hyperparameters[param] = np.random.choice(distribution)

        return hyperparameters


    def _compute_random_search_importance(self) -> Dict[str, float]:
        """Compute parameter importance for random search"""
        if len(self.evaluation_history) < 2:
            return {}

        # Extract hyperparameters and scores
        X = []
        y = []
        for eval_data in self.evaluation_history:
            X.append(list(eval_data['hyperparameters'].values()))
            y.append(eval_data['score'])

        X = np.array(X)
        y = np.array(y)

        # Compute correlation-based importance
        importances = {}
        param_names = list(self.evaluation_history[0]['hyperparameters'].keys())

        for i, param_name in enumerate(param_names):
            if X.shape[0] > 1:
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                importances[param_name] = corr if not np.isnan(corr) else 0.0
            else:
                importances[param_name] = 0.0

        # Normalize
        total_importance = sum(importances.values())
        if total_importance > 0:
            importances = {k: v / total_importance for k, v in importances.items()}

        return importances

    def _validate_best_hyperparameters(
        self,
        hyperparameters: Dict[str, Any],
        data: StudyData
    ) -> ValidationResults:
        """Perform comprehensive validation of best hyperparameters"""
        # Create configurations
        circuit_config, optimization_config = self._create_configs_from_hyperparameters(hyperparameters)

        # Create validation configuration
        validation_config = ValidationConfig(
            n_folds=self.config.cv_folds,
            bootstrap_samples=100,
            confidence_level=0.95
        )

        # Create trainer factory
        def trainer_factory():
            return VQCTrainer(circuit_config, optimization_config)

        # Perform validation
        validation_pipeline = ValidationPipeline(validation_config)
        models = {'best_model': trainer_factory}
        validation_results = validation_pipeline.run_comprehensive_validation(models, data)

        return validation_results['cross_validation_results']['best_model']


class GridSearchOptimizer(BaseHyperparameterOptimizer):
    """Grid search hyperparameter optimization"""

    def optimize(
        self,
        objective_function: Callable,
        data: StudyData
    ) -> OptimizationResult:
        """Perform grid search optimization"""
        self.logger.info("Starting grid search hyperparameter optimization")
        self.start_time = time.time()

        # Generate grid of all hyperparameter combinations
        grid_combinations = self._generate_grid_combinations()
        total_combinations = len(grid_combinations)

        self.logger.info(f"Testing {total_combinations} hyperparameter combinations")

        for i, hyperparameters in enumerate(grid_combinations):
            # Limit to n_calls if specified
            if i >= self.config.n_calls:
                self.logger.info(f"Reached max evaluations limit ({self.config.n_calls})")
                break

            # Evaluate
            score = self._evaluate_hyperparameters(hyperparameters, objective_function, data)

            if self.config.verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                progress = (i + 1) / min(total_combinations, self.config.n_calls) * 100
                self.logger.info(f"Progress: {progress:.1f}% ({i + 1}/{min(total_combinations, self.config.n_calls)}), Best score: {self.best_score:.6f}")

        # Create result
        best_validation_results = self._validate_best_hyperparameters(self.best_hyperparameters, data)

        optimization_result = OptimizationResult(
            best_hyperparameters=self.best_hyperparameters,
            best_score=self.best_score,
            best_validation_results=best_validation_results,
            hyperparameter_history=[eval_data['hyperparameters'] for eval_data in self.evaluation_history],
            score_history=[eval_data['score'] for eval_data in self.evaluation_history],
            evaluation_times=[eval_data['evaluation_time'] for eval_data in self.evaluation_history],
            parameter_importance=self._compute_grid_search_importance(),
            correlation_matrix=np.corrcoef([list(h.values()) for h in [eval_data['hyperparameters'] for eval_data in self.evaluation_history]], rowvar=False),
            feature_names=list(self.best_hyperparameters.keys()),
            total_optimization_time=time.time() - self.start_time,
            n_evaluations=len(self.evaluation_history),
            convergence_iteration=self._find_convergence_iteration(),
            optimization_config=self.config,
            search_space=self.search_space,
            objectives=self.objectives
        )

        self.logger.info(f"Grid search completed in {optimization_result.total_optimization_time:.1f}s")
        return optimization_result

    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters for grid search"""
        from itertools import product

        # Define grid values for each parameter
        param_grids = {}

        # Continuous parameters - create discrete grid
        n_qubits_vals = list(range(self.search_space.n_qubits[0], self.search_space.n_qubits[1] + 1))
        n_layers_vals = list(range(self.search_space.n_layers[0], min(self.search_space.n_layers[1] + 1, 6)))  # Limit layers for speed

        # Learning rate - logarithmic grid
        lr_min, lr_max = self.search_space.learning_rate
        lr_vals = [lr_min * (lr_max/lr_min)**(i/4) for i in range(5)]  # 5 values

        # Discrete parameters
        param_grids = {
            'n_qubits': n_qubits_vals,
            'n_layers': n_layers_vals,
            'ansatz': self.search_space.ansatz_options,
            'encoding': self.search_space.encoding_options,
            'learning_rate': lr_vals,
            'optimizer_type': self.search_space.optimizer_options,
            'batch_size': [16, 32, 64] if hasattr(self.search_space, 'batch_size') else [32],
            'max_iterations': [25, 50, 100] if hasattr(self.search_space, 'max_iterations') else [50]
        }

        # Generate all combinations
        param_names = list(param_grids.keys())
        combinations = []

        for combination in product(*param_grids.values()):
            hyperparams = dict(zip(param_names, combination))
            combinations.append(hyperparams)

        return combinations

    def _compute_grid_search_importance(self) -> Dict[str, float]:
        """Compute parameter importance for grid search"""
        if len(self.evaluation_history) < 2:
            return {}

        # Group evaluations by parameter values to compute importance
        param_importance = {}
        param_names = list(self.best_hyperparameters.keys())

        for param_name in param_names:
            # Get all unique values for this parameter
            param_values = {}
            for eval_data in self.evaluation_history:
                value = eval_data['hyperparameters'].get(param_name)
                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(eval_data['score'])

            # Calculate variance between groups vs within groups
            if len(param_values) > 1:
                all_scores = [eval_data['score'] for eval_data in self.evaluation_history]
                total_variance = np.var(all_scores)

                # Between group variance
                group_means = [np.mean(scores) for scores in param_values.values()]
                between_variance = np.var(group_means)

                # Parameter importance as ratio
                if total_variance > 0:
                    param_importance[param_name] = between_variance / total_variance
                else:
                    param_importance[param_name] = 0.0
            else:
                param_importance[param_name] = 0.0

        # Normalize
        total_importance = sum(param_importance.values())
        if total_importance > 0:
            param_importance = {k: v / total_importance for k, v in param_importance.items()}

        return param_importance


class MultiObjectiveOptimizer(BaseHyperparameterOptimizer):
    """Multi-objective hyperparameter optimization using NSGA-II"""

    def __init__(
        self,
        search_space: HyperparameterSpace,
        objectives: OptimizationObjectives,
        config: HyperparameterOptimizationConfig
    ):
        super().__init__(search_space, objectives, config)

        if not PYMOO_AVAILABLE:
            self.logger.warning("pymoo not available. Using simplified multi-objective optimization.")

    def optimize(
        self,
        objective_function: Callable,
        data: StudyData
    ) -> OptimizationResult:
        """Perform multi-objective optimization"""
        self.logger.info("Starting multi-objective hyperparameter optimization")
        self.start_time = time.time()

        if PYMOO_AVAILABLE:
            return self._optimize_with_pymoo(objective_function, data)
        else:
            return self._optimize_with_weighted_sum(objective_function, data)

    def _optimize_with_pymoo(
        self,
        objective_function: Callable,
        data: StudyData
    ) -> OptimizationResult:
        """Optimize using pymoo NSGA-II"""

        # Define problem class
        class VQCOptimizationProblem(Problem):
            def __init__(self, optimizer_instance, objective_function, data):
                self.optimizer_instance = optimizer_instance
                self.objective_function = objective_function
                self.data = data

                # Get search space bounds
                space = optimizer_instance.search_space
                n_vars = len(space.to_random_space())

                super().__init__(
                    n_var=n_vars,
                    n_obj=len(optimizer_instance.objectives.secondary_metrics) + 1,  # +1 for primary
                    xl=np.zeros(n_vars),  # Lower bounds (will be normalized)
                    xu=np.ones(n_vars)    # Upper bounds (will be normalized)
                )

            def _evaluate(self, X, out, *args, **kwargs):
                objectives = []

                for x in X:
                    # Convert normalized parameters back to actual hyperparameters
                    hyperparameters = self._denormalize_parameters(x)

                    # Evaluate multiple objectives
                    obj_values = self._evaluate_multiple_objectives(hyperparameters)
                    objectives.append(obj_values)

                out["F"] = np.array(objectives)

            def _denormalize_parameters(self, x_normalized):
                # Convert normalized [0,1] parameters back to actual values
                # This is a simplified implementation
                space = self.optimizer_instance.search_space
                hyperparameters = {}

                # For demonstration, using the first few parameters
                hyperparameters['n_qubits'] = int(space.n_qubits[0] + x_normalized[0] * (space.n_qubits[1] - space.n_qubits[0]))
                hyperparameters['n_layers'] = int(space.n_layers[0] + x_normalized[1] * (space.n_layers[1] - space.n_layers[0]))
                hyperparameters['learning_rate'] = space.learning_rate[0] * (space.learning_rate[1] / space.learning_rate[0]) ** x_normalized[2]

                # Add default values for other parameters
                hyperparameters.update({
                    'ansatz': np.random.choice(space.ansatz_options),
                    'encoding': np.random.choice(space.encoding_options),
                    'optimizer_type': np.random.choice(space.optimizer_options),
                    'batch_size': 32,
                    'max_iterations': 100,
                    'regularization_weight': 0.01,
                    'lr_scheduler': 'constant',
                    'lr_decay_rate': 0.95,
                    'momentum': 0.9,
                    'gradient_clipping': 1.0,
                    'enable_mini_batches': True,
                    'track_gradient_variance': True
                })

                return hyperparameters

            def _evaluate_multiple_objectives(self, hyperparameters):
                # Create trainer
                circuit_config, optimization_config = self.optimizer_instance._create_configs_from_hyperparameters(hyperparameters)
                trainer = VQCTrainer(circuit_config, optimization_config)

                # Measure training time
                start_time = time.time()
                primary_score = self.objective_function(trainer, self.data)
                training_time = time.time() - start_time

                # Calculate quantum resources (simplified)
                quantum_resources = hyperparameters['n_qubits'] * hyperparameters['n_layers']

                # Return objectives to minimize
                objectives = [primary_score, training_time, quantum_resources]
                return objectives

        # Create and solve problem
        problem = VQCOptimizationProblem(self, objective_function, data)
        algorithm = NSGA2(
            pop_size=self.objectives.population_size,
            eliminate_duplicates=True
        )

        result = moo_minimize(
            problem,
            algorithm,
            termination=('n_gen', self.objectives.n_generations),
            verbose=self.config.verbose,
            seed=self.config.random_state
        )

        # Extract Pareto front
        pareto_front = []
        pareto_scores = []

        for i in range(min(len(result.X), self.objectives.pareto_front_size)):
            hyperparameters = problem._denormalize_parameters(result.X[i])
            pareto_front.append(hyperparameters)
            pareto_scores.append(result.F[i].tolist())

        # Select best solution (can use different criteria)
        best_idx = np.argmin(result.F[:, 0])  # Best primary objective
        best_hyperparameters = problem._denormalize_parameters(result.X[best_idx])
        best_score = result.F[best_idx][0]

        # Validate best solution
        best_validation_results = self._validate_best_hyperparameters(best_hyperparameters, data)

        optimization_result = OptimizationResult(
            best_hyperparameters=best_hyperparameters,
            best_score=best_score,
            best_validation_results=best_validation_results,
            hyperparameter_history=pareto_front,
            score_history=[scores[0] for scores in pareto_scores],
            evaluation_times=[1.0] * len(pareto_front),  # Placeholder
            pareto_front=pareto_front,
            pareto_scores=pareto_scores,
            parameter_importance={},  # TODO: Implement for MOO
            correlation_matrix=np.eye(len(best_hyperparameters)),  # Placeholder
            feature_names=list(best_hyperparameters.keys()),
            total_optimization_time=time.time() - self.start_time,
            n_evaluations=len(pareto_front),
            convergence_iteration=None,
            optimization_config=self.config,
            search_space=self.search_space,
            objectives=self.objectives
        )

        self.logger.info(f"Multi-objective optimization completed in {optimization_result.total_optimization_time:.1f}s")
        return optimization_result

    def _optimize_with_weighted_sum(
        self,
        objective_function: Callable,
        data: StudyData
    ) -> OptimizationResult:
        """Simplified multi-objective optimization using weighted sum"""
        self.logger.info("Using weighted sum approach for multi-objective optimization")

        # Create weighted objective function
        def weighted_objective(trainer, data):
            # Primary objective
            primary_score = objective_function(trainer, data)

            # Additional objectives (simplified)
            training_time = 1.0  # Placeholder
            quantum_resources = trainer.circuit_config.n_qubits * trainer.circuit_config.n_layers

            # Combine with weights
            total_score = (
                self.objectives.metric_weights.get(self.objectives.primary_metric, 1.0) * primary_score +
                self.objectives.metric_weights.get("training_time", 0.1) * training_time +
                self.objectives.metric_weights.get("quantum_resources", 0.05) * quantum_resources
            )

            return total_score

        # Use single-objective optimizer
        random_optimizer = RandomSearchOptimizer(self.search_space, self.objectives, self.config)
        return random_optimizer.optimize(weighted_objective, data)


class SensitivityAnalyzer:
    """Analyze hyperparameter sensitivity and importance"""

    def __init__(self, optimization_result: OptimizationResult):
        self.result = optimization_result
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_sensitivity(self) -> Dict[str, Any]:
        """Perform comprehensive sensitivity analysis"""
        self.logger.info("Starting sensitivity analysis")

        analysis_results = {
            'parameter_importance': self.result.parameter_importance,
            'correlation_analysis': self._analyze_correlations(),
            'sensitivity_indices': self._calculate_sensitivity_indices(),
            'interaction_effects': self._analyze_interaction_effects(),
            'robustness_analysis': self._analyze_robustness()
        }

        return analysis_results

    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between parameters and performance"""
        if len(self.result.hyperparameter_history) < 2:
            return {}

        # Create parameter matrix
        param_names = list(self.result.hyperparameter_history[0].keys())
        param_matrix = []

        for hyperparams in self.result.hyperparameter_history:
            row = []
            for param in param_names:
                value = hyperparams[param]
                # Convert categorical to numeric
                if isinstance(value, str):
                    value = hash(value) % 1000  # Simple hash for categorical
                elif isinstance(value, bool):
                    value = int(value)
                row.append(float(value))
            param_matrix.append(row)

        param_matrix = np.array(param_matrix)
        scores = np.array(self.result.score_history)

        # Calculate correlations
        correlations = {}
        for i, param in enumerate(param_names):
            if param_matrix.shape[0] > 1:
                corr = np.corrcoef(param_matrix[:, i], scores)[0, 1]
                correlations[param] = corr if not np.isnan(corr) else 0.0
            else:
                correlations[param] = 0.0

        return {
            'parameter_score_correlations': correlations,
            'correlation_matrix': self.result.correlation_matrix,
            'parameter_names': param_names
        }

    def _calculate_sensitivity_indices(self) -> Dict[str, float]:
        """Calculate Sobol sensitivity indices"""
        # Simplified sensitivity calculation
        # In practice, would use library like SALib

        if len(self.result.hyperparameter_history) < 10:
            return {}

        param_names = list(self.result.hyperparameter_history[0].keys())
        sensitivity_indices = {}

        # Calculate variance-based sensitivity (simplified)
        for param in param_names:
            values = []
            scores = []

            for i, hyperparams in enumerate(self.result.hyperparameter_history):
                value = hyperparams[param]
                # Convert to numeric
                if isinstance(value, str):
                    value = hash(value) % 1000
                elif isinstance(value, bool):
                    value = int(value)
                values.append(float(value))
                scores.append(self.result.score_history[i])

            if len(set(values)) > 1:  # Parameter varies
                # Calculate conditional variance
                unique_values = list(set(values))
                conditional_vars = []

                for unique_val in unique_values:
                    indices = [i for i, v in enumerate(values) if v == unique_val]
                    if len(indices) > 1:
                        conditional_scores = [scores[i] for i in indices]
                        conditional_vars.append(np.var(conditional_scores))

                if conditional_vars:
                    # First-order sensitivity index
                    total_var = np.var(scores)
                    if total_var > 0:
                        sensitivity = 1 - np.mean(conditional_vars) / total_var
                        sensitivity_indices[param] = max(0, sensitivity)
                    else:
                        sensitivity_indices[param] = 0.0
                else:
                    sensitivity_indices[param] = 0.0
            else:
                sensitivity_indices[param] = 0.0

        return sensitivity_indices

    def _analyze_interaction_effects(self) -> Dict[str, float]:
        """Analyze parameter interaction effects"""
        # Simplified interaction analysis
        if len(self.result.hyperparameter_history) < 20:
            return {}

        param_names = list(self.result.hyperparameter_history[0].keys())
        interactions = {}

        # Look at pairwise interactions
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names[i+1:], i+1):
                interaction_key = f"{param1}_x_{param2}"

                # Calculate interaction effect (simplified)
                # This would be more sophisticated in practice
                correlation = self.result.parameter_importance.get(param1, 0) * \
                             self.result.parameter_importance.get(param2, 0)

                interactions[interaction_key] = correlation

        return interactions

    def _analyze_robustness(self) -> Dict[str, Any]:
        """Analyze robustness of best hyperparameters"""
        best_score = self.result.best_score

        # Find scores within certain percentage of best
        tolerance = 0.05  # 5% tolerance
        threshold = best_score * (1 + tolerance)

        robust_configs = []
        for i, score in enumerate(self.result.score_history):
            if score <= threshold:
                robust_configs.append(self.result.hyperparameter_history[i])

        # Analyze variation in robust configurations
        robustness_metrics = {
            'n_robust_configs': len(robust_configs),
            'robustness_ratio': len(robust_configs) / len(self.result.score_history),
            'parameter_stability': {}
        }

        if robust_configs:
            # Calculate parameter stability
            param_names = list(robust_configs[0].keys())
            for param in param_names:
                values = [config[param] for config in robust_configs]
                # Calculate coefficient of variation for numeric parameters
                if all(isinstance(v, (int, float)) for v in values):
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / (abs(mean_val) + 1e-8)
                    robustness_metrics['parameter_stability'][param] = 1 / (1 + cv)  # Higher = more stable
                else:
                    # For categorical parameters, calculate mode frequency
                    unique_vals, counts = np.unique(values, return_counts=True)
                    max_freq = np.max(counts) / len(values)
                    robustness_metrics['parameter_stability'][param] = max_freq

        return robustness_metrics

    def generate_sensitivity_report(self, output_dir: str = "sensitivity_analysis"):
        """Generate comprehensive sensitivity analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Perform analysis
        analysis_results = self.analyze_sensitivity()

        # Save results
        with open(output_path / 'sensitivity_analysis.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(analysis_results)
            json.dump(serializable_results, f, indent=2)

        # Generate plots
        self._plot_parameter_importance(analysis_results, output_path)
        self._plot_correlation_heatmap(analysis_results, output_path)
        self._plot_sensitivity_indices(analysis_results, output_path)

        self.logger.info(f"Sensitivity analysis report generated in: {output_path}")

    def _make_json_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        else:
            return obj

    def _plot_parameter_importance(self, analysis_results, output_path):
        """Plot parameter importance"""
        importance = self.result.parameter_importance
        if not importance:
            return

        plt.figure(figsize=(12, 8))
        params = list(importance.keys())
        values = list(importance.values())

        bars = plt.barh(params, values, color='skyblue', edgecolor='navy')
        plt.xlabel('Importance')
        plt.title('Hyperparameter Importance Ranking', fontsize=14, fontweight='bold')

        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center')

        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'parameter_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_correlation_heatmap(self, analysis_results, output_path):
        """Plot correlation heatmap"""
        correlation_data = analysis_results.get('correlation_analysis', {})
        correlations = correlation_data.get('parameter_score_correlations', {})

        if not correlations:
            return

        # Create correlation matrix for visualization
        params = list(correlations.keys())
        corr_values = list(correlations.values())

        plt.figure(figsize=(10, 8))
        plt.bar(range(len(params)), corr_values, color='lightcoral', edgecolor='darkred')
        plt.xticks(range(len(params)), params, rotation=45, ha='right')
        plt.ylabel('Correlation with Performance')
        plt.title('Parameter-Performance Correlations', fontsize=14, fontweight='bold')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_sensitivity_indices(self, analysis_results, output_path):
        """Plot sensitivity indices"""
        sensitivity = analysis_results.get('sensitivity_indices', {})
        if not sensitivity:
            return

        plt.figure(figsize=(12, 8))
        params = list(sensitivity.keys())
        values = list(sensitivity.values())

        bars = plt.barh(params, values, color='lightgreen', edgecolor='darkgreen')
        plt.xlabel('Sensitivity Index')
        plt.title('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')

        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center')

        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'sensitivity_indices.png', dpi=300, bbox_inches='tight')
        plt.close()


# Utility functions and convenience classes

class HyperparameterOptimizerFactory:
    """Factory for creating hyperparameter optimizers"""

    @staticmethod
    def create_optimizer(
        method: str,
        search_space: HyperparameterSpace,
        objectives: OptimizationObjectives,
        config: HyperparameterOptimizationConfig
    ) -> BaseHyperparameterOptimizer:
        """Create hyperparameter optimizer of specified type"""

        logger = logging.getLogger(__name__)

        # Smart fallback for Bayesian optimization
        if method == "bayesian":
            if not SKOPT_AVAILABLE:
                logger.warning("scikit-optimize not available for Bayesian optimization, falling back to random search")
                method = "random"
            else:
                try:
                    return BayesianOptimizer(search_space, objectives, config)
                except ImportError as e:
                    logger.warning(f"Bayesian optimization failed ({e}), falling back to random search")
                    method = "random"

        # Multi-objective fallback
        if method == "multi_objective":
            if not PYMOO_AVAILABLE:
                logger.warning("pymoo not available for multi-objective optimization, falling back to random search")
                method = "random"
            else:
                try:
                    return MultiObjectiveOptimizer(search_space, objectives, config)
                except ImportError as e:
                    logger.warning(f"Multi-objective optimization failed ({e}), falling back to random search")
                    method = "random"

        # Always available methods
        if method == "random":
            return RandomSearchOptimizer(search_space, objectives, config)
        elif method == "grid":
            return GridSearchOptimizer(search_space, objectives, config)
        else:
            logger.warning(f"Unknown optimization method: {method}, falling back to random search")
            return RandomSearchOptimizer(search_space, objectives, config)


def create_default_search_space() -> HyperparameterSpace:
    """Create a default hyperparameter search space"""
    return HyperparameterSpace(
        n_qubits=(2, 6),
        n_layers=(1, 8),
        ansatz_options=["ry_cnot", "strongly_entangling"],
        encoding_options=["angle", "amplitude"],
        learning_rate=(1e-4, 1e-1),
        optimizer_options=["adam", "adagrad"],
        batch_size=(16, 64),
        max_iterations=(50, 200)
    )


def create_performance_objectives() -> OptimizationObjectives:
    """Create default performance-focused objectives"""
    return OptimizationObjectives(
        primary_metric="mse",
        secondary_metrics=["mae", "training_time"],
        metric_weights={
            "mse": 1.0,
            "mae": 0.3,
            "training_time": 0.1,
            "quantum_resources": 0.05
        }
    )


def optimize_vqc_hyperparameters(
    data: StudyData,
    method: str = "bayesian",
    n_calls: int = 50,
    output_dir: str = "hyperparameter_results"
) -> OptimizationResult:
    """
    Convenience function for VQC hyperparameter optimization

    Args:
        data: Training data
        method: Optimization method ("bayesian", "random", "multi_objective")
        n_calls: Number of optimization calls
        output_dir: Output directory for results

    Returns:
        Optimization results
    """
    # Create default configurations
    search_space = create_default_search_space()
    objectives = create_performance_objectives()
    config = HyperparameterOptimizationConfig(
        optimization_method=method,
        n_calls=n_calls,
        verbose=True
    )

    # Define objective function
    def objective_function(trainer: VQCTrainer, data: StudyData) -> float:
        try:
            # Train and evaluate
            result = trainer.train(data)

            # Return primary metric (MSE)
            # In practice, this would be more sophisticated validation
            return result.final_cost

        except Exception as e:
            logging.warning(f"Training failed: {e}")
            return float('inf')

    # Create and run optimizer
    optimizer = HyperparameterOptimizerFactory.create_optimizer(
        method, search_space, objectives, config
    )

    optimization_result = optimizer.optimize(objective_function, data)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'optimization_results.json', 'w') as f:
        # Convert to serializable format
        serializable_result = {
            'best_hyperparameters': optimization_result.best_hyperparameters,
            'best_score': optimization_result.best_score,
            'parameter_importance': optimization_result.parameter_importance,
            'total_time': optimization_result.total_optimization_time,
            'n_evaluations': optimization_result.n_evaluations
        }
        json.dump(serializable_result, f, indent=2)

    # Generate sensitivity analysis
    sensitivity_analyzer = SensitivityAnalyzer(optimization_result)
    sensitivity_analyzer.generate_sensitivity_report(str(output_path / "sensitivity"))

    return optimization_result


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("Hyperparameter optimization framework created successfully!")
    print(f"Scikit-optimize available: {SKOPT_AVAILABLE}")
    print(f"Pymoo available: {PYMOO_AVAILABLE}")

    # Create example configurations
    search_space = create_default_search_space()
    objectives = create_performance_objectives()
    config = HyperparameterOptimizationConfig(n_calls=10, verbose=True)

    print(f"Search space: {search_space.n_qubits[0]}-{search_space.n_qubits[1]} qubits, {search_space.n_layers[0]}-{search_space.n_layers[1]} layers")
    print(f"Optimization: {config.optimization_method} with {config.n_calls} calls")
    print(f"Primary objective: {objectives.primary_metric}")