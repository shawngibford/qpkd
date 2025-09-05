"""
Dosing optimization algorithms for PK/PD modeling.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from typing import Dict, List, Tuple, Callable, Optional, Any
import logging
from dataclasses import dataclass

from ..quantum.core.data_structures import DosingResults


@dataclass
class OptimizationConfig:
    """Configuration for dosing optimization."""
    target_threshold: float = 3.3  # ng/mL biomarker threshold
    population_coverage: float = 0.9  # Fraction of population to satisfy
    dose_bounds: Tuple[float, float] = (0.1, 100.0)  # mg dosing bounds
    safety_margin: float = 0.1  # Safety factor
    optimization_method: str = 'differential_evolution'  # 'scipy', 'differential', 'basinhopping'
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6


class DosingOptimizer:
    """Optimization algorithms for finding optimal dosing regimens."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize dosing optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
    def optimize_daily_dose(self, 
                           prediction_function: Callable[[float], np.ndarray],
                           **kwargs) -> DosingResults:
        """Optimize daily dosing regimen.
        
        Args:
            prediction_function: Function that takes dose and returns biomarker predictions
            **kwargs: Additional optimization parameters
            
        Returns:
            DosingResults with optimal daily dose
        """
        def objective(dose_array):
            """Objective function to minimize."""
            dose = dose_array[0]
            try:
                biomarker_predictions = prediction_function(dose)
                
                # Calculate fraction of population below threshold
                below_threshold = np.sum(biomarker_predictions < self.config.target_threshold)
                coverage = below_threshold / len(biomarker_predictions)
                
                # Penalty for not meeting coverage requirement
                coverage_penalty = max(0, self.config.population_coverage - coverage) ** 2
                
                # Minimize dose while achieving coverage (with safety margin)
                dose_penalty = dose * 0.01  # Slight preference for lower doses
                
                return coverage_penalty * 1000 + dose_penalty
                
            except Exception as e:
                self.logger.warning(f"Objective function error at dose {dose}: {e}")
                return 1e6  # Large penalty for failed evaluations
                
        # Run optimization
        result = self._run_optimization(objective, self.config.dose_bounds)
        
        optimal_dose = result.x[0]
        
        # Validate result
        final_predictions = prediction_function(optimal_dose)
        coverage_achieved = np.mean(final_predictions < self.config.target_threshold)
        
        return DosingResults(
            daily_dose=optimal_dose,
            weekly_dose=optimal_dose * 7,
            predicted_biomarkers=final_predictions,
            coverage_achieved=coverage_achieved,
            optimization_success=result.success,
            optimization_details={
                'method': self.config.optimization_method,
                'iterations': result.nfev if hasattr(result, 'nfev') else None,
                'objective_value': result.fun,
                'message': result.message if hasattr(result, 'message') else str(result)
            }
        )
        
    def optimize_weekly_dose(self, 
                            prediction_function: Callable[[float], np.ndarray],
                            daily_equivalent: float = None,
                            **kwargs) -> DosingResults:
        """Optimize weekly dosing regimen.
        
        Args:
            prediction_function: Function that takes weekly dose and returns biomarker predictions
            daily_equivalent: If provided, start optimization near this daily equivalent
            **kwargs: Additional optimization parameters
            
        Returns:
            DosingResults with optimal weekly dose
        """
        # Adjust bounds for weekly dosing
        weekly_bounds = (self.config.dose_bounds[0] * 7, self.config.dose_bounds[1] * 7)
        
        def objective(dose_array):
            """Objective function for weekly dosing."""
            weekly_dose = dose_array[0]
            try:
                biomarker_predictions = prediction_function(weekly_dose)
                
                # Calculate coverage
                coverage = np.mean(biomarker_predictions < self.config.target_threshold)
                coverage_penalty = max(0, self.config.population_coverage - coverage) ** 2
                
                # Penalty for high doses
                dose_penalty = weekly_dose * 0.001
                
                return coverage_penalty * 1000 + dose_penalty
                
            except Exception as e:
                self.logger.warning(f"Weekly objective error at dose {weekly_dose}: {e}")
                return 1e6
                
        # Set initial guess near daily equivalent if provided
        initial_guess = daily_equivalent * 7 if daily_equivalent else None
        
        result = self._run_optimization(objective, weekly_bounds, initial_guess)
        
        optimal_weekly_dose = result.x[0]
        
        # Validate result
        final_predictions = prediction_function(optimal_weekly_dose)
        coverage_achieved = np.mean(final_predictions < self.config.target_threshold)
        
        return DosingResults(
            daily_dose=optimal_weekly_dose / 7,
            weekly_dose=optimal_weekly_dose,
            predicted_biomarkers=final_predictions,
            coverage_achieved=coverage_achieved,
            optimization_success=result.success,
            optimization_details={
                'method': self.config.optimization_method,
                'iterations': result.nfev if hasattr(result, 'nfev') else None,
                'objective_value': result.fun,
                'weekly_optimized': True
            }
        )
        
    def optimize_population_specific(self,
                                   prediction_functions: Dict[str, Callable[[float], np.ndarray]],
                                   population_weights: Dict[str, float] = None,
                                   **kwargs) -> Dict[str, DosingResults]:
        """Optimize doses for multiple population subgroups.
        
        Args:
            prediction_functions: Dict mapping population names to prediction functions
            population_weights: Relative importance of each population group
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary of DosingResults for each population
        """
        if population_weights is None:
            population_weights = {pop: 1.0 for pop in prediction_functions.keys()}
            
        results = {}
        
        for population_name, pred_func in prediction_functions.items():
            self.logger.info(f"Optimizing doses for population: {population_name}")
            
            # Adjust config for population-specific requirements
            pop_config = OptimizationConfig(
                target_threshold=self.config.target_threshold,
                population_coverage=self.config.population_coverage,
                dose_bounds=self.config.dose_bounds
            )
            
            pop_optimizer = DosingOptimizer(pop_config)
            
            # Optimize daily dose for this population
            daily_result = pop_optimizer.optimize_daily_dose(pred_func)
            
            # Optimize weekly dose using daily result as starting point
            weekly_result = pop_optimizer.optimize_weekly_dose(
                pred_func, 
                daily_equivalent=daily_result.daily_dose
            )
            
            # Combine results
            combined_result = DosingResults(
                daily_dose=daily_result.daily_dose,
                weekly_dose=weekly_result.weekly_dose,
                predicted_biomarkers=daily_result.predicted_biomarkers,
                coverage_achieved=daily_result.coverage_achieved,
                optimization_success=daily_result.optimization_success and weekly_result.optimization_success,
                optimization_details={
                    'population': population_name,
                    'weight': population_weights[population_name],
                    'daily_optimization': daily_result.optimization_details,
                    'weekly_optimization': weekly_result.optimization_details
                }
            )
            
            results[population_name] = combined_result
            
        return results
        
    def robust_optimization(self,
                           prediction_function: Callable[[float], np.ndarray],
                           uncertainty_samples: int = 100,
                           confidence_level: float = 0.95,
                           **kwargs) -> DosingResults:
        """Perform robust optimization accounting for uncertainty.
        
        Args:
            prediction_function: Function that returns predictions with uncertainty
            uncertainty_samples: Number of samples for uncertainty quantification
            confidence_level: Confidence level for robust optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            Robust DosingResults
        """
        def robust_objective(dose_array):
            """Robust objective function accounting for uncertainty."""
            dose = dose_array[0]
            
            # Sample predictions multiple times to account for uncertainty
            coverages = []
            
            for _ in range(uncertainty_samples):
                try:
                    predictions = prediction_function(dose)
                    coverage = np.mean(predictions < self.config.target_threshold)
                    coverages.append(coverage)
                except:
                    coverages.append(0.0)  # Failed prediction
                    
            coverages = np.array(coverages)
            
            # Use lower confidence bound for robust optimization
            alpha = 1 - confidence_level
            robust_coverage = np.percentile(coverages, alpha * 100)
            
            # Penalty for not meeting robust coverage requirement
            coverage_penalty = max(0, self.config.population_coverage - robust_coverage) ** 2
            dose_penalty = dose * 0.01
            
            return coverage_penalty * 1000 + dose_penalty
            
        result = self._run_optimization(robust_objective, self.config.dose_bounds)
        
        optimal_dose = result.x[0]
        
        # Final validation with uncertainty quantification
        final_coverages = []
        final_predictions_list = []
        
        for _ in range(uncertainty_samples):
            predictions = prediction_function(optimal_dose)
            coverage = np.mean(predictions < self.config.target_threshold)
            final_coverages.append(coverage)
            final_predictions_list.append(predictions)
            
        mean_coverage = np.mean(final_coverages)
        robust_coverage = np.percentile(final_coverages, (1 - confidence_level) * 100)
        
        return DosingResults(
            daily_dose=optimal_dose,
            weekly_dose=optimal_dose * 7,
            predicted_biomarkers=np.mean(final_predictions_list, axis=0),
            coverage_achieved=mean_coverage,
            optimization_success=result.success,
            optimization_details={
                'method': 'robust_' + self.config.optimization_method,
                'confidence_level': confidence_level,
                'robust_coverage': robust_coverage,
                'coverage_std': np.std(final_coverages),
                'uncertainty_samples': uncertainty_samples
            }
        )
        
    def _run_optimization(self, 
                         objective_func: Callable, 
                         bounds: Tuple[float, float],
                         initial_guess: float = None) -> Any:
        """Run optimization using specified method.
        
        Args:
            objective_func: Function to minimize
            bounds: (min, max) bounds for optimization
            initial_guess: Initial guess for optimization
            
        Returns:
            Optimization result object
        """
        bounds_tuple = [bounds]
        
        if initial_guess is None:
            initial_guess = (bounds[0] + bounds[1]) / 2
            
        if self.config.optimization_method == 'scipy':
            result = minimize(
                objective_func,
                [initial_guess],
                method='L-BFGS-B',
                bounds=bounds_tuple,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.convergence_tolerance}
            )
            
        elif self.config.optimization_method == 'differential_evolution':
            result = differential_evolution(
                objective_func,
                bounds_tuple,
                maxiter=self.config.max_iterations,
                tol=self.config.convergence_tolerance,
                seed=42
            )
            
        elif self.config.optimization_method == 'basinhopping':
            result = basinhopping(
                objective_func,
                [initial_guess],
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_tuple},
                niter=min(100, self.config.max_iterations // 10)
            )
            
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
            
        return result
        
    def multi_objective_optimization(self,
                                   prediction_function: Callable[[float], np.ndarray],
                                   objectives: List[str] = None,
                                   weights: List[float] = None,
                                   **kwargs) -> DosingResults:
        """Multi-objective optimization for dosing.
        
        Args:
            prediction_function: Function that returns biomarker predictions
            objectives: List of objectives ('coverage', 'dose_minimization', 'safety')
            weights: Weights for each objective
            **kwargs: Additional optimization parameters
            
        Returns:
            Multi-objective optimized DosingResults
        """
        if objectives is None:
            objectives = ['coverage', 'dose_minimization']
            
        if weights is None:
            weights = [1.0] * len(objectives)
            
        def multi_objective(dose_array):
            """Multi-objective function."""
            dose = dose_array[0]
            
            try:
                predictions = prediction_function(dose)
                coverage = np.mean(predictions < self.config.target_threshold)
                
                objective_values = []
                
                for obj in objectives:
                    if obj == 'coverage':
                        # Maximize coverage (minimize negative coverage)
                        obj_val = max(0, self.config.population_coverage - coverage) ** 2
                        
                    elif obj == 'dose_minimization':
                        # Minimize dose
                        obj_val = dose / self.config.dose_bounds[1]  # Normalized
                        
                    elif obj == 'safety':
                        # Penalize doses that might cause adverse effects
                        safety_threshold = self.config.dose_bounds[1] * 0.8
                        obj_val = max(0, dose - safety_threshold) / safety_threshold
                        
                    else:
                        obj_val = 0.0
                        
                    objective_values.append(obj_val)
                    
                # Weighted sum of objectives
                total_objective = sum(w * obj for w, obj in zip(weights, objective_values))
                return total_objective
                
            except Exception as e:
                self.logger.warning(f"Multi-objective error at dose {dose}: {e}")
                return 1e6
                
        result = self._run_optimization(multi_objective, self.config.dose_bounds)
        
        optimal_dose = result.x[0]
        final_predictions = prediction_function(optimal_dose)
        coverage_achieved = np.mean(final_predictions < self.config.target_threshold)
        
        return DosingResults(
            daily_dose=optimal_dose,
            weekly_dose=optimal_dose * 7,
            predicted_biomarkers=final_predictions,
            coverage_achieved=coverage_achieved,
            optimization_success=result.success,
            optimization_details={
                'method': 'multi_objective_' + self.config.optimization_method,
                'objectives': objectives,
                'weights': weights,
                'objective_value': result.fun
            }
        )