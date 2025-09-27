"""
Population-level optimization for PK/PD modeling across different demographics.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from quantum.core.base import PKPDData
from utils.logging_system import DosingResults


@dataclass
class PopulationSegment:
    """Definition of a population segment for optimization."""
    name: str
    weight_range: Tuple[float, float]
    age_range: Tuple[float, float] = (18, 100)
    sex_filter: Optional[int] = None  # 0=female, 1=male, None=both
    concomitant_allowed: bool = True
    prevalence: float = 1.0  # Relative prevalence in population


class PopulationOptimizer:
    """Optimization across different population segments and demographics."""
    
    def __init__(self, 
                 population_segments: List[PopulationSegment] = None,
                 optimization_objective: str = 'weighted_coverage',
                 safety_constraints: Dict[str, float] = None):
        """Initialize population optimizer.
        
        Args:
            population_segments: List of population segments to optimize for
            optimization_objective: 'weighted_coverage', 'min_max_fairness', 'total_coverage'
            safety_constraints: Safety constraints for optimization
        """
        self.population_segments = population_segments or self._default_segments()
        self.optimization_objective = optimization_objective
        self.safety_constraints = safety_constraints or {'max_dose': 100.0, 'safety_margin': 0.1}
        self.logger = logging.getLogger(__name__)
        
    def _default_segments(self) -> List[PopulationSegment]:
        """Create default population segments."""
        return [
            PopulationSegment(
                name="standard_weight",
                weight_range=(50, 100),
                prevalence=0.6
            ),
            PopulationSegment(
                name="extended_weight",
                weight_range=(70, 140),
                prevalence=0.4
            ),
            PopulationSegment(
                name="no_concomitant",
                weight_range=(50, 100),
                concomitant_allowed=False,
                prevalence=0.3
            ),
            PopulationSegment(
                name="elderly",
                weight_range=(50, 100),
                age_range=(65, 100),
                prevalence=0.25
            )
        ]
        
    def optimize_population_dosing(self,
                                  prediction_models: Dict[str, Callable],
                                  target_threshold: float = 3.3,
                                  population_coverage: float = 0.9) -> Dict[str, Any]:
        """Optimize dosing across all population segments.
        
        Args:
            prediction_models: Dict mapping segment names to prediction functions
            target_threshold: Biomarker threshold (ng/mL)
            population_coverage: Required population coverage
            
        Returns:
            Dictionary with optimization results for each segment
        """
        segment_results = {}
        
        for segment in self.population_segments:
            if segment.name not in prediction_models:
                self.logger.warning(f"No prediction model for segment {segment.name}")
                continue
                
            self.logger.info(f"Optimizing dosing for population segment: {segment.name}")
            
            # Optimize for this segment
            prediction_func = prediction_models[segment.name]
            
            segment_result = self._optimize_single_segment(
                segment, 
                prediction_func, 
                target_threshold, 
                population_coverage
            )
            
            segment_results[segment.name] = segment_result
            
        # Global optimization across segments
        global_result = self._global_population_optimization(
            segment_results, 
            prediction_models, 
            target_threshold, 
            population_coverage
        )
        
        return {
            'segment_results': segment_results,
            'global_optimization': global_result,
            'population_segments': [seg.__dict__ for seg in self.population_segments]
        }
        
    def _optimize_single_segment(self,
                                segment: PopulationSegment,
                                prediction_func: Callable,
                                target_threshold: float,
                                population_coverage: float) -> DosingResults:
        """Optimize dosing for a single population segment."""
        
        def objective(dose_array):
            """Objective function for single segment optimization."""
            dose = dose_array[0]
            
            try:
                # Get predictions for this dose
                predictions = prediction_func(dose)
                
                # Calculate coverage
                coverage = np.mean(predictions < target_threshold)
                
                # Penalty for not meeting coverage requirement
                coverage_penalty = max(0, population_coverage - coverage) ** 2
                
                # Penalty for high doses (safety)
                dose_penalty = (dose / self.safety_constraints['max_dose']) ** 2
                
                # Special penalty for safety-critical segments
                if 'elderly' in segment.name or 'pediatric' in segment.name:
                    dose_penalty *= 2.0
                    
                return coverage_penalty * 1000 + dose_penalty * 10
                
            except Exception as e:
                self.logger.warning(f"Objective evaluation failed for {segment.name}: {e}")
                return 1e6
                
        # Run optimization
        bounds = [(0.1, self.safety_constraints['max_dose'])]
        
        result = minimize(
            objective,
            [10.0],  # Initial guess
            method='L-BFGS-B',
            bounds=bounds
        )
        
        optimal_dose = result.x[0]
        
        # Validate result
        try:
            final_predictions = prediction_func(optimal_dose)
            coverage_achieved = np.mean(final_predictions < target_threshold)
        except:
            final_predictions = np.array([target_threshold + 1])  # Failed prediction
            coverage_achieved = 0.0
            
        return DosingResults(
            daily_dose=optimal_dose,
            weekly_dose=optimal_dose * 7,
            predicted_biomarkers=final_predictions,
            coverage_achieved=coverage_achieved,
            optimization_success=result.success,
            optimization_details={
                'segment': segment.name,
                'weight_range': segment.weight_range,
                'prevalence': segment.prevalence,
                'objective_value': result.fun,
                'optimization_method': 'single_segment_L-BFGS-B'
            }
        )
        
    def _global_population_optimization(self,
                                      segment_results: Dict[str, DosingResults],
                                      prediction_models: Dict[str, Callable],
                                      target_threshold: float,
                                      population_coverage: float) -> Dict[str, Any]:
        """Perform global optimization across all population segments."""
        
        def global_objective(dose_array):
            """Global objective function considering all segments."""
            dose = dose_array[0]
            
            total_weighted_penalty = 0.0
            total_weight = 0.0
            
            for segment in self.population_segments:
                if segment.name not in prediction_models:
                    continue
                    
                try:
                    predictions = prediction_models[segment.name](dose)
                    coverage = np.mean(predictions < target_threshold)
                    
                    # Penalty for this segment
                    coverage_penalty = max(0, population_coverage - coverage) ** 2
                    
                    # Weight by segment prevalence
                    weighted_penalty = coverage_penalty * segment.prevalence
                    total_weighted_penalty += weighted_penalty
                    total_weight += segment.prevalence
                    
                except Exception as e:
                    # Heavy penalty for failed predictions
                    total_weighted_penalty += 1000 * segment.prevalence
                    total_weight += segment.prevalence
                    
            # Normalize by total weight
            if total_weight > 0:
                average_penalty = total_weighted_penalty / total_weight
            else:
                average_penalty = 1e6
                
            # Global dose penalty
            dose_penalty = (dose / self.safety_constraints['max_dose']) ** 2
            
            # Fairness penalty - penalize large differences between segments
            fairness_penalty = self._calculate_fairness_penalty(dose, prediction_models, target_threshold)
            
            return average_penalty * 1000 + dose_penalty * 10 + fairness_penalty * 100
            
        # Run global optimization
        bounds = [(0.1, self.safety_constraints['max_dose'])]
        
        result = minimize(
            global_objective,
            [10.0],
            method='L-BFGS-B',
            bounds=bounds
        )
        
        optimal_global_dose = result.x[0]
        
        # Evaluate performance across all segments
        segment_performances = {}
        
        for segment in self.population_segments:
            if segment.name not in prediction_models:
                continue
                
            try:
                predictions = prediction_models[segment.name](optimal_global_dose)
                coverage = np.mean(predictions < target_threshold)
                segment_performances[segment.name] = {
                    'coverage': coverage,
                    'meets_target': coverage >= population_coverage,
                    'predicted_biomarkers': predictions
                }
            except:
                segment_performances[segment.name] = {
                    'coverage': 0.0,
                    'meets_target': False,
                    'predicted_biomarkers': np.array([])
                }
                
        # Calculate overall population-weighted coverage
        weighted_coverage = sum(
            perf['coverage'] * segment.prevalence 
            for segment, perf in zip(self.population_segments, segment_performances.values())
        ) / sum(segment.prevalence for segment in self.population_segments)
        
        return {
            'global_optimal_dose': optimal_global_dose,
            'global_weekly_dose': optimal_global_dose * 7,
            'weighted_coverage': weighted_coverage,
            'segment_performances': segment_performances,
            'optimization_success': result.success,
            'fairness_score': self._calculate_fairness_score(segment_performances),
            'optimization_details': {
                'method': 'global_population_optimization',
                'objective_value': result.fun,
                'total_segments': len(self.population_segments)
            }
        }
        
    def _calculate_fairness_penalty(self,
                                   dose: float,
                                   prediction_models: Dict[str, Callable],
                                   target_threshold: float) -> float:
        """Calculate fairness penalty to ensure equitable coverage across segments."""
        
        coverages = []
        
        for segment in self.population_segments:
            if segment.name not in prediction_models:
                continue
                
            try:
                predictions = prediction_models[segment.name](dose)
                coverage = np.mean(predictions < target_threshold)
                coverages.append(coverage)
            except:
                coverages.append(0.0)
                
        if len(coverages) < 2:
            return 0.0
            
        # Use coefficient of variation as fairness measure
        mean_coverage = np.mean(coverages)
        std_coverage = np.std(coverages)
        
        if mean_coverage > 0:
            fairness_penalty = (std_coverage / mean_coverage) ** 2
        else:
            fairness_penalty = 1.0
            
        return fairness_penalty
        
    def _calculate_fairness_score(self, segment_performances: Dict[str, Any]) -> float:
        """Calculate overall fairness score across segments."""
        
        coverages = [perf['coverage'] for perf in segment_performances.values()]
        
        if len(coverages) < 2:
            return 1.0
            
        # Fairness score based on minimum coverage relative to mean
        min_coverage = min(coverages)
        mean_coverage = np.mean(coverages)
        
        if mean_coverage > 0:
            fairness_score = min_coverage / mean_coverage
        else:
            fairness_score = 0.0
            
        return fairness_score
        
    def stratified_optimization(self,
                               data: PKPDData,
                               n_clusters: int = 4,
                               prediction_func: Callable = None) -> Dict[str, Any]:
        """Perform stratified optimization by clustering patients."""
        
        if prediction_func is None:
            raise ValueError("Prediction function required for stratified optimization")
            
        # Cluster patients based on features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data.features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Optimize for each cluster
        cluster_results = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_subjects = [data.subjects[i] for i in np.where(cluster_mask)[0]]
            
            self.logger.info(f"Optimizing for cluster {cluster_id} ({len(cluster_subjects)} subjects)")
            
            # Create cluster-specific prediction function
            def cluster_prediction_func(dose):
                predictions = prediction_func(dose)
                return predictions[cluster_mask]
                
            # Optimize for this cluster
            cluster_result = self._optimize_cluster(
                cluster_id,
                cluster_prediction_func,
                len(cluster_subjects)
            )
            
            cluster_results[f"cluster_{cluster_id}"] = cluster_result
            
        return {
            'cluster_results': cluster_results,
            'cluster_assignments': clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'n_clusters': n_clusters
        }
        
    def _optimize_cluster(self,
                         cluster_id: int,
                         prediction_func: Callable,
                         n_subjects: int) -> DosingResults:
        """Optimize dosing for a specific cluster."""
        
        def objective(dose_array):
            dose = dose_array[0]
            
            try:
                predictions = prediction_func(dose)
                coverage = np.mean(predictions < 3.3)  # Fixed threshold
                
                # Weight penalty by cluster size
                size_weight = min(1.0, n_subjects / 10.0)
                coverage_penalty = max(0, 0.9 - coverage) ** 2 * size_weight
                
                dose_penalty = dose * 0.01
                
                return coverage_penalty * 1000 + dose_penalty
                
            except:
                return 1e6
                
        bounds = [(0.1, 100.0)]
        result = minimize(objective, [10.0], method='L-BFGS-B', bounds=bounds)
        
        optimal_dose = result.x[0]
        
        try:
            final_predictions = prediction_func(optimal_dose)
            coverage_achieved = np.mean(final_predictions < 3.3)
        except:
            final_predictions = np.array([])
            coverage_achieved = 0.0
            
        return DosingResults(
            daily_dose=optimal_dose,
            weekly_dose=optimal_dose * 7,
            predicted_biomarkers=final_predictions,
            coverage_achieved=coverage_achieved,
            optimization_success=result.success,
            optimization_details={
                'cluster_id': cluster_id,
                'cluster_size': n_subjects,
                'method': 'cluster_optimization'
            }
        )
        
    def adaptive_population_dosing(self,
                                  initial_data: PKPDData,
                                  prediction_models: Dict[str, Callable],
                                  adaptation_rounds: int = 5) -> Dict[str, Any]:
        """Adaptive population dosing that learns from previous rounds."""
        
        adaptation_history = []
        current_doses = {}
        
        # Initialize with standard optimization
        initial_result = self.optimize_population_dosing(prediction_models)
        adaptation_history.append(initial_result)
        
        for segment in self.population_segments:
            if segment.name in initial_result['segment_results']:
                current_doses[segment.name] = initial_result['segment_results'][segment.name].daily_dose
                
        # Adaptive rounds
        for round_num in range(adaptation_rounds):
            self.logger.info(f"Adaptive optimization round {round_num + 1}")
            
            # Update prediction models based on previous results
            updated_models = self._update_prediction_models(
                prediction_models, 
                adaptation_history,
                current_doses
            )
            
            # Re-optimize with updated models
            round_result = self.optimize_population_dosing(updated_models)
            adaptation_history.append(round_result)
            
            # Update current doses
            for segment in self.population_segments:
                if segment.name in round_result['segment_results']:
                    new_dose = round_result['segment_results'][segment.name].daily_dose
                    
                    # Adaptive step size based on convergence
                    step_size = 0.5 ** round_num  # Decreasing step size
                    adapted_dose = current_doses.get(segment.name, new_dose)
                    current_doses[segment.name] = adapted_dose + step_size * (new_dose - adapted_dose)
                    
        return {
            'final_doses': current_doses,
            'adaptation_history': adaptation_history,
            'convergence_metrics': self._calculate_convergence_metrics(adaptation_history),
            'total_rounds': adaptation_rounds
        }
        
    def _update_prediction_models(self,
                                 original_models: Dict[str, Callable],
                                 history: List[Dict[str, Any]],
                                 current_doses: Dict[str, float]) -> Dict[str, Callable]:
        """Update prediction models based on optimization history."""
        
        # For now, return original models
        # In a real implementation, this could incorporate:
        # - Bayesian updates based on observed performance
        # - Uncertainty adjustments
        # - Model recalibration
        
        return original_models.copy()
        
    def _calculate_convergence_metrics(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate convergence metrics for adaptive optimization."""
        
        if len(history) < 2:
            return {'converged': False}
            
        # Calculate dose change between last two rounds
        last_round = history[-1]
        prev_round = history[-2]
        
        dose_changes = []
        
        for segment_name in last_round['segment_results'].keys():
            if segment_name in prev_round['segment_results']:
                last_dose = last_round['segment_results'][segment_name].daily_dose
                prev_dose = prev_round['segment_results'][segment_name].daily_dose
                
                relative_change = abs(last_dose - prev_dose) / prev_dose if prev_dose > 0 else 0
                dose_changes.append(relative_change)
                
        max_change = max(dose_changes) if dose_changes else 1.0
        mean_change = np.mean(dose_changes) if dose_changes else 1.0
        
        return {
            'converged': max_change < 0.01,  # 1% convergence threshold
            'max_relative_change': max_change,
            'mean_relative_change': mean_change,
            'optimization_rounds': len(history)
        }