#!/usr/bin/env python3
"""
Phase 2D: Advanced Analytics and Quantum Advantage Metrics

This module provides comprehensive analytics capabilities for VQCdd, including:
- Quantum advantage measurement and characterization
- Scientific insight generation and hypothesis testing
- Advanced performance metrics and benchmarking
- Automated experiment analysis and reporting
- Statistical significance testing for quantum vs classical approaches
- Resource efficiency analysis and scaling behavior
- Publication-ready scientific analysis

Author: VQCdd Development Team
Created: 2025-09-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from pathlib import Path
import json
import pickle
from scipy import stats

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy objects"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.complex_):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        return super().default(obj)
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
import itertools
from collections import defaultdict, deque
import concurrent.futures
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumAdvantageMetric(Enum):
    """Enumeration of quantum advantage metrics"""
    ACCURACY_IMPROVEMENT = "accuracy_improvement"
    CONVERGENCE_SPEED = "convergence_speed"
    PARAMETER_EFFICIENCY = "parameter_efficiency"
    EXPRESSIVITY = "expressivity"
    GENERALIZATION = "generalization"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    NOISE_RESILIENCE = "noise_resilience"
    SCALING_BEHAVIOR = "scaling_behavior"

@dataclass
class ExperimentResult:
    """Container for experiment results and metadata"""
    experiment_id: str
    timestamp: datetime
    approach: str
    configuration: Dict[str, Any]
    metrics: Dict[str, float]
    raw_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp.isoformat(),
            'approach': self.approach,
            'configuration': self.configuration,
            'metrics': self.metrics,
            'raw_data': self.raw_data,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class QuantumAdvantageReport:
    """Comprehensive quantum advantage analysis report"""
    advantage_metrics: Dict[QuantumAdvantageMetric, float]
    statistical_significance: Dict[QuantumAdvantageMetric, Dict[str, float]]
    confidence_intervals: Dict[QuantumAdvantageMetric, Tuple[float, float]]
    resource_analysis: Dict[str, Any]
    scaling_analysis: Dict[str, Any]
    recommendations: List[str]
    figures: Dict[str, str]  # Figure names to file paths
    raw_results: List[ExperimentResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'advantage_metrics': {k.value: v for k, v in self.advantage_metrics.items()},
            'statistical_significance': {k.value: v for k, v in self.statistical_significance.items()},
            'confidence_intervals': {k.value: v for k, v in self.confidence_intervals.items()},
            'resource_analysis': self.resource_analysis,
            'scaling_analysis': self.scaling_analysis,
            'recommendations': self.recommendations,
            'figures': self.figures,
            'raw_results': [r.to_dict() for r in self.raw_results]
        }

class AdvancedAnalytics:
    """
    Advanced analytics framework for VQCdd quantum pharmacokinetic modeling

    This class provides comprehensive analysis capabilities including:
    - Quantum advantage measurement across multiple metrics
    - Statistical significance testing and confidence intervals
    - Resource efficiency analysis and scaling behavior
    - Scientific insight generation and hypothesis testing
    - Automated experiment comparison and benchmarking
    """

    def __init__(self,
                 results_dir: str = "results/analytics",
                 figures_dir: str = "results/figures",
                 confidence_level: float = 0.95,
                 significance_threshold: float = 0.05,
                 n_bootstrap: int = 1000):
        """
        Initialize advanced analytics framework

        Args:
            results_dir: Directory for saving analysis results
            figures_dir: Directory for saving figures
            confidence_level: Confidence level for intervals
            significance_threshold: Statistical significance threshold
            n_bootstrap: Number of bootstrap samples
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.confidence_level = confidence_level
        self.significance_threshold = significance_threshold
        self.n_bootstrap = n_bootstrap

        # Experiment database
        self.experiments: List[ExperimentResult] = []
        self.experiment_db_path = self.results_dir / "experiment_database.json"

        # Load existing experiments if available
        self._load_experiment_database()

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        logger.info(f"Advanced Analytics initialized with {len(self.experiments)} existing experiments")

    def _load_experiment_database(self):
        """Load experiment database from file"""
        try:
            if self.experiment_db_path.exists():
                with open(self.experiment_db_path, 'r') as f:
                    data = json.load(f)
                    self.experiments = [ExperimentResult.from_dict(exp) for exp in data]
                logger.info(f"Loaded {len(self.experiments)} experiments from database")
        except Exception as e:
            logger.warning(f"Could not load experiment database: {e}")
            self.experiments = []

    def _save_experiment_database(self):
        """Save experiment database to file"""
        try:
            with open(self.experiment_db_path, 'w') as f:
                json.dump([exp.to_dict() for exp in self.experiments], f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Saved {len(self.experiments)} experiments to database")
        except Exception as e:
            logger.error(f"Could not save experiment database: {e}")

    def register_experiment(self,
                          experiment_id: str,
                          approach: str,
                          configuration: Dict[str, Any],
                          metrics: Dict[str, float],
                          raw_data: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None) -> ExperimentResult:
        """
        Register a new experiment result

        Args:
            experiment_id: Unique identifier for experiment
            approach: Approach type (e.g., 'quantum_vqc', 'classical_ml')
            configuration: Experiment configuration parameters
            metrics: Performance metrics
            raw_data: Raw experimental data
            metadata: Additional metadata

        Returns:
            ExperimentResult object
        """
        if metadata is None:
            metadata = {}

        result = ExperimentResult(
            experiment_id=experiment_id,
            timestamp=datetime.now(),
            approach=approach,
            configuration=configuration,
            metrics=metrics,
            raw_data=raw_data,
            metadata=metadata
        )

        self.experiments.append(result)
        self._save_experiment_database()

        logger.info(f"Registered experiment {experiment_id} with approach {approach}")
        return result

    def compute_quantum_advantage(self,
                                quantum_experiments: List[str],
                                classical_experiments: List[str],
                                metrics: Optional[List[QuantumAdvantageMetric]] = None) -> QuantumAdvantageReport:
        """
        Compute comprehensive quantum advantage analysis

        Args:
            quantum_experiments: List of quantum experiment IDs
            classical_experiments: List of classical experiment IDs
            metrics: Specific metrics to analyze (default: all)

        Returns:
            QuantumAdvantageReport with comprehensive analysis
        """
        if metrics is None:
            metrics = list(QuantumAdvantageMetric)

        logger.info(f"Computing quantum advantage for {len(quantum_experiments)} quantum vs {len(classical_experiments)} classical experiments")

        # Filter experiments
        quantum_results = [exp for exp in self.experiments if exp.experiment_id in quantum_experiments]
        classical_results = [exp for exp in self.experiments if exp.experiment_id in classical_experiments]

        if not quantum_results or not classical_results:
            raise ValueError("No quantum or classical experiments found")

        # Compute advantage metrics
        advantage_metrics = {}
        statistical_significance = {}
        confidence_intervals = {}

        for metric in metrics:
            advantage, significance, ci = self._compute_single_advantage_metric(
                quantum_results, classical_results, metric
            )
            advantage_metrics[metric] = advantage
            statistical_significance[metric] = significance
            confidence_intervals[metric] = ci

        # Resource analysis
        resource_analysis = self._analyze_resource_efficiency(quantum_results, classical_results)

        # Scaling analysis
        scaling_analysis = self._analyze_scaling_behavior(quantum_results, classical_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            advantage_metrics, statistical_significance, resource_analysis, scaling_analysis
        )

        # Generate figures
        figures = self._generate_advantage_figures(
            quantum_results, classical_results, advantage_metrics, scaling_analysis
        )

        report = QuantumAdvantageReport(
            advantage_metrics=advantage_metrics,
            statistical_significance=statistical_significance,
            confidence_intervals=confidence_intervals,
            resource_analysis=resource_analysis,
            scaling_analysis=scaling_analysis,
            recommendations=recommendations,
            figures=figures,
            raw_results=quantum_results + classical_results
        )

        # Save report
        self._save_advantage_report(report)

        return report

    def _compute_single_advantage_metric(self,
                                       quantum_results: List[ExperimentResult],
                                       classical_results: List[ExperimentResult],
                                       metric: QuantumAdvantageMetric) -> Tuple[float, Dict[str, float], Tuple[float, float]]:
        """Compute single quantum advantage metric"""

        metric_extractors = {
            QuantumAdvantageMetric.ACCURACY_IMPROVEMENT: self._extract_accuracy_metric,
            QuantumAdvantageMetric.CONVERGENCE_SPEED: self._extract_convergence_metric,
            QuantumAdvantageMetric.PARAMETER_EFFICIENCY: self._extract_parameter_efficiency_metric,
            QuantumAdvantageMetric.EXPRESSIVITY: self._extract_expressivity_metric,
            QuantumAdvantageMetric.GENERALIZATION: self._extract_generalization_metric,
            QuantumAdvantageMetric.RESOURCE_EFFICIENCY: self._extract_resource_efficiency_metric,
            QuantumAdvantageMetric.NOISE_RESILIENCE: self._extract_noise_resilience_metric,
            QuantumAdvantageMetric.SCALING_BEHAVIOR: self._extract_scaling_metric
        }

        if metric not in metric_extractors:
            raise ValueError(f"Unknown metric: {metric}")

        # Extract metric values
        quantum_values = [metric_extractors[metric](exp) for exp in quantum_results]
        classical_values = [metric_extractors[metric](exp) for exp in classical_results]

        # Remove None values
        quantum_values = [v for v in quantum_values if v is not None]
        classical_values = [v for v in classical_values if v is not None]

        if not quantum_values or not classical_values:
            logger.warning(f"No valid values for metric {metric.value}")
            return 0.0, {'p_value': 1.0, 'test_statistic': 0.0}, (0.0, 0.0)

        # Compute advantage (relative improvement)
        quantum_mean = np.mean(quantum_values)
        classical_mean = np.mean(classical_values)

        if classical_mean == 0:
            advantage = float('inf') if quantum_mean > 0 else 0.0
        else:
            advantage = (quantum_mean - classical_mean) / abs(classical_mean)

        # Statistical significance testing
        statistic, p_value = stats.mannwhitneyu(
            quantum_values, classical_values, alternative='two-sided'
        )

        significance = {
            'p_value': p_value,
            'test_statistic': float(statistic),
            'significant': p_value < self.significance_threshold
        }

        # Bootstrap confidence interval for advantage
        bootstrap_advantages = []
        for _ in range(self.n_bootstrap):
            q_sample = np.random.choice(quantum_values, len(quantum_values), replace=True)
            c_sample = np.random.choice(classical_values, len(classical_values), replace=True)

            q_mean = np.mean(q_sample)
            c_mean = np.mean(c_sample)

            if c_mean == 0:
                boot_advantage = float('inf') if q_mean > 0 else 0.0
            else:
                boot_advantage = (q_mean - c_mean) / abs(c_mean)

            if not np.isinf(boot_advantage):
                bootstrap_advantages.append(boot_advantage)

        if bootstrap_advantages:
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_advantages, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_advantages, 100 * (1 - alpha / 2))
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = (advantage, advantage)

        return advantage, significance, confidence_interval

    def _extract_accuracy_metric(self, exp: ExperimentResult) -> Optional[float]:
        """Extract accuracy metric from experiment"""
        # Try multiple accuracy metric names
        for key in ['accuracy', 'r2_score', 'final_accuracy', 'test_accuracy']:
            if key in exp.metrics:
                return exp.metrics[key]

        # Compute from raw data if available
        if 'predictions' in exp.raw_data and 'targets' in exp.raw_data:
            pred = np.array(exp.raw_data['predictions'])
            targ = np.array(exp.raw_data['targets'])
            return r2_score(targ, pred)

        return None

    def _extract_convergence_metric(self, exp: ExperimentResult) -> Optional[float]:
        """Extract convergence speed metric"""
        # Try convergence-related metrics
        for key in ['convergence_iterations', 'training_iterations', 'epochs_to_convergence']:
            if key in exp.metrics:
                return 1.0 / exp.metrics[key]  # Invert for "speed"

        # Extract from training history
        if 'training_history' in exp.raw_data:
            history = exp.raw_data['training_history']
            if isinstance(history, list) and len(history) > 1:
                # Find when loss stopped improving significantly
                losses = [h.get('loss', float('inf')) for h in history]
                for i in range(1, len(losses)):
                    if abs(losses[i] - losses[i-1]) < 0.001:  # Convergence threshold
                        return 1.0 / i

        return None

    def _extract_parameter_efficiency_metric(self, exp: ExperimentResult) -> Optional[float]:
        """Extract parameter efficiency metric"""
        accuracy = self._extract_accuracy_metric(exp)
        if accuracy is None:
            return None

        # Try to get parameter count
        n_params = exp.configuration.get('n_parameters')
        if n_params is None:
            n_params = exp.metrics.get('n_parameters')

        if n_params is not None and n_params > 0:
            return accuracy / n_params  # Accuracy per parameter

        return None

    def _extract_expressivity_metric(self, exp: ExperimentResult) -> Optional[float]:
        """Extract expressivity metric"""
        # Look for expressivity-specific metrics
        for key in ['expressivity', 'effective_dimension', 'feature_variance']:
            if key in exp.metrics:
                return exp.metrics[key]

        # Estimate from circuit configuration
        config = exp.configuration
        if 'n_qubits' in config and 'n_layers' in config:
            # Simple expressivity estimate
            return config['n_qubits'] * config['n_layers']

        return None

    def _extract_generalization_metric(self, exp: ExperimentResult) -> Optional[float]:
        """Extract generalization metric"""
        # Try generalization gap metrics
        train_acc = exp.metrics.get('train_accuracy')
        test_acc = exp.metrics.get('test_accuracy')

        if train_acc is not None and test_acc is not None:
            return 1.0 - abs(train_acc - test_acc)  # 1 - generalization gap

        # Try validation metrics
        val_acc = exp.metrics.get('validation_accuracy')
        if train_acc is not None and val_acc is not None:
            return 1.0 - abs(train_acc - val_acc)

        return None

    def _extract_resource_efficiency_metric(self, exp: ExperimentResult) -> Optional[float]:
        """Extract resource efficiency metric"""
        accuracy = self._extract_accuracy_metric(exp)
        if accuracy is None:
            return None

        # Try to get computational cost
        for key in ['training_time', 'computation_time', 'wall_time']:
            if key in exp.metrics:
                time_cost = exp.metrics[key]
                if time_cost > 0:
                    return accuracy / time_cost  # Accuracy per unit time

        return None

    def _extract_noise_resilience_metric(self, exp: ExperimentResult) -> Optional[float]:
        """Extract noise resilience metric"""
        # Look for noise-specific metrics
        for key in ['noise_resilience', 'clean_vs_noisy_accuracy', 'error_mitigation_effectiveness']:
            if key in exp.metrics:
                return exp.metrics[key]

        # Compute from clean vs noisy results if available
        clean_acc = exp.metrics.get('clean_accuracy')
        noisy_acc = exp.metrics.get('noisy_accuracy')

        if clean_acc is not None and noisy_acc is not None and clean_acc > 0:
            return noisy_acc / clean_acc  # Relative accuracy under noise

        return None

    def _extract_scaling_metric(self, exp: ExperimentResult) -> Optional[float]:
        """Extract scaling behavior metric"""
        # Look for scaling-specific metrics
        for key in ['scaling_exponent', 'complexity_scaling', 'data_efficiency']:
            if key in exp.metrics:
                return exp.metrics[key]

        # Estimate from problem size vs performance
        problem_size = exp.configuration.get('problem_size', exp.configuration.get('n_patients'))
        accuracy = self._extract_accuracy_metric(exp)

        if problem_size is not None and accuracy is not None and problem_size > 0:
            return accuracy * np.log(problem_size)  # Log-scaled performance

        return None

    def _analyze_resource_efficiency(self,
                                   quantum_results: List[ExperimentResult],
                                   classical_results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze resource efficiency comparison"""
        analysis = {
            'quantum_resource_profile': {},
            'classical_resource_profile': {},
            'efficiency_comparison': {},
            'cost_benefit_analysis': {}
        }

        # Extract resource metrics
        def extract_resources(results: List[ExperimentResult]) -> Dict[str, List[float]]:
            resources = defaultdict(list)
            for exp in results:
                for key in ['training_time', 'memory_usage', 'cpu_time', 'gpu_time']:
                    if key in exp.metrics:
                        resources[key].append(exp.metrics[key])
            return dict(resources)

        quantum_resources = extract_resources(quantum_results)
        classical_resources = extract_resources(classical_results)

        analysis['quantum_resource_profile'] = {
            key: {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values)
            } for key, values in quantum_resources.items() if values
        }

        analysis['classical_resource_profile'] = {
            key: {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values)
            } for key, values in classical_resources.items() if values
        }

        # Efficiency comparison
        for resource in set(quantum_resources.keys()) & set(classical_resources.keys()):
            q_values = quantum_resources[resource]
            c_values = classical_resources[resource]

            if q_values and c_values:
                q_mean = np.mean(q_values)
                c_mean = np.mean(c_values)

                analysis['efficiency_comparison'][resource] = {
                    'quantum_mean': q_mean,
                    'classical_mean': c_mean,
                    'relative_efficiency': c_mean / q_mean if q_mean > 0 else float('inf'),
                    'significance_test': stats.mannwhitneyu(q_values, c_values)._asdict()
                }

        return analysis

    def _analyze_scaling_behavior(self,
                                quantum_results: List[ExperimentResult],
                                classical_results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze scaling behavior of quantum vs classical approaches"""
        analysis = {
            'quantum_scaling': {},
            'classical_scaling': {},
            'comparative_scaling': {},
            'asymptotic_behavior': {}
        }

        def extract_scaling_data(results: List[ExperimentResult]) -> Tuple[List[float], List[float]]:
            """Extract problem size and performance data"""
            sizes = []
            performances = []

            for exp in results:
                # Try to extract problem size
                size = None
                for key in ['n_patients', 'problem_size', 'data_size', 'n_qubits']:
                    if key in exp.configuration:
                        size = exp.configuration[key]
                        break

                # Extract performance
                perf = self._extract_accuracy_metric(exp)

                if size is not None and perf is not None:
                    sizes.append(size)
                    performances.append(perf)

            return sizes, performances

        # Extract scaling data
        q_sizes, q_perfs = extract_scaling_data(quantum_results)
        c_sizes, c_perfs = extract_scaling_data(classical_results)

        # Fit scaling models
        if len(q_sizes) >= 3:
            analysis['quantum_scaling'] = self._fit_scaling_models(q_sizes, q_perfs, 'quantum')

        if len(c_sizes) >= 3:
            analysis['classical_scaling'] = self._fit_scaling_models(c_sizes, c_perfs, 'classical')

        # Comparative analysis
        if len(q_sizes) >= 3 and len(c_sizes) >= 3:
            analysis['comparative_scaling'] = self._compare_scaling_behaviors(
                (q_sizes, q_perfs), (c_sizes, c_perfs)
            )

        return analysis

    def _fit_scaling_models(self, sizes: List[float], performances: List[float], approach: str) -> Dict[str, Any]:
        """Fit various scaling models to performance data"""
        models = {}

        x = np.array(sizes)
        y = np.array(performances)

        # Sort by size
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

        # Define scaling models
        def linear(x, a, b):
            return a * x + b

        def logarithmic(x, a, b):
            return a * np.log(x) + b

        def power_law(x, a, b):
            return a * np.power(x, b)

        def exponential(x, a, b):
            return a * np.exp(b * x)

        # Fit models
        model_functions = {
            'linear': linear,
            'logarithmic': logarithmic,
            'power_law': power_law,
            'exponential': exponential
        }

        for name, func in model_functions.items():
            try:
                if name == 'logarithmic' and np.any(x <= 0):
                    continue
                if name == 'power_law' and np.any(x <= 0):
                    continue

                popt, pcov = curve_fit(func, x, y, maxfev=2000)
                y_pred = func(x, *popt)
                r2 = r2_score(y, y_pred)

                models[name] = {
                    'parameters': popt.tolist(),
                    'covariance': pcov.tolist(),
                    'r2_score': r2,
                    'rmse': np.sqrt(mean_squared_error(y, y_pred))
                }

            except Exception as e:
                logger.warning(f"Could not fit {name} model for {approach}: {e}")
                continue

        # Select best model
        if models:
            best_model = max(models.keys(), key=lambda k: models[k]['r2_score'])
            models['best_model'] = best_model

        return models

    def _compare_scaling_behaviors(self,
                                 quantum_data: Tuple[List[float], List[float]],
                                 classical_data: Tuple[List[float], List[float]]) -> Dict[str, Any]:
        """Compare scaling behaviors between quantum and classical approaches"""
        q_sizes, q_perfs = quantum_data
        c_sizes, c_perfs = classical_data

        comparison = {}

        # Find common size range
        min_size = max(min(q_sizes), min(c_sizes))
        max_size = min(max(q_sizes), max(c_sizes))

        if min_size < max_size:
            # Interpolate to common grid
            common_sizes = np.linspace(min_size, max_size, 10)

            try:
                q_interp = np.interp(common_sizes, sorted(q_sizes),
                                   [q_perfs[i] for i in np.argsort(q_sizes)])
                c_interp = np.interp(common_sizes, sorted(c_sizes),
                                   [c_perfs[i] for i in np.argsort(c_sizes)])

                # Compute advantage across sizes
                advantages = (q_interp - c_interp) / np.maximum(c_interp, 0.001)

                comparison['size_range'] = (min_size, max_size)
                comparison['advantage_trend'] = {
                    'sizes': common_sizes.tolist(),
                    'advantages': advantages.tolist(),
                    'mean_advantage': float(np.mean(advantages)),
                    'advantage_slope': float(np.polyfit(common_sizes, advantages, 1)[0])
                }

                # Statistical test for trend
                trend_corr, trend_p = stats.pearsonr(common_sizes, advantages)
                comparison['trend_significance'] = {
                    'correlation': trend_corr,
                    'p_value': trend_p,
                    'significant': trend_p < self.significance_threshold
                }

            except Exception as e:
                logger.warning(f"Could not perform scaling comparison: {e}")

        return comparison

    def _generate_recommendations(self,
                                advantage_metrics: Dict[QuantumAdvantageMetric, float],
                                statistical_significance: Dict[QuantumAdvantageMetric, Dict[str, float]],
                                resource_analysis: Dict[str, Any],
                                scaling_analysis: Dict[str, Any]) -> List[str]:
        """Generate scientific recommendations based on analysis"""
        recommendations = []

        # Analyze quantum advantage patterns
        significant_advantages = [
            metric for metric, significance in statistical_significance.items()
            if significance.get('significant', False) and advantage_metrics[metric] > 0
        ]

        if significant_advantages:
            recommendations.append(
                f"Quantum advantage demonstrated in {len(significant_advantages)} metrics: "
                f"{', '.join([m.value for m in significant_advantages])}"
            )

        # Resource efficiency recommendations
        if 'efficiency_comparison' in resource_analysis:
            efficient_resources = [
                resource for resource, data in resource_analysis['efficiency_comparison'].items()
                if data.get('relative_efficiency', 0) > 1.0
            ]
            if efficient_resources:
                recommendations.append(
                    f"Quantum approach shows resource efficiency in: {', '.join(efficient_resources)}"
                )

        # Scaling behavior recommendations
        if 'comparative_scaling' in scaling_analysis:
            scaling_data = scaling_analysis['comparative_scaling']
            if 'advantage_trend' in scaling_data:
                slope = scaling_data['advantage_trend'].get('advantage_slope', 0)
                if slope > 0:
                    recommendations.append(
                        "Quantum advantage increases with problem size - promising for large-scale applications"
                    )
                elif slope < 0:
                    recommendations.append(
                        "Quantum advantage decreases with problem size - may be limited to specific problem scales"
                    )

        # Performance threshold recommendations
        accuracy_advantage = advantage_metrics.get(QuantumAdvantageMetric.ACCURACY_IMPROVEMENT, 0)
        if accuracy_advantage > 0.1:  # 10% improvement
            recommendations.append(
                f"Substantial accuracy improvement ({accuracy_advantage:.1%}) suggests quantum approach is viable"
            )
        elif accuracy_advantage > 0.05:  # 5% improvement
            recommendations.append(
                f"Moderate accuracy improvement ({accuracy_advantage:.1%}) - consider cost-benefit trade-offs"
            )

        # Add general recommendations
        recommendations.extend([
            "Continue investigation with larger problem sizes to validate scaling behavior",
            "Implement noise mitigation strategies for NISQ device deployment",
            "Consider hybrid quantum-classical algorithms for optimal resource utilization"
        ])

        return recommendations

    def _generate_advantage_figures(self,
                                  quantum_results: List[ExperimentResult],
                                  classical_results: List[ExperimentResult],
                                  advantage_metrics: Dict[QuantumAdvantageMetric, float],
                                  scaling_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate publication-ready figures for quantum advantage analysis"""
        figures = {}

        # Set up figure style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Quantum Advantage Radar Chart
        fig_path = self.figures_dir / "quantum_advantage_radar.png"
        self._create_advantage_radar_chart(advantage_metrics, fig_path)
        figures['advantage_radar'] = str(fig_path)

        # 2. Performance Comparison Box Plots
        fig_path = self.figures_dir / "performance_comparison.png"
        self._create_performance_comparison_plot(quantum_results, classical_results, fig_path)
        figures['performance_comparison'] = str(fig_path)

        # 3. Scaling Behavior Analysis
        if scaling_analysis.get('comparative_scaling'):
            fig_path = self.figures_dir / "scaling_behavior.png"
            self._create_scaling_analysis_plot(scaling_analysis, fig_path)
            figures['scaling_behavior'] = str(fig_path)

        # 4. Resource Efficiency Heatmap
        fig_path = self.figures_dir / "resource_efficiency.png"
        self._create_resource_efficiency_plot(quantum_results, classical_results, fig_path)
        figures['resource_efficiency'] = str(fig_path)

        # 5. Statistical Significance Summary
        fig_path = self.figures_dir / "statistical_significance.png"
        self._create_significance_summary_plot(advantage_metrics, fig_path)
        figures['statistical_significance'] = str(fig_path)

        logger.info(f"Generated {len(figures)} analysis figures")
        return figures

    def _create_advantage_radar_chart(self,
                                    advantage_metrics: Dict[QuantumAdvantageMetric, float],
                                    fig_path: Path):
        """Create radar chart showing quantum advantage across metrics"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Prepare data
        metrics = list(advantage_metrics.keys())
        values = [max(0, advantage_metrics[m]) for m in metrics]  # Only positive advantages
        labels = [m.value.replace('_', ' ').title() for m in metrics]

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label='Quantum Advantage')
        ax.fill(angles, values, alpha=0.25)

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(values) * 1.1 if values else 1)
        ax.set_title('Quantum Advantage Across Metrics', size=16, fontweight='bold', pad=20)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_performance_comparison_plot(self,
                                          quantum_results: List[ExperimentResult],
                                          classical_results: List[ExperimentResult],
                                          fig_path: Path):
        """Create box plot comparing quantum vs classical performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        # Define metrics to compare
        metrics_to_plot = [
            ('accuracy', 'Accuracy'),
            ('training_time', 'Training Time (s)'),
            ('convergence_iterations', 'Convergence Iterations'),
            ('parameter_efficiency', 'Parameter Efficiency')
        ]

        for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Extract metric values
            quantum_values = []
            classical_values = []

            for exp in quantum_results:
                if metric_key == 'accuracy':
                    val = self._extract_accuracy_metric(exp)
                elif metric_key == 'parameter_efficiency':
                    val = self._extract_parameter_efficiency_metric(exp)
                else:
                    val = exp.metrics.get(metric_key)

                if val is not None:
                    quantum_values.append(val)

            for exp in classical_results:
                if metric_key == 'accuracy':
                    val = self._extract_accuracy_metric(exp)
                elif metric_key == 'parameter_efficiency':
                    val = self._extract_parameter_efficiency_metric(exp)
                else:
                    val = exp.metrics.get(metric_key)

                if val is not None:
                    classical_values.append(val)

            # Create box plot
            if quantum_values or classical_values:
                data = []
                labels = []

                if quantum_values:
                    data.append(quantum_values)
                    labels.append('Quantum')

                if classical_values:
                    data.append(classical_values)
                    labels.append('Classical')

                ax.boxplot(data, labels=labels)
                ax.set_title(metric_label, fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(metric_label, fontweight='bold')

        plt.suptitle('Quantum vs Classical Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_scaling_analysis_plot(self, scaling_analysis: Dict[str, Any], fig_path: Path):
        """Create scaling behavior analysis plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Advantage vs Problem Size
        if 'comparative_scaling' in scaling_analysis:
            comp_data = scaling_analysis['comparative_scaling']
            if 'advantage_trend' in comp_data:
                trend_data = comp_data['advantage_trend']
                sizes = trend_data['sizes']
                advantages = trend_data['advantages']

                ax1.plot(sizes, advantages, 'o-', linewidth=2, markersize=6)
                ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Advantage')
                ax1.set_xlabel('Problem Size')
                ax1.set_ylabel('Quantum Advantage')
                ax1.set_title('Quantum Advantage vs Problem Size')
                ax1.grid(True, alpha=0.3)
                ax1.legend()

        # Plot 2: Scaling Model Comparison
        quantum_scaling = scaling_analysis.get('quantum_scaling', {})
        classical_scaling = scaling_analysis.get('classical_scaling', {})

        if quantum_scaling and classical_scaling:
            # Compare best model R² scores
            metrics = ['linear', 'logarithmic', 'power_law', 'exponential']
            quantum_r2 = [quantum_scaling.get(m, {}).get('r2_score', 0) for m in metrics]
            classical_r2 = [classical_scaling.get(m, {}).get('r2_score', 0) for m in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            ax2.bar(x - width/2, quantum_r2, width, label='Quantum', alpha=0.8)
            ax2.bar(x + width/2, classical_r2, width, label='Classical', alpha=0.8)

            ax2.set_xlabel('Scaling Model')
            ax2.set_ylabel('R² Score')
            ax2.set_title('Scaling Model Fit Quality')
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_resource_efficiency_plot(self,
                                       quantum_results: List[ExperimentResult],
                                       classical_results: List[ExperimentResult],
                                       fig_path: Path):
        """Create resource efficiency comparison heatmap"""
        # Extract resource metrics
        resources = ['training_time', 'memory_usage', 'cpu_time', 'gpu_time']
        approaches = ['Quantum', 'Classical']

        data_matrix = np.zeros((len(resources), len(approaches)))

        for i, resource in enumerate(resources):
            # Quantum values
            q_values = [exp.metrics.get(resource) for exp in quantum_results
                       if exp.metrics.get(resource) is not None]
            if q_values:
                data_matrix[i, 0] = np.mean(q_values)

            # Classical values
            c_values = [exp.metrics.get(resource) for exp in classical_results
                       if exp.metrics.get(resource) is not None]
            if c_values:
                data_matrix[i, 1] = np.mean(c_values)

        # Normalize by row (relative efficiency)
        row_sums = data_matrix.sum(axis=1)
        normalized_data = np.divide(data_matrix, row_sums[:, np.newaxis],
                                  out=np.zeros_like(data_matrix), where=row_sums[:, np.newaxis]!=0)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(normalized_data, cmap='RdYlBu_r', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(approaches)))
        ax.set_yticks(np.arange(len(resources)))
        ax.set_xticklabels(approaches)
        ax.set_yticklabels([r.replace('_', ' ').title() for r in resources])

        # Add text annotations
        for i in range(len(resources)):
            for j in range(len(approaches)):
                text = ax.text(j, i, f'{normalized_data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')

        ax.set_title('Resource Efficiency Comparison\n(Normalized by Resource Type)',
                    fontweight='bold', pad=20)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Relative Resource Usage', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_significance_summary_plot(self,
                                        advantage_metrics: Dict[QuantumAdvantageMetric, float],
                                        fig_path: Path):
        """Create statistical significance summary plot"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data
        metrics = list(advantage_metrics.keys())
        advantages = [advantage_metrics[m] for m in metrics]
        labels = [m.value.replace('_', ' ').title() for m in metrics]

        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        colors = ['green' if adv > 0 else 'red' for adv in advantages]

        bars = ax.barh(y_pos, advantages, color=colors, alpha=0.7)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Quantum Advantage (Relative Improvement)')
        ax.set_title('Quantum Advantage Summary Across Metrics', fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for bar, advantage in zip(bars, advantages):
            width = bar.get_width()
            label_x = width + 0.01 if width >= 0 else width - 0.01
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{advantage:.2%}', ha='left' if width >= 0 else 'right',
                   va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_advantage_report(self, report: QuantumAdvantageReport):
        """Save quantum advantage report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"quantum_advantage_report_{timestamp}.json"

        try:
            with open(report_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Saved quantum advantage report to {report_path}")
        except Exception as e:
            logger.error(f"Could not save advantage report: {e}")

    def generate_scientific_insights(self,
                                   experiments: Optional[List[str]] = None,
                                   focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate scientific insights and hypotheses from experimental data

        Args:
            experiments: List of experiment IDs to analyze (default: all)
            focus_areas: Specific areas to focus on (e.g., 'convergence', 'expressivity')

        Returns:
            Dictionary containing scientific insights and recommendations
        """
        if experiments is None:
            target_experiments = self.experiments
        else:
            target_experiments = [exp for exp in self.experiments if exp.experiment_id in experiments]

        if not target_experiments:
            raise ValueError("No experiments found for analysis")

        insights = {
            'summary_statistics': self._compute_summary_statistics(target_experiments),
            'correlation_analysis': self._perform_correlation_analysis(target_experiments),
            'hypothesis_tests': self._perform_hypothesis_tests(target_experiments),
            'pattern_detection': self._detect_patterns(target_experiments),
            'recommendations': [],
            'future_directions': []
        }

        # Generate recommendations based on insights
        insights['recommendations'] = self._generate_scientific_recommendations(insights)
        insights['future_directions'] = self._generate_future_directions(insights)

        # Save insights
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        insights_path = self.results_dir / f"scientific_insights_{timestamp}.json"

        try:
            with open(insights_path, 'w') as f:
                json.dump(insights, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Generated scientific insights and saved to {insights_path}")
        except Exception as e:
            logger.error(f"Could not save scientific insights: {e}")

        return insights

    def _compute_summary_statistics(self, experiments: List[ExperimentResult]) -> Dict[str, Any]:
        """Compute comprehensive summary statistics"""
        stats_summary = {
            'experiment_count': len(experiments),
            'approaches': {},
            'performance_metrics': {},
            'configuration_analysis': {}
        }

        # Approach breakdown
        approach_counts = defaultdict(int)
        for exp in experiments:
            approach_counts[exp.approach] += 1
        stats_summary['approaches'] = dict(approach_counts)

        # Performance metrics summary
        metric_names = set()
        for exp in experiments:
            metric_names.update(exp.metrics.keys())

        for metric in metric_names:
            values = [exp.metrics[metric] for exp in experiments if metric in exp.metrics]
            if values:
                stats_summary['performance_metrics'][metric] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }

        return stats_summary

    def _perform_correlation_analysis(self, experiments: List[ExperimentResult]) -> Dict[str, Any]:
        """Perform correlation analysis between configuration and performance"""
        correlation_results = {}

        # Extract numerical configuration parameters
        config_data = []
        performance_data = []

        for exp in experiments:
            config_dict = {}

            # Extract numerical config parameters
            for key, value in exp.configuration.items():
                if isinstance(value, (int, float)):
                    config_dict[key] = value

            # Extract primary performance metric
            primary_metric = self._extract_accuracy_metric(exp)
            if primary_metric is not None and config_dict:
                config_data.append(config_dict)
                performance_data.append(primary_metric)

        if len(config_data) >= 3:  # Need at least 3 points for correlation
            # Convert to DataFrame for easier analysis
            config_df = pd.DataFrame(config_data)

            # Compute correlations
            correlations = {}
            for column in config_df.columns:
                if config_df[column].nunique() > 1:  # Only if there's variation
                    corr, p_value = stats.pearsonr(config_df[column], performance_data)
                    correlations[column] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < self.significance_threshold
                    }

            correlation_results['parameter_correlations'] = correlations

            # Find strongest correlations
            if correlations:
                strongest_positive = max(correlations.items(),
                                       key=lambda x: x[1]['correlation'] if x[1]['significant'] else -1)
                strongest_negative = min(correlations.items(),
                                       key=lambda x: x[1]['correlation'] if x[1]['significant'] else 1)

                correlation_results['strongest_positive'] = strongest_positive
                correlation_results['strongest_negative'] = strongest_negative

        return correlation_results

    def _perform_hypothesis_tests(self, experiments: List[ExperimentResult]) -> Dict[str, Any]:
        """Perform statistical hypothesis tests"""
        hypothesis_results = {}

        # Group experiments by approach
        approach_groups = defaultdict(list)
        for exp in experiments:
            approach_groups[exp.approach].append(exp)

        # Test if different approaches have significantly different performance
        if len(approach_groups) >= 2:
            approach_names = list(approach_groups.keys())
            approach_performances = []

            for approach in approach_names:
                perfs = [self._extract_accuracy_metric(exp) for exp in approach_groups[approach]]
                perfs = [p for p in perfs if p is not None]
                approach_performances.append(perfs)

            # Kruskal-Wallis test for multiple groups
            if all(len(perfs) > 0 for perfs in approach_performances):
                try:
                    statistic, p_value = stats.kruskal(*approach_performances)
                    hypothesis_results['approach_difference_test'] = {
                        'test': 'Kruskal-Wallis',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < self.significance_threshold,
                        'interpretation': 'Approaches have significantly different performance' if p_value < self.significance_threshold else 'No significant difference between approaches'
                    }
                except Exception as e:
                    logger.warning(f"Could not perform Kruskal-Wallis test: {e}")

        # Test for normality of performance distributions
        all_performances = []
        for exp in experiments:
            perf = self._extract_accuracy_metric(exp)
            if perf is not None:
                all_performances.append(perf)

        if len(all_performances) >= 8:  # Minimum for reliable normality test
            statistic, p_value = stats.shapiro(all_performances)
            hypothesis_results['normality_test'] = {
                'test': 'Shapiro-Wilk',
                'statistic': statistic,
                'p_value': p_value,
                'normal': p_value > self.significance_threshold,
                'interpretation': 'Performance distribution is normal' if p_value > self.significance_threshold else 'Performance distribution is not normal'
            }

        return hypothesis_results

    def _detect_patterns(self, experiments: List[ExperimentResult]) -> Dict[str, Any]:
        """Detect patterns and anomalies in experimental data"""
        patterns = {
            'temporal_trends': {},
            'configuration_clusters': {},
            'performance_outliers': {},
            'convergence_patterns': {}
        }

        # Temporal trends
        if len(experiments) >= 5:
            # Sort by timestamp
            sorted_experiments = sorted(experiments, key=lambda x: x.timestamp)

            # Extract performance over time
            timestamps = [(exp.timestamp - sorted_experiments[0].timestamp).total_seconds() / 3600
                         for exp in sorted_experiments]  # Hours since first experiment
            performances = [self._extract_accuracy_metric(exp) for exp in sorted_experiments]
            performances = [p for p in performances if p is not None]

            if len(performances) >= 3:
                # Test for temporal trend
                corr, p_value = stats.pearsonr(timestamps[:len(performances)], performances)
                patterns['temporal_trends'] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'trend': 'improving' if corr > 0 and p_value < self.significance_threshold else
                            'declining' if corr < 0 and p_value < self.significance_threshold else 'no_trend'
                }

        # Performance outliers
        all_performances = [self._extract_accuracy_metric(exp) for exp in experiments]
        all_performances = [p for p in all_performances if p is not None]

        if len(all_performances) >= 5:
            # Identify outliers using IQR method
            q25, q75 = np.percentile(all_performances, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            outlier_experiments = []
            for exp in experiments:
                perf = self._extract_accuracy_metric(exp)
                if perf is not None and (perf < lower_bound or perf > upper_bound):
                    outlier_experiments.append({
                        'experiment_id': exp.experiment_id,
                        'performance': perf,
                        'type': 'high' if perf > upper_bound else 'low'
                    })

            patterns['performance_outliers'] = {
                'count': len(outlier_experiments),
                'outliers': outlier_experiments,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }

        return patterns

    def _generate_scientific_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate scientific recommendations based on insights"""
        recommendations = []

        # Based on correlation analysis
        if 'correlation_analysis' in insights and 'parameter_correlations' in insights['correlation_analysis']:
            correlations = insights['correlation_analysis']['parameter_correlations']

            significant_correlations = [
                (param, data) for param, data in correlations.items()
                if data['significant'] and abs(data['correlation']) > 0.5
            ]

            if significant_correlations:
                correlation_text = ', '.join([f'{param} (r={data["correlation"]:.2f})' for param, data in significant_correlations])
                recommendations.append(
                    f"Strong correlations found between configuration parameters and performance: {correlation_text}"
                )

        # Based on hypothesis tests
        if 'hypothesis_tests' in insights:
            if 'approach_difference_test' in insights['hypothesis_tests']:
                test = insights['hypothesis_tests']['approach_difference_test']
                if test['significant']:
                    recommendations.append(
                        "Significant performance differences detected between approaches - "
                        "investigate specific configurations that drive these differences"
                    )

        # Based on pattern detection
        if 'pattern_detection' in insights:
            patterns = insights['pattern_detection']

            if 'temporal_trends' in patterns and patterns['temporal_trends'].get('trend') == 'improving':
                recommendations.append(
                    "Performance shows improving trend over time - "
                    "continue current optimization strategy"
                )

            if 'performance_outliers' in patterns and patterns['performance_outliers'].get('count', 0) > 0:
                outlier_count = patterns['performance_outliers'].get('count', 0)
                recommendations.append(
                    f"Identified {outlier_count} performance outliers - "
                    "investigate configurations that lead to exceptional performance"
                )

        return recommendations

    def _generate_future_directions(self, insights: Dict[str, Any]) -> List[str]:
        """Generate future research directions"""
        directions = [
            "Investigate quantum advantage at larger problem scales",
            "Implement noise-aware circuit design for NISQ devices",
            "Explore hybrid quantum-classical optimization strategies",
            "Develop problem-specific quantum feature encodings",
            "Study quantum expressivity bounds for PK/PD modeling"
        ]

        # Add specific directions based on insights
        if 'summary_statistics' in insights:
            approaches = insights['summary_statistics']['approaches']
            if 'quantum_vqc' in approaches and approaches['quantum_vqc'] < 10:
                directions.append(
                    "Expand quantum VQC experimental dataset for robust statistical analysis"
                )

        return directions

# Utility functions for integration with existing VQCdd components
def create_analytics_experiment_from_trainer(trainer,
                                           experiment_id: str,
                                           test_data: Dict[str, Any],
                                           analytics: AdvancedAnalytics) -> ExperimentResult:
    """
    Create analytics experiment result from VQCTrainer

    Args:
        trainer: Trained VQCTrainer instance
        experiment_id: Unique experiment identifier
        test_data: Test dataset for evaluation
        analytics: AdvancedAnalytics instance

    Returns:
        ExperimentResult registered with analytics
    """
    # Extract configuration
    config = {
        'n_qubits': trainer.quantum_circuit.config.n_qubits,
        'n_layers': trainer.quantum_circuit.config.n_layers,
        'ansatz': trainer.quantum_circuit.config.ansatz,
        'encoding': trainer.quantum_circuit.config.encoding,
        'n_parameters': len(trainer.best_parameters) if hasattr(trainer, 'best_parameters') else None,
        'max_iterations': trainer.config.max_iterations,
        'learning_rate': trainer.config.learning_rate
    }

    # Extract metrics
    metrics = {
        'final_cost': trainer.optimization_history[-1]['cost'] if trainer.optimization_history else None,
        'training_iterations': len(trainer.optimization_history),
        'convergence_iterations': trainer._find_convergence_iteration() if hasattr(trainer, '_find_convergence_iteration') else None
    }

    # Add test performance if available
    if test_data:
        # Evaluate on test data
        features = test_data.get('features')
        targets = test_data.get('targets')

        if features is not None and targets is not None and hasattr(trainer, 'predict'):
            predictions = trainer.predict(features)
            metrics['test_accuracy'] = r2_score(targets, predictions)
            metrics['test_mse'] = mean_squared_error(targets, predictions)
            metrics['test_mae'] = mean_absolute_error(targets, predictions)

    # Extract raw data
    raw_data = {
        'training_history': trainer.optimization_history,
        'best_parameters': trainer.best_parameters.tolist() if hasattr(trainer, 'best_parameters') else None,
        'circuit_config': config
    }

    # Add predictions if available
    if test_data and 'features' in test_data and hasattr(trainer, 'predict'):
        raw_data['predictions'] = trainer.predict(test_data['features']).tolist()
        raw_data['targets'] = test_data['targets'].tolist()

    # Register experiment
    return analytics.register_experiment(
        experiment_id=experiment_id,
        approach='quantum_vqc',
        configuration=config,
        metrics=metrics,
        raw_data=raw_data,
        metadata={'framework': 'VQCdd', 'version': '2.0'}
    )

if __name__ == "__main__":
    # Example usage and testing
    analytics = AdvancedAnalytics()

    # Create example experiments for demonstration
    logger.info("Creating example experiments for analytics demonstration...")

    # Quantum experiments
    for i in range(5):
        config = {
            'n_qubits': 4 + i,
            'n_layers': 2 + i,
            'ansatz': 'ry_cnot',
            'learning_rate': 0.1 - i * 0.01
        }

        metrics = {
            'accuracy': 0.85 + np.random.normal(0, 0.05),
            'training_time': 100 + np.random.normal(0, 20),
            'convergence_iterations': 50 + np.random.randint(-10, 10)
        }

        raw_data = {
            'training_history': [{'cost': 1.0 - j * 0.1} for j in range(10)],
            'predictions': np.random.normal(0.8, 0.1, 50).tolist(),
            'targets': np.random.normal(0.8, 0.1, 50).tolist()
        }

        analytics.register_experiment(
            experiment_id=f'quantum_exp_{i}',
            approach='quantum_vqc',
            configuration=config,
            metrics=metrics,
            raw_data=raw_data
        )

    # Classical experiments
    for i in range(5):
        config = {
            'model_type': 'random_forest',
            'n_estimators': 100 + i * 20,
            'max_depth': 5 + i
        }

        metrics = {
            'accuracy': 0.80 + np.random.normal(0, 0.03),
            'training_time': 50 + np.random.normal(0, 10),
            'convergence_iterations': 30 + np.random.randint(-5, 5)
        }

        raw_data = {
            'predictions': np.random.normal(0.75, 0.1, 50).tolist(),
            'targets': np.random.normal(0.75, 0.1, 50).tolist()
        }

        analytics.register_experiment(
            experiment_id=f'classical_exp_{i}',
            approach='classical_ml',
            configuration=config,
            metrics=metrics,
            raw_data=raw_data
        )

    # Perform quantum advantage analysis
    logger.info("Performing quantum advantage analysis...")
    quantum_ids = [f'quantum_exp_{i}' for i in range(5)]
    classical_ids = [f'classical_exp_{i}' for i in range(5)]

    advantage_report = analytics.compute_quantum_advantage(quantum_ids, classical_ids)

    logger.info("Quantum Advantage Analysis Results:")
    for metric, advantage in advantage_report.advantage_metrics.items():
        significance = advantage_report.statistical_significance[metric]
        logger.info(f"  {metric.value}: {advantage:.3f} (p={significance['p_value']:.3f})")

    # Generate scientific insights
    logger.info("Generating scientific insights...")
    insights = analytics.generate_scientific_insights()

    logger.info(f"Generated {len(insights['recommendations'])} recommendations")
    for rec in insights['recommendations']:
        logger.info(f"  - {rec}")

    logger.info("Phase 2D Advanced Analytics implementation completed successfully!")