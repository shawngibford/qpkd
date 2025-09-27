"""
Validation Framework Module for VQCdd Phase 2C

This module implements comprehensive validation capabilities for quantum
pharmacokinetic models, including cross-validation, statistical testing,
and generalization analysis. It provides robust tools for model assessment
and comparison across different data distributions.

Key Features:
- K-fold cross-validation with stratified sampling
- Bootstrap confidence intervals and hypothesis testing
- Statistical significance testing (t-tests, Wilcoxon, permutation tests)
- Generalization analysis across population subgroups
- Model comparison framework with effect size calculations
- Real data integration preparation and validation
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
from itertools import combinations

# Statistical and ML libraries
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import bootstrap, ttest_rel, wilcoxon, permutation_test
import matplotlib.pyplot as plt
import seaborn as sns

# Import VQCdd modules
from data_handler import StudyData, PatientData, QuantumFeatureEncoder
from optimizer import VQCTrainer, OptimizationConfig
from quantum_circuit import VQCircuit, CircuitConfig
from parameter_mapping import QuantumParameterMapper, ParameterBounds
from pkpd_models import PKPDModel, TwoCompartmentPK, InhibitoryEmaxPD


@dataclass
class ValidationConfig:
    """Configuration for validation procedures"""

    # Cross-validation parameters
    n_folds: int = 5                           # Number of CV folds
    stratify_by: Optional[str] = "weight_group"  # Stratification variable
    random_state: int = 42                     # Random seed for reproducibility
    shuffle: bool = True                       # Shuffle data before splitting

    # Bootstrap parameters
    bootstrap_samples: int = 1000              # Number of bootstrap samples
    confidence_level: float = 0.95            # Confidence level for intervals
    bootstrap_method: str = "percentile"      # "percentile", "bca", "basic"

    # Statistical testing parameters
    alpha: float = 0.05                       # Significance level
    correction_method: str = "bonferroni"     # Multiple testing correction
    effect_size_threshold: float = 0.2        # Small effect size threshold

    # Generalization analysis parameters
    test_population_splits: List[str] = field(default_factory=lambda: [
        "weight_standard",     # 50-100 kg
        "weight_extended",     # 70-140 kg
        "concomitant_yes",     # With concomitant medication
        "concomitant_no",      # Without concomitant medication
        "age_young",           # <65 years (if available)
        "age_elderly"          # >=65 years (if available)
    ])

    # Model comparison parameters
    compare_classical: bool = True             # Compare with classical methods
    compare_noise_models: bool = True          # Compare ideal vs noisy quantum
    performance_metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "r2", "mape", "concordance"
    ])

    # Validation output parameters
    save_detailed_results: bool = True         # Save detailed fold results
    generate_plots: bool = True               # Generate validation plots
    output_dir: str = "validation_results"    # Output directory


@dataclass
class ValidationResults:
    """Container for validation results"""

    # Cross-validation results
    cv_scores: Dict[str, np.ndarray]          # CV scores for each metric
    cv_mean: Dict[str, float]                 # Mean CV scores
    cv_std: Dict[str, float]                  # Standard deviation of CV scores
    cv_confidence_intervals: Dict[str, Tuple[float, float]]  # Bootstrap CIs

    # Statistical testing results
    statistical_tests: Dict[str, Dict[str, Any]]  # Test results
    effect_sizes: Dict[str, float]            # Effect sizes (Cohen's d)
    significance_summary: Dict[str, bool]     # Significance flags

    # Generalization analysis
    generalization_scores: Dict[str, Dict[str, float]]  # Scores by population
    population_comparisons: Dict[str, Dict[str, Any]]   # Statistical comparisons

    # Model comparison results
    model_comparison: Dict[str, Dict[str, float]]  # Performance by model type
    ranking: List[Tuple[str, float]]          # Model ranking by performance

    # Additional metadata
    validation_config: ValidationConfig       # Configuration used
    training_history: Dict[str, Any]          # Training details
    timestamp: str                            # Validation timestamp


class BaseValidator(ABC):
    """Abstract base class for validation procedures"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate(self, model: Any, data: StudyData) -> ValidationResults:
        """Perform validation procedure"""
        pass

    def _setup_output_directory(self) -> Path:
        """Create output directory for validation results"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


class KFoldCrossValidator(BaseValidator):
    """K-fold cross-validation with stratified sampling"""

    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.fold_results = []

    def validate(self, trainer_factory: Callable, data: StudyData) -> ValidationResults:
        """
        Perform K-fold cross-validation

        Args:
            trainer_factory: Function that creates a VQCTrainer instance
            data: Study data for validation

        Returns:
            ValidationResults containing CV scores and statistics
        """
        self.logger.info(f"Starting {self.config.n_folds}-fold cross-validation")

        # Prepare data for stratification
        X, y, stratify_labels = self._prepare_data_for_cv(data)

        # Create cross-validation splitter
        if self.config.stratify_by and stratify_labels is not None:
            cv_splitter = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            splits = list(cv_splitter.split(X, stratify_labels))
        else:
            cv_splitter = KFold(
                n_splits=self.config.n_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            splits = list(cv_splitter.split(X))

        # Perform cross-validation
        fold_scores = defaultdict(list)
        fold_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            self.logger.info(f"Processing fold {fold_idx + 1}/{self.config.n_folds}")

            # Split data
            train_data = self._create_subset_data(data, train_idx)
            val_data = self._create_subset_data(data, val_idx)

            # Train model
            trainer = trainer_factory()
            training_result = trainer.train(train_data)

            # Evaluate on validation set
            val_scores = self._evaluate_model(trainer, val_data)

            # Store results
            for metric, score in val_scores.items():
                fold_scores[metric].append(score)

            fold_models.append({
                'trainer': trainer,
                'training_result': training_result,
                'val_scores': val_scores,
                'train_indices': train_idx,
                'val_indices': val_idx
            })

        self.fold_results = fold_models

        # Calculate cross-validation statistics
        cv_results = self._calculate_cv_statistics(fold_scores)

        # Perform bootstrap confidence intervals
        bootstrap_cis = self._bootstrap_confidence_intervals(fold_scores)
        cv_results.cv_confidence_intervals = bootstrap_cis

        self.logger.info("Cross-validation completed successfully")
        return cv_results

    def _prepare_data_for_cv(self, data: StudyData) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Prepare data for cross-validation splitting"""
        # Extract features and targets
        n_patients = len(data.patients)
        patient_features = []
        targets = []
        stratify_labels = None

        for patient in data.patients:
            # Simple feature representation for CV splitting
            features = [patient.body_weight, float(patient.concomitant_med)]
            patient_features.append(features)

            # Use mean PD biomarker as target (handle NaN values)
            valid_biomarkers = patient.pd_biomarkers[~np.isnan(patient.pd_biomarkers)]
            if len(valid_biomarkers) > 0:
                targets.append(np.mean(valid_biomarkers))
            else:
                targets.append(0.0)  # Fallback for missing data

        X = np.array(patient_features)
        y = np.array(targets)

        # Create stratification labels if specified
        if self.config.stratify_by == "weight_group":
            stratify_labels = np.array([
                0 if patient.body_weight < 75 else 1
                for patient in data.patients
            ])
        elif self.config.stratify_by == "concomitant":
            stratify_labels = np.array([
                int(patient.concomitant_med)
                for patient in data.patients
            ])

        return X, y, stratify_labels

    def _create_subset_data(self, data: StudyData, indices: np.ndarray) -> StudyData:
        """Create subset of study data using given indices"""
        subset_patients = [data.patients[i] for i in indices]
        return StudyData(
            patients=subset_patients,
            study_design=data.study_design,
            data_quality=data.data_quality,
            study_metadata=data.study_metadata
        )

    def _evaluate_model(self, trainer: VQCTrainer, data: StudyData) -> Dict[str, float]:
        """Evaluate trained model on validation data"""
        scores = {}

        # Get model predictions
        predictions = []
        targets = []

        for patient in data.patients:
            # Get patient features for prediction
            features, _, _ = trainer.feature_encoder.encode_patient_data(patient)
            patient_features = features

            # Make prediction
            pred = trainer.predict_parameters(patient_features)
            predictions.append(pred)

            # Get target (simplified - use mean biomarker suppression)
            valid_biomarkers = patient.pd_biomarkers[~np.isnan(patient.pd_biomarkers)]
            if len(valid_biomarkers) > 0:
                targets.append(np.mean(valid_biomarkers))
            else:
                targets.append(0.0)

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Calculate metrics
        if "mse" in self.config.performance_metrics:
            scores["mse"] = mean_squared_error(targets, predictions)

        if "mae" in self.config.performance_metrics:
            scores["mae"] = mean_absolute_error(targets, predictions)

        if "r2" in self.config.performance_metrics:
            scores["r2"] = r2_score(targets, predictions)

        if "mape" in self.config.performance_metrics:
            scores["mape"] = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

        if "concordance" in self.config.performance_metrics:
            scores["concordance"] = self._calculate_concordance(targets, predictions)

        return scores

    def _calculate_concordance(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate concordance correlation coefficient"""
        # Pearson correlation coefficient
        r = np.corrcoef(targets, predictions)[0, 1]

        # Means and variances
        mean_true = np.mean(targets)
        mean_pred = np.mean(predictions)
        var_true = np.var(targets)
        var_pred = np.var(predictions)

        # Concordance correlation coefficient
        concordance = (2 * r * np.sqrt(var_true) * np.sqrt(var_pred)) / (
            var_true + var_pred + (mean_true - mean_pred)**2
        )

        return concordance

    def _calculate_cv_statistics(self, fold_scores: Dict[str, List[float]]) -> ValidationResults:
        """Calculate cross-validation statistics"""
        cv_scores = {metric: np.array(scores) for metric, scores in fold_scores.items()}
        cv_mean = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        cv_std = {metric: np.std(scores) for metric, scores in cv_scores.items()}

        return ValidationResults(
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            cv_confidence_intervals={},  # Will be filled by bootstrap
            statistical_tests={},
            effect_sizes={},
            significance_summary={},
            generalization_scores={},
            population_comparisons={},
            model_comparison={},
            ranking=[],
            validation_config=self.config,
            training_history={},
            timestamp=pd.Timestamp.now().isoformat()
        )

    def _bootstrap_confidence_intervals(self, fold_scores: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for CV scores"""
        confidence_intervals = {}

        for metric, scores in fold_scores.items():
            # Create bootstrap distribution
            bootstrap_samples = []
            scores_array = np.array(scores)

            for _ in range(self.config.bootstrap_samples):
                # Resample with replacement
                bootstrap_sample = np.random.choice(
                    scores_array,
                    size=len(scores_array),
                    replace=True
                )
                bootstrap_samples.append(np.mean(bootstrap_sample))

            bootstrap_samples = np.array(bootstrap_samples)

            # Calculate confidence interval
            alpha = 1 - self.config.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_samples, lower_percentile)
            ci_upper = np.percentile(bootstrap_samples, upper_percentile)

            confidence_intervals[metric] = (ci_lower, ci_upper)

        return confidence_intervals


class StatisticalValidator(BaseValidator):
    """Statistical hypothesis testing and significance analysis"""

    def __init__(self, config: ValidationConfig):
        super().__init__(config)

    def validate(self, results_dict: Dict[str, ValidationResults]) -> Dict[str, Any]:
        """
        Perform statistical validation comparing multiple models

        Args:
            results_dict: Dictionary mapping model names to ValidationResults

        Returns:
            Dictionary containing statistical test results
        """
        self.logger.info("Starting statistical validation")

        # Perform pairwise comparisons
        pairwise_tests = self._perform_pairwise_tests(results_dict)

        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(results_dict)

        # Multiple testing correction
        corrected_results = self._apply_multiple_testing_correction(pairwise_tests)

        # Generate summary
        significance_summary = self._generate_significance_summary(corrected_results)

        statistical_results = {
            'pairwise_tests': pairwise_tests,
            'corrected_tests': corrected_results,
            'effect_sizes': effect_sizes,
            'significance_summary': significance_summary,
            'alpha_level': self.config.alpha,
            'correction_method': self.config.correction_method
        }

        self.logger.info("Statistical validation completed")
        return statistical_results

    def _perform_pairwise_tests(self, results_dict: Dict[str, ValidationResults]) -> Dict[str, Dict[str, Any]]:
        """Perform pairwise statistical tests between models"""
        pairwise_tests = {}
        model_names = list(results_dict.keys())

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                pair_key = f"{model1}_vs_{model2}"
                pairwise_tests[pair_key] = {}

                # Test each metric
                for metric in self.config.performance_metrics:
                    if metric in results_dict[model1].cv_scores and metric in results_dict[model2].cv_scores:
                        scores1 = results_dict[model1].cv_scores[metric]
                        scores2 = results_dict[model2].cv_scores[metric]

                        # Paired t-test
                        t_stat, t_pvalue = ttest_rel(scores1, scores2)

                        # Wilcoxon signed-rank test (non-parametric)
                        try:
                            w_stat, w_pvalue = wilcoxon(scores1, scores2)
                        except ValueError:
                            w_stat, w_pvalue = np.nan, np.nan

                        # Permutation test
                        def test_statistic(x, y):
                            return np.mean(x) - np.mean(y)

                        try:
                            perm_result = permutation_test(
                                (scores1, scores2),
                                test_statistic,
                                n_resamples=1000,
                                random_state=self.config.random_state
                            )
                            perm_pvalue = perm_result.pvalue
                        except:
                            perm_pvalue = np.nan

                        pairwise_tests[pair_key][metric] = {
                            't_test': {'statistic': t_stat, 'pvalue': t_pvalue},
                            'wilcoxon': {'statistic': w_stat, 'pvalue': w_pvalue},
                            'permutation': {'pvalue': perm_pvalue},
                            'mean_diff': np.mean(scores1) - np.mean(scores2)
                        }

        return pairwise_tests

    def _calculate_effect_sizes(self, results_dict: Dict[str, ValidationResults]) -> Dict[str, Dict[str, float]]:
        """Calculate Cohen's d effect sizes for model comparisons"""
        effect_sizes = {}
        model_names = list(results_dict.keys())

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                pair_key = f"{model1}_vs_{model2}"
                effect_sizes[pair_key] = {}

                for metric in self.config.performance_metrics:
                    if metric in results_dict[model1].cv_scores and metric in results_dict[model2].cv_scores:
                        scores1 = results_dict[model1].cv_scores[metric]
                        scores2 = results_dict[model2].cv_scores[metric]

                        # Cohen's d
                        pooled_std = np.sqrt(
                            ((len(scores1) - 1) * np.var(scores1, ddof=1) +
                             (len(scores2) - 1) * np.var(scores2, ddof=1)) /
                            (len(scores1) + len(scores2) - 2)
                        )

                        if pooled_std > 0:
                            cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                        else:
                            cohens_d = 0.0

                        effect_sizes[pair_key][metric] = cohens_d

        return effect_sizes

    def _apply_multiple_testing_correction(self, pairwise_tests: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Apply multiple testing correction to p-values"""
        corrected_tests = {}

        # Collect all p-values for correction
        all_pvalues = []
        test_info = []

        for pair_key, pair_tests in pairwise_tests.items():
            for metric, tests in pair_tests.items():
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict) and 'pvalue' in test_result:
                        all_pvalues.append(test_result['pvalue'])
                        test_info.append((pair_key, metric, test_name))

        all_pvalues = np.array(all_pvalues)

        # Apply correction
        if self.config.correction_method == "bonferroni":
            corrected_pvalues = all_pvalues * len(all_pvalues)
            corrected_pvalues = np.minimum(corrected_pvalues, 1.0)
        elif self.config.correction_method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(all_pvalues)
            corrected_pvalues = np.zeros_like(all_pvalues)

            for i, idx in enumerate(sorted_indices):
                corrected_pvalues[idx] = all_pvalues[idx] * (len(all_pvalues) - i)
                if i > 0:
                    corrected_pvalues[idx] = max(
                        corrected_pvalues[idx],
                        corrected_pvalues[sorted_indices[i-1]]
                    )
            corrected_pvalues = np.minimum(corrected_pvalues, 1.0)
        else:
            # No correction
            corrected_pvalues = all_pvalues

        # Rebuild corrected results
        for i, (pair_key, metric, test_name) in enumerate(test_info):
            if pair_key not in corrected_tests:
                corrected_tests[pair_key] = {}
            if metric not in corrected_tests[pair_key]:
                corrected_tests[pair_key][metric] = {}

            original_result = pairwise_tests[pair_key][metric][test_name].copy()
            original_result['corrected_pvalue'] = corrected_pvalues[i]
            original_result['significant'] = corrected_pvalues[i] < self.config.alpha

            corrected_tests[pair_key][metric][test_name] = original_result

        return corrected_tests

    def _generate_significance_summary(self, corrected_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of significant differences"""
        summary = {
            'significant_pairs': [],
            'non_significant_pairs': [],
            'metrics_with_differences': set(),
            'most_significant_test': None,
            'least_significant_test': None
        }

        min_pvalue = float('inf')
        max_pvalue = 0.0

        for pair_key, pair_tests in corrected_results.items():
            pair_significant = False

            for metric, tests in pair_tests.items():
                for test_name, test_result in tests.items():
                    if test_result.get('significant', False):
                        pair_significant = True
                        summary['metrics_with_differences'].add(metric)

                        # Track most/least significant
                        pvalue = test_result['corrected_pvalue']
                        if pvalue < min_pvalue:
                            min_pvalue = pvalue
                            summary['most_significant_test'] = {
                                'pair': pair_key,
                                'metric': metric,
                                'test': test_name,
                                'pvalue': pvalue
                            }
                        if pvalue > max_pvalue:
                            max_pvalue = pvalue
                            summary['least_significant_test'] = {
                                'pair': pair_key,
                                'metric': metric,
                                'test': test_name,
                                'pvalue': pvalue
                            }

            if pair_significant:
                summary['significant_pairs'].append(pair_key)
            else:
                summary['non_significant_pairs'].append(pair_key)

        summary['metrics_with_differences'] = list(summary['metrics_with_differences'])
        return summary


class GeneralizationAnalyzer(BaseValidator):
    """Analyze model generalization across different population subgroups"""

    def __init__(self, config: ValidationConfig):
        super().__init__(config)

    def validate(self, trainer: VQCTrainer, data: StudyData) -> Dict[str, Any]:
        """
        Analyze generalization across different population splits

        Args:
            trainer: Trained VQC model
            data: Complete study data

        Returns:
            Dictionary containing generalization analysis results
        """
        self.logger.info("Starting generalization analysis")

        # Create population splits
        population_splits = self._create_population_splits(data)

        # Evaluate on each population
        population_scores = {}
        for split_name, split_data in population_splits.items():
            if len(split_data.patients) > 0:
                scores = self._evaluate_on_population(trainer, split_data)
                population_scores[split_name] = scores
                self.logger.info(f"Evaluated on {split_name}: {len(split_data.patients)} patients")

        # Perform population comparisons
        population_comparisons = self._compare_populations(population_scores)

        # Analyze population bias
        bias_analysis = self._analyze_population_bias(population_scores)

        generalization_results = {
            'population_scores': population_scores,
            'population_comparisons': population_comparisons,
            'bias_analysis': bias_analysis,
            'population_sizes': {
                name: len(split.patients)
                for name, split in population_splits.items()
            }
        }

        self.logger.info("Generalization analysis completed")
        return generalization_results

    def _create_population_splits(self, data: StudyData) -> Dict[str, StudyData]:
        """Create different population subgroups for analysis"""
        splits = {}

        for split_name in self.config.test_population_splits:
            if split_name == "weight_standard":
                # Standard weight range: 50-100 kg
                patients = [p for p in data.patients if 50 <= p.body_weight <= 100]
            elif split_name == "weight_extended":
                # Extended weight range: 70-140 kg
                patients = [p for p in data.patients if 70 <= p.body_weight <= 140]
            elif split_name == "concomitant_yes":
                # With concomitant medication
                patients = [p for p in data.patients if p.concomitant_med]
            elif split_name == "concomitant_no":
                # Without concomitant medication
                patients = [p for p in data.patients if not p.concomitant_med]
            elif split_name == "weight_low":
                # Lower weight quartile
                weights = [p.body_weight for p in data.patients]
                weight_25th = np.percentile(weights, 25)
                patients = [p for p in data.patients if p.body_weight <= weight_25th]
            elif split_name == "weight_high":
                # Upper weight quartile
                weights = [p.body_weight for p in data.patients]
                weight_75th = np.percentile(weights, 75)
                patients = [p for p in data.patients if p.body_weight >= weight_75th]
            else:
                # Default: use all patients
                patients = data.patients

            if patients:
                splits[split_name] = StudyData(
                    patients=patients,
                    study_metadata=data.study_metadata
                )

        return splits

    def _evaluate_on_population(self, trainer: VQCTrainer, data: StudyData) -> Dict[str, float]:
        """Evaluate model performance on specific population"""
        predictions = []
        targets = []

        for patient in data.patients:
            # Get patient features
            features, _, _ = trainer.feature_encoder.encode_patient_data(patient)
            patient_features = features

            # Make prediction
            pred = trainer.predict_parameters(patient_features)
            predictions.append(pred)

            # Get target
            valid_biomarkers = patient.pd_biomarkers[~np.isnan(patient.pd_biomarkers)]
            if len(valid_biomarkers) > 0:
                targets.append(np.mean(valid_biomarkers))
            else:
                targets.append(0.0)

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Calculate metrics
        scores = {}
        scores["mse"] = mean_squared_error(targets, predictions)
        scores["mae"] = mean_absolute_error(targets, predictions)
        scores["r2"] = r2_score(targets, predictions)
        scores["mape"] = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

        return scores

    def _compare_populations(self, population_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Compare performance across different populations"""
        comparisons = {}

        population_names = list(population_scores.keys())

        for i, pop1 in enumerate(population_names):
            for j, pop2 in enumerate(population_names[i+1:], i+1):
                comparison_key = f"{pop1}_vs_{pop2}"
                comparisons[comparison_key] = {}

                for metric in self.config.performance_metrics:
                    if metric in population_scores[pop1] and metric in population_scores[pop2]:
                        score1 = population_scores[pop1][metric]
                        score2 = population_scores[pop2][metric]

                        # Calculate relative difference
                        rel_diff = (score1 - score2) / (abs(score2) + 1e-8) * 100

                        comparisons[comparison_key][metric] = {
                            'score1': score1,
                            'score2': score2,
                            'absolute_diff': score1 - score2,
                            'relative_diff_percent': rel_diff,
                            'better_population': pop1 if score1 < score2 else pop2  # Assuming lower is better
                        }

        return comparisons

    def _analyze_population_bias(self, population_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze potential bias across populations"""
        bias_analysis = {}

        # Calculate coefficient of variation across populations
        for metric in self.config.performance_metrics:
            metric_scores = []
            for pop_name, scores in population_scores.items():
                if metric in scores:
                    metric_scores.append(scores[metric])

            if len(metric_scores) > 1:
                mean_score = np.mean(metric_scores)
                std_score = np.std(metric_scores)
                cv = std_score / (abs(mean_score) + 1e-8) * 100

                bias_analysis[metric] = {
                    'coefficient_of_variation': cv,
                    'mean_across_populations': mean_score,
                    'std_across_populations': std_score,
                    'min_score': np.min(metric_scores),
                    'max_score': np.max(metric_scores),
                    'score_range': np.max(metric_scores) - np.min(metric_scores)
                }

        # Identify most/least biased metrics
        if bias_analysis:
            cvs = {metric: data['coefficient_of_variation'] for metric, data in bias_analysis.items()}
            bias_analysis['most_biased_metric'] = max(cvs, key=cvs.get)
            bias_analysis['least_biased_metric'] = min(cvs, key=cvs.get)
            bias_analysis['overall_bias_score'] = np.mean(list(cvs.values()))

        return bias_analysis


class ValidationPipeline:
    """Complete validation pipeline orchestrating all validation components"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize validators
        self.cv_validator = KFoldCrossValidator(config)
        self.statistical_validator = StatisticalValidator(config)
        self.generalization_analyzer = GeneralizationAnalyzer(config)

        # Results storage
        self.results = {}
        self.comparison_results = {}

    def run_comprehensive_validation(
        self,
        models: Dict[str, Callable],
        data: StudyData
    ) -> Dict[str, Any]:
        """
        Run complete validation pipeline

        Args:
            models: Dictionary mapping model names to trainer factory functions
            data: Study data for validation

        Returns:
            Complete validation results
        """
        self.logger.info("Starting comprehensive validation pipeline")

        # Phase 1: Cross-validation for each model
        cv_results = {}
        for model_name, trainer_factory in models.items():
            self.logger.info(f"Cross-validating model: {model_name}")
            cv_results[model_name] = self.cv_validator.validate(trainer_factory, data)

        # Phase 2: Statistical comparison
        self.logger.info("Performing statistical comparisons")
        statistical_results = self.statistical_validator.validate(cv_results)

        # Phase 3: Generalization analysis (use best model)
        best_model_name = self._identify_best_model(cv_results)
        self.logger.info(f"Analyzing generalization for best model: {best_model_name}")

        # Train best model on full dataset for generalization analysis
        best_trainer = models[best_model_name]()
        best_trainer.train(data)
        generalization_results = self.generalization_analyzer.validate(best_trainer, data)

        # Phase 4: Model ranking and recommendations
        model_ranking = self._rank_models(cv_results)
        recommendations = self._generate_recommendations(cv_results, statistical_results, generalization_results)

        # Compile comprehensive results
        comprehensive_results = {
            'cross_validation_results': cv_results,
            'statistical_analysis': statistical_results,
            'generalization_analysis': generalization_results,
            'model_ranking': model_ranking,
            'best_model': best_model_name,
            'recommendations': recommendations,
            'validation_config': self.config,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Save results if requested
        if self.config.save_detailed_results:
            self._save_results(comprehensive_results)

        # Generate plots if requested
        if self.config.generate_plots:
            self._generate_validation_plots(comprehensive_results)

        self.logger.info("Comprehensive validation pipeline completed")
        return comprehensive_results

    def _identify_best_model(self, cv_results: Dict[str, ValidationResults]) -> str:
        """Identify best performing model based on CV results"""
        # Use primary metric (first in list) for ranking
        primary_metric = self.config.performance_metrics[0]

        best_model = None
        best_score = float('inf')  # Assuming lower is better

        for model_name, results in cv_results.items():
            if primary_metric in results.cv_mean:
                score = results.cv_mean[primary_metric]
                if score < best_score:
                    best_score = score
                    best_model = model_name

        return best_model

    def _rank_models(self, cv_results: Dict[str, ValidationResults]) -> List[Tuple[str, float]]:
        """Rank models by performance"""
        primary_metric = self.config.performance_metrics[0]

        model_scores = []
        for model_name, results in cv_results.items():
            if primary_metric in results.cv_mean:
                score = results.cv_mean[primary_metric]
                model_scores.append((model_name, score))

        # Sort by score (assuming lower is better)
        model_scores.sort(key=lambda x: x[1])

        return model_scores

    def _generate_recommendations(
        self,
        cv_results: Dict[str, ValidationResults],
        statistical_results: Dict[str, Any],
        generalization_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate validation recommendations"""
        recommendations = {}

        # Model selection recommendation
        best_model = self._identify_best_model(cv_results)
        recommendations['best_model'] = f"Recommended model: {best_model}"

        # Statistical significance
        significant_pairs = statistical_results.get('significance_summary', {}).get('significant_pairs', [])
        if significant_pairs:
            recommendations['statistical_significance'] = f"Found {len(significant_pairs)} statistically significant model differences"
        else:
            recommendations['statistical_significance'] = "No statistically significant differences found between models"

        # Generalization assessment
        bias_analysis = generalization_results.get('bias_analysis', {})
        if 'overall_bias_score' in bias_analysis:
            bias_score = bias_analysis['overall_bias_score']
            if bias_score > 20:
                recommendations['generalization'] = "High population bias detected - consider model refinement"
            elif bias_score > 10:
                recommendations['generalization'] = "Moderate population bias - monitor performance across subgroups"
            else:
                recommendations['generalization'] = "Good generalization across population subgroups"

        # Sample size recommendations
        total_patients = sum(len(results.cv_scores[list(results.cv_scores.keys())[0]])
                           for results in cv_results.values() if results.cv_scores)
        if total_patients < 100:
            recommendations['sample_size'] = "Consider collecting more data - current sample may be insufficient"

        return recommendations

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save validation results to disk"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        results_file = output_dir / "validation_results.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Validation results saved to {results_file}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(v) for v in obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

    def _generate_validation_plots(self, results: Dict[str, Any]) -> None:
        """Generate validation visualization plots"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Cross-validation scores comparison
        self._plot_cv_comparison(results['cross_validation_results'], output_dir)

        # Plot 2: Population generalization analysis
        if 'generalization_analysis' in results:
            self._plot_generalization_analysis(results['generalization_analysis'], output_dir)

        # Plot 3: Model ranking
        self._plot_model_ranking(results['model_ranking'], output_dir)

        self.logger.info(f"Validation plots saved to {output_dir}")

    def _plot_cv_comparison(self, cv_results: Dict[str, ValidationResults], output_dir: Path) -> None:
        """Plot cross-validation results comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Validation Results Comparison', fontsize=16, fontweight='bold')

        metrics = self.config.performance_metrics[:4]  # Plot first 4 metrics

        for idx, metric in enumerate(metrics):
            if idx >= 4:
                break

            ax = axes[idx // 2, idx % 2]

            # Collect data for boxplot
            data_for_plot = []
            labels = []

            for model_name, results in cv_results.items():
                if metric in results.cv_scores:
                    data_for_plot.append(results.cv_scores[metric])
                    labels.append(model_name)

            if data_for_plot:
                ax.boxplot(data_for_plot, labels=labels)
                ax.set_title(f'{metric.upper()} Scores')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'cv_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_generalization_analysis(self, generalization_results: Dict[str, Any], output_dir: Path) -> None:
        """Plot generalization analysis results"""
        if 'population_scores' not in generalization_results:
            return

        population_scores = generalization_results['population_scores']

        # Create heatmap of performance across populations
        populations = list(population_scores.keys())
        metrics = self.config.performance_metrics

        # Prepare data matrix
        data_matrix = []
        for pop in populations:
            row = []
            for metric in metrics:
                if metric in population_scores[pop]:
                    row.append(population_scores[pop][metric])
                else:
                    row.append(np.nan)
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            data_matrix,
            xticklabels=metrics,
            yticklabels=populations,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r'
        )
        plt.title('Model Performance Across Population Subgroups', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Population Subgroups')
        plt.tight_layout()
        plt.savefig(output_dir / 'generalization_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_ranking(self, model_ranking: List[Tuple[str, float]], output_dir: Path) -> None:
        """Plot model ranking"""
        if not model_ranking:
            return

        models, scores = zip(*model_ranking)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(models)), scores, color='skyblue', edgecolor='navy')
        plt.xlabel('Models')
        plt.ylabel('Score (Lower is Better)')
        plt.title('Model Performance Ranking', fontsize=14, fontweight='bold')
        plt.xticks(range(len(models)), models, rotation=45)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()


# Utility functions for validation

def create_synthetic_validation_data(
    n_patients: int = 100,
    weight_range: Tuple[float, float] = (50, 120),
    concomitant_ratio: float = 0.5,
    random_state: int = 42
) -> StudyData:
    """
    Create synthetic data for validation testing

    Args:
        n_patients: Number of synthetic patients
        weight_range: Range of body weights (kg)
        concomitant_ratio: Fraction of patients with concomitant medication
        random_state: Random seed for reproducibility

    Returns:
        StudyData object with synthetic patients
    """
    np.random.seed(random_state)

    patients = []
    for i in range(n_patients):
        # Generate patient characteristics
        body_weight = np.random.uniform(weight_range[0], weight_range[1])
        concomitant_med = np.random.random() < concomitant_ratio

        # Generate time points (simulate typical PK study)
        n_timepoints = np.random.randint(5, 15)
        time_points = np.sort(np.random.uniform(0, 24, n_timepoints))

        # Generate doses (simplified dosing regimen)
        doses = np.zeros_like(time_points)
        doses[0] = np.random.uniform(10, 100)  # Initial dose

        # Generate synthetic PK concentrations with some noise
        half_life = np.random.uniform(2, 8)  # hours
        pk_concentrations = doses[0] * np.exp(-0.693 * time_points / half_life)
        pk_concentrations += np.random.normal(0, 0.1 * pk_concentrations)

        # Generate synthetic PD biomarkers (related to PK concentrations)
        baseline_biomarker = np.random.uniform(5, 10)  # ng/mL
        emax = np.random.uniform(0.5, 0.9)
        ec50 = np.random.uniform(0.5, 2.0)

        pd_biomarkers = baseline_biomarker * (1 - emax * pk_concentrations / (ec50 + pk_concentrations))
        pd_biomarkers += np.random.normal(0, 0.1 * pd_biomarkers)

        # Add some missing data
        missing_mask = np.random.random(n_timepoints) < 0.1
        pk_concentrations[missing_mask] = np.nan
        pd_biomarkers[missing_mask] = np.nan

        # Create NONMEM-compatible fields
        evid = np.zeros(n_timepoints, dtype=int)  # 0=observation, 1=dose
        evid[0] = 1  # First timepoint is dose

        mdv = missing_mask.astype(int)  # Missing dependent variable flag

        amt = np.zeros(n_timepoints)  # Amount for dosing events
        amt[0] = doses[0]  # Only first timepoint has dose amount

        cmt = np.ones(n_timepoints, dtype=int)  # Compartment (1=dose, 2=PK, 3=PD)
        cmt[pk_concentrations > 0] = 2  # PK observations
        cmt[pd_biomarkers > 0] = 3  # PD observations

        dvid = np.zeros(n_timepoints, dtype=int)  # DV type (0=dose, 1=PK, 2=PD)
        dvid[evid == 0] = 1  # Observation records get DVID=1 (PK) or 2 (PD)
        dvid[~np.isnan(pd_biomarkers)] = 2  # PD biomarker observations

        patient = PatientData(
            patient_id=i + 1,
            body_weight=body_weight,
            concomitant_med=concomitant_med,
            time_points=time_points,
            doses=doses,
            pk_concentrations=pk_concentrations,
            pd_biomarkers=pd_biomarkers,
            evid=evid,
            mdv=mdv,
            amt=amt,
            cmt=cmt,
            dvid=dvid
        )

        patients.append(patient)

    return StudyData(
        patients=patients,
        study_design={'synthetic_data': True, 'n_patients': n_patients},
        data_quality={'complete': True, 'validated': True},
        study_metadata={'source': 'validation_synthetic', 'generated_for': 'validation'}
    )


def compare_validation_results(
    results1: ValidationResults,
    results2: ValidationResults,
    config: ValidationConfig
) -> Dict[str, Any]:
    """
    Compare two validation results statistically

    Args:
        results1: First validation results
        results2: Second validation results
        config: Validation configuration

    Returns:
        Dictionary containing comparison results
    """
    statistical_validator = StatisticalValidator(config)

    comparison_dict = {
        'model1': results1,
        'model2': results2
    }

    return statistical_validator.validate(comparison_dict)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create synthetic data for testing
    synthetic_data = create_synthetic_validation_data(n_patients=50)

    # Create validation configuration
    config = ValidationConfig(
        n_folds=3,
        bootstrap_samples=100,  # Reduced for testing
        test_population_splits=["weight_standard", "concomitant_yes", "concomitant_no"]
    )

    print("Validation framework created successfully!")
    print(f"Synthetic data: {len(synthetic_data.patients)} patients")
    print(f"Validation config: {config.n_folds}-fold CV, {config.bootstrap_samples} bootstrap samples")