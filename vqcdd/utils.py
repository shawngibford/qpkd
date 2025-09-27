"""
Utility Functions for VQCdd

This module provides various utility functions for data processing, validation,
logging, metrics calculation, and scientific analysis support.

Key Features:
- Data validation and quality checks
- Statistical analysis utilities
- Pharmacokinetic calculation helpers
- Plotting and visualization utilities
- File I/O and configuration management
- Scientific computing helpers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import pickle
from pathlib import Path
import warnings
from dataclasses import asdict
import scipy.stats as stats
from scipy import optimize
import time


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for VQCdd

    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger('vqcdd')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Timer:
    """Context manager for timing code execution"""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.description} completed in {elapsed:.3f} seconds")

    def elapsed(self) -> float:
        """Get elapsed time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def validate_pk_parameters(params: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict:
    """
    Validate PK parameters against physiological bounds

    Args:
        params: PK parameters to validate
        bounds: Parameter bounds dictionary

    Returns:
        Validation report
    """
    validation_report = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'parameter_scores': {}
    }

    for param_name, value in params.items():
        if param_name in bounds:
            min_val, max_val = bounds[param_name]

            # Check bounds
            if value < min_val or value > max_val:
                validation_report['valid'] = False
                validation_report['errors'].append(
                    f"{param_name}={value:.3f} outside bounds [{min_val}, {max_val}]"
                )
                score = 0.0
            else:
                # Calculate parameter score (1.0 = center of range, decreases toward bounds)
                center = (min_val + max_val) / 2
                range_size = max_val - min_val
                distance_from_center = abs(value - center)
                score = max(0.0, 1.0 - 2 * distance_from_center / range_size)

            validation_report['parameter_scores'][param_name] = score

    # Overall quality score
    if validation_report['parameter_scores']:
        validation_report['overall_score'] = np.mean(list(validation_report['parameter_scores'].values()))
    else:
        validation_report['overall_score'] = 0.0

    return validation_report


def calculate_pharmacokinetic_metrics(time: np.ndarray, concentrations: np.ndarray,
                                    dose: float) -> Dict[str, float]:
    """
    Calculate standard pharmacokinetic metrics

    Args:
        time: Time points (hours)
        concentrations: Concentration values (mg/L)
        dose: Administered dose (mg)

    Returns:
        Dictionary of PK metrics
    """
    # Remove any non-positive concentrations for log calculations
    valid_mask = concentrations > 0
    time_valid = time[valid_mask]
    conc_valid = concentrations[valid_mask]

    if len(conc_valid) < 2:
        return {'error': 'Insufficient valid data points'}

    metrics = {}

    try:
        # Cmax - Maximum concentration
        cmax_idx = np.argmax(conc_valid)
        metrics['Cmax'] = float(conc_valid[cmax_idx])
        metrics['Tmax'] = float(time_valid[cmax_idx])

        # AUC - Area under curve (trapezoidal rule)
        metrics['AUC_0_last'] = float(np.trapz(conc_valid, time_valid))

        # Terminal half-life (using last 3-5 points)
        if len(conc_valid) >= 4:
            n_points = min(5, len(conc_valid))
            log_conc = np.log(conc_valid[-n_points:])
            time_terminal = time_valid[-n_points:]

            # Linear regression on log-concentration vs time
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_terminal, log_conc)

            if slope < 0:  # Valid elimination phase
                metrics['lambda_z'] = -slope
                metrics['t_half'] = 0.693 / (-slope)
                metrics['r_squared'] = r_value ** 2

                # Extrapolate AUC to infinity
                auc_extrap = conc_valid[-1] / (-slope)
                metrics['AUC_0_inf'] = metrics['AUC_0_last'] + auc_extrap
                metrics['percent_extrapolated'] = 100 * auc_extrap / metrics['AUC_0_inf']

        # Clearance and Volume of distribution
        if 'AUC_0_inf' in metrics:
            metrics['Clearance'] = dose / metrics['AUC_0_inf']
            if 'lambda_z' in metrics:
                metrics['Vd'] = metrics['Clearance'] / metrics['lambda_z']

        # Bioavailability-related metrics
        metrics['MRT'] = calculate_mean_residence_time(time_valid, conc_valid)

    except Exception as e:
        metrics['calculation_error'] = str(e)

    return metrics


def calculate_mean_residence_time(time: np.ndarray, concentrations: np.ndarray) -> float:
    """Calculate mean residence time"""
    # AUMC - Area under moment curve
    aumc = np.trapz(time * concentrations, time)
    auc = np.trapz(concentrations, time)

    if auc > 0:
        return aumc / auc
    else:
        return np.nan


def calculate_biomarker_metrics(time: np.ndarray, biomarker_levels: np.ndarray,
                              baseline: float, target: float) -> Dict[str, float]:
    """
    Calculate pharmacodynamic biomarker metrics

    Args:
        time: Time points (hours)
        biomarker_levels: Biomarker values (ng/mL)
        baseline: Baseline biomarker level
        target: Target biomarker level

    Returns:
        Dictionary of PD metrics
    """
    metrics = {}

    try:
        # Minimum biomarker level
        min_idx = np.argmin(biomarker_levels)
        metrics['min_biomarker'] = float(biomarker_levels[min_idx])
        metrics['time_to_min'] = float(time[min_idx])

        # Maximum suppression
        max_suppression = (baseline - metrics['min_biomarker']) / baseline
        metrics['max_suppression_percent'] = max_suppression * 100

        # Time below target
        below_target_mask = biomarker_levels < target
        if below_target_mask.any():
            time_below_target = np.sum(np.diff(time)[:-1][below_target_mask[1:]])
            metrics['time_below_target'] = float(time_below_target)

            # Time to reach target
            first_below_idx = np.where(below_target_mask)[0]
            if len(first_below_idx) > 0:
                metrics['time_to_target'] = float(time[first_below_idx[0]])
        else:
            metrics['time_below_target'] = 0.0
            metrics['time_to_target'] = np.inf

        # Area below baseline
        baseline_array = np.full_like(biomarker_levels, baseline)
        suppression_area = np.trapz(np.maximum(0, baseline_array - biomarker_levels), time)
        metrics['suppression_area'] = float(suppression_area)

        # Recovery metrics (if biomarker returns to >90% baseline)
        recovery_threshold = 0.9 * baseline
        recovery_mask = biomarker_levels > recovery_threshold
        if recovery_mask.any() and below_target_mask.any():
            recovery_indices = np.where(recovery_mask)[0]
            below_indices = np.where(below_target_mask)[0]

            if len(recovery_indices) > 0 and len(below_indices) > 0:
                last_below = below_indices[-1]
                recovery_after_suppression = recovery_indices[recovery_indices > last_below]

                if len(recovery_after_suppression) > 0:
                    metrics['time_to_recovery'] = float(time[recovery_after_suppression[0]])

    except Exception as e:
        metrics['calculation_error'] = str(e)

    return metrics


def generate_population_summary(patient_data_list: List[Dict]) -> Dict:
    """
    Generate population summary statistics

    Args:
        patient_data_list: List of patient data dictionaries

    Returns:
        Population summary
    """
    if not patient_data_list:
        return {'error': 'No patient data provided'}

    # Extract patient characteristics
    weights = [p.get('body_weight', np.nan) for p in patient_data_list]
    comed_status = [p.get('concomitant_med', False) for p in patient_data_list]

    summary = {
        'n_patients': len(patient_data_list),
        'demographics': {
            'weight_mean': np.nanmean(weights),
            'weight_std': np.nanstd(weights),
            'weight_median': np.nanmedian(weights),
            'weight_range': (np.nanmin(weights), np.nanmax(weights)),
            'concomitant_med_prevalence': np.mean(comed_status)
        }
    }

    # Age statistics if available
    ages = [p.get('age', np.nan) for p in patient_data_list]
    if not all(np.isnan(ages)):
        summary['demographics']['age_mean'] = np.nanmean(ages)
        summary['demographics']['age_std'] = np.nanstd(ages)
        summary['demographics']['age_range'] = (np.nanmin(ages), np.nanmax(ages))

    return summary


def create_pk_plot(time: np.ndarray, concentrations: np.ndarray,
                  title: str = "PK Profile", log_scale: bool = True) -> plt.Figure:
    """
    Create pharmacokinetic concentration-time plot

    Args:
        time: Time points
        concentrations: Concentration values
        title: Plot title
        log_scale: Use log scale for y-axis

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot concentration-time profile
    ax.plot(time, concentrations, 'bo-', linewidth=2, markersize=6, label='Concentration')

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Concentration (mg/L)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if log_scale and np.all(concentrations > 0):
        ax.set_yscale('log')
        ax.set_ylabel('Concentration (mg/L) - Log Scale')

    plt.tight_layout()
    return fig


def create_pd_plot(time: np.ndarray, biomarker_levels: np.ndarray,
                  baseline: Optional[float] = None, target: Optional[float] = None,
                  title: str = "PD Profile") -> plt.Figure:
    """
    Create pharmacodynamic biomarker-time plot

    Args:
        time: Time points
        biomarker_levels: Biomarker values
        baseline: Baseline level (optional)
        target: Target level (optional)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot biomarker-time profile
    ax.plot(time, biomarker_levels, 'ro-', linewidth=2, markersize=6, label='Biomarker')

    # Add reference lines
    if baseline is not None:
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({baseline})')

    if target is not None:
        ax.axhline(y=target, color='green', linestyle='--', alpha=0.7, label=f'Target ({target})')

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Biomarker Level (ng/mL)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def create_dose_response_plot(doses: np.ndarray, coverage: np.ndarray,
                            target_coverage: float = 0.9) -> plt.Figure:
    """
    Create dose-response curve plot

    Args:
        doses: Dose values
        coverage: Population coverage values
        target_coverage: Target coverage level

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot dose-response curve
    ax.plot(doses, coverage * 100, 'b-', linewidth=2, label='Population Coverage')
    ax.scatter(doses, coverage * 100, color='blue', s=50, alpha=0.7)

    # Add target line
    ax.axhline(y=target_coverage * 100, color='red', linestyle='--',
              alpha=0.7, label=f'Target ({target_coverage:.0%})')

    ax.set_xlabel('Dose (mg)')
    ax.set_ylabel('Population Coverage (%)')
    ax.set_title('Dose-Response Relationship')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set reasonable axis limits
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max(doses) * 1.1)

    plt.tight_layout()
    return fig


def save_results(results: Dict, filename: str, format: str = "json") -> None:
    """
    Save results to file

    Args:
        results: Results dictionary
        filename: Output filename
        format: File format ("json", "pickle", "csv")
    """
    filepath = Path(filename)

    if format == "json":
        # Convert numpy arrays to lists for JSON serialization
        json_results = convert_numpy_to_list(results)
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(json_results, f, indent=2)

    elif format == "pickle":
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(results, f)

    elif format == "csv":
        # Convert to DataFrame if possible
        if isinstance(results, dict) and all(isinstance(v, (list, np.ndarray)) for v in results.values()):
            df = pd.DataFrame(results)
            df.to_csv(filepath.with_suffix('.csv'), index=False)
        else:
            raise ValueError("Cannot convert results to CSV format")

    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(filename: str, format: str = "json") -> Dict:
    """
    Load results from file

    Args:
        filename: Input filename
        format: File format ("json", "pickle", "csv")

    Returns:
        Loaded results
    """
    filepath = Path(filename)

    if format == "json":
        with open(filepath, 'r') as f:
            return json.load(f)

    elif format == "pickle":
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    elif format == "csv":
        df = pd.read_csv(filepath)
        return df.to_dict('list')

    else:
        raise ValueError(f"Unsupported format: {format}")


def convert_numpy_to_list(obj: Any) -> Any:
    """
    Recursively convert numpy arrays to lists for JSON serialization

    Args:
        obj: Object to convert

    Returns:
        Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for data

    Args:
        data: Data array
        confidence: Confidence level (0-1)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1.0 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)

    return float(lower_bound), float(upper_bound)


def bootstrap_sampling(data: np.ndarray, statistic_func: callable,
                      n_bootstrap: int = 1000, confidence: float = 0.95) -> Dict:
    """
    Perform bootstrap sampling for statistical inference

    Args:
        data: Original data
        statistic_func: Function to calculate statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Bootstrap results
    """
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate confidence interval
    ci_lower, ci_upper = calculate_confidence_interval(bootstrap_stats, confidence)

    return {
        'original_statistic': statistic_func(data),
        'bootstrap_mean': np.mean(bootstrap_stats),
        'bootstrap_std': np.std(bootstrap_stats),
        'confidence_interval': (ci_lower, ci_upper),
        'bootstrap_samples': bootstrap_stats
    }


def check_data_quality(data: Union[np.ndarray, List], verbose: bool = True) -> Dict:
    """
    Perform data quality checks

    Args:
        data: Data to check
        verbose: Print detailed information

    Returns:
        Data quality report
    """
    data_array = np.asarray(data)

    report = {
        'total_points': len(data_array),
        'missing_values': np.sum(np.isnan(data_array)),
        'infinite_values': np.sum(np.isinf(data_array)),
        'negative_values': np.sum(data_array < 0),
        'zero_values': np.sum(data_array == 0),
        'unique_values': len(np.unique(data_array[np.isfinite(data_array)])),
    }

    # Calculate percentages
    total = report['total_points']
    if total > 0:
        report['missing_percent'] = (report['missing_values'] / total) * 100
        report['infinite_percent'] = (report['infinite_values'] / total) * 100

    # Data quality score
    valid_data = total - report['missing_values'] - report['infinite_values']
    report['quality_score'] = (valid_data / total) if total > 0 else 0.0

    if verbose:
        print("Data Quality Report:")
        print(f"  Total points: {report['total_points']}")
        print(f"  Missing values: {report['missing_values']} ({report.get('missing_percent', 0):.1f}%)")
        print(f"  Infinite values: {report['infinite_values']} ({report.get('infinite_percent', 0):.1f}%)")
        print(f"  Quality score: {report['quality_score']:.3f}")

    return report


if __name__ == "__main__":
    # Example usage and testing
    print("VQCdd Utilities Module")
    print("=" * 40)

    # Test timer
    with Timer("Example computation"):
        time.sleep(0.1)  # Simulate work

    # Test PK metrics calculation
    time_points = np.array([0, 1, 2, 4, 8, 12, 24])
    concentrations = np.array([0, 8.5, 7.2, 5.8, 3.9, 2.6, 0.9])
    dose = 10.0

    pk_metrics = calculate_pharmacokinetic_metrics(time_points, concentrations, dose)
    print(f"PK Metrics: {pk_metrics}")

    # Test biomarker metrics
    biomarker_levels = np.array([15.0, 12.1, 8.3, 5.2, 3.1, 2.8, 4.5])
    baseline = 15.0
    target = 3.3

    pd_metrics = calculate_biomarker_metrics(time_points, biomarker_levels, baseline, target)
    print(f"PD Metrics: {pd_metrics}")

    # Test data quality check
    test_data = np.array([1, 2, 3, np.nan, 5, np.inf, 7, 8])
    quality_report = check_data_quality(test_data)

    print("Testing completed successfully!")