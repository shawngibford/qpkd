"""
Data validation utilities for PK/PD datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings

from ..quantum.core.data_structures import PKPDData


class DataValidator:
    """Validation utilities for PK/PD data quality and integrity."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: If True, raises exceptions for warnings
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)
        
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive validation of raw dataset.
        
        Args:
            df: Raw dataset DataFrame
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check required columns
        required_cols = ['ID', 'TIME', 'DOSE', 'CONC', 'BIOMARKER', 'WEIGHT', 'AGE']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            error = f"Missing required columns: {missing_cols}"
            results['errors'].append(error)
            results['valid'] = False
            if self.strict_mode:
                raise ValueError(error)
                
        # Check for duplicate entries
        duplicates = df.duplicated(subset=['ID', 'TIME'])
        if duplicates.any():
            warning = f"Found {duplicates.sum()} duplicate time points"
            results['warnings'].append(warning)
            if self.strict_mode:
                raise ValueError(warning)
                
        # Validate data types and ranges
        self._validate_numeric_columns(df, results)
        self._validate_time_series(df, results)
        self._validate_biological_plausibility(df, results)
        
        # Generate statistics
        results['statistics'] = self._generate_dataset_statistics(df)
        
        return results
        
    def validate_pkpd_data(self, data: PKPDData) -> Dict[str, Any]:
        """Validate processed PKPDData object.
        
        Args:
            data: PKPDData object
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check dimensions consistency
        n_subjects = len(data.subjects)
        if data.features.shape[0] != n_subjects:
            error = f"Feature dimension mismatch: {data.features.shape[0]} vs {n_subjects} subjects"
            results['errors'].append(error)
            results['valid'] = False
            
        if data.concentrations.shape[0] != n_subjects:
            error = f"Concentration dimension mismatch: {data.concentrations.shape[0]} vs {n_subjects} subjects"
            results['errors'].append(error)
            results['valid'] = False
            
        if data.biomarkers.shape[0] != n_subjects:
            error = f"Biomarker dimension mismatch: {data.biomarkers.shape[0]} vs {n_subjects} subjects"
            results['errors'].append(error)
            results['valid'] = False
            
        # Check for NaN or infinite values
        self._check_for_invalid_values(data, results)
        
        # Validate quantum encoding compatibility
        self._validate_quantum_encoding(data, results)
        
        # Generate statistics
        results['statistics'] = self._generate_pkpd_statistics(data)
        
        return results
        
    def _validate_numeric_columns(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Validate numeric columns for proper data types and ranges."""
        numeric_cols = ['TIME', 'DOSE', 'CONC', 'BIOMARKER', 'WEIGHT', 'AGE']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
                
            # Check for non-numeric data
            if not pd.api.types.is_numeric_dtype(df[col]):
                error = f"Column {col} should be numeric"
                results['errors'].append(error)
                results['valid'] = False
                continue
                
            # Check for negative values where inappropriate
            if col in ['TIME', 'DOSE', 'CONC', 'WEIGHT', 'AGE'] and (df[col] < 0).any():
                warning = f"Found negative values in {col}"
                results['warnings'].append(warning)
                
            # Check for extremely large values (potential outliers)
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            
            if col == 'CONC' and q99 > 1000:
                warning = f"Very high concentrations detected (max: {df[col].max():.2f})"
                results['warnings'].append(warning)
                
            if col == 'BIOMARKER' and q99 > 100:
                warning = f"Very high biomarker values detected (max: {df[col].max():.2f})"
                results['warnings'].append(warning)
                
    def _validate_time_series(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Validate time series structure and consistency."""
        for subject_id in df['ID'].unique():
            subject_data = df[df['ID'] == subject_id].sort_values('TIME')
            
            # Check for non-monotonic time
            if not subject_data['TIME'].is_monotonic_increasing:
                warning = f"Non-monotonic time series for subject {subject_id}"
                results['warnings'].append(warning)
                
            # Check for missing baseline (t=0)
            if subject_data['TIME'].min() > 0:
                warning = f"Missing baseline measurement for subject {subject_id}"
                results['warnings'].append(warning)
                
            # Check for reasonable sampling frequency
            time_diffs = subject_data['TIME'].diff().dropna()
            if time_diffs.min() < 0.1:  # Less than 0.1 time units
                warning = f"Very high sampling frequency for subject {subject_id}"
                results['warnings'].append(warning)
                
    def _validate_biological_plausibility(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Check for biologically plausible values."""
        
        # Age ranges
        if 'AGE' in df.columns:
            age_range = (df['AGE'].min(), df['AGE'].max())
            if age_range[0] < 18 or age_range[1] > 100:
                warning = f"Unusual age range: {age_range[0]:.1f} - {age_range[1]:.1f} years"
                results['warnings'].append(warning)
                
        # Weight ranges
        if 'WEIGHT' in df.columns:
            weight_range = (df['WEIGHT'].min(), df['WEIGHT'].max())
            if weight_range[0] < 30 or weight_range[1] > 200:
                warning = f"Unusual weight range: {weight_range[0]:.1f} - {weight_range[1]:.1f} kg"
                results['warnings'].append(warning)
                
        # Dose ranges
        if 'DOSE' in df.columns:
            dose_range = (df['DOSE'].min(), df['DOSE'].max())
            if dose_range[1] / dose_range[0] > 100:  # More than 100-fold dose range
                warning = f"Very wide dose range: {dose_range[0]:.2f} - {dose_range[1]:.2f}"
                results['warnings'].append(warning)
                
    def _check_for_invalid_values(self, data: PKPDData, results: Dict[str, Any]) -> None:
        """Check for NaN or infinite values in processed data."""
        
        arrays_to_check = [
            ('features', data.features),
            ('concentrations', data.concentrations),
            ('biomarkers', data.biomarkers)
        ]
        
        for name, array in arrays_to_check:
            if np.isnan(array).any():
                error = f"NaN values found in {name}"
                results['errors'].append(error)
                results['valid'] = False
                
            if np.isinf(array).any():
                error = f"Infinite values found in {name}"
                results['errors'].append(error)
                results['valid'] = False
                
    def _validate_quantum_encoding(self, data: PKPDData, results: Dict[str, Any]) -> None:
        """Validate data for quantum circuit encoding."""
        
        # Check if values are in reasonable range for quantum encoding
        max_feature_val = np.max(np.abs(data.features))
        if max_feature_val > 10:
            warning = f"Large feature values may cause quantum encoding issues (max: {max_feature_val:.2f})"
            results['warnings'].append(warning)
            
        # Check for zero variance features
        feature_vars = np.var(data.features, axis=0)
        zero_var_features = np.sum(feature_vars < 1e-10)
        if zero_var_features > 0:
            warning = f"Found {zero_var_features} features with near-zero variance"
            results['warnings'].append(warning)
            
    def _generate_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics."""
        stats = {
            'n_subjects': df['ID'].nunique() if 'ID' in df.columns else 0,
            'n_observations': len(df),
            'time_range': (df['TIME'].min(), df['TIME'].max()) if 'TIME' in df.columns else None,
            'dose_statistics': df['DOSE'].describe().to_dict() if 'DOSE' in df.columns else None,
            'biomarker_statistics': df['BIOMARKER'].describe().to_dict() if 'BIOMARKER' in df.columns else None,
            'missing_values': df.isnull().sum().to_dict()
        }
        return stats
        
    def _generate_pkpd_statistics(self, data: PKPDData) -> Dict[str, Any]:
        """Generate statistics for processed PKPDData."""
        stats = {
            'n_subjects': len(data.subjects),
            'feature_shape': data.features.shape,
            'concentration_shape': data.concentrations.shape,
            'biomarker_shape': data.biomarkers.shape,
            'feature_means': np.mean(data.features, axis=0).tolist(),
            'feature_stds': np.std(data.features, axis=0).tolist(),
            'biomarker_range': (float(np.min(data.biomarkers)), float(np.max(data.biomarkers))),
            'metadata': data.metadata
        }
        return stats
        
    def check_quantum_readiness(self, data: PKPDData) -> bool:
        """Check if data is ready for quantum processing.
        
        Args:
            data: PKPDData object
            
        Returns:
            True if data is quantum-ready
        """
        validation_results = self.validate_pkpd_data(data)
        
        if not validation_results['valid']:
            self.logger.error("Data validation failed")
            return False
            
        # Additional quantum-specific checks
        if data.features.shape[1] > 20:
            self.logger.warning("High-dimensional features may require dimensionality reduction")
            
        if len(data.subjects) < 10:
            self.logger.warning("Very small dataset may not benefit from quantum methods")
            
        return True