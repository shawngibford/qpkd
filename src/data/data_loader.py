"""
Data loader for PK/PD clinical trial datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from ..quantum.core.data_structures import PKPDData


class PKPDDataLoader:
    """Loads and prepares PK/PD clinical trial data."""
    
    def __init__(self, data_path: str = "data/EstData.csv"):
        """Initialize data loader.
        
        Args:
            data_path: Path to the clinical trial dataset
        """
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self) -> pd.DataFrame:
        """Load the EstData.csv dataset.
        
        Returns:
            Raw dataset as pandas DataFrame
        """
        try:
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            self.logger.error(f"Dataset not found at {self.data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
            
    def prepare_pkpd_data(self, 
                         weight_range: Tuple[float, float] = (50, 100),
                         concomitant_allowed: bool = True) -> PKPDData:
        """Prepare data for PK/PD modeling.
        
        Args:
            weight_range: Body weight range (min_kg, max_kg)
            concomitant_allowed: Whether concomitant medication is allowed
            
        Returns:
            Structured PKPDData object
        """
        df = self.load_dataset()
        
        # Filter by weight range
        df_filtered = df[
            (df['WEIGHT'] >= weight_range[0]) & 
            (df['WEIGHT'] <= weight_range[1])
        ].copy()
        
        # Filter by concomitant medication
        if not concomitant_allowed:
            df_filtered = df_filtered[df_filtered['CONMED'] == 0]
            
        self.logger.info(f"Filtered to {len(df_filtered)} subjects")
        
        # Extract features and targets
        features = self._extract_features(df_filtered)
        concentrations = self._extract_concentrations(df_filtered)
        biomarkers = self._extract_biomarkers(df_filtered)
        
        return PKPDData(
            subjects=df_filtered['ID'].unique().tolist(),
            features=features,
            concentrations=concentrations,
            biomarkers=biomarkers,
            metadata={
                'weight_range': weight_range,
                'concomitant_allowed': concomitant_allowed,
                'n_subjects': len(df_filtered['ID'].unique())
            }
        )
        
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract subject features for modeling."""
        feature_cols = ['WEIGHT', 'AGE', 'SEX', 'DOSE', 'CONMED']
        features = df[feature_cols].groupby('ID').first().values
        return features.astype(np.float32)
        
    def _extract_concentrations(self, df: pd.DataFrame) -> np.ndarray:
        """Extract drug concentration time series."""
        conc_data = []
        for subject_id in df['ID'].unique():
            subject_data = df[df['ID'] == subject_id].sort_values('TIME')
            concentrations = subject_data['CONC'].values
            conc_data.append(concentrations)
        
        # Pad to same length
        max_len = max(len(c) for c in conc_data)
        padded_conc = np.zeros((len(conc_data), max_len))
        for i, conc in enumerate(conc_data):
            padded_conc[i, :len(conc)] = conc
            
        return padded_conc.astype(np.float32)
        
    def _extract_biomarkers(self, df: pd.DataFrame) -> np.ndarray:
        """Extract biomarker measurements."""
        biomarker_data = []
        for subject_id in df['ID'].unique():
            subject_data = df[df['ID'] == subject_id].sort_values('TIME')
            biomarkers = subject_data['BIOMARKER'].values
            biomarker_data.append(biomarkers)
            
        # Pad to same length
        max_len = max(len(b) for b in biomarker_data)
        padded_biomarkers = np.zeros((len(biomarker_data), max_len))
        for i, bio in enumerate(biomarker_data):
            padded_biomarkers[i, :len(bio)] = bio
            
        return padded_biomarkers.astype(np.float32)
        
    def get_scenario_data(self, scenario: str) -> PKPDData:
        """Get data for specific challenge scenarios.
        
        Args:
            scenario: One of 'baseline', 'extended_weight', 'no_conmed', 'threshold_75'
        """
        scenario_configs = {
            'baseline': {
                'weight_range': (50, 100),
                'concomitant_allowed': True
            },
            'extended_weight': {
                'weight_range': (70, 140),
                'concomitant_allowed': True
            },
            'no_conmed': {
                'weight_range': (50, 100),
                'concomitant_allowed': False
            },
            'threshold_75': {
                'weight_range': (50, 100),
                'concomitant_allowed': True
            }
        }
        
        if scenario not in scenario_configs:
            raise ValueError(f"Unknown scenario: {scenario}")
            
        config = scenario_configs[scenario]
        return self.prepare_pkpd_data(**config)