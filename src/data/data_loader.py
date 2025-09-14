"""
Data loader for PK/PD clinical trial datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from quantum.core.base import PKPDData


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
            (df['BW'] >= weight_range[0]) & 
            (df['BW'] <= weight_range[1])
        ].copy()
        
        # Filter by concomitant medication
        if not concomitant_allowed:
            df_filtered = df_filtered[df_filtered['COMED'] == 0]
            
        self.logger.info(f"Filtered to {len(df_filtered)} subjects")
        
        # Extract features and targets
        features = self._extract_features(df_filtered)
        concentrations = self._extract_concentrations(df_filtered)
        biomarkers = self._extract_biomarkers(df_filtered)
        
        # Create a data object that matches VQC model expectations
        class PKPDDataCompat:
            def __init__(self, subjects, features, concentrations, biomarkers, time_points, doses, body_weights, concomitant_meds):
                self.subjects = subjects
                self.features = features 
                self.concentrations = concentrations
                self.biomarkers = biomarkers
                # VQC model expectations
                self.time_points = time_points
                self.pk_concentrations = concentrations  # Same as concentrations
                self.pd_biomarkers = biomarkers  # Same as biomarkers
                self.doses = doses
                self.body_weights = body_weights  
                self.concomitant_meds = concomitant_meds
                # Add metadata for preprocessor compatibility
                self.metadata = {
                    'n_subjects': len(subjects),
                    'n_features': features.shape[1] if hasattr(features, 'shape') else len(features[0]),
                    'weight_range': (float(np.min(body_weights)), float(np.max(body_weights))),
                    'data_source': 'PKPDDataLoader'
                }
                
        # Extract additional data for VQC compatibility
        time_data = self._extract_time_points(df_filtered)
        dose_data = self._extract_doses(df_filtered) 
        weight_data = self._extract_body_weights(df_filtered)
        conmed_data = self._extract_concomitant_meds(df_filtered)
                
        # The VQC model expects features to include time, so we need to construct them properly
        # Since we have time series data, we'll use the features as fallback but the VQC
        # encode_data method will reconstruct [time, dose, body_weight, concomitant_med] from
        # the individual time_points, doses, body_weights, concomitant_meds arrays
        return PKPDDataCompat(
            subjects=df_filtered['ID'].unique(),
            features=features,  # This is per-subject [weight, dose, conmed] for compatibility
            concentrations=concentrations,
            biomarkers=biomarkers,
            time_points=time_data,
            doses=dose_data,
            body_weights=weight_data,
            concomitant_meds=conmed_data
        )
        
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract subject features for modeling using only available data columns."""
        # Get features per subject - only use columns that exist in the data file
        # Available columns: ID, BW, COMED, DOSE, TIME, DV, EVID, MDV, AMT, CMT, DVID
        subject_features = df[['BW', 'DOSE', 'COMED']].groupby(df['ID']).first()

        # Use only the actual data columns available: Body Weight, Dose, Concomitant Medication
        # Format: ['Weight', 'Dose', 'Conmed'] - no age/sex since not in data
        features = subject_features[['BW', 'DOSE', 'COMED']].values
        return features.astype(np.float32)
        
    def _extract_concentrations(self, df: pd.DataFrame) -> np.ndarray:
        """Extract drug concentration time series."""
        conc_data = []
        for subject_id in df['ID'].unique():
            subject_data = df[df['ID'] == subject_id].sort_values('TIME')
            concentrations = subject_data['DV'].values  # Using DV as concentration/measurement
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
            biomarkers = subject_data['DV'].values  # Using DV as biomarker measurement
            biomarker_data.append(biomarkers)
            
        # Pad to same length
        max_len = max(len(b) for b in biomarker_data)
        padded_biomarkers = np.zeros((len(biomarker_data), max_len))
        for i, bio in enumerate(biomarker_data):
            padded_biomarkers[i, :len(bio)] = bio
            
        return padded_biomarkers.astype(np.float32)
    
    def _extract_time_points(self, df: pd.DataFrame) -> np.ndarray:
        """Extract time points for each subject."""
        time_data = []
        for subject_id in df['ID'].unique():
            subject_data = df[df['ID'] == subject_id].sort_values('TIME')
            times = subject_data['TIME'].values
            time_data.append(times)
        
        # Pad to same length
        max_len = max(len(t) for t in time_data)
        padded_times = np.zeros((len(time_data), max_len))
        for i, times in enumerate(time_data):
            padded_times[i, :len(times)] = times
            
        return padded_times.astype(np.float32)
    
    def _extract_doses(self, df: pd.DataFrame) -> np.ndarray:
        """Extract dose information per subject."""
        dose_data = []
        for subject_id in df['ID'].unique():
            subject_data = df[df['ID'] == subject_id].sort_values('TIME')
            doses = subject_data['DOSE'].values
            dose_data.append(doses)
        
        # Pad to same length
        max_len = max(len(d) for d in dose_data)
        padded_doses = np.zeros((len(dose_data), max_len))
        for i, doses in enumerate(dose_data):
            padded_doses[i, :len(doses)] = doses
            
        return padded_doses.astype(np.float32)
    
    def _extract_body_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Extract body weight per subject (constant per subject)."""
        weights = df[['ID', 'BW']].drop_duplicates('ID')['BW'].values
        return weights.astype(np.float32)
    
    def _extract_concomitant_meds(self, df: pd.DataFrame) -> np.ndarray:
        """Extract concomitant medication status per subject."""
        conmed = df[['ID', 'COMED']].drop_duplicates('ID')['COMED'].values
        return conmed.astype(np.float32)
        
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