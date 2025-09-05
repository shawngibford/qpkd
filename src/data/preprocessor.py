"""
Data preprocessing utilities for PK/PD modeling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional, List
import logging

from ..quantum.core.data_structures import PKPDData


class DataPreprocessor:
    """Preprocessing utilities for PK/PD data."""
    
    def __init__(self, scaling_method: str = 'standard'):
        """Initialize preprocessor.
        
        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
        """
        self.scaling_method = scaling_method
        self.feature_scaler = None
        self.concentration_scaler = None
        self.biomarker_scaler = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers
        scaler_map = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
        
        if scaling_method not in scaler_map:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
            
        Scaler = scaler_map[scaling_method]
        self.feature_scaler = Scaler()
        self.concentration_scaler = Scaler()
        self.biomarker_scaler = Scaler()
        
    def fit_transform(self, data: PKPDData) -> PKPDData:
        """Fit scalers and transform data.
        
        Args:
            data: Raw PKPDData object
            
        Returns:
            Scaled PKPDData object
        """
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(data.features)
        
        # Scale concentrations (handle padded zeros)
        conc_mask = data.concentrations > 0
        scaled_concentrations = data.concentrations.copy()
        if conc_mask.any():
            valid_conc = data.concentrations[conc_mask].reshape(-1, 1)
            scaled_valid_conc = self.concentration_scaler.fit_transform(valid_conc)
            scaled_concentrations[conc_mask] = scaled_valid_conc.flatten()
            
        # Scale biomarkers (handle padded zeros)
        bio_mask = data.biomarkers > 0
        scaled_biomarkers = data.biomarkers.copy()
        if bio_mask.any():
            valid_bio = data.biomarkers[bio_mask].reshape(-1, 1)
            scaled_valid_bio = self.biomarker_scaler.fit_transform(valid_bio)
            scaled_biomarkers[bio_mask] = scaled_valid_bio.flatten()
            
        return PKPDData(
            subjects=data.subjects,
            features=scaled_features,
            concentrations=scaled_concentrations,
            biomarkers=scaled_biomarkers,
            metadata={**data.metadata, 'scaled': True, 'scaling_method': self.scaling_method}
        )
        
    def transform(self, data: PKPDData) -> PKPDData:
        """Transform data using fitted scalers.
        
        Args:
            data: Raw PKPDData object
            
        Returns:
            Scaled PKPDData object
        """
        if self.feature_scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
        # Transform features
        scaled_features = self.feature_scaler.transform(data.features)
        
        # Transform concentrations
        conc_mask = data.concentrations > 0
        scaled_concentrations = data.concentrations.copy()
        if conc_mask.any():
            valid_conc = data.concentrations[conc_mask].reshape(-1, 1)
            scaled_valid_conc = self.concentration_scaler.transform(valid_conc)
            scaled_concentrations[conc_mask] = scaled_valid_conc.flatten()
            
        # Transform biomarkers
        bio_mask = data.biomarkers > 0
        scaled_biomarkers = data.biomarkers.copy()
        if bio_mask.any():
            valid_bio = data.biomarkers[bio_mask].reshape(-1, 1)
            scaled_valid_bio = self.biomarker_scaler.transform(valid_bio)
            scaled_biomarkers[bio_mask] = scaled_valid_bio.flatten()
            
        return PKPDData(
            subjects=data.subjects,
            features=scaled_features,
            concentrations=scaled_concentrations,
            biomarkers=scaled_biomarkers,
            metadata={**data.metadata, 'scaled': True, 'scaling_method': self.scaling_method}
        )
        
    def inverse_transform_biomarkers(self, scaled_biomarkers: np.ndarray) -> np.ndarray:
        """Inverse transform biomarker predictions back to original scale.
        
        Args:
            scaled_biomarkers: Scaled biomarker values
            
        Returns:
            Original scale biomarker values
        """
        if self.biomarker_scaler is None:
            raise ValueError("Biomarker scaler not fitted.")
            
        return self.biomarker_scaler.inverse_transform(scaled_biomarkers.reshape(-1, 1)).flatten()
        
    def create_train_test_split(self, 
                               data: PKPDData, 
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[PKPDData, PKPDData]:
        """Split data into training and testing sets.
        
        Args:
            data: PKPDData object
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        n_subjects = len(data.subjects)
        train_indices, test_indices = train_test_split(
            range(n_subjects), 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Create training data
        train_data = PKPDData(
            subjects=[data.subjects[i] for i in train_indices],
            features=data.features[train_indices],
            concentrations=data.concentrations[train_indices],
            biomarkers=data.biomarkers[train_indices],
            metadata={**data.metadata, 'split': 'train', 'n_subjects': len(train_indices)}
        )
        
        # Create testing data
        test_data = PKPDData(
            subjects=[data.subjects[i] for i in test_indices],
            features=data.features[test_indices],
            concentrations=data.concentrations[test_indices],
            biomarkers=data.biomarkers[test_indices],
            metadata={**data.metadata, 'split': 'test', 'n_subjects': len(test_indices)}
        )
        
        return train_data, test_data
        
    def augment_data(self, 
                     data: PKPDData, 
                     augmentation_factor: int = 2,
                     noise_level: float = 0.05) -> PKPDData:
        """Generate augmented data to increase dataset size.
        
        Args:
            data: Original PKPDData
            augmentation_factor: How many times to multiply the dataset
            noise_level: Standard deviation of Gaussian noise to add
            
        Returns:
            Augmented PKPDData object
        """
        augmented_subjects = data.subjects.copy()
        augmented_features = data.features.copy()
        augmented_concentrations = data.concentrations.copy()
        augmented_biomarkers = data.biomarkers.copy()
        
        for i in range(augmentation_factor - 1):
            # Add Gaussian noise to features
            noise_features = data.features + np.random.normal(0, noise_level, data.features.shape)
            noise_concentrations = data.concentrations + np.random.normal(0, noise_level, data.concentrations.shape)
            noise_biomarkers = data.biomarkers + np.random.normal(0, noise_level, data.biomarkers.shape)
            
            # Ensure non-negative values where appropriate
            noise_concentrations = np.maximum(0, noise_concentrations)
            noise_biomarkers = np.maximum(0, noise_biomarkers)
            
            # Append augmented data
            augmented_subjects.extend([f"{subj}_aug{i+1}" for subj in data.subjects])
            augmented_features = np.vstack([augmented_features, noise_features])
            augmented_concentrations = np.vstack([augmented_concentrations, noise_concentrations])
            augmented_biomarkers = np.vstack([augmented_biomarkers, noise_biomarkers])
            
        return PKPDData(
            subjects=augmented_subjects,
            features=augmented_features,
            concentrations=augmented_concentrations,
            biomarkers=augmented_biomarkers,
            metadata={**data.metadata, 'augmented': True, 'augmentation_factor': augmentation_factor}
        )
        
    def create_time_series_windows(self, 
                                  data: PKPDData, 
                                  window_size: int = 5,
                                  step_size: int = 1) -> PKPDData:
        """Create sliding windows for time series modeling.
        
        Args:
            data: PKPDData object
            window_size: Size of each time window
            step_size: Step size between windows
            
        Returns:
            Windowed PKPDData object
        """
        windowed_subjects = []
        windowed_features = []
        windowed_concentrations = []
        windowed_biomarkers = []
        
        for i, subject in enumerate(data.subjects):
            subject_conc = data.concentrations[i]
            subject_bio = data.biomarkers[i]
            subject_feat = data.features[i]
            
            # Find valid time points (non-zero)
            valid_indices = np.where(subject_conc > 0)[0]
            
            if len(valid_indices) >= window_size:
                for start_idx in range(0, len(valid_indices) - window_size + 1, step_size):
                    end_idx = start_idx + window_size
                    
                    windowed_subjects.append(f"{subject}_w{start_idx}")
                    windowed_features.append(subject_feat)
                    windowed_concentrations.append(subject_conc[valid_indices[start_idx:end_idx]])
                    windowed_biomarkers.append(subject_bio[valid_indices[start_idx:end_idx]])
                    
        # Pad to consistent window size
        max_window = max(len(w) for w in windowed_concentrations) if windowed_concentrations else window_size
        
        padded_conc = np.zeros((len(windowed_concentrations), max_window))
        padded_bio = np.zeros((len(windowed_biomarkers), max_window))
        
        for i, (conc, bio) in enumerate(zip(windowed_concentrations, windowed_biomarkers)):
            padded_conc[i, :len(conc)] = conc
            padded_bio[i, :len(bio)] = bio
            
        return PKPDData(
            subjects=windowed_subjects,
            features=np.array(windowed_features),
            concentrations=padded_conc,
            biomarkers=padded_bio,
            metadata={**data.metadata, 'windowed': True, 'window_size': window_size}
        )