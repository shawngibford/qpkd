"""
Base class for all quantum PK/PD modeling approaches
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import pennylane as qml
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for quantum PK/PD models"""
    n_qubits: int = 4  # Reduced from 8 to mitigate barren plateaus
    n_layers: int = 2  # Reduced from 4 to avoid gradient vanishing
    learning_rate: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    shots: int = 1024


@dataclass 
class PKPDData:
    """Structured PK/PD data container"""
    subjects: np.ndarray
    time_points: np.ndarray
    pk_concentrations: np.ndarray
    pd_biomarkers: np.ndarray
    doses: np.ndarray
    body_weights: np.ndarray
    concomitant_meds: np.ndarray
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'PKPDData':
        """Create PKPDData from EstData.csv DataFrame

        Args:
            df: DataFrame with columns ID, BW, COMED, DOSE, TIME, DV, EVID, MDV, AMT, CMT, DVID

        Returns:
            PKPDData object with structured arrays
        """
        # Extract unique subject IDs
        subjects = df['ID'].unique()

        # Extract time points (all unique times across subjects)
        time_points = np.sort(df['TIME'].unique())

        # Initialize arrays
        n_subjects = len(subjects)
        n_times = len(time_points)

        pk_concentrations = np.zeros((n_subjects, n_times))
        pd_biomarkers = np.zeros((n_subjects, n_times))
        doses = np.zeros(n_subjects)
        body_weights = np.zeros(n_subjects)
        concomitant_meds = np.zeros(n_subjects)

        for i, subject_id in enumerate(subjects):
            subject_data = df[df['ID'] == subject_id]

            # Extract subject-specific data
            body_weights[i] = subject_data['BW'].iloc[0]
            concomitant_meds[i] = subject_data['COMED'].iloc[0]

            # Get dosing information (from EVID=1 records)
            dose_records = subject_data[subject_data['EVID'] == 1]
            if not dose_records.empty:
                doses[i] = dose_records['AMT'].sum()  # Total dose

            # Map concentrations and biomarkers to time grid
            for _, row in subject_data.iterrows():
                time_idx = np.where(time_points == row['TIME'])[0]
                if len(time_idx) > 0:
                    idx = time_idx[0]
                    if row['EVID'] == 0 and row['MDV'] == 0:  # Observation record
                        if row['CMT'] == 2:  # PK compartment
                            pk_concentrations[i, idx] = row['DV']
                        elif row['CMT'] == 3:  # PD compartment
                            pd_biomarkers[i, idx] = row['DV']

        return cls(
            subjects=subjects,
            time_points=time_points,
            pk_concentrations=pk_concentrations,
            pd_biomarkers=pd_biomarkers,
            doses=doses,
            body_weights=body_weights,
            concomitant_meds=concomitant_meds
        )


@dataclass
class OptimizationResult:
    """Results from quantum optimization"""
    optimal_daily_dose: float
    optimal_weekly_dose: float
    population_coverage: float
    parameter_estimates: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    convergence_info: Dict[str, Any]
    quantum_metrics: Dict[str, float]  # quantum-specific metrics


class QuantumPKPDBase(ABC):
    """
    Abstract base class for quantum-enhanced PK/PD modeling approaches
    
    All five approaches inherit from this base class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = None
        self.circuit = None
        self.parameters = None
        self.is_trained = False
        
    @abstractmethod
    def setup_quantum_device(self) -> qml.device:
        """Setup PennyLane quantum device"""
        pass
    
    @abstractmethod
    def build_quantum_circuit(self, n_qubits: int, n_layers: int) -> callable:
        """Build the quantum circuit for this approach"""
        pass
    
    @abstractmethod
    def encode_data(self, data: PKPDData) -> np.ndarray:
        """Encode PK/PD data into quantum-compatible format"""
        pass
    
    @abstractmethod
    def cost_function(self, params: np.ndarray, data: PKPDData) -> float:
        """Define the cost function to minimize"""
        pass
    
    @abstractmethod
    def optimize_parameters(self, data: PKPDData) -> Dict[str, Any]:
        """Run quantum optimization"""
        pass
    
    @abstractmethod
    def predict_biomarker(self, 
                         dose: float, 
                         time: np.ndarray,
                         covariates: Dict[str, float]) -> np.ndarray:
        """Predict biomarker levels for given dose and covariates"""
        pass
    
    @abstractmethod
    def optimize_dosing(self, 
                       target_threshold: float = 3.3,
                       population_coverage: float = 0.9) -> OptimizationResult:
        """Optimize dosing regimen to meet target criteria"""
        pass
    
    def fit(self, data: PKPDData) -> 'QuantumPKPDBase':
        """Fit the quantum model to data"""
        self.setup_quantum_device()
        self.circuit = self.build_quantum_circuit(self.config.n_qubits, self.config.n_layers)
        optimization_result = self.optimize_parameters(data)
        self.parameters = optimization_result['optimal_params']
        self.is_trained = True
        return self
        
    def evaluate_population_coverage(self,
                                   dose: float,
                                   dosing_interval: float,
                                   population_params: Dict[str, np.ndarray],
                                   threshold: float = 3.3) -> float:
        """
        Evaluate what percentage of population achieves biomarker suppression

        Args:
            dose: Dose amount (mg)
            dosing_interval: 24h (daily) or 168h (weekly)
            population_params: Dictionary of population parameter distributions
            threshold: Biomarker threshold (3.3 ng/mL)

        Returns:
            Fraction of population achieving target suppression
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluating population coverage")

        n_simulations = len(population_params.get('body_weight', [100]))
        below_threshold_count = 0

        # Generate time points for steady-state evaluation
        time_points = np.linspace(0, dosing_interval, int(dosing_interval))

        for i in range(n_simulations):
            # Extract individual covariates
            covariates = {
                'body_weight': population_params.get('body_weight', [70])[i % len(population_params.get('body_weight', [70]))],
                'concomitant_med': population_params.get('concomitant_med', [0])[i % len(population_params.get('concomitant_med', [0]))]
            }

            try:
                # Predict biomarker response for this individual
                biomarker_response = self.predict_biomarker(dose, time_points, covariates)

                # Check if minimum biomarker level is below threshold
                min_biomarker = np.min(biomarker_response)
                if min_biomarker <= threshold:
                    below_threshold_count += 1

            except Exception:
                # If prediction fails, assume this individual doesn't achieve target
                continue

        return below_threshold_count / n_simulations
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive results report"""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating report")
            
        return {
            'approach': self.__class__.__name__,
            'quantum_framework': 'PennyLane',
            'model_config': self.config,
            'is_trained': self.is_trained,
            'parameters': self.parameters
        }