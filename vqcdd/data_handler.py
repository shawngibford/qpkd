"""
Data Handler Module for VQCdd

This module handles loading, preprocessing, and validation of pharmacokinetic
and pharmacodynamic data from the EstData.csv file. It provides a clean interface
for quantum parameter estimation and includes utilities for synthetic data generation.

Key Features:
- EstData.csv loading and parsing
- Data validation and quality checks
- Feature engineering for quantum circuits
- Synthetic population generation
- Train/test splitting and cross-validation support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings

# Import PK/PD model components
from pkpd_models import PKParameters, PDParameters


@dataclass
class PatientData:
    """Single patient's PK/PD data with full NONMEM features"""
    patient_id: int
    body_weight: float                    # BW: Body weight (kg)
    concomitant_med: bool                # COMED: Concomitant medication (True/False)
    time_points: np.ndarray              # TIME: Time points (hours)
    doses: np.ndarray                    # DOSE: Dose amounts (mg)
    pk_concentrations: np.ndarray        # DV for DVID=1: PK concentrations (mg/L)
    pd_biomarkers: np.ndarray           # DV for DVID=2: PD biomarkers (ng/mL)

    # Additional NONMEM columns for full feature set
    evid: np.ndarray                     # EVID: Event ID (0=observation, 1=dose)
    mdv: np.ndarray                      # MDV: Missing dependent variable flag
    amt: np.ndarray                      # AMT: Amount for dosing events
    cmt: np.ndarray                      # CMT: Compartment (1=dose, 2=PK, 3=PD)
    dvid: np.ndarray                     # DVID: DV type (0=dose, 1=PK, 2=PD)

    def __post_init__(self):
        """Validate data consistency for all NONMEM fields"""
        n_points = len(self.time_points)
        assert len(self.doses) == n_points, "Dose array length mismatch"
        assert len(self.pk_concentrations) == n_points, "PK concentration array length mismatch"
        assert len(self.pd_biomarkers) == n_points, "PD biomarker array length mismatch"
        assert len(self.evid) == n_points, "EVID array length mismatch"
        assert len(self.mdv) == n_points, "MDV array length mismatch"
        assert len(self.amt) == n_points, "AMT array length mismatch"
        assert len(self.cmt) == n_points, "CMT array length mismatch"
        assert len(self.dvid) == n_points, "DVID array length mismatch"


@dataclass
class StudyData:
    """Complete study dataset"""
    patients: List[PatientData]
    study_design: Dict
    data_quality: Dict
    study_metadata: Optional[Dict] = None

    def get_patient_count(self) -> int:
        """Get total number of patients"""
        return len(self.patients)

    def get_observation_count(self) -> int:
        """Get total number of observations"""
        return sum(len(p.time_points) for p in self.patients)

    def get_population_characteristics(self) -> Dict:
        """Get population summary statistics"""
        weights = [p.body_weight for p in self.patients]
        comed_prevalence = sum(p.concomitant_med for p in self.patients) / len(self.patients)

        return {
            'n_patients': len(self.patients),
            'weight_mean': np.mean(weights),
            'weight_std': np.std(weights),
            'weight_range': (np.min(weights), np.max(weights)),
            'concomitant_med_prevalence': comed_prevalence
        }


class EstDataLoader:
    """
    Loads and processes the EstData.csv file

    The EstData.csv file contains PK/PD data in NONMEM format with columns:
    ID, BW, COMED, DOSE, TIME, DV, EVID, MDV, AMT, CMT, DVID
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data loader

        Args:
            data_path: Path to EstData.csv (if None, uses default location)
        """
        if data_path is None:
            # Default path relative to project root
            self.data_path = Path(__file__).parent.parent / "data" / "EstData.csv"
        else:
            self.data_path = Path(data_path)

        self.logger = logging.getLogger(__name__)

    def load_data(self) -> StudyData:
        """
        Load and parse EstData.csv file

        Returns:
            StudyData object with all patients
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.logger.info(f"Loading data from {self.data_path}")

        # Load CSV file
        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file: {e}")

        # Validate required columns
        required_columns = ['ID', 'BW', 'COMED', 'DOSE', 'TIME', 'DV', 'EVID', 'CMT', 'DVID']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Parse data by patient
        patients = []
        patient_ids = df['ID'].unique()

        for patient_id in patient_ids:
            patient_data = self._parse_patient_data(df, patient_id)
            if patient_data is not None:
                patients.append(patient_data)

        # Create study design information
        study_design = self._extract_study_design(df)

        # Perform data quality assessment
        data_quality = self._assess_data_quality(df, patients)

        self.logger.info(f"Loaded {len(patients)} patients with {sum(len(p.time_points) for p in patients)} observations")

        return StudyData(patients=patients, study_design=study_design, data_quality=data_quality, study_metadata={'source': 'EstData.csv', 'loaded_at': pd.Timestamp.now().isoformat()})

    def _parse_patient_data(self, df: pd.DataFrame, patient_id: int) -> Optional[PatientData]:
        """
        Parse individual patient data from DataFrame

        Args:
            df: Complete DataFrame
            patient_id: Patient ID to extract

        Returns:
            PatientData object or None if parsing fails
        """
        try:
            patient_df = df[df['ID'] == patient_id].copy()

            if len(patient_df) == 0:
                return None

            # Extract patient characteristics (should be constant)
            body_weight = patient_df['BW'].iloc[0]
            concomitant_med = bool(patient_df['COMED'].iloc[0])

            # Extract time points and doses
            time_points = patient_df['TIME'].values
            doses = patient_df['DOSE'].values

            # Separate PK and PD observations based on DVID/CMT
            # DVID=1 or CMT=2: PK concentrations
            # DVID=2 or CMT=3: PD biomarkers
            pk_mask = (patient_df['DVID'] == 1) | (patient_df['CMT'] == 2)
            pd_mask = (patient_df['DVID'] == 2) | (patient_df['CMT'] == 3)

            # Initialize concentration and biomarker arrays with NaN
            pk_concentrations = np.full(len(patient_df), np.nan)
            pd_biomarkers = np.full(len(patient_df), np.nan)

            # Fill in observed values
            if pk_mask.any():
                pk_concentrations[pk_mask] = patient_df.loc[pk_mask, 'DV'].values

            if pd_mask.any():
                pd_biomarkers[pd_mask] = patient_df.loc[pd_mask, 'DV'].values

            # Extract all additional NONMEM fields
            evid = patient_df['EVID'].values
            mdv = patient_df['MDV'].values
            amt = patient_df['AMT'].values
            cmt = patient_df['CMT'].values
            dvid = patient_df['DVID'].values

            return PatientData(
                patient_id=int(patient_id),
                body_weight=float(body_weight),
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

        except Exception as e:
            self.logger.warning(f"Failed to parse patient {patient_id}: {e}")
            return None

    def _extract_study_design(self, df: pd.DataFrame) -> Dict:
        """Extract study design information"""
        return {
            'time_range': (df['TIME'].min(), df['TIME'].max()),
            'dose_levels': sorted(df[df['DOSE'] > 0]['DOSE'].unique()),
            'n_patients': df['ID'].nunique(),
            'total_observations': len(df),
            'observation_types': df['DVID'].unique().tolist()
        }

    def _assess_data_quality(self, df: pd.DataFrame, patients: List[PatientData]) -> Dict:
        """Assess data quality and completeness"""
        # Calculate missing data rates
        total_obs = len(df)
        missing_dv = df['DV'].isna().sum()

        # Patient-level statistics
        patient_obs_counts = [len(p.time_points) for p in patients]

        # PK/PD data availability
        pk_available = sum(1 for p in patients if not np.all(np.isnan(p.pk_concentrations)))
        pd_available = sum(1 for p in patients if not np.all(np.isnan(p.pd_biomarkers)))

        return {
            'missing_dv_rate': missing_dv / total_obs,
            'patients_with_pk': pk_available,
            'patients_with_pd': pd_available,
            'obs_per_patient_mean': np.mean(patient_obs_counts),
            'obs_per_patient_range': (np.min(patient_obs_counts), np.max(patient_obs_counts)),
            'data_quality_score': 1.0 - (missing_dv / total_obs)  # Simple quality score
        }


class QuantumFeatureEncoder:
    """
    Encodes patient data into features suitable for quantum circuits

    Transforms PK/PD data into fixed-length feature vectors that can be
    processed by quantum circuits for parameter estimation.
    """

    def __init__(self, feature_dim: int = 11, normalization: str = "robust"):
        """
        Initialize feature encoder for full NONMEM dataset

        Args:
            feature_dim: Number of features to encode (11 for full NONMEM)
            normalization: "robust", "standard", or "minmax"
        """
        self.feature_dim = feature_dim
        self.normalization = normalization
        self.fitted = False
        self.scaler_params = {}
        self.logger = logging.getLogger(__name__)

    def fit(self, study_data: StudyData) -> None:
        """
        Fit encoder to study data (learn normalization parameters)

        Args:
            study_data: Complete study dataset
        """
        # Extract all features for normalization fitting
        all_features = []

        for patient in study_data.patients:
            for i, time_point in enumerate(patient.time_points):
                features = self._extract_raw_features(patient, i)
                all_features.append(features)

        feature_matrix = np.array(all_features)

        # Fit normalization parameters
        if self.normalization == "robust":
            self.scaler_params = {
                'median': np.median(feature_matrix, axis=0),
                'mad': np.median(np.abs(feature_matrix - np.median(feature_matrix, axis=0)), axis=0)
            }
            # Prevent division by zero
            self.scaler_params['mad'] = np.maximum(self.scaler_params['mad'], 1e-6)

        elif self.normalization == "standard":
            self.scaler_params = {
                'mean': np.mean(feature_matrix, axis=0),
                'std': np.std(feature_matrix, axis=0)
            }
            self.scaler_params['std'] = np.maximum(self.scaler_params['std'], 1e-6)

        elif self.normalization == "minmax":
            self.scaler_params = {
                'min': np.min(feature_matrix, axis=0),
                'max': np.max(feature_matrix, axis=0)
            }
            range_vals = self.scaler_params['max'] - self.scaler_params['min']
            self.scaler_params['range'] = np.maximum(range_vals, 1e-6)

        self.fitted = True
        self.logger.info(f"Feature encoder fitted with {len(all_features)} observations")

    def encode_patient_data(self, patient: PatientData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode patient data into quantum-compatible features

        Args:
            patient: Patient data to encode

        Returns:
            Tuple of (features, pk_targets, pd_targets)
        """
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before encoding data")

        features_list = []
        pk_targets = []
        pd_targets = []

        for i in range(len(patient.time_points)):
            # Extract and normalize features
            raw_features = self._extract_raw_features(patient, i)
            normalized_features = self._normalize_features(raw_features)
            features_list.append(normalized_features)

            # Extract targets (handle missing values)
            pk_target = patient.pk_concentrations[i] if not np.isnan(patient.pk_concentrations[i]) else 0.0
            pd_target = patient.pd_biomarkers[i] if not np.isnan(patient.pd_biomarkers[i]) else 0.0

            pk_targets.append(pk_target)
            pd_targets.append(pd_target)

        return (np.array(features_list),
                np.array(pk_targets),
                np.array(pd_targets))

    def _extract_raw_features(self, patient: PatientData, time_index: int) -> np.ndarray:
        """Extract all 11 NONMEM features for a specific time point"""

        # All 11 NONMEM features in order:
        features = [
            float(patient.patient_id),              # 1. ID - Patient identifier
            patient.body_weight,                    # 2. BW - Body weight (kg)
            float(patient.concomitant_med),         # 3. COMED - Concomitant medication (0/1)
            patient.doses[time_index],              # 4. DOSE - Dose amount (mg)
            patient.time_points[time_index],        # 5. TIME - Time point (hours)
            patient.pk_concentrations[time_index] if not np.isnan(patient.pk_concentrations[time_index]) else 0.0,  # 6. DV (PK)
            float(patient.evid[time_index]),        # 7. EVID - Event ID (0=obs, 1=dose)
            float(patient.mdv[time_index]),         # 8. MDV - Missing DV flag
            patient.amt[time_index],                # 9. AMT - Amount for dosing events
            float(patient.cmt[time_index]),         # 10. CMT - Compartment (1=dose, 2=PK, 3=PD)
            float(patient.dvid[time_index])         # 11. DVID - DV type (0=dose, 1=PK, 2=PD)
        ]

        # Handle NaN values by replacing with 0.0
        features = [0.0 if np.isnan(f) else f for f in features]

        # Ensure we have exactly the expected number of features
        if len(features) != self.feature_dim:
            self.logger.warning(f"Feature dimension mismatch: got {len(features)}, expected {self.feature_dim}")
            if len(features) < self.feature_dim:
                features.extend([0.0] * (self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]

        return np.array(features, dtype=np.float64)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply learned normalization to features"""
        if self.normalization == "robust":
            return (features - self.scaler_params['median']) / self.scaler_params['mad']

        elif self.normalization == "standard":
            return (features - self.scaler_params['mean']) / self.scaler_params['std']

        elif self.normalization == "minmax":
            return (features - self.scaler_params['min']) / self.scaler_params['range']

        else:
            return features  # No normalization


class SyntheticDataGenerator:
    """
    Generates synthetic PK/PD data for testing and validation

    Creates virtual patients with known parameter values to test
    quantum parameter estimation algorithms.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize synthetic data generator

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.logger = logging.getLogger(__name__)

    def generate_virtual_population(self, n_patients: int = 100,
                                   weight_range: Tuple[float, float] = (50, 100),
                                   comed_prevalence: float = 0.3) -> StudyData:
        """
        Generate virtual patient population

        Args:
            n_patients: Number of virtual patients
            weight_range: Body weight range (kg)
            comed_prevalence: Prevalence of concomitant medication

        Returns:
            StudyData with synthetic patients
        """
        patients = []

        for patient_id in range(1, n_patients + 1):
            # Generate patient characteristics
            body_weight = np.random.uniform(weight_range[0], weight_range[1])
            concomitant_med = np.random.random() < comed_prevalence

            # Generate PK/PD parameters
            pk_params = self._generate_pk_parameters(body_weight)
            pd_params = self._generate_pd_parameters(concomitant_med)

            # Generate time series data
            patient_data = self._generate_patient_timeseries(
                patient_id, body_weight, concomitant_med, pk_params, pd_params
            )

            patients.append(patient_data)

        # Create study design
        study_design = {
            'time_range': (0, 48),
            'dose_levels': [5.0, 10.0, 20.0],
            'n_patients': n_patients,
            'synthetic': True
        }

        # Perfect data quality for synthetic data
        data_quality = {
            'missing_dv_rate': 0.0,
            'patients_with_pk': n_patients,
            'patients_with_pd': n_patients,
            'data_quality_score': 1.0,
            'synthetic': True
        }

        self.logger.info(f"Generated {n_patients} synthetic patients")

        return StudyData(patients=patients, study_design=study_design, data_quality=data_quality, study_metadata={'source': 'synthetic', 'generated_at': pd.Timestamp.now().isoformat(), 'weight_range': weight_range, 'comed_prevalence': comed_prevalence})

    def _generate_pk_parameters(self, body_weight: float) -> PKParameters:
        """Generate realistic PK parameters with inter-individual variability"""
        # Typical values with log-normal distribution
        ka = np.random.lognormal(np.log(1.0), 0.5)    # Absorption rate
        cl = np.random.lognormal(np.log(5.0), 0.3)    # Clearance
        v1 = np.random.lognormal(np.log(20.0), 0.2)   # Central volume
        q = np.random.lognormal(np.log(2.0), 0.4)     # Inter-compartmental clearance
        v2 = np.random.lognormal(np.log(50.0), 0.3)   # Peripheral volume

        return PKParameters(ka=ka, cl=cl, v1=v1, q=q, v2=v2)

    def _generate_pd_parameters(self, concomitant_med: bool) -> PDParameters:
        """Generate realistic PD parameters"""
        baseline = np.random.lognormal(np.log(15.0), 0.2)
        imax = np.random.beta(8, 2)  # Skewed toward higher values
        ic50 = np.random.lognormal(np.log(5.0), 0.4)
        gamma = np.random.gamma(2, 0.5)  # Hill coefficient

        # Adjust for concomitant medication
        if concomitant_med:
            baseline *= 1.3  # Higher baseline with concomitant meds

        return PDParameters(baseline=baseline, imax=imax, ic50=ic50, gamma=gamma)

    def _generate_patient_timeseries(self, patient_id: int, body_weight: float,
                                    concomitant_med: bool, pk_params: PKParameters,
                                    pd_params: PDParameters) -> PatientData:
        """Generate time series data for a single patient"""
        from pkpd_models import TwoCompartmentPK, InhibitoryEmaxPD, PKPDModel

        # Create PKPD model
        pk_model = TwoCompartmentPK("iv_bolus")
        pd_model = InhibitoryEmaxPD("direct")
        pkpd_model = PKPDModel(pk_model, pd_model)

        # Define sampling schedule
        dose_times = [0, 24]  # Two doses
        sample_times = [0, 1, 2, 4, 8, 12, 24, 25, 26, 28, 32, 36, 48]
        dose = 10.0  # mg

        # Create dose array
        doses = np.zeros(len(sample_times))
        for dose_time in dose_times:
            dose_idx = np.argmin(np.abs(np.array(sample_times) - dose_time))
            doses[dose_idx] = dose

        # Simulate true concentrations and biomarkers
        concentrations, biomarkers = pkpd_model.simulate_pkpd_profile(
            np.array(sample_times), dose, pk_params, pd_params,
            body_weight, concomitant_med
        )

        # Add measurement noise
        conc_noise = np.random.normal(0, 0.1 * concentrations)  # 10% CV
        bio_noise = np.random.normal(0, 0.15 * biomarkers)     # 15% CV

        noisy_concentrations = np.maximum(concentrations + conc_noise, 0)
        noisy_biomarkers = np.maximum(biomarkers + bio_noise, 0.1)

        # Create synthetic NONMEM fields to match real data structure
        n_points = len(sample_times)
        evid = np.array([1 if d > 0 else 0 for d in doses])  # 1 for doses, 0 for observations
        mdv = np.zeros(n_points)  # No missing data in synthetic
        amt = doses.copy()  # Amount same as dose for dosing events
        cmt = np.array([1 if d > 0 else 3 for d in doses])  # 1 for dosing, 3 for PD observations
        dvid = np.array([0 if d > 0 else 2 for d in doses])  # 0 for dosing, 2 for PD observations

        return PatientData(
            patient_id=patient_id,
            body_weight=body_weight,
            concomitant_med=concomitant_med,
            time_points=np.array(sample_times),
            doses=doses,
            pk_concentrations=noisy_concentrations,
            pd_biomarkers=noisy_biomarkers,
            evid=evid,
            mdv=mdv,
            amt=amt,
            cmt=cmt,
            dvid=dvid
        )


def train_test_split_patients(study_data: StudyData, test_fraction: float = 0.2,
                            random_state: Optional[int] = None) -> Tuple[StudyData, StudyData]:
    """
    Split patients into training and test sets

    Args:
        study_data: Complete study data
        test_fraction: Fraction of patients for testing
        random_state: Random seed

    Returns:
        Tuple of (train_data, test_data)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_patients = len(study_data.patients)
    n_test = int(n_patients * test_fraction)

    # Randomly select test patients
    test_indices = np.random.choice(n_patients, size=n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_patients), test_indices)

    train_patients = [study_data.patients[i] for i in train_indices]
    test_patients = [study_data.patients[i] for i in test_indices]

    # Create new StudyData objects
    train_data = StudyData(
        patients=train_patients,
        study_design=study_data.study_design.copy(),
        data_quality=study_data.data_quality.copy(),
        study_metadata=study_data.study_metadata
    )

    test_data = StudyData(
        patients=test_patients,
        study_design=study_data.study_design.copy(),
        data_quality=study_data.data_quality.copy(),
        study_metadata=study_data.study_metadata
    )

    return train_data, test_data


if __name__ == "__main__":
    # Example usage and testing
    print("VQCdd Data Handler Module")
    print("=" * 40)

    # Test with synthetic data
    generator = SyntheticDataGenerator(seed=42)
    study_data = generator.generate_virtual_population(n_patients=20)

    print(f"Generated study with {study_data.get_patient_count()} patients")
    print(f"Population characteristics: {study_data.get_population_characteristics()}")

    # Test feature encoding
    encoder = QuantumFeatureEncoder(feature_dim=11)
    encoder.fit(study_data)

    # Encode first patient
    patient = study_data.patients[0]
    features, pk_targets, pd_targets = encoder.encode_patient_data(patient)

    print(f"Patient {patient.patient_id}:")
    print(f"  Features shape: {features.shape}")
    print(f"  Example features: {features[0]}")
    print(f"  PK targets range: [{pk_targets.min():.3f}, {pk_targets.max():.3f}]")
    print(f"  PD targets range: [{pd_targets.min():.3f}, {pd_targets.max():.3f}]")

    # Test train/test split
    train_data, test_data = train_test_split_patients(study_data, test_fraction=0.3)
    print(f"Split: {train_data.get_patient_count()} train, {test_data.get_patient_count()} test patients")

    # Try loading real data (if available)
    try:
        loader = EstDataLoader()
        real_data = loader.load_data()
        print(f"Real data: {real_data.get_patient_count()} patients")
        print(f"Data quality: {real_data.data_quality}")
    except FileNotFoundError:
        print("Real EstData.csv not found - using synthetic data only")