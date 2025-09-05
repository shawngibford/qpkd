"""
Population pharmacokinetic and pharmacodynamic models.
"""

import numpy as np
from scipy.stats import multivariate_normal, lognorm
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from .compartment_models import OneCompartmentModel, TwoCompartmentModel


@dataclass
class PopulationParameters:
    """Population parameter structure."""
    typical_values: Dict[str, float]  # Population typical values
    between_subject_variability: Dict[str, float]  # BSV (CV%)
    residual_variability: float  # Residual error (CV%)
    covariates: Dict[str, Dict[str, float]] = None  # Covariate effects


class PopulationPKModel:
    """Population pharmacokinetic model with inter-individual variability."""
    
    def __init__(self, 
                 base_model: str = 'one_compartment',
                 population_params: PopulationParameters = None):
        """Initialize population PK model.
        
        Args:
            base_model: Base structural model ('one_compartment', 'two_compartment')
            population_params: Population parameter definitions
        """
        self.base_model_type = base_model
        self.population_params = population_params or self._default_population_params()
        self.logger = logging.getLogger(__name__)
        
        # Initialize base model
        if base_model == 'one_compartment':
            self.base_model = OneCompartmentModel()
        elif base_model == 'two_compartment':
            self.base_model = TwoCompartmentModel()
        else:
            raise ValueError(f"Unknown base model: {base_model}")
            
    def _default_population_params(self) -> PopulationParameters:
        """Create default population parameters."""
        return PopulationParameters(
            typical_values={
                'CL': 10.0,    # L/h
                'V': 50.0,     # L
                'ka': 1.0      # 1/h
            },
            between_subject_variability={
                'CL': 30.0,    # 30% CV
                'V': 25.0,     # 25% CV
                'ka': 50.0     # 50% CV
            },
            residual_variability=20.0,  # 20% CV
            covariates={
                'WEIGHT': {'CL': 0.75, 'V': 1.0},  # Allometric scaling
                'AGE': {'CL': -0.01},  # Age effect on clearance
                'SEX': {'CL': -0.2}    # Sex effect (female vs male)
            }
        )
        
    def simulate_population(self,
                           n_subjects: int,
                           covariates: Dict[str, np.ndarray] = None,
                           time_points: np.ndarray = None,
                           dose_schedule: Dict[float, float] = None) -> Dict[str, Any]:
        """Simulate a population of subjects.
        
        Args:
            n_subjects: Number of subjects to simulate
            covariates: Subject covariates (weight, age, sex, etc.)
            time_points: Time points for simulation
            dose_schedule: Dosing schedule
            
        Returns:
            Dictionary with subject data and population statistics
        """
        if time_points is None:
            time_points = np.linspace(0, 24, 25)  # 24 hours, hourly
            
        if dose_schedule is None:
            dose_schedule = {0.0: 100.0}  # 100mg at t=0
            
        if covariates is None:
            covariates = self._generate_default_covariates(n_subjects)
            
        # Generate individual parameters
        individual_params = self._generate_individual_parameters(n_subjects, covariates)
        
        # Simulate each subject
        subject_data = {}
        population_concentrations = []
        
        for subject_id in range(n_subjects):
            # Set individual parameters
            subject_params = {param: individual_params[param][subject_id] 
                            for param in individual_params.keys()}
            
            # Create individual model
            individual_model = self._create_individual_model(subject_params)
            
            # Simulate concentration
            concentrations = individual_model.simulate_concentration(time_points, dose_schedule)
            
            # Add residual error
            concentrations_with_error = self._add_residual_error(concentrations)
            
            subject_data[subject_id] = {
                'parameters': subject_params,
                'covariates': {cov: covariates[cov][subject_id] for cov in covariates.keys()},
                'time': time_points,
                'concentrations': concentrations_with_error,
                'true_concentrations': concentrations
            }
            
            population_concentrations.append(concentrations_with_error)
            
        # Calculate population statistics
        population_concentrations = np.array(population_concentrations)
        
        return {
            'subject_data': subject_data,
            'population_statistics': self._calculate_population_statistics(population_concentrations),
            'parameters_summary': self._summarize_parameters(individual_params),
            'time_points': time_points,
            'n_subjects': n_subjects
        }
        
    def _generate_default_covariates(self, n_subjects: int) -> Dict[str, np.ndarray]:
        """Generate default subject covariates."""
        np.random.seed(42)  # For reproducibility
        
        covariates = {
            'WEIGHT': np.random.normal(75, 15, n_subjects),  # Weight: 75±15 kg
            'AGE': np.random.uniform(18, 75, n_subjects),    # Age: 18-75 years
            'SEX': np.random.binomial(1, 0.5, n_subjects),   # Sex: 50% male
            'HEIGHT': np.random.normal(170, 10, n_subjects)  # Height: 170±10 cm
        }
        
        # Ensure realistic ranges
        covariates['WEIGHT'] = np.clip(covariates['WEIGHT'], 40, 150)
        covariates['HEIGHT'] = np.clip(covariates['HEIGHT'], 150, 200)
        
        return covariates
        
    def _generate_individual_parameters(self, 
                                      n_subjects: int,
                                      covariates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate individual subject parameters."""
        
        # Get parameter names
        param_names = list(self.population_params.typical_values.keys())
        n_params = len(param_names)
        
        # Create correlation matrix (simplified - assume independence for now)
        correlation_matrix = np.eye(n_params)
        
        # Generate random effects (log-normal distribution)
        individual_params = {}
        
        for i, param in enumerate(param_names):
            tv = self.population_params.typical_values[param]  # Typical value
            bsv = self.population_params.between_subject_variability[param] / 100.0  # BSV as fraction
            
            # Log-normal random effects
            log_param_values = np.random.normal(
                loc=np.log(tv),
                scale=bsv,
                size=n_subjects
            )
            
            # Apply covariate effects
            if self.population_params.covariates:
                for cov_name, cov_effects in self.population_params.covariates.items():
                    if param in cov_effects and cov_name in covariates:
                        cov_effect = cov_effects[param]
                        cov_values = covariates[cov_name]
                        
                        if cov_name == 'WEIGHT':
                            # Allometric scaling
                            reference_weight = 70.0  # kg
                            log_param_values += cov_effect * np.log(cov_values / reference_weight)
                            
                        elif cov_name == 'AGE':
                            # Linear age effect
                            reference_age = 40.0  # years
                            log_param_values += cov_effect * (cov_values - reference_age)
                            
                        elif cov_name == 'SEX':
                            # Categorical effect (0=female, 1=male)
                            log_param_values += cov_effect * cov_values
                            
            # Convert back to normal scale
            individual_params[param] = np.exp(log_param_values)
            
        return individual_params
        
    def _create_individual_model(self, parameters: Dict[str, float]):
        """Create individual model with specific parameters."""
        if self.base_model_type == 'one_compartment':
            return OneCompartmentModel(parameters)
        elif self.base_model_type == 'two_compartment':
            return TwoCompartmentModel(parameters)
        else:
            raise ValueError(f"Unknown base model type: {self.base_model_type}")
            
    def _add_residual_error(self, concentrations: np.ndarray) -> np.ndarray:
        """Add residual error to concentrations."""
        cv = self.population_params.residual_variability / 100.0
        
        # Proportional error model
        error = np.random.normal(1.0, cv, len(concentrations))
        concentrations_with_error = concentrations * error
        
        # Ensure non-negative concentrations
        concentrations_with_error = np.maximum(concentrations_with_error, 0.0)
        
        return concentrations_with_error
        
    def _calculate_population_statistics(self, concentrations: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate population concentration statistics."""
        return {
            'mean': np.mean(concentrations, axis=0),
            'median': np.median(concentrations, axis=0),
            'std': np.std(concentrations, axis=0),
            'percentile_5': np.percentile(concentrations, 5, axis=0),
            'percentile_25': np.percentile(concentrations, 25, axis=0),
            'percentile_75': np.percentile(concentrations, 75, axis=0),
            'percentile_95': np.percentile(concentrations, 95, axis=0)
        }
        
    def _summarize_parameters(self, individual_params: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Summarize individual parameter distributions."""
        summary = {}
        
        for param, values in individual_params.items():
            summary[param] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'cv_percent': (np.std(values) / np.mean(values)) * 100,
                'min': np.min(values),
                'max': np.max(values)
            }
            
        return summary
        
    def fit_population_model(self,
                           subject_data: Dict[int, Dict[str, Any]],
                           method: str = 'naive_pooled') -> Dict[str, Any]:
        """Fit population model to observed data.
        
        Args:
            subject_data: Dictionary with subject concentration-time data
            method: Fitting method ('naive_pooled', 'two_stage', 'nlme')
            
        Returns:
            Fitted population parameters and results
        """
        if method == 'naive_pooled':
            return self._fit_naive_pooled(subject_data)
        elif method == 'two_stage':
            return self._fit_two_stage(subject_data)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
            
    def _fit_naive_pooled(self, subject_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Naive pooled data approach."""
        
        # Pool all data
        all_times = []
        all_concentrations = []
        all_dose_schedules = []
        
        for subject_id, data in subject_data.items():
            all_times.extend(data['time'])
            all_concentrations.extend(data['concentrations'])
            # Assume same dosing schedule for simplicity
            all_dose_schedules.append({0.0: 100.0})
            
        # Fit single model to pooled data
        pooled_times = np.array(all_times)
        pooled_concentrations = np.array(all_concentrations)
        
        # Use first dose schedule as representative
        dose_schedule = all_dose_schedules[0]
        
        fitted_params = self.base_model.fit_parameters(
            pooled_times,
            pooled_concentrations,
            dose_schedule
        )
        
        return {
            'method': 'naive_pooled',
            'fitted_parameters': fitted_params,
            'population_parameters': PopulationParameters(
                typical_values=fitted_params,
                between_subject_variability={param: 30.0 for param in fitted_params.keys()},
                residual_variability=20.0
            )
        }
        
    def _fit_two_stage(self, subject_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Two-stage approach: fit individuals then pool."""
        
        individual_fitted_params = {}
        
        # Stage 1: Fit each subject individually
        for subject_id, data in subject_data.items():
            try:
                # Assume simple dose schedule
                dose_schedule = {0.0: 100.0}
                
                fitted_params = self.base_model.fit_parameters(
                    data['time'],
                    data['concentrations'],
                    dose_schedule
                )
                
                individual_fitted_params[subject_id] = fitted_params
                
            except Exception as e:
                self.logger.warning(f"Failed to fit subject {subject_id}: {e}")
                continue
                
        # Stage 2: Calculate population statistics
        if individual_fitted_params:
            param_names = list(list(individual_fitted_params.values())[0].keys())
            
            population_stats = {}
            for param in param_names:
                param_values = [params[param] for params in individual_fitted_params.values() 
                              if param in params]
                
                if param_values:
                    population_stats[param] = {
                        'typical_value': np.mean(param_values),
                        'bsv_cv': (np.std(param_values) / np.mean(param_values)) * 100
                    }
                    
            typical_values = {param: stats['typical_value'] 
                            for param, stats in population_stats.items()}
            bsv_values = {param: stats['bsv_cv'] 
                         for param, stats in population_stats.items()}
            
            population_params = PopulationParameters(
                typical_values=typical_values,
                between_subject_variability=bsv_values,
                residual_variability=20.0  # Default
            )
            
            return {
                'method': 'two_stage',
                'individual_parameters': individual_fitted_params,
                'population_parameters': population_params,
                'population_statistics': population_stats
            }
        else:
            raise ValueError("No subjects could be fitted individually")


class PopulationPDModel:
    """Population pharmacodynamic model."""
    
    def __init__(self,
                 pd_model_type: str = 'emax',
                 population_params: PopulationParameters = None):
        """Initialize population PD model.
        
        Args:
            pd_model_type: PD model type ('emax', 'linear', 'sigmoid_emax')
            population_params: Population PD parameters
        """
        self.pd_model_type = pd_model_type
        self.population_params = population_params or self._default_pd_params()
        self.logger = logging.getLogger(__name__)
        
    def _default_pd_params(self) -> PopulationParameters:
        """Default population PD parameters."""
        if self.pd_model_type == 'emax':
            return PopulationParameters(
                typical_values={
                    'EMAX': 100.0,  # Maximum effect
                    'EC50': 10.0,   # Concentration for 50% effect
                    'E0': 0.0       # Baseline effect
                },
                between_subject_variability={
                    'EMAX': 20.0,   # 20% CV
                    'EC50': 40.0,   # 40% CV
                    'E0': 30.0      # 30% CV
                },
                residual_variability=15.0
            )
        else:
            raise ValueError(f"Unknown PD model type: {self.pd_model_type}")
            
    def simulate_population_pd(self,
                              concentrations: np.ndarray,
                              n_subjects: int,
                              covariates: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """Simulate population PD response.
        
        Args:
            concentrations: Concentration time series for each subject
            n_subjects: Number of subjects
            covariates: Subject covariates
            
        Returns:
            Simulated PD responses
        """
        if covariates is None:
            covariates = {'WEIGHT': np.random.normal(75, 15, n_subjects)}
            
        # Generate individual PD parameters
        individual_params = self._generate_pd_parameters(n_subjects, covariates)
        
        # Simulate PD response for each subject
        pd_responses = []
        
        for subject_id in range(n_subjects):
            subject_concentrations = concentrations[subject_id]
            subject_params = {param: individual_params[param][subject_id] 
                            for param in individual_params.keys()}
            
            # Calculate PD response
            pd_response = self._calculate_pd_response(subject_concentrations, subject_params)
            
            # Add residual error
            pd_response_with_error = self._add_pd_residual_error(pd_response)
            
            pd_responses.append(pd_response_with_error)
            
        return {
            'pd_responses': np.array(pd_responses),
            'individual_pd_parameters': individual_params,
            'true_responses': [self._calculate_pd_response(concentrations[i], 
                                                         {param: individual_params[param][i] 
                                                          for param in individual_params.keys()})
                             for i in range(n_subjects)]
        }
        
    def _generate_pd_parameters(self, n_subjects: int, covariates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate individual PD parameters."""
        individual_params = {}
        
        for param in self.population_params.typical_values.keys():
            tv = self.population_params.typical_values[param]
            bsv = self.population_params.between_subject_variability[param] / 100.0
            
            # Log-normal distribution for positive parameters
            if param in ['EMAX', 'EC50']:
                log_param_values = np.random.normal(np.log(tv), bsv, n_subjects)
                individual_params[param] = np.exp(log_param_values)
            else:  # Normal distribution for E0
                individual_params[param] = np.random.normal(tv, tv * bsv, n_subjects)
                
        return individual_params
        
    def _calculate_pd_response(self, concentrations: np.ndarray, parameters: Dict[str, float]) -> np.ndarray:
        """Calculate PD response from concentrations."""
        
        if self.pd_model_type == 'emax':
            EMAX = parameters['EMAX']
            EC50 = parameters['EC50']
            E0 = parameters.get('E0', 0.0)
            
            # Emax model: E = E0 + (EMAX * C) / (EC50 + C)
            effect = E0 + (EMAX * concentrations) / (EC50 + concentrations)
            
        elif self.pd_model_type == 'linear':
            slope = parameters.get('SLOPE', 1.0)
            intercept = parameters.get('INTERCEPT', 0.0)
            
            effect = intercept + slope * concentrations
            
        elif self.pd_model_type == 'sigmoid_emax':
            EMAX = parameters['EMAX']
            EC50 = parameters['EC50']
            E0 = parameters.get('E0', 0.0)
            HILL = parameters.get('HILL', 1.0)
            
            # Sigmoid Emax: E = E0 + (EMAX * C^HILL) / (EC50^HILL + C^HILL)
            effect = E0 + (EMAX * np.power(concentrations, HILL)) / (np.power(EC50, HILL) + np.power(concentrations, HILL))
            
        else:
            raise ValueError(f"Unknown PD model type: {self.pd_model_type}")
            
        return effect
        
    def _add_pd_residual_error(self, pd_response: np.ndarray) -> np.ndarray:
        """Add residual error to PD response."""
        cv = self.population_params.residual_variability / 100.0
        
        # Additive error model
        error = np.random.normal(0.0, np.abs(pd_response) * cv, len(pd_response))
        pd_response_with_error = pd_response + error
        
        return pd_response_with_error