"""
Biomarker and pharmacodynamic response models.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from abc import ABC, abstractmethod


class BiomarkerModel(ABC):
    """Abstract base class for biomarker response models."""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize biomarker model.
        
        Args:
            parameters: Model parameters
        """
        self.parameters = parameters or {}
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def predict_response(self, 
                        concentrations: np.ndarray,
                        time_points: np.ndarray = None) -> np.ndarray:
        """Predict biomarker response from drug concentrations.
        
        Args:
            concentrations: Drug concentration time series
            time_points: Time points (optional)
            
        Returns:
            Predicted biomarker responses
        """
        pass
        
    @abstractmethod
    def fit_model(self,
                 concentrations: np.ndarray,
                 biomarker_data: np.ndarray,
                 time_points: np.ndarray = None) -> Dict[str, float]:
        """Fit model parameters to observed data.
        
        Args:
            concentrations: Drug concentrations
            biomarker_data: Observed biomarker measurements
            time_points: Time points
            
        Returns:
            Fitted parameters
        """
        pass


class DirectResponseModel(BiomarkerModel):
    """Direct response model: E = E0 + Slope × C."""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize direct response model.
        
        Expected parameters:
        - E0: Baseline biomarker level
        - SLOPE: Linear slope parameter
        """
        super().__init__(parameters)
        
        # Default parameters
        if 'E0' not in self.parameters:
            self.parameters['E0'] = 10.0  # Baseline biomarker level
        if 'SLOPE' not in self.parameters:
            self.parameters['SLOPE'] = -0.5  # Negative for suppression
            
    def predict_response(self, 
                        concentrations: np.ndarray,
                        time_points: np.ndarray = None) -> np.ndarray:
        """Predict biomarker using direct linear model."""
        
        E0 = self.parameters['E0']
        slope = self.parameters['SLOPE']
        
        # Linear response model
        response = E0 + slope * concentrations
        
        # Ensure non-negative biomarker levels
        response = np.maximum(response, 0.0)
        
        return response
        
    def fit_model(self,
                 concentrations: np.ndarray,
                 biomarker_data: np.ndarray,
                 time_points: np.ndarray = None) -> Dict[str, float]:
        """Fit direct response model parameters."""
        
        def objective(params):
            """Objective function for parameter fitting."""
            E0, slope = params
            
            # Update parameters
            old_params = self.parameters.copy()
            self.parameters.update({'E0': E0, 'SLOPE': slope})
            
            try:
                # Predict biomarker response
                predicted = self.predict_response(concentrations, time_points)
                
                # Calculate residuals
                residuals = biomarker_data - predicted
                mse = np.mean(residuals**2)
                
                # Restore old parameters
                self.parameters = old_params
                
                return mse
                
            except Exception as e:
                self.logger.warning(f"Direct model fitting error: {e}")
                self.parameters = old_params
                return 1e6
                
        # Initial guess
        initial_guess = [self.parameters['E0'], self.parameters['SLOPE']]
        
        # Parameter bounds
        bounds = [(0.1, 100.0), (-10.0, 10.0)]  # E0 > 0, slope can be negative
        
        # Optimize
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            fitted_params = {'E0': result.x[0], 'SLOPE': result.x[1]}
            self.parameters.update(fitted_params)
            return fitted_params
        else:
            self.logger.warning("Direct model fitting failed")
            return self.parameters


class EmaxModel(BiomarkerModel):
    """Emax pharmacodynamic model: E = E0 + (EMAX × C) / (EC50 + C)."""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize Emax model.
        
        Expected parameters:
        - E0: Baseline biomarker level
        - EMAX: Maximum effect
        - EC50: Concentration producing 50% of maximum effect
        """
        super().__init__(parameters)
        
        # Default parameters
        defaults = {
            'E0': 10.0,     # Baseline
            'EMAX': -8.0,   # Maximum suppression (negative for biomarker reduction)
            'EC50': 5.0     # IC50
        }
        
        for param, default_val in defaults.items():
            if param not in self.parameters:
                self.parameters[param] = default_val
                
    def predict_response(self, 
                        concentrations: np.ndarray,
                        time_points: np.ndarray = None) -> np.ndarray:
        """Predict biomarker using Emax model."""
        
        E0 = self.parameters['E0']
        EMAX = self.parameters['EMAX']
        EC50 = self.parameters['EC50']
        
        # Emax model
        effect = EMAX * concentrations / (EC50 + concentrations)
        response = E0 + effect
        
        # Ensure non-negative biomarker levels
        response = np.maximum(response, 0.0)
        
        return response
        
    def fit_model(self,
                 concentrations: np.ndarray,
                 biomarker_data: np.ndarray,
                 time_points: np.ndarray = None) -> Dict[str, float]:
        """Fit Emax model parameters."""
        
        def objective(params):
            """Objective function for Emax fitting."""
            E0, EMAX, EC50 = params
            
            old_params = self.parameters.copy()
            self.parameters.update({'E0': E0, 'EMAX': EMAX, 'EC50': EC50})
            
            try:
                predicted = self.predict_response(concentrations, time_points)
                residuals = biomarker_data - predicted
                mse = np.mean(residuals**2)
                
                self.parameters = old_params
                return mse
                
            except Exception as e:
                self.logger.warning(f"Emax fitting error: {e}")
                self.parameters = old_params
                return 1e6
                
        # Initial guess
        initial_guess = [
            self.parameters['E0'],
            self.parameters['EMAX'],
            self.parameters['EC50']
        ]
        
        # Parameter bounds
        bounds = [
            (0.1, 100.0),   # E0 > 0
            (-20.0, 20.0),  # EMAX can be positive or negative
            (0.1, 100.0)    # EC50 > 0
        ]
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            fitted_params = {
                'E0': result.x[0],
                'EMAX': result.x[1],
                'EC50': result.x[2]
            }
            self.parameters.update(fitted_params)
            return fitted_params
        else:
            self.logger.warning("Emax model fitting failed")
            return self.parameters


class IndirectResponseModel(BiomarkerModel):
    """Indirect response model with turnover kinetics."""
    
    def __init__(self, 
                 model_type: str = 'inhibition_production',
                 parameters: Dict[str, float] = None):
        """Initialize indirect response model.
        
        Args:
            model_type: Type of indirect model
                - 'inhibition_production': Drug inhibits biomarker production
                - 'stimulation_production': Drug stimulates biomarker production
                - 'inhibition_loss': Drug inhibits biomarker elimination
                - 'stimulation_loss': Drug stimulates biomarker elimination
                
        Expected parameters:
        - KIN: Zero-order production rate
        - KOUT: First-order elimination rate
        - IC50/EC50: Concentration for 50% inhibition/stimulation
        - IMAX/EMAX: Maximum inhibition/stimulation
        """
        super().__init__(parameters)
        self.model_type = model_type
        
        # Default parameters
        defaults = {
            'KIN': 10.0,    # Production rate (units/h)
            'KOUT': 0.1,    # Elimination rate (1/h)
            'IC50': 5.0,    # IC50 or EC50
            'IMAX': 0.9     # Maximum inhibition (0-1) or stimulation (>1)
        }
        
        for param, default_val in defaults.items():
            if param not in self.parameters:
                self.parameters[param] = default_val
                
    def predict_response(self, 
                        concentrations: np.ndarray,
                        time_points: np.ndarray) -> np.ndarray:
        """Predict biomarker using indirect response model."""
        
        if time_points is None:
            raise ValueError("Time points required for indirect response model")
            
        # Initial condition (steady-state baseline)
        KIN = self.parameters['KIN']
        KOUT = self.parameters['KOUT']
        
        baseline = KIN / KOUT
        y0 = [baseline]
        
        # Create concentration interpolation function
        def concentration_at_time(t):
            """Get concentration at specific time point."""
            if len(concentrations) == 1:
                return concentrations[0]
            else:
                # Linear interpolation
                idx = np.searchsorted(time_points, t, side='right') - 1
                idx = np.clip(idx, 0, len(concentrations) - 1)
                return concentrations[idx]
                
        # ODE system
        def indirect_ode(y, t):
            """Indirect response ODE."""
            biomarker = y[0]
            concentration = concentration_at_time(t)
            
            IC50 = self.parameters['IC50']
            IMAX = self.parameters['IMAX']
            
            # Drug effect on production or elimination
            if self.model_type == 'inhibition_production':
                # Drug inhibits production: R_in = KIN * (1 - IMAX * C/(IC50 + C))
                drug_effect = 1 - IMAX * concentration / (IC50 + concentration)
                R_in = KIN * drug_effect
                R_out = KOUT * biomarker
                
            elif self.model_type == 'stimulation_production':
                # Drug stimulates production: R_in = KIN * (1 + EMAX * C/(EC50 + C))
                drug_effect = 1 + IMAX * concentration / (IC50 + concentration)
                R_in = KIN * drug_effect
                R_out = KOUT * biomarker
                
            elif self.model_type == 'inhibition_loss':
                # Drug inhibits elimination: R_out = KOUT * (1 - IMAX * C/(IC50 + C)) * biomarker
                drug_effect = 1 - IMAX * concentration / (IC50 + concentration)
                R_in = KIN
                R_out = KOUT * drug_effect * biomarker
                
            elif self.model_type == 'stimulation_loss':
                # Drug stimulates elimination: R_out = KOUT * (1 + EMAX * C/(EC50 + C)) * biomarker
                drug_effect = 1 + IMAX * concentration / (IC50 + concentration)
                R_in = KIN
                R_out = KOUT * drug_effect * biomarker
                
            else:
                raise ValueError(f"Unknown indirect model type: {self.model_type}")
                
            # Rate of change
            dA_dt = R_in - R_out
            
            return [dA_dt]
            
        # Solve ODE
        solution = odeint(indirect_ode, y0, time_points)
        
        return solution[:, 0]  # Return biomarker levels
        
    def fit_model(self,
                 concentrations: np.ndarray,
                 biomarker_data: np.ndarray,
                 time_points: np.ndarray) -> Dict[str, float]:
        """Fit indirect response model parameters."""
        
        def objective(params):
            """Objective function for indirect model fitting."""
            KIN, KOUT, IC50, IMAX = params
            
            old_params = self.parameters.copy()
            self.parameters.update({
                'KIN': KIN,
                'KOUT': KOUT,
                'IC50': IC50,
                'IMAX': IMAX
            })
            
            try:
                predicted = self.predict_response(concentrations, time_points)
                residuals = biomarker_data - predicted
                mse = np.mean(residuals**2)
                
                self.parameters = old_params
                return mse
                
            except Exception as e:
                self.logger.warning(f"Indirect model fitting error: {e}")
                self.parameters = old_params
                return 1e6
                
        # Initial guess
        initial_guess = [
            self.parameters['KIN'],
            self.parameters['KOUT'],
            self.parameters['IC50'],
            self.parameters['IMAX']
        ]
        
        # Parameter bounds
        bounds = [
            (0.1, 1000.0),  # KIN > 0
            (0.001, 10.0),  # KOUT > 0
            (0.1, 100.0),   # IC50 > 0
            (0.01, 2.0)     # IMAX: 0-1 for inhibition, >1 for stimulation
        ]
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            fitted_params = {
                'KIN': result.x[0],
                'KOUT': result.x[1],
                'IC50': result.x[2],
                'IMAX': result.x[3]
            }
            self.parameters.update(fitted_params)
            return fitted_params
        else:
            self.logger.warning("Indirect response model fitting failed")
            return self.parameters
            
    def steady_state_response(self, concentration: float) -> float:
        """Calculate steady-state biomarker level at given concentration."""
        
        KIN = self.parameters['KIN']
        KOUT = self.parameters['KOUT']
        IC50 = self.parameters['IC50']
        IMAX = self.parameters['IMAX']
        
        if self.model_type == 'inhibition_production':
            drug_effect = 1 - IMAX * concentration / (IC50 + concentration)
            steady_state = (KIN * drug_effect) / KOUT
            
        elif self.model_type == 'stimulation_production':
            drug_effect = 1 + IMAX * concentration / (IC50 + concentration)
            steady_state = (KIN * drug_effect) / KOUT
            
        elif self.model_type == 'inhibition_loss':
            drug_effect = 1 - IMAX * concentration / (IC50 + concentration)
            steady_state = KIN / (KOUT * drug_effect)
            
        elif self.model_type == 'stimulation_loss':
            drug_effect = 1 + IMAX * concentration / (IC50 + concentration)
            steady_state = KIN / (KOUT * drug_effect)
            
        else:
            steady_state = KIN / KOUT  # Baseline
            
        return steady_state


class DelayedResponseModel(BiomarkerModel):
    """Delayed response model with effect compartment."""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize delayed response model.
        
        Expected parameters:
        - E0: Baseline effect
        - EMAX: Maximum effect
        - EC50: EC50 in effect compartment
        - KEO: Effect compartment equilibration rate constant
        """
        super().__init__(parameters)
        
        defaults = {
            'E0': 10.0,
            'EMAX': -8.0,
            'EC50': 5.0,
            'KEO': 0.1  # 1/h
        }
        
        for param, default_val in defaults.items():
            if param not in self.parameters:
                self.parameters[param] = default_val
                
    def predict_response(self, 
                        concentrations: np.ndarray,
                        time_points: np.ndarray) -> np.ndarray:
        """Predict biomarker with effect compartment delay."""
        
        if time_points is None:
            raise ValueError("Time points required for delayed response model")
            
        KEO = self.parameters['KEO']
        
        # Initial condition (no drug in effect compartment)
        y0 = [0.0]
        
        def concentration_at_time(t):
            """Get plasma concentration at time t."""
            if len(concentrations) == 1:
                return concentrations[0]
            else:
                idx = np.searchsorted(time_points, t, side='right') - 1
                idx = np.clip(idx, 0, len(concentrations) - 1)
                return concentrations[idx]
                
        # Effect compartment ODE
        def effect_compartment_ode(y, t):
            """Effect compartment differential equation."""
            Ce = y[0]  # Concentration in effect compartment
            Cp = concentration_at_time(t)  # Plasma concentration
            
            dCe_dt = KEO * (Cp - Ce)
            
            return [dCe_dt]
            
        # Solve for effect compartment concentrations
        solution = odeint(effect_compartment_ode, y0, time_points)
        effect_concentrations = solution[:, 0]
        
        # Calculate PD response from effect compartment concentrations
        E0 = self.parameters['E0']
        EMAX = self.parameters['EMAX']
        EC50 = self.parameters['EC50']
        
        effects = EMAX * effect_concentrations / (EC50 + effect_concentrations)
        biomarker_response = E0 + effects
        
        # Ensure non-negative
        biomarker_response = np.maximum(biomarker_response, 0.0)
        
        return biomarker_response
        
    def fit_model(self,
                 concentrations: np.ndarray,
                 biomarker_data: np.ndarray,
                 time_points: np.ndarray) -> Dict[str, float]:
        """Fit delayed response model parameters."""
        
        def objective(params):
            """Objective function."""
            E0, EMAX, EC50, KEO = params
            
            old_params = self.parameters.copy()
            self.parameters.update({
                'E0': E0,
                'EMAX': EMAX,
                'EC50': EC50,
                'KEO': KEO
            })
            
            try:
                predicted = self.predict_response(concentrations, time_points)
                residuals = biomarker_data - predicted
                mse = np.mean(residuals**2)
                
                self.parameters = old_params
                return mse
                
            except Exception as e:
                self.logger.warning(f"Delayed model fitting error: {e}")
                self.parameters = old_params
                return 1e6
                
        initial_guess = [
            self.parameters['E0'],
            self.parameters['EMAX'],
            self.parameters['EC50'],
            self.parameters['KEO']
        ]
        
        bounds = [
            (0.1, 100.0),   # E0 > 0
            (-20.0, 20.0),  # EMAX
            (0.1, 100.0),   # EC50 > 0
            (0.01, 5.0)     # KEO > 0
        ]
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            fitted_params = {
                'E0': result.x[0],
                'EMAX': result.x[1],
                'EC50': result.x[2],
                'KEO': result.x[3]
            }
            self.parameters.update(fitted_params)
            return fitted_params
        else:
            self.logger.warning("Delayed response model fitting failed")
            return self.parameters