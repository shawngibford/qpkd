"""
Classical compartment models for pharmacokinetic analysis.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any
import logging
from abc import ABC, abstractmethod


class CompartmentModel(ABC):
    """Abstract base class for compartment models."""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize compartment model.
        
        Args:
            parameters: Model parameters (clearance, volume, etc.)
        """
        self.parameters = parameters or {}
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def ode_system(self, y: np.ndarray, t: float, dose_func: callable) -> np.ndarray:
        """Define the ODE system for the compartment model.
        
        Args:
            y: State vector (concentrations in each compartment)
            t: Time
            dose_func: Function returning dose rate at time t
            
        Returns:
            Derivative vector dy/dt
        """
        pass
        
    @abstractmethod
    def initial_conditions(self, dose: float) -> np.ndarray:
        """Get initial conditions for the ODE system.
        
        Args:
            dose: Initial dose
            
        Returns:
            Initial state vector
        """
        pass
        
    def simulate_concentration(self, 
                             time_points: np.ndarray,
                             dose_schedule: Dict[float, float],
                             **kwargs) -> np.ndarray:
        """Simulate drug concentration over time.
        
        Args:
            time_points: Array of time points
            dose_schedule: Dictionary mapping time -> dose
            **kwargs: Additional simulation parameters
            
        Returns:
            Concentration time series
        """
        def dose_function(t):
            """Dose input function."""
            # Find the most recent dose
            dose_times = sorted([dt for dt in dose_schedule.keys() if dt <= t])
            if dose_times:
                return dose_schedule[dose_times[-1]]
            return 0.0
            
        # Initial conditions
        y0 = self.initial_conditions(dose_schedule.get(0.0, 0.0))
        
        # Solve ODE
        solution = odeint(self.ode_system, y0, time_points, args=(dose_function,))
        
        # Return plasma concentration (typically first compartment)
        return solution[:, 0]
        
    def fit_parameters(self, 
                      time_data: np.ndarray,
                      concentration_data: np.ndarray,
                      dose_schedule: Dict[float, float],
                      initial_guess: Dict[str, float] = None) -> Dict[str, float]:
        """Fit model parameters to observed data.
        
        Args:
            time_data: Observed time points
            concentration_data: Observed concentrations
            dose_schedule: Dosing schedule
            initial_guess: Initial parameter estimates
            
        Returns:
            Fitted parameters
        """
        if initial_guess is None:
            initial_guess = self._default_initial_guess()
            
        def objective(params_array):
            """Objective function for parameter fitting."""
            param_names = list(initial_guess.keys())
            params_dict = dict(zip(param_names, params_array))
            
            # Update model parameters
            old_params = self.parameters.copy()
            self.parameters.update(params_dict)
            
            try:
                # Simulate with these parameters
                predicted = self.simulate_concentration(time_data, dose_schedule)
                
                # Calculate residuals
                residuals = concentration_data - predicted
                mse = np.mean(residuals**2)
                
                # Restore old parameters
                self.parameters = old_params
                
                return mse
                
            except Exception as e:
                self.logger.warning(f"Simulation failed with parameters {params_dict}: {e}")
                # Restore old parameters
                self.parameters = old_params
                return 1e6
                
        # Parameter bounds
        bounds = self._get_parameter_bounds(initial_guess)
        
        # Optimize
        result = minimize(
            objective,
            list(initial_guess.values()),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Update parameters with fitted values
        if result.success:
            param_names = list(initial_guess.keys())
            fitted_params = dict(zip(param_names, result.x))
            self.parameters.update(fitted_params)
            return fitted_params
        else:
            self.logger.warning(f"Parameter fitting failed: {result.message}")
            return initial_guess
            
    @abstractmethod
    def _default_initial_guess(self) -> Dict[str, float]:
        """Get default initial parameter guess."""
        pass
        
    @abstractmethod
    def _get_parameter_bounds(self, initial_guess: Dict[str, float]) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        pass


class OneCompartmentModel(CompartmentModel):
    """One-compartment pharmacokinetic model."""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize one-compartment model.
        
        Expected parameters:
        - CL: Clearance (L/h)
        - V: Volume of distribution (L)
        - ka: Absorption rate constant (1/h) - optional for oral dosing
        """
        super().__init__(parameters)
        
        # Default parameters
        default_params = {
            'CL': 10.0,  # L/h
            'V': 50.0,   # L
            'ka': 1.0    # 1/h (for oral dosing)
        }
        
        for param, default_val in default_params.items():
            if param not in self.parameters:
                self.parameters[param] = default_val
                
    def ode_system(self, y: np.ndarray, t: float, dose_func: callable) -> np.ndarray:
        """One-compartment ODE system.
        
        State variables:
        y[0]: Amount in central compartment (mg)
        y[1]: Amount in depot compartment (mg) - for oral dosing
        """
        CL = self.parameters['CL']
        V = self.parameters['V']
        ka = self.parameters.get('ka', 0)  # Zero for IV dosing
        
        dose_rate = dose_func(t)
        
        if len(y) == 1:  # IV dosing
            dA1_dt = dose_rate - (CL / V) * y[0]
            return np.array([dA1_dt])
        else:  # Oral dosing
            dA_depot_dt = dose_rate - ka * y[1]
            dA1_dt = ka * y[1] - (CL / V) * y[0]
            return np.array([dA1_dt, dA_depot_dt])
            
    def initial_conditions(self, dose: float) -> np.ndarray:
        """Get initial conditions."""
        ka = self.parameters.get('ka', 0)
        
        if ka > 0:  # Oral dosing
            return np.array([0.0, dose])  # [central, depot]
        else:  # IV dosing
            return np.array([dose])  # [central]
            
    def simulate_concentration(self, 
                             time_points: np.ndarray,
                             dose_schedule: Dict[float, float],
                             **kwargs) -> np.ndarray:
        """Simulate plasma concentration."""
        amounts = super().simulate_concentration(time_points, dose_schedule, **kwargs)
        
        # Convert amount to concentration
        V = self.parameters['V']
        
        if amounts.ndim == 1:
            concentrations = amounts / V
        else:
            concentrations = amounts[:, 0] / V  # Central compartment
            
        return concentrations
        
    def _default_initial_guess(self) -> Dict[str, float]:
        """Default parameter initial guess."""
        return {
            'CL': 10.0,
            'V': 50.0,
            'ka': 1.0
        }
        
    def _get_parameter_bounds(self, initial_guess: Dict[str, float]) -> List[Tuple[float, float]]:
        """Parameter bounds for optimization."""
        bounds = []
        
        for param in initial_guess.keys():
            if param == 'CL':
                bounds.append((0.1, 100.0))
            elif param == 'V':
                bounds.append((1.0, 200.0))
            elif param == 'ka':
                bounds.append((0.01, 10.0))
            else:
                bounds.append((0.001, 1000.0))
                
        return bounds
        
    def steady_state_concentration(self, dose: float, dosing_interval: float) -> float:
        """Calculate steady-state concentration for multiple dosing.
        
        Args:
            dose: Dose amount (mg)
            dosing_interval: Dosing interval (h)
            
        Returns:
            Average steady-state concentration (mg/L)
        """
        CL = self.parameters['CL']
        V = self.parameters['V']
        ka = self.parameters.get('ka', np.inf)  # Large value for IV
        
        # Elimination rate constant
        ke = CL / V
        
        if ka == np.inf:  # IV bolus
            css_avg = dose / (CL * dosing_interval)
        else:  # Oral dosing
            F = self.parameters.get('F', 1.0)  # Bioavailability
            css_avg = F * dose / (CL * dosing_interval)
            
        return css_avg


class TwoCompartmentModel(CompartmentModel):
    """Two-compartment pharmacokinetic model."""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize two-compartment model.
        
        Expected parameters:
        - CL: Clearance (L/h)
        - V1: Central volume (L)
        - V2: Peripheral volume (L)
        - Q: Inter-compartmental clearance (L/h)
        - ka: Absorption rate constant (1/h) - optional
        """
        super().__init__(parameters)
        
        # Default parameters
        default_params = {
            'CL': 10.0,  # L/h
            'V1': 30.0,  # L
            'V2': 40.0,  # L
            'Q': 5.0,    # L/h
            'ka': 1.0    # 1/h
        }
        
        for param, default_val in default_params.items():
            if param not in self.parameters:
                self.parameters[param] = default_val
                
    def ode_system(self, y: np.ndarray, t: float, dose_func: callable) -> np.ndarray:
        """Two-compartment ODE system.
        
        State variables:
        y[0]: Amount in central compartment (mg)
        y[1]: Amount in peripheral compartment (mg)
        y[2]: Amount in depot compartment (mg) - for oral dosing
        """
        CL = self.parameters['CL']
        V1 = self.parameters['V1']
        V2 = self.parameters['V2']
        Q = self.parameters['Q']
        ka = self.parameters.get('ka', 0)
        
        dose_rate = dose_func(t)
        
        if len(y) == 2:  # IV dosing
            A1, A2 = y
            dA1_dt = dose_rate - (CL / V1) * A1 - (Q / V1) * A1 + (Q / V2) * A2
            dA2_dt = (Q / V1) * A1 - (Q / V2) * A2
            return np.array([dA1_dt, dA2_dt])
        else:  # Oral dosing
            A1, A2, A_depot = y
            dA1_dt = ka * A_depot - (CL / V1) * A1 - (Q / V1) * A1 + (Q / V2) * A2
            dA2_dt = (Q / V1) * A1 - (Q / V2) * A2
            dA_depot_dt = dose_rate - ka * A_depot
            return np.array([dA1_dt, dA2_dt, dA_depot_dt])
            
    def initial_conditions(self, dose: float) -> np.ndarray:
        """Get initial conditions."""
        ka = self.parameters.get('ka', 0)
        
        if ka > 0:  # Oral dosing
            return np.array([0.0, 0.0, dose])  # [central, peripheral, depot]
        else:  # IV dosing
            return np.array([dose, 0.0])  # [central, peripheral]
            
    def simulate_concentration(self, 
                             time_points: np.ndarray,
                             dose_schedule: Dict[float, float],
                             **kwargs) -> np.ndarray:
        """Simulate plasma concentration."""
        amounts = super().simulate_concentration(time_points, dose_schedule, **kwargs)
        
        # Convert amount to concentration (central compartment)
        V1 = self.parameters['V1']
        
        if amounts.ndim == 1:
            concentrations = amounts / V1
        else:
            concentrations = amounts[:, 0] / V1  # Central compartment
            
        return concentrations
        
    def _default_initial_guess(self) -> Dict[str, float]:
        """Default parameter initial guess."""
        return {
            'CL': 10.0,
            'V1': 30.0,
            'V2': 40.0,
            'Q': 5.0,
            'ka': 1.0
        }
        
    def _get_parameter_bounds(self, initial_guess: Dict[str, float]) -> List[Tuple[float, float]]:
        """Parameter bounds for optimization."""
        bounds = []
        
        for param in initial_guess.keys():
            if param == 'CL':
                bounds.append((0.1, 100.0))
            elif param in ['V1', 'V2']:
                bounds.append((1.0, 200.0))
            elif param == 'Q':
                bounds.append((0.1, 50.0))
            elif param == 'ka':
                bounds.append((0.01, 10.0))
            else:
                bounds.append((0.001, 1000.0))
                
        return bounds
        
    def distribution_half_life(self) -> float:
        """Calculate distribution half-life."""
        CL = self.parameters['CL']
        V1 = self.parameters['V1']
        V2 = self.parameters['V2']
        Q = self.parameters['Q']
        
        # Micro constants
        k10 = CL / V1
        k12 = Q / V1
        k21 = Q / V2
        
        # Hybrid rate constants
        alpha_beta_sum = k10 + k12 + k21
        alpha_beta_product = k10 * k21
        
        discriminant = (alpha_beta_sum**2 - 4 * alpha_beta_product)**0.5
        alpha = (alpha_beta_sum + discriminant) / 2
        beta = (alpha_beta_sum - discriminant) / 2
        
        # Distribution half-life (alpha phase)
        t_half_alpha = np.log(2) / alpha
        
        return t_half_alpha
        
    def elimination_half_life(self) -> float:
        """Calculate elimination half-life."""
        CL = self.parameters['CL']
        V1 = self.parameters['V1']
        V2 = self.parameters['V2']
        Q = self.parameters['Q']
        
        # Micro constants
        k10 = CL / V1
        k12 = Q / V1
        k21 = Q / V2
        
        # Hybrid rate constants
        alpha_beta_sum = k10 + k12 + k21
        alpha_beta_product = k10 * k21
        
        discriminant = (alpha_beta_sum**2 - 4 * alpha_beta_product)**0.5
        beta = (alpha_beta_sum - discriminant) / 2
        
        # Elimination half-life (beta phase)
        t_half_beta = np.log(2) / beta
        
        return t_half_beta
        
    def steady_state_concentration(self, dose: float, dosing_interval: float) -> float:
        """Calculate average steady-state concentration."""
        CL = self.parameters['CL']
        F = self.parameters.get('F', 1.0)  # Bioavailability
        
        css_avg = F * dose / (CL * dosing_interval)
        return css_avg