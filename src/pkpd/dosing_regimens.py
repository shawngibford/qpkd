"""
Dosing regimen modeling and optimization utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
import logging


@dataclass
class DoseEvent:
    """Single dose event."""
    time: float        # Time of dose (hours)
    amount: float      # Dose amount (mg)
    route: str = 'iv'  # Route of administration ('iv', 'oral', 'sc', etc.)


@dataclass
class DosingInterval:
    """Regular dosing interval."""
    dose_amount: float    # Dose per administration (mg)
    interval: float       # Time between doses (hours)
    n_doses: int         # Number of doses (0 = infinite)
    route: str = 'oral'  # Route of administration


class DosingRegimen:
    """Single dosing regimen management and optimization."""
    
    def __init__(self, 
                 regimen_type: str = 'single',
                 dose_events: List[DoseEvent] = None,
                 dosing_interval: DosingInterval = None):
        """Initialize dosing regimen.
        
        Args:
            regimen_type: 'single', 'multiple', 'irregular'
            dose_events: List of individual dose events
            dosing_interval: Regular dosing interval specification
        """
        self.regimen_type = regimen_type
        self.dose_events = dose_events or []
        self.dosing_interval = dosing_interval
        self.logger = logging.getLogger(__name__)
        
    def generate_dose_schedule(self, simulation_time: float = 168.0) -> Dict[float, float]:
        """Generate complete dose schedule.
        
        Args:
            simulation_time: Total simulation time (hours)
            
        Returns:
            Dictionary mapping time -> dose amount
        """
        dose_schedule = {}
        
        if self.regimen_type == 'single':
            # Single dose at t=0
            if self.dose_events:
                for event in self.dose_events:
                    dose_schedule[event.time] = event.amount
            else:
                dose_schedule[0.0] = 100.0  # Default single dose
                
        elif self.regimen_type == 'multiple':
            # Multiple regular doses
            if self.dosing_interval:
                interval = self.dosing_interval
                dose_times = np.arange(0, simulation_time, interval.interval)
                
                # Limit by number of doses if specified
                if interval.n_doses > 0:
                    dose_times = dose_times[:interval.n_doses]
                    
                for t in dose_times:
                    dose_schedule[t] = interval.dose_amount
                    
        elif self.regimen_type == 'irregular':
            # Irregular dosing from dose events
            for event in self.dose_events:
                if event.time <= simulation_time:
                    dose_schedule[event.time] = event.amount
                    
        return dose_schedule
        
    def optimize_single_dose(self,
                            target_concentration: float,
                            pk_model: Any,
                            time_point: float = 24.0,
                            dose_bounds: Tuple[float, float] = (1.0, 1000.0)) -> Dict[str, Any]:
        """Optimize single dose to achieve target concentration.
        
        Args:
            target_concentration: Target plasma concentration (mg/L)
            pk_model: Pharmacokinetic model for simulation
            time_point: Time point to evaluate concentration (hours)
            dose_bounds: (min_dose, max_dose) in mg
            
        Returns:
            Optimization results
        """
        def objective(dose_array):
            """Objective function for dose optimization."""
            dose = dose_array[0]
            
            try:
                # Simulate concentration
                time_points = np.array([0, time_point])
                dose_schedule = {0.0: dose}
                
                concentrations = pk_model.simulate_concentration(time_points, dose_schedule)
                predicted_conc = concentrations[-1]  # Concentration at time_point
                
                # Minimize squared error
                error = (predicted_conc - target_concentration) ** 2
                
                return error
                
            except Exception as e:
                self.logger.warning(f"Dose optimization error: {e}")
                return 1e6
                
        # Run optimization
        result = minimize(
            objective,
            [50.0],  # Initial guess
            method='L-BFGS-B',
            bounds=[dose_bounds]
        )
        
        optimal_dose = result.x[0]
        
        # Validate result
        try:
            time_points = np.array([0, time_point])
            dose_schedule = {0.0: optimal_dose}
            concentrations = pk_model.simulate_concentration(time_points, dose_schedule)
            achieved_concentration = concentrations[-1]
        except:
            achieved_concentration = 0.0
            
        return {
            'optimal_dose': optimal_dose,
            'target_concentration': target_concentration,
            'achieved_concentration': achieved_concentration,
            'time_point': time_point,
            'optimization_success': result.success,
            'relative_error': abs(achieved_concentration - target_concentration) / target_concentration if target_concentration > 0 else float('inf')
        }
        
    def optimize_multiple_dose(self,
                              target_steady_state: float,
                              pk_model: Any,
                              dosing_interval: float = 24.0,
                              dose_bounds: Tuple[float, float] = (1.0, 500.0)) -> Dict[str, Any]:
        """Optimize multiple dosing to achieve steady-state concentration.
        
        Args:
            target_steady_state: Target steady-state concentration (mg/L)
            pk_model: PK model
            dosing_interval: Dosing interval (hours)
            dose_bounds: Dose bounds (mg)
            
        Returns:
            Optimization results
        """
        def objective(dose_array):
            """Objective function for multiple dose optimization."""
            dose = dose_array[0]
            
            try:
                # Simulate to steady state (assume 7 half-lives)
                if hasattr(pk_model, 'parameters') and 'CL' in pk_model.parameters and 'V' in pk_model.parameters:
                    # Estimate half-life for simulation time
                    t_half = np.log(2) * pk_model.parameters['V'] / pk_model.parameters['CL']
                    sim_time = max(7 * t_half, 168)  # At least 1 week
                else:
                    sim_time = 168  # Default 1 week
                    
                # Create multiple dose schedule
                dose_times = np.arange(0, sim_time, dosing_interval)
                dose_schedule = {t: dose for t in dose_times}
                
                # Simulate
                time_points = np.linspace(0, sim_time, int(sim_time) + 1)
                concentrations = pk_model.simulate_concentration(time_points, dose_schedule)
                
                # Calculate average concentration in last dosing interval
                last_interval_start = dose_times[-1]
                last_interval_mask = time_points >= last_interval_start
                
                if np.sum(last_interval_mask) > 1:
                    steady_state_conc = np.mean(concentrations[last_interval_mask])
                else:
                    steady_state_conc = concentrations[-1]
                    
                # Minimize squared error
                error = (steady_state_conc - target_steady_state) ** 2
                
                return error
                
            except Exception as e:
                self.logger.warning(f"Multiple dose optimization error: {e}")
                return 1e6
                
        # Run optimization
        result = minimize(
            objective,
            [target_steady_state * 10],  # Initial guess based on target
            method='L-BFGS-B',
            bounds=[dose_bounds]
        )
        
        optimal_dose = result.x[0]
        
        # Validate result
        try:
            # Simulate optimal regimen
            sim_time = 168
            dose_times = np.arange(0, sim_time, dosing_interval)
            dose_schedule = {t: optimal_dose for t in dose_times}
            time_points = np.linspace(0, sim_time, int(sim_time) + 1)
            concentrations = pk_model.simulate_concentration(time_points, dose_schedule)
            
            # Calculate steady-state metrics
            last_interval_start = dose_times[-1]
            last_interval_mask = time_points >= last_interval_start
            achieved_steady_state = np.mean(concentrations[last_interval_mask])
            
            # Calculate peak and trough
            peak_conc = np.max(concentrations[last_interval_mask])
            trough_conc = np.min(concentrations[last_interval_mask])
            
        except:
            achieved_steady_state = 0.0
            peak_conc = 0.0
            trough_conc = 0.0
            
        return {
            'optimal_dose': optimal_dose,
            'dosing_interval': dosing_interval,
            'target_steady_state': target_steady_state,
            'achieved_steady_state': achieved_steady_state,
            'peak_concentration': peak_conc,
            'trough_concentration': trough_conc,
            'optimization_success': result.success,
            'relative_error': abs(achieved_steady_state - target_steady_state) / target_steady_state if target_steady_state > 0 else float('inf')
        }
        
    def calculate_bioequivalence(self, 
                                reference_regimen: 'DosingRegimen',
                                pk_model: Any,
                                simulation_time: float = 72.0) -> Dict[str, Any]:
        """Calculate bioequivalence metrics between two regimens.
        
        Args:
            reference_regimen: Reference dosing regimen
            pk_model: PK model for simulation
            simulation_time: Simulation time (hours)
            
        Returns:
            Bioequivalence metrics
        """
        # Simulate both regimens
        time_points = np.linspace(0, simulation_time, int(simulation_time * 4) + 1)  # 15-min intervals
        
        # Test regimen (self)
        test_schedule = self.generate_dose_schedule(simulation_time)
        test_concentrations = pk_model.simulate_concentration(time_points, test_schedule)
        
        # Reference regimen
        ref_schedule = reference_regimen.generate_dose_schedule(simulation_time)
        ref_concentrations = pk_model.simulate_concentration(time_points, ref_schedule)
        
        # Calculate AUC using trapezoidal rule
        test_auc = np.trapz(test_concentrations, time_points)
        ref_auc = np.trapz(ref_concentrations, time_points)
        
        # Calculate Cmax
        test_cmax = np.max(test_concentrations)
        ref_cmax = np.max(ref_concentrations)
        
        # Time to Cmax
        test_tmax = time_points[np.argmax(test_concentrations)]
        ref_tmax = time_points[np.argmax(ref_concentrations)]
        
        # Bioequivalence ratios
        auc_ratio = test_auc / ref_auc if ref_auc > 0 else 0
        cmax_ratio = test_cmax / ref_cmax if ref_cmax > 0 else 0
        
        # Bioequivalence criteria (90% CI should be within 0.8-1.25)
        auc_bioequivalent = 0.8 <= auc_ratio <= 1.25
        cmax_bioequivalent = 0.8 <= cmax_ratio <= 1.25
        
        return {
            'test_auc': test_auc,
            'reference_auc': ref_auc,
            'auc_ratio': auc_ratio,
            'test_cmax': test_cmax,
            'reference_cmax': ref_cmax,
            'cmax_ratio': cmax_ratio,
            'test_tmax': test_tmax,
            'reference_tmax': ref_tmax,
            'auc_bioequivalent': auc_bioequivalent,
            'cmax_bioequivalent': cmax_bioequivalent,
            'overall_bioequivalent': auc_bioequivalent and cmax_bioequivalent
        }


class MultipleDosingRegimen:
    """Advanced multiple dosing regimen optimization."""
    
    def __init__(self):
        """Initialize multiple dosing regimen optimizer."""
        self.logger = logging.getLogger(__name__)
        
    def optimize_loading_maintenance(self,
                                   target_concentration: float,
                                   pk_model: Any,
                                   maintenance_interval: float = 24.0,
                                   target_time: float = 2.0) -> Dict[str, Any]:
        """Optimize loading dose + maintenance regimen.
        
        Args:
            target_concentration: Target concentration (mg/L)
            pk_model: PK model
            maintenance_interval: Maintenance dosing interval (hours)
            target_time: Time to reach target (hours)
            
        Returns:
            Optimized loading and maintenance doses
        """
        def objective(doses):
            """Objective for loading + maintenance optimization."""
            loading_dose, maintenance_dose = doses
            
            try:
                # Create dosing schedule
                simulation_time = 168  # 1 week
                maintenance_times = np.arange(maintenance_interval, simulation_time, maintenance_interval)
                
                dose_schedule = {0.0: loading_dose}  # Loading dose
                for t in maintenance_times:
                    dose_schedule[t] = maintenance_dose
                    
                # Simulate
                time_points = np.linspace(0, simulation_time, int(simulation_time * 4) + 1)
                concentrations = pk_model.simulate_concentration(time_points, dose_schedule)
                
                # Find concentration at target time
                target_idx = np.argmin(np.abs(time_points - target_time))
                conc_at_target = concentrations[target_idx]
                
                # Penalty for missing target at target_time
                target_penalty = (conc_at_target - target_concentration) ** 2
                
                # Penalty for steady-state deviation (last 24 hours)
                steady_state_mask = time_points >= (simulation_time - 24)
                ss_concentrations = concentrations[steady_state_mask]
                ss_mean = np.mean(ss_concentrations)
                ss_penalty = (ss_mean - target_concentration) ** 2
                
                # Total penalty
                return target_penalty + 0.1 * ss_penalty
                
            except Exception as e:
                self.logger.warning(f"Loading/maintenance optimization error: {e}")
                return 1e6
                
        # Initial guess
        initial_guess = [target_concentration * 50, target_concentration * 20]  # Rough estimates
        
        # Bounds
        bounds = [(1.0, 2000.0), (1.0, 1000.0)]  # Loading, maintenance
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        loading_dose = result.x[0]
        maintenance_dose = result.x[1]
        
        # Validate result
        try:
            simulation_time = 168
            maintenance_times = np.arange(maintenance_interval, simulation_time, maintenance_interval)
            dose_schedule = {0.0: loading_dose}
            for t in maintenance_times:
                dose_schedule[t] = maintenance_dose
                
            time_points = np.linspace(0, simulation_time, int(simulation_time * 4) + 1)
            concentrations = pk_model.simulate_concentration(time_points, dose_schedule)
            
            # Performance metrics
            target_idx = np.argmin(np.abs(time_points - target_time))
            achieved_target_conc = concentrations[target_idx]
            
            steady_state_mask = time_points >= (simulation_time - 24)
            ss_mean = np.mean(concentrations[steady_state_mask])
            
        except:
            achieved_target_conc = 0.0
            ss_mean = 0.0
            
        return {
            'loading_dose': loading_dose,
            'maintenance_dose': maintenance_dose,
            'maintenance_interval': maintenance_interval,
            'target_concentration': target_concentration,
            'achieved_target_concentration': achieved_target_conc,
            'steady_state_concentration': ss_mean,
            'optimization_success': result.success
        }
        
    def optimize_dose_escalation(self,
                               target_concentrations: List[float],
                               escalation_times: List[float],
                               pk_model: Any) -> Dict[str, Any]:
        """Optimize dose escalation schedule.
        
        Args:
            target_concentrations: Target concentrations at each time point
            escalation_times: Times for dose escalation
            pk_model: PK model
            
        Returns:
            Optimized dose escalation schedule
        """
        n_doses = len(escalation_times)
        
        def objective(doses):
            """Objective for dose escalation optimization."""
            if len(doses) != n_doses:
                return 1e6
                
            try:
                # Create dose schedule
                dose_schedule = {}
                for i, (time, dose) in enumerate(zip(escalation_times, doses)):
                    dose_schedule[time] = dose
                    
                # Simulate
                simulation_time = max(escalation_times) + 72  # 3 days after last dose
                time_points = np.linspace(0, simulation_time, int(simulation_time * 2) + 1)
                concentrations = pk_model.simulate_concentration(time_points, dose_schedule)
                
                # Calculate penalties for each target
                total_penalty = 0.0
                
                for i, (target_time, target_conc) in enumerate(zip(escalation_times, target_concentrations)):
                    # Find concentration at target time (after some delay)
                    evaluation_time = target_time + 2.0  # 2 hours post-dose
                    eval_idx = np.argmin(np.abs(time_points - evaluation_time))
                    
                    if eval_idx < len(concentrations):
                        achieved_conc = concentrations[eval_idx]
                        penalty = (achieved_conc - target_conc) ** 2
                        total_penalty += penalty
                        
                return total_penalty
                
            except Exception as e:
                self.logger.warning(f"Dose escalation optimization error: {e}")
                return 1e6
                
        # Initial guess - escalating doses
        initial_guess = [target_concentrations[i] * 20 * (i + 1) for i in range(n_doses)]
        
        # Bounds
        bounds = [(1.0, 1000.0) for _ in range(n_doses)]
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        optimal_doses = result.x
        
        # Validate result
        try:
            dose_schedule = dict(zip(escalation_times, optimal_doses))
            simulation_time = max(escalation_times) + 72
            time_points = np.linspace(0, simulation_time, int(simulation_time * 2) + 1)
            concentrations = pk_model.simulate_concentration(time_points, dose_schedule)
            
            # Calculate achieved concentrations
            achieved_concentrations = []
            for target_time in escalation_times:
                evaluation_time = target_time + 2.0
                eval_idx = np.argmin(np.abs(time_points - evaluation_time))
                if eval_idx < len(concentrations):
                    achieved_concentrations.append(concentrations[eval_idx])
                else:
                    achieved_concentrations.append(0.0)
                    
        except:
            achieved_concentrations = [0.0] * n_doses
            
        return {
            'escalation_times': escalation_times,
            'optimal_doses': optimal_doses.tolist(),
            'target_concentrations': target_concentrations,
            'achieved_concentrations': achieved_concentrations,
            'optimization_success': result.success,
            'dose_schedule': dict(zip(escalation_times, optimal_doses))
        }
        
    def compare_regimens(self,
                        regimens: List[DosingRegimen],
                        pk_model: Any,
                        pd_model: Any = None,
                        simulation_time: float = 168.0) -> Dict[str, Any]:
        """Compare multiple dosing regimens.
        
        Args:
            regimens: List of dosing regimens to compare
            pk_model: PK model
            pd_model: Optional PD model
            simulation_time: Simulation time (hours)
            
        Returns:
            Comparison results
        """
        results = {}
        time_points = np.linspace(0, simulation_time, int(simulation_time * 2) + 1)
        
        for i, regimen in enumerate(regimens):
            regimen_name = f"regimen_{i + 1}"
            
            try:
                # Generate dose schedule and simulate
                dose_schedule = regimen.generate_dose_schedule(simulation_time)
                concentrations = pk_model.simulate_concentration(time_points, dose_schedule)
                
                # Calculate PK metrics
                auc = np.trapz(concentrations, time_points)
                cmax = np.max(concentrations)
                tmax = time_points[np.argmax(concentrations)]
                
                # Calculate total dose
                total_dose = sum(dose_schedule.values())
                
                # PD response if model provided
                pd_response = None
                if pd_model:
                    try:
                        pd_response = pd_model.predict_response(concentrations, time_points)
                        pd_auc = np.trapz(pd_response, time_points)
                    except:
                        pd_auc = 0.0
                else:
                    pd_auc = 0.0
                    
                results[regimen_name] = {
                    'dose_schedule': dose_schedule,
                    'total_dose': total_dose,
                    'auc': auc,
                    'cmax': cmax,
                    'tmax': tmax,
                    'concentrations': concentrations,
                    'pd_response': pd_response,
                    'pd_auc': pd_auc,
                    'regimen_type': regimen.regimen_type
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to simulate regimen {i + 1}: {e}")
                results[regimen_name] = {'error': str(e)}
                
        return results