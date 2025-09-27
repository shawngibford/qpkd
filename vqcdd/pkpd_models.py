"""
Pharmacokinetic and Pharmacodynamic Models for VQCdd

This module implements classical PK/PD models that form the foundation for
quantum-enhanced parameter estimation. It includes compartment models for
drug disposition and effect models for biomarker response.

Key Features:
- Two-compartment pharmacokinetic model with allometric scaling
- Inhibitory Emax pharmacodynamic model
- Population variability and covariate effects
- Analytical solutions for computational efficiency
- Comprehensive validation and error handling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod


@dataclass
class PKParameters:
    """Pharmacokinetic parameters with units and descriptions"""
    ka: float    # Absorption rate constant (1/h)
    cl: float    # Clearance (L/h)
    v1: float    # Central volume of distribution (L)
    q: float     # Inter-compartmental clearance (L/h)
    v2: float    # Peripheral volume of distribution (L)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'ka': self.ka, 'cl': self.cl, 'v1': self.v1,
            'q': self.q, 'v2': self.v2
        }

    def scale_for_bodyweight(self, body_weight: float, reference_weight: float = 70.0) -> 'PKParameters':
        """
        Apply allometric scaling for body weight

        Args:
            body_weight: Patient body weight (kg)
            reference_weight: Reference weight for scaling (kg)

        Returns:
            Scaled PK parameters
        """
        # Validate inputs to prevent numerical issues
        if not (isinstance(body_weight, (int, float)) and np.isfinite(body_weight) and body_weight > 0):
            raise ValueError(f"Body weight must be positive finite number, got {body_weight}")

        if not (isinstance(reference_weight, (int, float)) and np.isfinite(reference_weight) and reference_weight > 0):
            raise ValueError(f"Reference weight must be positive finite number, got {reference_weight}")

        # Clamp body weight to reasonable physiological range to prevent extreme scaling
        body_weight = max(min(body_weight, 200.0), 30.0)  # 30-200 kg range

        bw_ratio = body_weight / reference_weight

        # Additional safety check for the ratio
        if not (0.1 < bw_ratio < 5.0):  # Allow 10% to 500% of reference weight
            # Clamp to reasonable range
            bw_ratio = max(min(bw_ratio, 5.0), 0.1)

        return PKParameters(
            ka=self.ka,  # Absorption rate typically not weight-scaled
            cl=self.cl * (bw_ratio ** 0.75),     # Clearance scales with metabolic rate
            v1=self.v1 * bw_ratio,               # Volume scales with size
            q=self.q * (bw_ratio ** 0.75),       # Inter-compartmental clearance
            v2=self.v2 * bw_ratio                # Peripheral volume scales with size
        )


@dataclass
class PDParameters:
    """Pharmacodynamic parameters with units and descriptions"""
    baseline: float    # Baseline biomarker level (ng/mL)
    imax: float       # Maximum inhibition (fraction, 0-1)
    ic50: float       # Concentration for 50% inhibition (mg/L)
    gamma: float      # Hill coefficient (sigmoidicity)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'baseline': self.baseline, 'imax': self.imax,
            'ic50': self.ic50, 'gamma': self.gamma
        }

    def adjust_for_covariates(self, concomitant_med: bool = False,
                             age: Optional[float] = None) -> 'PDParameters':
        """
        Adjust PD parameters for patient covariates

        Args:
            concomitant_med: Whether patient is on concomitant medication
            age: Patient age (years), if available

        Returns:
            Adjusted PD parameters
        """
        # Copy current parameters
        adjusted = PDParameters(
            baseline=self.baseline,
            imax=self.imax,
            ic50=self.ic50,
            gamma=self.gamma
        )

        # Concomitant medication effect on baseline
        if concomitant_med:
            adjusted.baseline *= 1.3  # 30% increase in baseline

        # Age effect on drug sensitivity (if age provided)
        if age is not None and age > 65:
            adjusted.ic50 *= 0.8  # Increased sensitivity in elderly

        return adjusted


class PKModel(ABC):
    """Abstract base class for pharmacokinetic models"""

    @abstractmethod
    def concentration_time_profile(self, time: np.ndarray, dose: float,
                                  pk_params: PKParameters,
                                  **kwargs) -> np.ndarray:
        """Calculate concentration-time profile"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name for logging"""
        pass


class TwoCompartmentPK(PKModel):
    """
    Two-compartment pharmacokinetic model with analytical solution

    Implements the classical two-compartment model for drugs that exhibit
    distribution into peripheral tissues. More realistic than single-compartment
    for most drugs.
    """

    def __init__(self, administration_route: str = "iv_bolus"):
        """
        Initialize two-compartment PK model

        Args:
            administration_route: "iv_bolus", "oral", or "iv_infusion"
        """
        self.administration_route = administration_route
        self.logger = logging.getLogger(__name__)

    def concentration_time_profile(self, time: np.ndarray, dose: float,
                                  pk_params: PKParameters,
                                  body_weight: float = 70.0,
                                  **kwargs) -> np.ndarray:
        """
        Calculate plasma concentration-time profile

        Args:
            time: Time points (hours)
            dose: Administered dose (mg)
            pk_params: PK parameters
            body_weight: Patient body weight (kg)

        Returns:
            Plasma concentrations (mg/L)
        """
        # Scale parameters for body weight
        scaled_params = pk_params.scale_for_bodyweight(body_weight)

        # Calculate hybrid rate constants
        k10 = scaled_params.cl / scaled_params.v1
        k12 = scaled_params.q / scaled_params.v1
        k21 = scaled_params.q / scaled_params.v2

        # Calculate eigenvalues (λ1, λ2)
        a = k10 + k12 + k21
        b = k10 * k21
        discriminant = a**2 - 4*b

        # Ensure positive discriminant for real eigenvalues
        if discriminant < 0:
            self.logger.warning(f"Negative discriminant: {discriminant}, using absolute value")
            discriminant = abs(discriminant)

        sqrt_discriminant = np.sqrt(discriminant)
        lambda1 = 0.5 * (a + sqrt_discriminant)
        lambda2 = 0.5 * (a - sqrt_discriminant)

        # Calculate distribution coefficients
        if abs(lambda1 - lambda2) < 1e-10:
            # Special case: repeated roots (rare but possible)
            alpha = dose / scaled_params.v1
            concentrations = alpha * (1 + lambda1 * time) * np.exp(-lambda1 * time)
        else:
            # Standard two-compartment solution
            A = (k21 - lambda1) / (lambda2 - lambda1) * dose / scaled_params.v1
            B = (k21 - lambda2) / (lambda1 - lambda2) * dose / scaled_params.v1

            concentrations = A * np.exp(-lambda1 * time) + B * np.exp(-lambda2 * time)

        # Handle different administration routes
        if self.administration_route == "oral":
            # Add absorption phase
            ka = scaled_params.ka
            F = kwargs.get('bioavailability', 1.0)  # Bioavailability
            absorption_factor = F * ka / (ka - lambda1) * (np.exp(-lambda1 * time) - np.exp(-ka * time))
            concentrations *= absorption_factor

        # Ensure non-negative concentrations
        return np.maximum(concentrations, 0.0)

    def get_model_name(self) -> str:
        return f"TwoCompartment_{self.administration_route}"

    def steady_state_concentration(self, dose: float, dosing_interval: float,
                                  pk_params: PKParameters, body_weight: float = 70.0) -> float:
        """
        Calculate steady-state concentration for multiple dosing

        Args:
            dose: Single dose (mg)
            dosing_interval: Time between doses (hours)
            pk_params: PK parameters
            body_weight: Patient body weight (kg)

        Returns:
            Average steady-state concentration (mg/L)
        """
        # Use superposition principle for multiple dosing
        time_points = np.arange(0, 5 * dosing_interval, dosing_interval/10)  # 5 dosing intervals
        single_dose_profile = self.concentration_time_profile(
            time_points, dose, pk_params, body_weight
        )

        # Approximate steady-state as average over last dosing interval
        steady_state_idx = int(4 * len(time_points) / 5)  # Last 20% of time points
        return np.mean(single_dose_profile[steady_state_idx:])


class PDModel(ABC):
    """Abstract base class for pharmacodynamic models"""

    @abstractmethod
    def biomarker_response(self, concentrations: np.ndarray,
                          pd_params: PDParameters,
                          **kwargs) -> np.ndarray:
        """Calculate biomarker response to drug concentrations"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name for logging"""
        pass


class InhibitoryEmaxPD(PDModel):
    """
    Inhibitory Emax pharmacodynamic model

    Implements the classical sigmoid Emax model for drug effects that
    inhibit biomarker production or activity. Commonly used for
    biomarker suppression studies.
    """

    def __init__(self, model_type: str = "direct"):
        """
        Initialize Emax PD model

        Args:
            model_type: "direct" for immediate effect, "indirect" for delayed effect
        """
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)

    def biomarker_response(self, concentrations: np.ndarray,
                          pd_params: PDParameters,
                          concomitant_med: bool = False,
                          time: Optional[np.ndarray] = None,
                          **kwargs) -> np.ndarray:
        """
        Calculate biomarker response using inhibitory Emax model

        Args:
            concentrations: Drug concentrations (mg/L)
            pd_params: PD parameters
            concomitant_med: Whether patient is on concomitant medication
            time: Time points (for indirect models)

        Returns:
            Biomarker levels (ng/mL)
        """
        # Adjust parameters for covariates
        adjusted_params = pd_params.adjust_for_covariates(concomitant_med)

        # Calculate inhibition using Hill equation
        inhibition = (adjusted_params.imax *
                     concentrations ** adjusted_params.gamma /
                     (adjusted_params.ic50 ** adjusted_params.gamma +
                      concentrations ** adjusted_params.gamma))

        # Ensure inhibition is between 0 and 1
        inhibition = np.clip(inhibition, 0.0, 1.0)

        if self.model_type == "direct":
            # Direct effect: immediate response
            biomarker = adjusted_params.baseline * (1 - inhibition)
        else:
            # Indirect effect: delayed response (simplified)
            # In practice, would solve differential equation
            delay_factor = kwargs.get('delay_factor', 0.8)
            biomarker = adjusted_params.baseline * (1 - delay_factor * inhibition)

        # Ensure positive biomarker levels
        return np.maximum(biomarker, 0.1)  # Minimum physiological level

    def get_model_name(self) -> str:
        return f"InhibitoryEmax_{self.model_type}"

    def calculate_efficacy_threshold(self, pd_params: PDParameters,
                                   target_suppression: float = 0.7,
                                   concomitant_med: bool = False) -> float:
        """
        Calculate concentration needed for target biomarker suppression

        Args:
            pd_params: PD parameters
            target_suppression: Target suppression (0-1)
            concomitant_med: Concomitant medication status

        Returns:
            Required concentration (mg/L)
        """
        adjusted_params = pd_params.adjust_for_covariates(concomitant_med)

        # Solve Emax equation for required concentration
        if target_suppression >= adjusted_params.imax:
            self.logger.warning(f"Target suppression {target_suppression} exceeds Imax {adjusted_params.imax}")
            return np.inf

        # Rearrange Hill equation: C = IC50 * (I/(Imax-I))^(1/gamma)
        inhibition_ratio = target_suppression / (adjusted_params.imax - target_suppression)
        required_concentration = adjusted_params.ic50 * (inhibition_ratio ** (1/adjusted_params.gamma))

        return required_concentration


class PKPDModel:
    """
    Combined pharmacokinetic-pharmacodynamic model

    Integrates PK and PD models to simulate the complete drug disposition
    and effect chain for dosing optimization.
    """

    def __init__(self, pk_model: PKModel, pd_model: PDModel):
        """
        Initialize PKPD model

        Args:
            pk_model: Pharmacokinetic model instance
            pd_model: Pharmacodynamic model instance
        """
        self.pk_model = pk_model
        self.pd_model = pd_model
        self.logger = logging.getLogger(__name__)

    def simulate_pkpd_profile(self, time: np.ndarray, dose: float,
                             pk_params: PKParameters, pd_params: PDParameters,
                             body_weight: float = 70.0,
                             concomitant_med: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate complete PK/PD profile

        Args:
            time: Time points (hours)
            dose: Administered dose (mg)
            pk_params: PK parameters
            pd_params: PD parameters
            body_weight: Patient body weight (kg)
            concomitant_med: Concomitant medication status

        Returns:
            Tuple of (concentrations, biomarker_levels)
        """
        # Simulate PK profile
        concentrations = self.pk_model.concentration_time_profile(
            time, dose, pk_params, body_weight
        )

        # Simulate PD response
        biomarker_levels = self.pd_model.biomarker_response(
            concentrations, pd_params, concomitant_med, time
        )

        return concentrations, biomarker_levels

    def optimize_dose_for_target(self, target_biomarker: float,
                                pk_params: PKParameters, pd_params: PDParameters,
                                dosing_interval: float = 24.0,
                                body_weight: float = 70.0,
                                concomitant_med: bool = False,
                                dose_range: Tuple[float, float] = (1.0, 100.0)) -> Dict:
        """
        Optimize dose to achieve target biomarker suppression

        Args:
            target_biomarker: Target biomarker level (ng/mL)
            pk_params: PK parameters
            pd_params: PD parameters
            dosing_interval: Dosing interval (hours)
            body_weight: Patient body weight (kg)
            concomitant_med: Concomitant medication status
            dose_range: Min and max doses to consider (mg)

        Returns:
            Dictionary with optimization results
        """
        # Validate target biomarker level
        adjusted_params = pd_params.adjust_for_covariates(concomitant_med)
        min_possible_biomarker = adjusted_params.baseline * (1 - adjusted_params.imax)

        if target_biomarker < min_possible_biomarker:
            self.logger.warning(f"Target biomarker {target_biomarker:.2f} below minimum achievable {min_possible_biomarker:.2f}")

        if target_biomarker >= adjusted_params.baseline:
            self.logger.warning(f"Target biomarker {target_biomarker:.2f} at or above baseline {adjusted_params.baseline:.2f}")

        # Validate dose range based on body weight
        weight_adjusted_dose_range = (
            max(dose_range[0], 0.1 * body_weight / 70.0),  # Minimum 0.1 mg/kg equivalent
            min(dose_range[1], 2.0 * body_weight)          # Maximum 2.0 mg/kg
        )
        from scipy.optimize import minimize_scalar

        def objective(dose):
            """Objective function: minimize difference from target"""
            # Simulate steady-state
            if hasattr(self.pk_model, 'steady_state_concentration'):
                ss_conc = self.pk_model.steady_state_concentration(
                    dose, dosing_interval, pk_params, body_weight
                )
            else:
                # Approximate with single dose at steady-state time
                time_ss = np.array([dosing_interval * 5])
                ss_conc = self.pk_model.concentration_time_profile(
                    time_ss, dose, pk_params, body_weight
                )[0]

            # Calculate biomarker response
            biomarker = self.pd_model.biomarker_response(
                np.array([ss_conc]), pd_params, concomitant_med
            )[0]

            return abs(biomarker - target_biomarker)

        # Optimize dose with weight-adjusted range
        result = minimize_scalar(objective, bounds=weight_adjusted_dose_range, method='bounded')

        # Calculate final metrics
        optimal_dose = result.x
        if hasattr(self.pk_model, 'steady_state_concentration'):
            final_concentration = self.pk_model.steady_state_concentration(
                optimal_dose, dosing_interval, pk_params, body_weight
            )
        else:
            time_ss = np.array([dosing_interval * 5])
            final_concentration = self.pk_model.concentration_time_profile(
                time_ss, optimal_dose, pk_params, body_weight
            )[0]

        final_biomarker = self.pd_model.biomarker_response(
            np.array([final_concentration]), pd_params, concomitant_med
        )[0]

        # Calculate success metrics
        target_error = abs(final_biomarker - target_biomarker)
        target_tolerance = max(0.3, 0.1 * target_biomarker)  # Adaptive tolerance
        target_achieved = target_error < target_tolerance

        return {
            'optimal_dose': optimal_dose,
            'final_biomarker': final_biomarker,
            'final_concentration': final_concentration,
            'target_biomarker': target_biomarker,
            'target_error': target_error,
            'target_achieved': target_achieved,
            'optimization_success': result.success,
            'dose_range_used': weight_adjusted_dose_range,
            'min_achievable_biomarker': min_possible_biomarker,
            'body_weight_adjusted': body_weight != 70.0
        }

    def get_model_info(self) -> Dict:
        """Get information about the PKPD model components"""
        return {
            'pk_model': self.pk_model.get_model_name(),
            'pd_model': self.pd_model.get_model_name(),
            'combined_name': f"{self.pk_model.get_model_name()}_{self.pd_model.get_model_name()}"
        }


# Utility functions for model validation and testing

def validate_pk_parameters(pk_params: PKParameters) -> List[str]:
    """
    Validate PK parameters for physiological plausibility

    Args:
        pk_params: PK parameters to validate

    Returns:
        List of validation warnings
    """
    warnings = []

    # Check parameter ranges
    if pk_params.ka < 0.01 or pk_params.ka > 20:
        warnings.append(f"Absorption rate ka={pk_params.ka:.3f} outside typical range (0.01-20 h⁻¹)")

    if pk_params.cl < 0.1 or pk_params.cl > 100:
        warnings.append(f"Clearance cl={pk_params.cl:.3f} outside typical range (0.1-100 L/h)")

    if pk_params.v1 < 1 or pk_params.v1 > 200:
        warnings.append(f"Central volume v1={pk_params.v1:.3f} outside typical range (1-200 L)")

    # Check parameter relationships
    ke = pk_params.cl / pk_params.v1  # Elimination rate constant
    if ke > 10:
        warnings.append(f"Very high elimination rate: ke={ke:.3f} h⁻¹")

    t_half = 0.693 / ke  # Half-life
    if t_half < 0.1 or t_half > 100:
        warnings.append(f"Half-life t1/2={t_half:.3f} h outside typical range (0.1-100 h)")

    return warnings


def validate_pd_parameters(pd_params: PDParameters) -> List[str]:
    """
    Validate PD parameters for physiological plausibility

    Args:
        pd_params: PD parameters to validate

    Returns:
        List of validation warnings
    """
    warnings = []

    if pd_params.imax < 0 or pd_params.imax > 1:
        warnings.append(f"Imax={pd_params.imax:.3f} outside valid range (0-1)")

    if pd_params.ic50 <= 0:
        warnings.append(f"IC50={pd_params.ic50:.3f} must be positive")

    if pd_params.gamma < 0.1 or pd_params.gamma > 10:
        warnings.append(f"Hill coefficient gamma={pd_params.gamma:.3f} outside typical range (0.1-10)")

    if pd_params.baseline <= 0:
        warnings.append(f"Baseline={pd_params.baseline:.3f} must be positive")

    return warnings


if __name__ == "__main__":
    # Example usage and testing
    print("VQCdd PK/PD Models Module")
    print("=" * 40)

    # Create example parameters
    pk_params = PKParameters(ka=1.0, cl=5.0, v1=20.0, q=2.0, v2=50.0)
    pd_params = PDParameters(baseline=15.0, imax=0.8, ic50=5.0, gamma=1.5)

    print("PK Parameters:", pk_params.to_dict())
    print("PD Parameters:", pd_params.to_dict())

    # Validate parameters
    pk_warnings = validate_pk_parameters(pk_params)
    pd_warnings = validate_pd_parameters(pd_params)

    if pk_warnings:
        print("PK Warnings:", pk_warnings)
    if pd_warnings:
        print("PD Warnings:", pd_warnings)

    # Create models
    pk_model = TwoCompartmentPK("iv_bolus")
    pd_model = InhibitoryEmaxPD("direct")
    pkpd_model = PKPDModel(pk_model, pd_model)

    print(f"Model: {pkpd_model.get_model_info()}")

    # Simulate profile
    time = np.linspace(0, 48, 100)  # 48 hours
    dose = 10.0  # mg
    concentrations, biomarkers = pkpd_model.simulate_pkpd_profile(
        time, dose, pk_params, pd_params
    )

    print(f"Peak concentration: {concentrations.max():.3f} mg/L")
    print(f"Minimum biomarker: {biomarkers.min():.3f} ng/mL")

    # Optimize dose for target
    target_biomarker = 3.3  # ng/mL
    optimization_result = pkpd_model.optimize_dose_for_target(
        target_biomarker, pk_params, pd_params
    )

    print(f"Optimization result: {optimization_result}")