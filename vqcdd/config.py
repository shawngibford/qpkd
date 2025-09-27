#!/usr/bin/env python3
"""
VQCdd Configuration Management System

This module provides centralized configuration management for VQCdd, replacing
magic numbers scattered throughout the codebase with organized, well-documented
configuration parameters.

Author: VQCdd Development Team
Created: 2025-09-19
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryManagementConfig:
    """Configuration for memory management throughout VQCdd"""
    # History buffer sizes
    gradient_history_size: int = 50              # Gradient monitoring history
    cost_history_size: int = 1000               # Training cost history
    performance_history_size: int = 200         # Circuit performance history
    evaluation_history_size: int = 500          # Hyperparameter evaluation history
    gradient_magnitude_history_size: int = 200  # Gradient magnitude tracking

    # Circuit depth adaptation history
    training_history_size: int = 200            # Training progress history
    depth_adjustment_history_size: int = 100    # Circuit depth changes


@dataclass
class ValidationConfig:
    """Configuration for parameter validation and warnings"""
    # Parameter mapping thresholds
    imax_warning_threshold: float = 0.95        # Warning threshold for Imax parameter

    # PK parameter validation ranges
    ka_range: Tuple[float, float] = (0.01, 20.0)      # Absorption rate range (h⁻¹)
    cl_range: Tuple[float, float] = (0.1, 100.0)      # Clearance range (L/h)
    v1_range: Tuple[float, float] = (1.0, 200.0)      # Central volume range (L)
    elimination_rate_max: float = 10.0                 # Maximum elimination rate
    half_life_range: Tuple[float, float] = (0.1, 100.0)  # Half-life range (h)

    # PD parameter validation ranges
    imax_range: Tuple[float, float] = (0.0, 1.0)      # Maximum inhibition range
    gamma_range: Tuple[float, float] = (0.1, 10.0)    # Hill coefficient range

    # Body weight validation
    body_weight_range: Tuple[float, float] = (30.0, 200.0)  # Physiological weight range
    body_weight_ratio_range: Tuple[float, float] = (0.1, 5.0)  # Scaling ratio limits


@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters"""
    # Convergence criteria
    default_max_iterations: int = 50            # Reduced from 100 for faster training
    default_early_stopping_patience: int = 10   # Reduced from 20
    default_convergence_threshold: float = 1e-4 # Relaxed from 1e-6
    min_improvement_threshold: float = 1e-5     # Minimum improvement for patience

    # Gradient monitoring
    barren_plateau_threshold: float = 1e-6      # Barren plateau detection
    gradient_health_threshold: float = 0.3      # Gradient health score threshold

    # Dose optimization
    default_dose_range: Tuple[float, float] = (1.0, 100.0)  # Default dose range (mg)
    dose_tolerance_factor: float = 0.1          # Adaptive tolerance factor
    target_tolerance_base: float = 0.3          # Base tolerance for target achievement


@dataclass
class CircuitConfig:
    """Configuration for quantum circuit parameters"""
    # Default circuit dimensions
    default_n_qubits: int = 4                   # Default qubit count
    default_n_layers: int = 2                   # Default layer count

    # Circuit depth adaptation
    min_layers: int = 1                         # Minimum circuit layers
    max_layers: int = 10                        # Maximum circuit layers

    # Performance thresholds
    expressivity_threshold: float = 0.5         # Expressivity benchmark threshold

    # Encoding parameters
    default_encoding_scaling: float = 1.0       # Default feature scaling
    max_encoding_range: float = 2 * 3.14159     # Maximum encoding range (2π)


@dataclass
class AnalyticsConfig:
    """Configuration for analytics and reporting"""
    # Statistical analysis
    default_confidence_level: float = 0.95      # Default confidence level
    significance_threshold: float = 0.05        # Statistical significance threshold
    bootstrap_samples: int = 1000               # Bootstrap resampling count

    # Figure generation
    default_dpi: int = 300                      # Figure resolution
    default_figsize: Tuple[int, int] = (12, 8)  # Default figure size

    # Performance analysis
    quantum_advantage_threshold: float = 0.05   # Minimum advantage for significance
    substantial_improvement_threshold: float = 0.1  # Substantial improvement (10%)


@dataclass
class DataHandlingConfig:
    """Configuration for data handling and processing"""
    # Weight ranges for different scenarios
    standard_weight_range: Tuple[float, float] = (50.0, 100.0)   # Standard population
    extended_weight_range: Tuple[float, float] = (70.0, 140.0)   # Extended population

    # Target biomarker levels
    default_target_biomarker: float = 3.3       # Default target (ng/mL)
    target_coverage_default: float = 0.9        # Default population coverage (90%)
    target_coverage_alternative: float = 0.75   # Alternative coverage (75%)

    # Data validation
    min_data_points: int = 3                    # Minimum data points for analysis
    outlier_iqr_factor: float = 1.5            # IQR factor for outlier detection


@dataclass
class VQCddConfig:
    """Master configuration for VQCdd system"""
    memory: MemoryManagementConfig = field(default_factory=MemoryManagementConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    data: DataHandlingConfig = field(default_factory=DataHandlingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'memory': self.memory.__dict__,
            'validation': self.validation.__dict__,
            'optimization': self.optimization.__dict__,
            'circuit': self.circuit.__dict__,
            'analytics': self.analytics.__dict__,
            'data': self.data.__dict__
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VQCddConfig':
        """Create configuration from dictionary"""
        return cls(
            memory=MemoryManagementConfig(**data.get('memory', {})),
            validation=ValidationConfig(**data.get('validation', {})),
            optimization=OptimizationConfig(**data.get('optimization', {})),
            circuit=CircuitConfig(**data.get('circuit', {})),
            analytics=AnalyticsConfig(**data.get('analytics', {})),
            data=DataHandlingConfig(**data.get('data', {}))
        )

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    @classmethod
    def load(cls, filepath: str) -> 'VQCddConfig':
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Configuration loaded from {filepath}")
            return cls.from_dict(data)
        except FileNotFoundError:
            logger.warning(f"Configuration file {filepath} not found, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}, using defaults")
            return cls()


class ConfigManager:
    """Global configuration manager for VQCdd"""

    _instance: Optional['ConfigManager'] = None
    _config: Optional[VQCddConfig] = None

    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager"""
        if self._config is None:
            self._config = VQCddConfig()
            logger.info("Configuration manager initialized with default settings")

    @property
    def config(self) -> VQCddConfig:
        """Get current configuration"""
        return self._config

    def load_config(self, filepath: str) -> None:
        """Load configuration from file"""
        self._config = VQCddConfig.load(filepath)

    def save_config(self, filepath: str) -> None:
        """Save current configuration to file"""
        if self._config:
            self._config.save(filepath)

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for section, values in kwargs.items():
            if hasattr(self._config, section):
                section_config = getattr(self._config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                        logger.info(f"Updated {section}.{key} = {value}")
                    else:
                        logger.warning(f"Unknown configuration parameter: {section}.{key}")
            else:
                logger.warning(f"Unknown configuration section: {section}")


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience function for accessing configuration
def get_config() -> VQCddConfig:
    """Get the global VQCdd configuration"""
    return config_manager.config


# Convenience functions for common configuration access patterns
def get_memory_config() -> MemoryManagementConfig:
    """Get memory management configuration"""
    return get_config().memory


def get_validation_config() -> ValidationConfig:
    """Get validation configuration"""
    return get_config().validation


def get_optimization_config() -> OptimizationConfig:
    """Get optimization configuration"""
    return get_config().optimization


def get_circuit_config() -> CircuitConfig:
    """Get circuit configuration"""
    return get_config().circuit


def get_analytics_config() -> AnalyticsConfig:
    """Get analytics configuration"""
    return get_config().analytics


def get_data_config() -> DataHandlingConfig:
    """Get data handling configuration"""
    return get_config().data


if __name__ == "__main__":
    # Example usage and testing
    print("VQCdd Configuration Management System")
    print("=" * 50)

    # Create and display default configuration
    config = VQCddConfig()
    print("Default Configuration:")
    print(f"  Memory - Gradient History Size: {config.memory.gradient_history_size}")
    print(f"  Validation - Imax Warning Threshold: {config.validation.imax_warning_threshold}")
    print(f"  Optimization - Max Iterations: {config.optimization.default_max_iterations}")
    print(f"  Circuit - Default Qubits: {config.circuit.default_n_qubits}")
    print(f"  Analytics - Confidence Level: {config.analytics.default_confidence_level}")
    print(f"  Data - Target Biomarker: {config.data.default_target_biomarker}")

    # Test configuration manager
    manager = ConfigManager()
    print(f"\nConfiguration Manager initialized: {manager.config is not None}")

    # Test updating configuration
    manager.update_config(
        optimization={'default_max_iterations': 75},
        validation={'imax_warning_threshold': 0.9}
    )

    print(f"Updated Max Iterations: {manager.config.optimization.default_max_iterations}")
    print(f"Updated Imax Threshold: {manager.config.validation.imax_warning_threshold}")

    # Test save/load
    test_config_path = "test_vqcdd_config.json"
    try:
        manager.save_config(test_config_path)

        # Load and verify
        new_manager = ConfigManager()
        new_manager.load_config(test_config_path)

        print(f"Loaded Max Iterations: {new_manager.config.optimization.default_max_iterations}")

        # Cleanup
        Path(test_config_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"Configuration save/load test failed: {e}")

    print("\nConfiguration management system ready!")