"""
Dependency Management Utilities for VQCdd

This module provides graceful handling of optional dependencies with clear fallback
behavior and user guidance. It prevents hard failures when optional packages
are not available while maintaining core functionality.

Author: VQCdd Development Team
Created: 2025-09-19
"""

import warnings
import logging
from typing import Dict, Optional, Callable, Any
from functools import wraps
import importlib.util

logger = logging.getLogger(__name__)


class OptionalDependency:
    """
    Manages optional dependencies with graceful fallback behavior

    This class checks for package availability and provides informative
    messages when packages are missing, along with installation guidance.
    """

    def __init__(self,
                 package_name: str,
                 feature_name: str,
                 install_command: str,
                 fallback_message: str):
        """
        Initialize optional dependency manager

        Args:
            package_name: Name of the Python package to check
            feature_name: User-friendly name of the feature
            install_command: Command to install the package
            fallback_message: Message explaining fallback behavior
        """
        self.package_name = package_name
        self.feature_name = feature_name
        self.install_command = install_command
        self.fallback_message = fallback_message
        self.available = self._check_availability()

        if not self.available:
            self._warn_unavailable()

    def _check_availability(self) -> bool:
        """Check if the package is available for import"""
        try:
            spec = importlib.util.find_spec(self.package_name)
            return spec is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False

    def _warn_unavailable(self):
        """Warn user about unavailable dependency"""
        warning_msg = (
            f"{self.feature_name} not available. {self.fallback_message} "
            f"To enable full functionality, install with: {self.install_command}"
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=3)
        logger.info(f"Optional dependency '{self.package_name}' not found - using fallback implementation")

    def require(self) -> bool:
        """
        Check if dependency is required for a specific operation

        Returns:
            True if available, False if should use fallback
        """
        return self.available

    def import_or_none(self):
        """
        Import the package if available, otherwise return None

        Returns:
            The imported module or None if not available
        """
        if self.available:
            try:
                return importlib.import_module(self.package_name)
            except ImportError:
                logger.warning(f"Import failed for {self.package_name} despite availability check")
                return None
        return None


class DependencyManager:
    """
    Central manager for all VQCdd optional dependencies

    This class provides a single point of truth for dependency checking
    and fallback behavior across the entire VQCdd codebase.
    """

    def __init__(self):
        """Initialize dependency manager with all known optional dependencies"""
        self.dependencies = {
            'skopt': OptionalDependency(
                package_name='skopt',
                feature_name='Bayesian Optimization',
                install_command='pip install scikit-optimize',
                fallback_message='Will use simplified grid search optimization.'
            ),

            'pymoo': OptionalDependency(
                package_name='pymoo',
                feature_name='Multi-objective Optimization',
                install_command='pip install pymoo',
                fallback_message='Will use single-objective optimization only.'
            ),

            'latex': OptionalDependency(
                package_name='subprocess',  # Special case - we check for latex binary
                feature_name='LaTeX Rendering',
                install_command='Install LaTeX distribution (e.g., MacTeX, TeX Live)',
                fallback_message='Will use matplotlib mathtext for equation rendering.'
            )
        }

        # Special handling for LaTeX (not a Python package)
        self.dependencies['latex'].available = self._check_latex_availability()
        if not self.dependencies['latex'].available:
            self.dependencies['latex']._warn_unavailable()

    def _check_latex_availability(self) -> bool:
        """Check if LaTeX is available on the system"""
        try:
            import subprocess
            result = subprocess.run(['latex', '--version'],
                                  capture_output=True,
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def is_available(self, dependency_name: str) -> bool:
        """Check if a specific dependency is available"""
        if dependency_name not in self.dependencies:
            logger.warning(f"Unknown dependency: {dependency_name}")
            return False
        return self.dependencies[dependency_name].available

    def get_dependency(self, dependency_name: str) -> Optional[OptionalDependency]:
        """Get dependency manager for specific package"""
        return self.dependencies.get(dependency_name)

    def import_optional(self, dependency_name: str):
        """Import optional dependency if available"""
        dep = self.get_dependency(dependency_name)
        if dep:
            return dep.import_or_none()
        return None

    def get_summary(self) -> Dict[str, bool]:
        """Get summary of all dependency availability"""
        return {name: dep.available for name, dep in self.dependencies.items()}


# Global dependency manager instance
dependency_manager = DependencyManager()


def requires_dependency(dependency_name: str, fallback_return=None):
    """
    Decorator that checks for optional dependencies before function execution

    Args:
        dependency_name: Name of the required dependency
        fallback_return: Value to return if dependency is not available

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if dependency_manager.is_available(dependency_name):
                return func(*args, **kwargs)
            else:
                dep = dependency_manager.get_dependency(dependency_name)
                logger.info(f"Skipping {func.__name__} - {dep.feature_name if dep else dependency_name} not available")
                return fallback_return
        return wrapper
    return decorator


def check_dependencies() -> Dict[str, bool]:
    """
    Check all dependencies and return status

    Returns:
        Dictionary mapping dependency names to availability status
    """
    return dependency_manager.get_summary()


def print_dependency_status():
    """Print a summary of all dependency statuses"""
    status = check_dependencies()

    print("VQCdd Dependency Status:")
    print("=" * 40)

    for dep_name, available in status.items():
        dep = dependency_manager.get_dependency(dep_name)
        status_symbol = "✅" if available else "❌"
        feature_name = dep.feature_name if dep else dep_name
        print(f"{status_symbol} {feature_name}")

        if not available and dep:
            print(f"   Install: {dep.install_command}")
            print(f"   Fallback: {dep.fallback_message}")

    print("=" * 40)


# Convenience functions for common dependencies
def has_bayesian_optimization() -> bool:
    """Check if Bayesian optimization is available"""
    return dependency_manager.is_available('skopt')


def has_multi_objective_optimization() -> bool:
    """Check if multi-objective optimization is available"""
    return dependency_manager.is_available('pymoo')


def has_latex() -> bool:
    """Check if LaTeX rendering is available"""
    return dependency_manager.is_available('latex')


def import_skopt():
    """Import scikit-optimize if available"""
    return dependency_manager.import_optional('skopt')


def import_pymoo():
    """Import pymoo if available"""
    return dependency_manager.import_optional('pymoo')


if __name__ == "__main__":
    # Demo the dependency management system
    print("Testing VQCdd Dependency Management")
    print_dependency_status()

    # Test decorator
    @requires_dependency('skopt', fallback_return={'method': 'grid_search'})
    def bayesian_optimization_example():
        return {'method': 'bayesian', 'optimizer': 'gp_minimize'}

    result = bayesian_optimization_example()
    print(f"\nBayesian optimization result: {result}")