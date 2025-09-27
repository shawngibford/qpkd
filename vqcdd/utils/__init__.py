"""
VQCdd Utilities Package

This package contains utility modules for the VQCdd quantum pharmacokinetic modeling framework.
"""

from .dependencies import (
    dependency_manager,
    check_dependencies,
    print_dependency_status,
    has_bayesian_optimization,
    has_multi_objective_optimization,
    has_latex,
    import_skopt,
    import_pymoo,
    requires_dependency
)

__all__ = [
    'dependency_manager',
    'check_dependencies',
    'print_dependency_status',
    'has_bayesian_optimization',
    'has_multi_objective_optimization',
    'has_latex',
    'import_skopt',
    'import_pymoo',
    'requires_dependency'
]