"""
Quantum PK/PD Package - LSQI Challenge 2025
Quantum-Enhanced Pharmacokinetics-Pharmacodynamics Modeling
"""

__version__ = "1.0.0"
__author__ = "Quantum PK/PD Research Team"

# Import main modules
from . import data
from . import quantum
from . import utils
from . import optimization
from . import pkpd

__all__ = [
    "data",
    "quantum", 
    "utils",
    "optimization",
    "pkpd"
]