#!/usr/bin/env python3
"""
Setup script for QPKD (Quantum Pharmacokinetics-Pharmacodynamics) package
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="qpkd",
    version="1.0.0",
    author="Quantum PK/PD Research Team",
    description="Quantum-Enhanced PK/PD Modeling for Optimal Dosing Regimens",
    long_description="LSQI Challenge 2025: Quantum computing methods for pharmacokinetics-pharmacodynamics modeling",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)