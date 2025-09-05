"""
Quantum Manager for Coordinating All 5 Approaches

Central manager class that coordinates and compares all quantum approaches
for PK/PD modeling. Provides unified interface for running experiments
and comparative analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import yaml

from .core.base import PKPDData, OptimizationResult
from .approach1_vqc import VQCParameterEstimator
from .approach2_qml import QuantumNeuralNetwork  
from .approach3_qode import QuantumODESolver
from .approach4_qaoa import QUBOFormulator
from .approach5_tensor_zx import TensorPopulationModel


@dataclass
class ExperimentConfig:
    """Configuration for quantum PK/PD experiments"""
    approaches_to_run: List[str] = None  # Which approaches to execute
    comparison_metrics: List[str] = None  # Metrics for comparison
    population_scenarios: List[str] = None  # Population scenarios to test
    output_dir: str = "results/"
    save_intermediate: bool = True
    parallel_execution: bool = False


class QuantumPKPDManager:
    """
    Manager class coordinating all 5 quantum approaches
    
    Provides unified interface for:
    - Loading and preprocessing data
    - Running individual approaches
    - Comparative analysis 
    - Results aggregation and reporting
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.experiment_config = None
        
        # Initialize approach instances
        self.approaches = {}
        self.results = {}
        self.data = None
        
        # Default approaches to run
        self.available_approaches = {
            'vqc': 'Variational Quantum Circuit Parameter Estimation',
            'qml': 'Quantum Machine Learning Population PK',  
            'qode': 'Quantum-Enhanced Differential Equation Solver',
            'qaoa': 'Quantum Annealing Multi-Objective Optimization',
            'tensor_zx': 'Tensor Network Population Modeling with ZX Calculus'
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using defaults.")
            config = self._default_config()
        
        return config
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if file not found"""
        return {
            'quantum': {'device_wires': 16, 'shots': 1024},
            'clinical': {'biomarker_threshold': 3.3, 'target_suppression_primary': 0.9},
            'modeling': {'compartments': [1, 2]},
            'population': {'baseline_weight_range': [50, 100], 'extended_weight_range': [70, 140]}
        }
    
    def setup_experiment(self, experiment_config: ExperimentConfig):
        """Setup experimental configuration"""
        self.experiment_config = experiment_config
        
        # Default approaches if none specified
        if experiment_config.approaches_to_run is None:
            experiment_config.approaches_to_run = list(self.available_approaches.keys())
            
        # Default comparison metrics
        if experiment_config.comparison_metrics is None:
            experiment_config.comparison_metrics = [
                'optimal_daily_dose',
                'optimal_weekly_dose', 
                'population_coverage',
                'computation_time',
                'convergence_quality'
            ]
            
        # Default population scenarios  
        if experiment_config.population_scenarios is None:
            experiment_config.population_scenarios = [
                'baseline_50_100kg',
                'extended_70_140kg',
                'no_concomitant_med',
                'with_concomitant_med'
            ]
    
    def load_data(self, data_path: str = "data/EstData.csv") -> PKPDData:
        """Load and preprocess PK/PD data"""
        # Load EstData.csv
        df = pd.read_csv(data_path)
        
        # Convert to PKPDData structure
        self.data = self._convert_dataframe_to_pkpd_data(df)
        
        return self.data
        
    def _convert_dataframe_to_pkpd_data(self, df: pd.DataFrame) -> PKPDData:
        """Convert EstData.csv DataFrame to PKPDData structure"""
        # Separate PK and PD data based on DVID
        pk_mask = df['DVID'] == 1  # PK data
        pd_mask = df['DVID'] == 2  # PD data
        
        # Extract arrays
        subjects = df['ID'].values
        time_points = df['TIME'].values
        doses = df['DOSE'].values
        body_weights = df['BW'].values
        concomitant_meds = df['COMED'].values
        
        # Separate PK and PD observations
        pk_concentrations = np.full(len(df), np.nan)
        pd_biomarkers = np.full(len(df), np.nan)
        
        pk_concentrations[pk_mask] = df.loc[pk_mask, 'DV'].values
        pd_biomarkers[pd_mask] = df.loc[pd_mask, 'DV'].values
        
        return PKPDData(
            subjects=subjects,
            time_points=time_points,
            pk_concentrations=pk_concentrations,
            pd_biomarkers=pd_biomarkers, 
            doses=doses,
            body_weights=body_weights,
            concomitant_meds=concomitant_meds
        )
    
    def initialize_approaches(self):
        """Initialize all selected quantum approaches"""
        if self.experiment_config is None:
            raise ValueError("Experiment config not set. Call setup_experiment() first.")
            
        for approach_name in self.experiment_config.approaches_to_run:
            if approach_name == 'vqc':
                from .approach1_vqc.vqc_parameter_estimator import VQCParameterEstimator, VQCConfig
                config = VQCConfig(n_qubits=8, n_layers=4)
                self.approaches[approach_name] = VQCParameterEstimator(config)
                
            elif approach_name == 'qml':
                from .approach2_qml.quantum_neural_network import QuantumNeuralNetwork, QNNConfig
                config = QNNConfig(n_qubits=10, variational_layers=4)
                self.approaches[approach_name] = QuantumNeuralNetwork(config)
                
            elif approach_name == 'qode':
                from .approach3_qode.quantum_ode_solver import QuantumODESolver, QODEConfig  
                config = QODEConfig(n_qubits=12, time_evolution_steps=100)
                self.approaches[approach_name] = QuantumODESolver(config)
                
            elif approach_name == 'qaoa':
                from .approach4_qaoa.qubo_formulator import QUBOFormulator, QUBOConfig
                config = QUBOConfig(n_qubits=8, dose_discretization=0.5)
                self.approaches[approach_name] = QUBOFormulator(config)
                
            elif approach_name == 'tensor_zx':
                from .approach5_tensor_zx.tensor_population_model import TensorPopulationModel, TensorConfig
                config = TensorConfig(bond_dimension=16, tensor_structure='mps')
                self.approaches[approach_name] = TensorPopulationModel(config)
    
    def run_single_approach(self, approach_name: str, 
                          population_scenario: str = 'baseline_50_100kg') -> OptimizationResult:
        """Run a single quantum approach on specified population scenario"""
        if approach_name not in self.approaches:
            raise ValueError(f"Approach {approach_name} not initialized")
            
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Filter data based on population scenario
        scenario_data = self._filter_data_by_scenario(self.data, population_scenario)
        
        # Train the approach
        approach = self.approaches[approach_name]
        approach.fit(scenario_data)
        
        # Get optimization results
        result = approach.optimize_dosing(
            target_threshold=self.config['clinical']['biomarker_threshold'],
            population_coverage=self.config['clinical']['target_suppression_primary']
        )
        
        return result
    
    def run_all_approaches(self) -> Dict[str, Dict[str, OptimizationResult]]:
        """Run all approaches on all population scenarios"""
        if not self.approaches:
            self.initialize_approaches()
            
        results = {}
        
        for approach_name in self.experiment_config.approaches_to_run:
            results[approach_name] = {}
            
            for scenario in self.experiment_config.population_scenarios:
                print(f"Running {approach_name} on {scenario}...")
                
                try:
                    result = self.run_single_approach(approach_name, scenario)
                    results[approach_name][scenario] = result
                    
                except Exception as e:
                    print(f"Error running {approach_name} on {scenario}: {e}")
                    results[approach_name][scenario] = None
        
        self.results = results
        return results
    
    def _filter_data_by_scenario(self, data: PKPDData, scenario: str) -> PKPDData:
        """Filter data based on population scenario"""
        # Create masks based on scenario
        if scenario == 'baseline_50_100kg':
            mask = (data.body_weights >= 50) & (data.body_weights <= 100)
        elif scenario == 'extended_70_140kg':
            mask = (data.body_weights >= 70) & (data.body_weights <= 140)
        elif scenario == 'no_concomitant_med':
            mask = data.concomitant_meds == 0
        elif scenario == 'with_concomitant_med':
            mask = data.concomitant_meds == 1
        else:
            mask = np.ones(len(data.subjects), dtype=bool)  # No filtering
            
        # Apply mask to all data arrays
        return PKPDData(
            subjects=data.subjects[mask],
            time_points=data.time_points[mask],
            pk_concentrations=data.pk_concentrations[mask],
            pd_biomarkers=data.pd_biomarkers[mask],
            doses=data.doses[mask],
            body_weights=data.body_weights[mask], 
            concomitant_meds=data.concomitant_meds[mask]
        )
    
    def compare_approaches(self) -> pd.DataFrame:
        """Generate comparative analysis of all approaches"""
        if not self.results:
            raise ValueError("No results available. Run approaches first.")
            
        comparison_data = []
        
        for approach_name, scenario_results in self.results.items():
            for scenario, result in scenario_results.items():
                if result is not None:
                    row = {
                        'approach': approach_name,
                        'scenario': scenario,
                        'optimal_daily_dose': result.optimal_daily_dose,
                        'optimal_weekly_dose': result.optimal_weekly_dose,
                        'population_coverage': result.population_coverage,
                        'quantum_advantage': self._calculate_quantum_advantage(result)
                    }
                    comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _calculate_quantum_advantage(self, result: OptimizationResult) -> float:
        """Calculate quantum advantage metric"""
        # Placeholder calculation - would compare against classical baseline
        base_metric = result.population_coverage * (1.0 / result.optimal_daily_dose)
        quantum_boost = result.quantum_metrics.get('expressivity', 1.0)
        return base_metric * quantum_boost
    
    def generate_report(self, output_path: str = "quantum_pkpd_report.html") -> str:
        """Generate comprehensive HTML report"""
        if not self.results:
            raise ValueError("No results to report")
            
        # Create comparison DataFrame
        comparison_df = self.compare_approaches()
        
        # Generate HTML report (placeholder)
        html_content = f"""
        <html>
        <head><title>Quantum PK/PD Modeling Results</title></head>
        <body>
        <h1>Quantum-Enhanced PK/PD Modeling Results</h1>
        
        <h2>Approach Comparison</h2>
        {comparison_df.to_html()}
        
        <h2>Challenge Questions Answered</h2>
        <ul>
        <li>Daily dosing: {comparison_df['optimal_daily_dose'].mean():.1f} mg</li>
        <li>Weekly dosing: {comparison_df['optimal_weekly_dose'].mean():.1f} mg</li>
        <li>Population coverage: {comparison_df['population_coverage'].mean():.1%}</li>
        </ul>
        
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        return output_path