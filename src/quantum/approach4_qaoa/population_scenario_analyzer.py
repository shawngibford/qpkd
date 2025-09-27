"""
Population Scenario Analyzer using QAOA

Analyzes multiple population scenarios simultaneously using Quantum Approximate
Optimization Algorithm to identify optimal dosing strategies across diverse populations.
"""

import numpy as np
import pennylane as qml
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import itertools
from collections import defaultdict

from ..core.base import QuantumPKPDBase, ModelConfig, PKPDData, OptimizationResult


@dataclass
class PopulationScenario:
    """Definition of a population scenario"""
    scenario_id: str
    weight_range: Tuple[float, float]
    concomitant_prevalence: float
    age_range: Optional[Tuple[float, float]] = None
    special_population: Optional[str] = None  # "elderly", "pediatric", "renal_impaired"
    target_coverage: float = 0.9
    description: str = ""


@dataclass
class ScenarioAnalysisConfig(ModelConfig):
    """Configuration for population scenario analysis"""
    qaoa_layers: int = 3
    optimization_objective: str = "multi_scenario"  # "multi_scenario", "robust_coverage", "pareto_optimal"
    scenario_weighting: str = "equal"  # "equal", "prevalence_based", "custom"
    constraint_handling: str = "penalty"  # "penalty", "feasibility_check", "adaptive"
    population_simulation_size: int = 500
    dose_resolution: int = 20
    robustness_analysis: bool = True


class PopulationScenarioAnalyzer:
    """
    QAOA-based analyzer for multiple population scenarios

    Simultaneously optimizes dosing across different population scenarios
    to find globally robust dosing strategies.
    """

    def __init__(self, config: ScenarioAnalysisConfig):
        self.config = config
        self.scenario_config = config
        self.device = None
        self.qaoa_circuit = None
        self.population_scenarios = []
        self.scenario_weights = {}
        self.optimization_results = {}

    def setup_quantum_device(self) -> qml.device:
        """Setup quantum device for QAOA optimization"""
        device = qml.device(
            "lightning.qubit",
            wires=self.config.n_qubits,
            shots=self.config.shots
        )
        self.device = device
        return device

    def add_population_scenario(self, scenario: PopulationScenario, weight: float = 1.0):
        """Add a population scenario to the analysis"""
        self.population_scenarios.append(scenario)
        self.scenario_weights[scenario.scenario_id] = weight

    def define_standard_scenarios(self):
        """Define standard population scenarios for drug development"""
        # Clear existing scenarios
        self.population_scenarios = []
        self.scenario_weights = {}

        # Standard scenarios based on regulatory guidance
        scenarios = [
            PopulationScenario(
                scenario_id="standard_population",
                weight_range=(50, 100),
                concomitant_prevalence=0.3,
                target_coverage=0.9,
                description="Standard population (50-100 kg, 30% concomitant meds)"
            ),
            PopulationScenario(
                scenario_id="extended_weight",
                weight_range=(70, 140),
                concomitant_prevalence=0.3,
                target_coverage=0.9,
                description="Extended weight range (70-140 kg)"
            ),
            PopulationScenario(
                scenario_id="no_concomitant",
                weight_range=(50, 100),
                concomitant_prevalence=0.0,
                target_coverage=0.9,
                description="No concomitant medication"
            ),
            PopulationScenario(
                scenario_id="high_concomitant",
                weight_range=(50, 100),
                concomitant_prevalence=0.8,
                target_coverage=0.9,
                description="High concomitant medication prevalence"
            ),
            PopulationScenario(
                scenario_id="reduced_coverage",
                weight_range=(50, 100),
                concomitant_prevalence=0.3,
                target_coverage=0.75,
                description="Reduced coverage target (75%)"
            )
        ]

        for scenario in scenarios:
            self.add_population_scenario(scenario, weight=1.0)

    def build_qaoa_circuit(self, n_qubits: int, n_layers: int) -> callable:
        """
        Build QAOA circuit for multi-scenario optimization

        Each qubit represents a dosing decision variable or scenario constraint
        """

        @qml.qnode(self.device)
        def qaoa_multi_scenario(params, scenario_encoding=None):
            """
            QAOA circuit for multi-scenario dosing optimization

            Args:
                params: QAOA parameters [gammas, betas]
                scenario_encoding: Encoding of scenario constraints and objectives
            """
            n_params_per_layer = 2  # gamma and beta for each layer
            if len(params) != n_params_per_layer * n_layers:
                raise ValueError(f"Expected {n_params_per_layer * n_layers} parameters, got {len(params)}")

            # Initialize uniform superposition
            for qubit in range(n_qubits):
                qml.Hadamard(wires=qubit)

            # QAOA layers
            for layer in range(n_layers):
                gamma = params[2 * layer]
                beta = params[2 * layer + 1]

                # Problem Hamiltonian (cost function)
                self._apply_cost_hamiltonian(gamma, scenario_encoding)

                # Mixer Hamiltonian
                self._apply_mixer_hamiltonian(beta)

            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qaoa_circuit = qaoa_multi_scenario
        return qaoa_multi_scenario

    def _apply_cost_hamiltonian(self, gamma: float, scenario_encoding: Optional[np.ndarray]):
        """Apply cost Hamiltonian for multi-scenario optimization"""
        n_qubits = self.config.n_qubits

        # Single-qubit terms (individual scenario objectives)
        for i in range(n_qubits):
            if scenario_encoding is not None and i < len(scenario_encoding):
                # Weight by scenario importance
                weight = scenario_encoding[i]
                qml.RZ(gamma * weight, wires=i)
            else:
                qml.RZ(gamma, wires=i)

        # Two-qubit terms (scenario interactions and constraints)
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                # Coupling between scenarios
                qml.CNOT(wires=[i, j])
                qml.RZ(gamma * 0.5, wires=j)  # Interaction strength
                qml.CNOT(wires=[i, j])

    def _apply_mixer_hamiltonian(self, beta: float):
        """Apply mixer Hamiltonian (transverse field)"""
        for qubit in range(self.config.n_qubits):
            qml.RX(2 * beta, wires=qubit)

    def encode_scenario_constraints(self, scenarios: List[PopulationScenario]) -> np.ndarray:
        """
        Encode population scenarios as quantum state preparation

        Each scenario contributes to the encoding based on its constraints and objectives
        """
        n_scenarios = len(scenarios)
        n_qubits = self.config.n_qubits

        # Create encoding vector
        encoding = np.zeros(n_qubits)

        for i, scenario in enumerate(scenarios[:n_qubits]):
            # Encode scenario characteristics
            weight_factor = (scenario.weight_range[1] - scenario.weight_range[0]) / 100.0  # Normalize
            concomitant_factor = scenario.concomitant_prevalence
            coverage_factor = scenario.target_coverage

            # Combine factors into encoding value
            scenario_encoding = weight_factor + concomitant_factor + coverage_factor
            encoding[i] = scenario_encoding

            # Apply scenario weight
            if scenario.scenario_id in self.scenario_weights:
                encoding[i] *= self.scenario_weights[scenario.scenario_id]

        # Normalize encoding
        if np.linalg.norm(encoding) > 0:
            encoding = encoding / np.linalg.norm(encoding) * np.pi

        return encoding

    def simulate_population_scenario(self, scenario: PopulationScenario, dose: float,
                                   dosing_interval: float, model: QuantumPKPDBase) -> Dict[str, float]:
        """
        Simulate a population scenario with given dosing parameters

        Args:
            scenario: Population scenario definition
            dose: Dose amount (mg)
            dosing_interval: Dosing interval (hours)
            model: Trained PK/PD model

        Returns:
            Simulation results including coverage and safety metrics
        """
        n_population = self.scenario_config.population_simulation_size

        # Generate population based on scenario
        body_weights = np.random.uniform(
            scenario.weight_range[0],
            scenario.weight_range[1],
            n_population
        )

        concomitant_meds = np.random.binomial(1, scenario.concomitant_prevalence, n_population)

        # Additional population characteristics
        if scenario.age_range:
            ages = np.random.uniform(scenario.age_range[0], scenario.age_range[1], n_population)
        else:
            ages = np.random.uniform(18, 80, n_population)  # Default adult range

        # Simulate biomarker response for each individual
        below_threshold_count = 0
        safety_violations = 0
        biomarker_responses = []

        time_points = np.linspace(0, dosing_interval, int(dosing_interval))

        for i in range(n_population):
            covariates = {
                'body_weight': body_weights[i],
                'concomitant_med': concomitant_meds[i],
                'age': ages[i]
            }

            try:
                biomarker_trajectory = model.predict_biomarker(dose, time_points, covariates)
                min_biomarker = np.min(biomarker_trajectory)
                max_biomarker = np.max(biomarker_trajectory)

                biomarker_responses.append(min_biomarker)

                # Efficacy: biomarker below threshold
                if min_biomarker <= 3.3:  # Target threshold
                    below_threshold_count += 1

                # Safety: check for excessive suppression or other safety concerns
                if min_biomarker < 0.5:  # Excessive suppression
                    safety_violations += 1

            except Exception:
                # Count as failure if prediction fails
                continue

        # Calculate metrics
        efficacy_coverage = below_threshold_count / n_population
        safety_rate = 1.0 - (safety_violations / n_population)

        # Additional metrics
        mean_response = np.mean(biomarker_responses) if biomarker_responses else 0
        response_variability = np.std(biomarker_responses) if biomarker_responses else 0

        return {
            'efficacy_coverage': efficacy_coverage,
            'safety_rate': safety_rate,
            'mean_biomarker_response': mean_response,
            'response_variability': response_variability,
            'population_size': n_population,
            'meets_target': efficacy_coverage >= scenario.target_coverage
        }

    def multi_scenario_objective_function(self, dose_params: np.ndarray,
                                        model: QuantumPKPDBase) -> float:
        """
        Multi-scenario objective function for QAOA optimization

        Args:
            dose_params: Encoded dosing parameters
            model: Trained PK/PD model

        Returns:
            Objective value (lower is better)
        """
        # Decode dosing parameters from quantum state
        daily_dose = self._decode_dose_from_quantum_params(dose_params, "daily")
        weekly_dose = self._decode_dose_from_quantum_params(dose_params, "weekly")

        total_objective = 0.0
        scenario_results = {}

        # Evaluate each scenario
        for scenario in self.population_scenarios:
            # Determine dosing interval based on scenario (simplified)
            dosing_interval = 24.0  # Daily dosing for most scenarios
            current_dose = daily_dose

            # Simulate scenario
            scenario_result = self.simulate_population_scenario(
                scenario, current_dose, dosing_interval, model
            )

            scenario_results[scenario.scenario_id] = scenario_result

            # Calculate scenario objective
            efficacy_penalty = max(0, scenario.target_coverage - scenario_result['efficacy_coverage'])
            safety_penalty = max(0, 0.05 - (1.0 - scenario_result['safety_rate']))  # Safety threshold
            variability_penalty = scenario_result['response_variability'] * 0.1

            scenario_objective = efficacy_penalty + safety_penalty + variability_penalty

            # Weight by scenario importance
            weight = self.scenario_weights.get(scenario.scenario_id, 1.0)
            total_objective += weight * scenario_objective

        # Store results for analysis
        self.optimization_results['last_evaluation'] = {
            'dose_params': dose_params.tolist(),
            'daily_dose': daily_dose,
            'weekly_dose': weekly_dose,
            'scenario_results': scenario_results,
            'total_objective': total_objective
        }

        return total_objective

    def _decode_dose_from_quantum_params(self, quantum_params: np.ndarray, dose_type: str) -> float:
        """
        Decode dose from quantum parameters

        Args:
            quantum_params: Quantum measurement results
            dose_type: "daily" or "weekly"

        Returns:
            Decoded dose value
        """
        if len(quantum_params) == 0:
            return 50.0  # Default dose

        # Use first few qubits for dose encoding
        dose_encoding = quantum_params[:min(4, len(quantum_params))]

        # Map from [-1, 1] to dose range
        if dose_type == "daily":
            min_dose, max_dose = 10, 200
        else:  # weekly
            min_dose, max_dose = 50, 1000

        # Average the measurements and map to dose range
        avg_measurement = np.mean(dose_encoding)
        normalized_value = (avg_measurement + 1) / 2  # Map [-1, 1] to [0, 1]
        dose = min_dose + normalized_value * (max_dose - min_dose)

        return dose

    def optimize_multi_scenario_dosing(self, model: QuantumPKPDBase) -> Dict[str, Any]:
        """
        Optimize dosing across multiple population scenarios using QAOA

        Args:
            model: Trained PK/PD model

        Returns:
            Multi-scenario optimization results
        """
        if not self.population_scenarios:
            self.define_standard_scenarios()

        if self.device is None:
            self.setup_quantum_device()

        if self.qaoa_circuit is None:
            self.qaoa_circuit = self.build_qaoa_circuit(self.config.n_qubits, self.scenario_config.qaoa_layers)

        # Encode scenarios
        scenario_encoding = self.encode_scenario_constraints(self.population_scenarios)

        # Initialize QAOA parameters
        n_params = 2 * self.scenario_config.qaoa_layers  # gamma and beta for each layer
        initial_params = np.random.uniform(0, 2*np.pi, n_params)

        # QAOA optimization
        optimizer = qml.AdamOptimizer(stepsize=0.1)
        best_params = initial_params.copy()
        best_objective = float('inf')
        optimization_history = []

        for iteration in range(self.config.max_iterations):
            # Get quantum measurements
            quantum_measurements = self.qaoa_circuit(initial_params, scenario_encoding)

            # Evaluate objective function
            objective_value = self.multi_scenario_objective_function(
                np.array(quantum_measurements), model
            )

            optimization_history.append(objective_value)

            if objective_value < best_objective:
                best_objective = objective_value
                best_params = initial_params.copy()

            # Update parameters
            initial_params = optimizer.step(
                lambda p: self.multi_scenario_objective_function(
                    self.qaoa_circuit(p, scenario_encoding), model
                ),
                initial_params
            )

            # Check convergence
            if iteration > 10 and abs(optimization_history[-1] - optimization_history[-10]) < self.config.convergence_threshold:
                break

        # Final evaluation with best parameters
        best_measurements = self.qaoa_circuit(best_params, scenario_encoding)
        final_objective = self.multi_scenario_objective_function(best_measurements, model)

        # Decode optimal doses
        optimal_daily_dose = self._decode_dose_from_quantum_params(best_measurements, "daily")
        optimal_weekly_dose = self._decode_dose_from_quantum_params(best_measurements, "weekly")

        return {
            'optimal_daily_dose': optimal_daily_dose,
            'optimal_weekly_dose': optimal_weekly_dose,
            'best_objective': best_objective,
            'optimization_history': optimization_history,
            'qaoa_params': best_params.tolist(),
            'scenario_results': self.optimization_results.get('last_evaluation', {}),
            'n_scenarios': len(self.population_scenarios),
            'convergence_iterations': iteration + 1
        }

    def robustness_analysis(self, optimal_doses: Dict[str, float],
                          model: QuantumPKPDBase,
                          perturbation_magnitude: float = 0.1) -> Dict[str, Any]:
        """
        Analyze robustness of optimal dosing across scenarios

        Args:
            optimal_doses: Optimal dose recommendations
            model: Trained PK/PD model
            perturbation_magnitude: Magnitude of dose perturbations

        Returns:
            Robustness analysis results
        """
        if not self.scenario_config.robustness_analysis:
            return {}

        robustness_results = {}
        base_daily_dose = optimal_doses.get('daily', 100)

        # Test dose perturbations
        perturbations = [-perturbation_magnitude, 0, perturbation_magnitude]
        perturbation_results = {}

        for perturbation in perturbations:
            perturbed_dose = base_daily_dose * (1 + perturbation)
            scenario_performances = {}

            for scenario in self.population_scenarios:
                performance = self.simulate_population_scenario(
                    scenario, perturbed_dose, 24.0, model
                )
                scenario_performances[scenario.scenario_id] = performance

            perturbation_results[f"perturbation_{perturbation:.1f}"] = scenario_performances

        # Calculate robustness metrics
        robustness_metrics = {}
        for scenario in self.population_scenarios:
            coverage_values = []
            for perturbation_key in perturbation_results:
                coverage = perturbation_results[perturbation_key][scenario.scenario_id]['efficacy_coverage']
                coverage_values.append(coverage)

            robustness_metrics[scenario.scenario_id] = {
                'coverage_std': np.std(coverage_values),
                'coverage_range': max(coverage_values) - min(coverage_values),
                'min_coverage': min(coverage_values),
                'robust': min(coverage_values) >= scenario.target_coverage * 0.95  # 95% of target
            }

        # Overall robustness score
        overall_robustness = np.mean([
            metrics['robust'] for metrics in robustness_metrics.values()
        ])

        return {
            'perturbation_results': perturbation_results,
            'robustness_metrics': robustness_metrics,
            'overall_robustness_score': overall_robustness,
            'perturbation_magnitude': perturbation_magnitude
        }

    def generate_scenario_analysis_report(self, model: QuantumPKPDBase) -> Dict[str, Any]:
        """
        Generate comprehensive multi-scenario analysis report

        Args:
            model: Trained PK/PD model

        Returns:
            Comprehensive analysis report
        """
        # Perform multi-scenario optimization
        optimization_results = self.optimize_multi_scenario_dosing(model)

        # Robustness analysis
        robustness_results = self.robustness_analysis(
            {
                'daily': optimization_results['optimal_daily_dose'],
                'weekly': optimization_results['optimal_weekly_dose']
            },
            model
        )

        # Scenario summary
        scenario_summary = []
        for scenario in self.population_scenarios:
            scenario_summary.append({
                'id': scenario.scenario_id,
                'description': scenario.description,
                'weight_range': scenario.weight_range,
                'concomitant_prevalence': scenario.concomitant_prevalence,
                'target_coverage': scenario.target_coverage,
                'weight': self.scenario_weights.get(scenario.scenario_id, 1.0)
            })

        return {
            'optimization_results': optimization_results,
            'robustness_analysis': robustness_results,
            'scenario_definitions': scenario_summary,
            'analysis_config': {
                'qaoa_layers': self.scenario_config.qaoa_layers,
                'population_simulation_size': self.scenario_config.population_simulation_size,
                'optimization_objective': self.scenario_config.optimization_objective,
                'n_qubits': self.config.n_qubits
            },
            'recommendations': {
                'daily_dose': optimization_results['optimal_daily_dose'],
                'weekly_dose': optimization_results['optimal_weekly_dose'],
                'confidence': 'high' if robustness_results.get('overall_robustness_score', 0) > 0.8 else 'medium'
            }
        }