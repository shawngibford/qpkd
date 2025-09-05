# Quantum-Enhanced PK/PD Modeling

This project develops quantum computing methods for pharmacokinetics-pharmacodynamics (PK/PD) modeling to optimize drug dosing regimens.

## Project Structure

```
qpkd/
├── data/                    # Clinical trial datasets
├── src/
│   ├── data/               # Data preprocessing and analysis
│   ├── quantum/            # Quantum computing components
│   ├── pkpd/              # PK/PD modeling modules
│   ├── optimization/      # Dosing optimization algorithms
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── config.yaml           # Configuration settings
```

## Challenge Objectives

1. **Daily Dosing**: Determine dose ensuring 90% of subjects achieve biomarker suppression below 3.3 ng/mL
2. **Weekly Dosing**: Find equivalent weekly dose with same effect
3. **Population Variation**: Analyze impact of different body weight distributions (70-140 kg)
4. **Concomitant Medication**: Assess effects when concomitant medication is not allowed
5. **Threshold Analysis**: Compare results when targeting 75% vs 90% of subjects

## Key Features

- Quantum machine learning for enhanced generalization with limited data
- Advanced PK/PD modeling with quantum-enhanced parameter estimation
- Population pharmacokinetic variability modeling
- Dosing regimen optimization algorithms
- Biomarker suppression prediction

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Optional: R with nlmixr2 package for traditional PK/PD modeling

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd /Users/shawngibford/dev/qpkd
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   The `EstData.csv` dataset should already be available in the `data/` directory.
   If missing, download from: https://github.com/Quantum-Innovation-Challenge/LSQI-Challenge-2025/tree/main/data

4. **Verify installation:**
   ```python
   python -c "import pennylane as qml; import numpy as np; print('Installation successful!')"
   ```

## Project Architecture

The codebase is organized into several key modules:

### Core Quantum Approaches (`src/quantum/`)
- **approach1_vqc/**: Variational Quantum Circuits for parameter estimation
- **approach2_qml/**: Quantum Machine Learning with data reuploading
- **approach3_qode/**: Quantum Ordinary Differential Equation solvers
- **approach4_qaoa/**: Quantum Approximate Optimization Algorithm
- **approach5_tensor_zx/**: Tensor Networks with ZX calculus

### Supporting Modules
- **src/data/**: Data loading, preprocessing, and validation
- **src/pkpd/**: Classical PK/PD models and compartment analysis
- **src/optimization/**: Dosing and hyperparameter optimization
- **src/utils/**: Logging, R integration, and utilities

## Usage Examples

### 1. Quick Start - Run Single Quantum Approach

```python
from src.quantum.approach1_vqc.vqc_parameter_estimator_full import VQCParameterEstimatorFull
from src.data.data_loader import PKPDDataLoader
from src.utils.logging_system import QuantumPKPDLogger

# Load data
loader = PKPDDataLoader("data/EstData.csv")
data = loader.prepare_pkpd_data(weight_range=(50, 100), concomitant_allowed=True)

# Initialize VQC approach
vqc_model = VQCParameterEstimatorFull(n_qubits=6, n_layers=4)

# Fit model and optimize dosing
vqc_model.fit(data)
results = vqc_model.optimize_dosing(target_threshold=3.3, population_coverage=0.9)

print(f"Optimal daily dose: {results.daily_dose:.2f} mg")
print(f"Optimal weekly dose: {results.weekly_dose:.2f} mg")
print(f"Coverage achieved: {results.coverage_achieved:.1%}")
```

### 2. Compare All 5 Quantum Approaches

```python
from src.quantum.core.data_structures import PKPDData
from src.data.data_loader import PKPDDataLoader
from src.optimization.population_optimizer import PopulationOptimizer

# Load data for different scenarios
loader = PKPDDataLoader("data/EstData.csv")
scenarios = {
    'baseline': loader.get_scenario_data('baseline'),
    'extended_weight': loader.get_scenario_data('extended_weight'),
    'no_conmed': loader.get_scenario_data('no_conmed')
}

# Initialize all approaches
approaches = {
    'VQC': VQCParameterEstimatorFull(n_qubits=6, n_layers=4),
    'QML': QuantumNeuralNetworkFull(n_qubits=8, n_layers=6),
    'QODE': QuantumODESolverFull(n_qubits=6, evolution_time=10.0),
    'QAOA': MultiObjectiveOptimizerFull(n_qubits=8, qaoa_layers=3),
    'Tensor': TensorPopulationModelFull(bond_dim=32, max_iterations=100)
}

# Run comparative analysis
results = {}
for name, model in approaches.items():
    print(f"Running {name} approach...")
    model.fit(scenarios['baseline'])
    dosing_result = model.optimize_dosing(target_threshold=3.3, population_coverage=0.9)
    results[name] = dosing_result
    print(f"{name} optimal dose: {dosing_result.daily_dose:.2f} mg/day")
```

### 3. Population-Specific Optimization

```python
from src.optimization.population_optimizer import PopulationOptimizer, PopulationSegment

# Define population segments
segments = [
    PopulationSegment(name="standard", weight_range=(50, 100), prevalence=0.6),
    PopulationSegment(name="heavy", weight_range=(70, 140), prevalence=0.4),
    PopulationSegment(name="no_conmed", weight_range=(50, 100), 
                     concomitant_allowed=False, prevalence=0.3)
]

# Initialize population optimizer
pop_optimizer = PopulationOptimizer(population_segments=segments)

# Create prediction models for each segment
prediction_models = {}
for segment in segments:
    data = loader.get_scenario_data(segment.name)
    model = VQCParameterEstimatorFull(n_qubits=6)
    model.fit(data)
    prediction_models[segment.name] = lambda dose: model.predict_biomarkers(dose)

# Optimize across all populations
population_results = pop_optimizer.optimize_population_dosing(
    prediction_models, target_threshold=3.3, population_coverage=0.9
)

print("Population-specific optimal doses:")
for segment_name, result in population_results['segment_results'].items():
    print(f"{segment_name}: {result.daily_dose:.2f} mg/day")
```

### 4. Hyperparameter Optimization

```python
from src.optimization.hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterSpace

# Define search space
param_space = HyperparameterSpace(
    learning_rate=(0.001, 0.1),
    n_layers=(2, 8),
    n_qubits=(4, 12),
    batch_size=[16, 32, 64],
    ansatz_type=['hardware_efficient', 'alternating']
)

# Initialize optimizer
hyperopt = HyperparameterOptimizer(
    optimization_method='bayesian',
    n_trials=50,
    cv_folds=5
)

# Optimize hyperparameters
optimization_results = hyperopt.optimize(
    model_class=VQCParameterEstimatorFull,
    data=data,
    param_space=param_space
)

print(f"Best parameters: {optimization_results['best_params']}")
print(f"Best score: {optimization_results['best_score']:.4f}")
```

### 5. Classical PK/PD Baseline Comparison

```python
from src.pkpd.compartment_models import OneCompartmentModel, TwoCompartmentModel
from src.pkpd.population_models import PopulationPKModel
from src.pkpd.biomarker_models import EmaxModel

# Fit classical models
pk_model = OneCompartmentModel()
biomarker_model = EmaxModel()

# Population PK analysis
pop_pk = PopulationPKModel(base_model='one_compartment')
population_sim = pop_pk.simulate_population(
    n_subjects=48, 
    time_points=np.linspace(0, 168, 169)
)

print(f"Population PK parameters:")
for param, stats in population_sim['parameters_summary'].items():
    print(f"{param}: {stats['mean']:.2f} ± {stats['std']:.2f}")
```

## Challenge Questions

The project addresses 5 specific dosing optimization questions:

### Question 1: Daily Dosing (Standard Population)
```python
data_q1 = loader.prepare_pkpd_data(weight_range=(50, 100), concomitant_allowed=True)
model = VQCParameterEstimatorFull(n_qubits=6)
model.fit(data_q1)
result_q1 = model.optimize_dosing(target_threshold=3.3, population_coverage=0.9)
print(f"Q1 Answer: {result_q1.daily_dose:.1f} mg/day")
```

### Question 2: Weekly Dosing
```python
result_q2 = model.optimize_weekly_dosing(target_threshold=3.3, population_coverage=0.9)
print(f"Q2 Answer: {result_q2.weekly_dose:.1f} mg/week")
```

### Question 3: Extended Weight Range
```python
data_q3 = loader.prepare_pkpd_data(weight_range=(70, 140), concomitant_allowed=True)
model.fit(data_q3)
result_q3 = model.optimize_dosing(target_threshold=3.3, population_coverage=0.9)
print(f"Q3 Answer: {result_q3.daily_dose:.1f} mg/day")
```

### Question 4: No Concomitant Medication
```python
data_q4 = loader.prepare_pkpd_data(weight_range=(50, 100), concomitant_allowed=False)
model.fit(data_q4)
result_q4 = model.optimize_dosing(target_threshold=3.3, population_coverage=0.9)
print(f"Q4 Answer: {result_q4.daily_dose:.1f} mg/day")
```

### Question 5: 75% Population Coverage
```python
result_q5 = model.optimize_dosing(target_threshold=3.3, population_coverage=0.75)
print(f"Q5 Answer: {result_q5.daily_dose:.1f} mg/day")
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_vqc_approach.py -v
python -m pytest tests/test_qnn_approach.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Logging and Results

All experiments are automatically logged with structured output:

```python
from src.utils.logging_system import QuantumPKPDLogger

# Initialize logger
logger = QuantumPKPDLogger(log_level='INFO', log_to_file=True)

# Log experiment
metadata = ExperimentMetadata(
    approach='VQC',
    dataset_info={'n_subjects': 48, 'scenario': 'baseline'},
    model_config={'n_qubits': 6, 'n_layers': 4}
)

logger.log_experiment_start(metadata)

# Results are automatically saved to logs/experiments/
```

## Advanced Features

### Custom Quantum Circuits
```python
def custom_ansatz(params, wires):
    """Define custom quantum ansatz."""
    qml.templates.AngleEmbedding(params[:len(wires)], wires=wires)
    for layer in range(len(params) // len(wires) - 1):
        for i in wires:
            qml.RY(params[len(wires) + layer * len(wires) + i], wires=i)
        qml.broadcast(qml.CNOT, wires, pattern="ring")

# Use in VQC approach
vqc_model = VQCParameterEstimatorFull(custom_ansatz=custom_ansatz)
```

### Ensemble Methods
```python
from src.quantum.approach2_qml.quantum_neural_network_full import QuantumNeuralNetworkFull

# Create ensemble
ensemble_models = [
    QuantumNeuralNetworkFull(n_qubits=6, architecture='layered'),
    QuantumNeuralNetworkFull(n_qubits=8, architecture='tree'),
    QuantumNeuralNetworkFull(n_qubits=6, architecture='alternating')
]

# Train ensemble
for model in ensemble_models:
    model.fit(data)

# Ensemble prediction
ensemble_predictions = [model.predict_biomarkers(dose=10.0) for model in ensemble_models]
final_prediction = np.mean(ensemble_predictions, axis=0)
```

## Troubleshooting

### Common Issues

1. **PennyLane Installation Issues:**
   ```bash
   pip install --upgrade pennylane pennylane-lightning
   ```

2. **Memory Issues with Large Quantum Circuits:**
   - Reduce `n_qubits` parameter
   - Use `device='lightning.qubit'` for faster simulation
   - Consider using `batch_size` parameter

3. **R Integration Problems:**
   ```r
   install.packages("nlmixr2")
   ```

4. **Optimization Convergence:**
   - Increase `max_iterations` parameter
   - Try different `optimization_method` ('adam', 'sgd', 'rmsprop')
   - Adjust learning rate

### Performance Optimization

- Use `n_jobs=-1` for parallel processing where available
- Set `device='lightning.qubit'` for faster quantum simulation
- Reduce dataset size for initial testing
- Use `early_stopping=True` in hyperparameter optimization

## Contributing

The codebase follows modular design principles:
- Each quantum approach is self-contained
- All models inherit from `QuantumPKPDBase`
- Results use standardized `DosingResults` format
- Comprehensive logging and error handling throughout

## References

- LSQI Challenge 2025: https://github.com/Quantum-Innovation-Challenge/LSQI-Challenge-2025
- PennyLane Documentation: https://pennylane.ai/
- nlmixr2 Documentation: https://nlmixr2.org/