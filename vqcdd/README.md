# VQCdd - Variational Quantum Circuit for Drug Dosing

A comprehensive quantum machine learning framework for pharmacokinetic/pharmacodynamic (PK/PD) modeling and optimal drug dosing determination using variational quantum circuits.

## Overview

VQCdd implements quantum-enhanced parameter estimation for early-stage clinical trial dosing optimization. The system integrates quantum variational circuits with classical PK/PD models to determine optimal drug dosing regimens across different patient populations.

### Key Features

- **Quantum Circuit Optimization**: Variational quantum circuits for PK/PD parameter estimation
- **Population Dosing**: Optimization for different weight ranges and concomitant medication scenarios
- **Hyperparameter Optimization**: Bayesian optimization for quantum circuit architecture
- **Noise Analysis**: NISQ device simulation and error mitigation
- **Statistical Validation**: Cross-validation and quantum vs classical comparisons
- **Clinical Scenarios**: Automated optimization for 5 core dosing questions (Q1-Q5)

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify quantum installation
python -c "import pennylane as qml; import numpy as np; print('Quantum setup OK')"

# Set Python path for modules
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Optimization Workflow

### **Important: No Separate Optimization Scripts Required**

VQCdd integrates all optimization directly into `main.py`. There are **two types of optimization**:

#### 1. Built-in VQC Training Optimization (Automatic)
- **Included in ALL modes** (demo, validation, noise, analysis, dosing)
- Optimizes quantum circuit parameters (rotation angles, etc.) during training
- Uses optimizers: adam, adagrad, qng, rmsprop, etc.
- Minimizes cost function for PK/PD parameter estimation
- **No separate step needed** - happens automatically

#### 2. Hyperparameter Optimization (HPO Mode)
- **Separate mode**: `--mode hpo`
- **Also included** in `--mode all --comprehensive`
- Optimizes configuration settings: learning rates, circuit architecture, etc.
- Finds optimal hyperparameters, then trains final model with those settings
- Two-stage process: hyperparameter search → final model training

### Single-Stage vs Two-Stage Optimization

```bash
# Single-stage: VQC parameter optimization only (automatic)
python main.py --mode demo              # Quick optimization demo
python main.py --mode validation        # Optimization + validation
python main.py --mode dosing            # Optimization + dosing scenarios

# Two-stage: Hyperparameter search + VQC optimization
python main.py --mode hpo --hpo-n-calls 50 --hpo-method bayesian

# Complete workflow with both stages
python main.py --mode all --comprehensive
```

## Test Modes

### Available Modes

| Mode | Description | Optimization Type |
|------|-------------|-------------------|
| `demo` | Quick synthetic data demonstration | Single-stage (VQC only) |
| `validation` | Cross-validation and statistical testing | Single-stage (VQC only) |
| `noise` | NISQ device noise analysis | Single-stage (VQC only) |
| `hpo` | Hyperparameter optimization | Two-stage (HPO + VQC) |
| `analysis` | Scientific analysis and quantum advantage | Single-stage (VQC only) |
| `dosing` | Population dosing optimization (Q1-Q5) | Single-stage (VQC only) |
| `all` | Complete experimental suite | Two-stage if `--comprehensive` |

### Command Line Parameters

#### **Data Controls**
- `--data-source` (synthetic|real) - Use synthetic or real EstData.csv
- `--n-patients` - Number of synthetic patients (default: 100)
- `--test-fraction` - Fraction for testing (default: 0.2)
- `--validation-fraction` - Fraction for validation (default: 0.2)

#### **Quantum Circuit Controls**
- `--n-qubits` - Number of qubits (default: 4)
- `--n-layers` - Circuit layers (default: 2)
- `--ansatz` - Circuit type: ry_cnot, strongly_entangling, hardware_efficient, etc.
- `--encoding` - Data encoding: angle, amplitude, iqp, basis, etc.

#### **Training Controls**
- `--max-iterations` - Training iterations (default: 50)
- `--learning-rate` - Learning rate (default: 0.01)
- `--optimizer-type` - Optimizer: adam, adagrad, rmsprop, gd, qng

#### **Experiment-Specific Controls**
- `--noise-level` (low|medium|high) - For noise analysis
- `--hpo-n-calls` - HPO evaluations (default: 20)
- `--hpo-method` (bayesian|random|grid) - HPO strategy

#### **Output & Analysis Controls**
- `--output-dir` - Results directory (default: "results")
- `--compare-classical` - Include classical method comparison
- `--generate-report` - Create detailed reports
- `--save-models` - Save trained model parameters
- `--comprehensive` - Run full comprehensive experiments
- `--verbose/-v` - Verbose output
- `--debug/-d` - Debug mode with full traceback
- `--parallel/-p` - Enable parallel processing
- `--seed` - Random seed (default: 42)

## Usage Examples

### Quick Start

```bash
# 1. Quick functionality test
python main.py --mode demo --verbose

# 2. Validation with real data
python main.py --mode validation --data-source real --generate-report

# 3. Complete workflow
python main.py --mode all --comprehensive --compare-classical
```

### Optimization Workflows

#### Single-Stage Optimization (Most Common)
```bash
# Demo with automatic VQC optimization
python main.py --mode demo --verbose --debug

# Dosing optimization for clinical scenarios
python main.py --mode dosing --n-qubits 6 --max-iterations 100 --save-models

# Validation with classical comparison
python main.py --mode validation --compare-classical --generate-report
```

#### Two-Stage Optimization (HPO)
```bash
# Find optimal hyperparameters (automatically trains final model)
python main.py --mode hpo --hpo-n-calls 50 --hpo-method bayesian --verbose

# Quick HPO for testing
python main.py --mode hpo --hpo-n-calls 10 --hpo-method random

# Use HPO results for subsequent runs
# (Check HPO output for best hyperparameters, then use them)
python main.py --mode dosing --n-qubits 6 --n-layers 3 --learning-rate 0.05
```

### Testing Different Configurations

#### Circuit Architecture Testing
```bash
# Test different ansatz types
python main.py --mode demo --ansatz strongly_entangling --n-qubits 6
python main.py --mode demo --ansatz hardware_efficient --n-layers 3

# Test different encodings
python main.py --mode demo --encoding amplitude --n-qubits 4
python main.py --mode demo --encoding iqp --encoding data_reuploading
```

#### Noise Analysis
```bash
# Different noise levels
python main.py --mode noise --noise-level low --verbose
python main.py --mode noise --noise-level medium --ansatz ry_cnot
python main.py --mode noise --noise-level high --n-qubits 4
```

### Comprehensive Testing

```bash
# Complete experimental suite (long runtime)
python main.py --mode all --comprehensive --compare-classical --generate-report --save-models

# Faster comprehensive testing (skips HPO and noise)
python main.py --mode all --compare-classical --generate-report
```

## Recommended Test Sequences

### Phase 1: Quick Validation
```bash
# 1. Basic functionality
python main.py --mode demo --verbose --debug

# 2. Data validation
python main.py --mode demo --data-source real --verbose
```

### Phase 2: Component Testing
```bash
# 3. Circuit variations
python main.py --mode demo --ansatz strongly_entangling --n-qubits 6
python main.py --mode demo --encoding amplitude --n-layers 3

# 4. Noise analysis
python main.py --mode noise --noise-level medium --verbose

# 5. Validation suite
python main.py --mode validation --compare-classical --generate-report
```

### Phase 3: Optimization
```bash
# 6. Hyperparameter optimization
python main.py --mode hpo --hpo-n-calls 20 --hpo-method bayesian

# 7. Dosing optimization
python main.py --mode dosing --save-models --generate-report
```

### Phase 4: Comprehensive Analysis
```bash
# 8. Scientific analysis
python main.py --mode analysis --compare-classical --comprehensive

# 9. Full test suite
python main.py --mode all --comprehensive --compare-classical --generate-report --save-models
```

## Clinical Scenarios (Dosing Mode)

The dosing mode automatically optimizes for these clinical scenarios:

- **Q1**: Daily dosing for standard population (50-100 kg, concomitant allowed)
- **Q2**: Weekly dosing equivalent
- **Q3**: Daily dosing for extended weight range (70-140 kg)
- **Q4**: Daily dosing without concomitant medication
- **Q5**: Daily dosing targeting 75% population coverage

## Output Structure

```
results/
└── vqcdd_YYYYMMDD_HHMMSS/
    ├── config/
    │   ├── experiment_config.json
    │   └── vqcdd_config.json
    ├── data/
    │   └── data_info.json
    ├── demo/              # --mode demo
    ├── validation/        # --mode validation
    ├── noise_analysis/    # --mode noise
    ├── hyperparameter_optimization/  # --mode hpo
    ├── analytics/         # --mode analysis
    ├── dosing_optimization/  # --mode dosing
    ├── reports/
    │   ├── experiment_summary.json
    │   └── experiment_report.txt
    └── vqcdd_YYYYMMDD_HHMMSS.log
```

## Architecture Overview

### Core Components

- **`main.py`**: Unified execution script with all experimental modes
- **`quantum_circuit.py`**: VQCircuit implementation with various ansätze
- **`optimizer.py`**: VQCTrainer and DosingOptimizer classes
- **`hyperparameter_optimization.py`**: Bayesian and multi-objective HPO
- **`validation.py`**: Cross-validation and statistical testing
- **`noise_analysis.py`**: NISQ device simulation and error mitigation
- **`pkpd_models.py`**: Classical PK/PD models for comparison
- **`data_handler.py`**: Data loading and synthetic generation
- **`analytics.py`**: Scientific analysis and quantum advantage characterization

### Key Classes

- **`VQCddMainExperiment`**: Main experimental framework
- **`VQCTrainer`**: Quantum circuit training and optimization
- **`DosingOptimizer`**: Population dosing optimization
- **`ValidationPipeline`**: Comprehensive validation framework
- **`HyperparameterOptimizer`**: Automated hyperparameter tuning

## Error Handling & Debugging

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Issues**: Check EstData.csv availability or use synthetic data
3. **Memory Issues**: Reduce n_qubits or n_patients for large experiments
4. **Convergence Issues**: Adjust learning_rate or max_iterations

### Debugging Commands

```bash
# Enable debug mode for detailed tracebacks
python main.py --mode demo --debug

# Verbose output for monitoring progress
python main.py --mode hpo --verbose --debug

# Check logs in output directory
tail -f results/vqcdd_*/vqcdd_*.log
```

## Performance Notes

- **Demo mode**: ~30 seconds
- **Validation mode**: ~2-5 minutes
- **HPO mode**: ~10-30 minutes (depends on n_calls)
- **All comprehensive**: ~1-2 hours
- Use `--parallel` flag for faster execution where possible
- Reduce n_qubits, n_patients, or n_calls for quicker testing

## License

MIT License - Open source for Quantum Innovation Challenge 2025