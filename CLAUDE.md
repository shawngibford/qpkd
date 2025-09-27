# CLAUDE.md - Project Configuration for Quantum PK/PD Modeling

## Development Philosophy

### Test-Driven Development (TDD)
- **ALWAYS** write tests before implementing new functionality
- Follow the Red-Green-Refactor cycle:
  1. **Red**: Write a failing test that defines the desired functionality
  2. **Green**: Write the minimal code to make the test pass
  3. **Refactor**: Clean up the code while keeping tests green
- All quantum approaches must have corresponding test files in `tests/`
- Test coverage should be maintained above 80% for all src/ modules

## Common Commands

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_vqc_approach.py -v
python -m pytest tests/test_qnn_approach.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run tests for specific quantum approach
python -m pytest tests/test_*approach*.py -k "test_optimization"
```

### Demo Execution
```bash
# Approach 1: VQC Demo
PYTHONPATH=src timeout 30 python3 notebooks/approach1_vqc_demo.py

# Approach 2: QML Demo
PYTHONPATH=src python3 notebooks/approach2_qml_demo.py

# Other approach demos
python3 notebooks/approach3_qode_demo.py
python3 notebooks/approach4_qaoa_demo.py
python3 notebooks/approach5_tensor_zx_demo.py
```

### Data Validation
```bash
# Test data handling
python3 test_data_simple.py
python3 test_real_data_only.py
python3 test_data_fixes.py
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify quantum installation
python -c "import pennylane as qml; import numpy as np; print('Quantum setup OK')"
```

## Project Structure Rules

### Code Organization
- **src/quantum/**: All quantum approach implementations
  - Each approach has its own subdirectory (approach1_vqc/, approach2_qml/, etc.)
  - All approaches inherit from `QuantumPKPDBase`
  - Use standardized `DosingResults` format
- **src/data/**: Data loading, preprocessing, validation only
- **src/pkpd/**: Classical PK/PD models and compartment analysis
- **src/optimization/**: Dosing and hyperparameter optimization algorithms
- **src/utils/**: Logging, R integration, utility functions
- **tests/**: Comprehensive test suite with TDD approach
- **notebooks/**: Jupyter notebooks for demonstrations and analysis

### Naming Conventions
- Quantum models: `*Full` suffix (e.g., `VQCParameterEstimatorFull`)
- Test files: `test_*.py` pattern
- Demo files: `*_demo.py` pattern
- Configuration: Use `config.yaml` for project settings

### Data Handling
- Primary dataset: `data/EstData.csv`
- Weight ranges: (50, 100) kg for standard, (70, 140) kg for extended
- Target threshold: 3.3 ng/mL biomarker suppression
- Population coverage targets: 90% (default), 75% (alternative)

## Challenge-Specific Rules

### Five Core Questions
1. **Q1**: Daily dosing for standard population (50-100 kg, concomitant allowed)
2. **Q2**: Weekly dosing equivalent
3. **Q3**: Daily dosing for extended weight range (70-140 kg)
4. **Q4**: Daily dosing without concomitant medication
5. **Q5**: Daily dosing targeting 75% population coverage

### Quantum Approaches
- **Approach 1 (VQC)**: Variational Quantum Circuits for parameter estimation
- **Approach 2 (QML)**: Quantum Machine Learning with data reuploading
- **Approach 3 (QODE)**: Quantum Ordinary Differential Equation solvers
- **Approach 4 (QAOA)**: Quantum Approximate Optimization Algorithm
- **Approach 5 (Tensor)**: Tensor Networks with ZX calculus

## Code Quality Standards

### Testing Requirements
- Unit tests for all quantum circuit components
- Integration tests for full dosing optimization workflows
- Validation tests against known classical PK/PD results
- Performance benchmarks for quantum vs classical approaches
- Mock data tests and real data validation tests

### Error Handling
- All quantum circuits must handle decoherence gracefully
- Data validation with clear error messages
- Logging of all optimization convergence issues
- Fallback to classical methods when quantum simulation fails

### Performance Guidelines
- Use `device='lightning.qubit'` for faster quantum simulation
- Implement batch processing for population-scale optimization
- Set reasonable timeouts for quantum circuit execution
- Monitor memory usage with large quantum circuits

## Git Workflow
- Main branch: `main`
- Branch naming: `feature/description-of-feature`
- **MANDATORY**: All commits must pass tests before pushing
- **MANDATORY**: Code reviews required for all quantum algorithm implementations
- **MANDATORY**: Pair programming enforced for quantum approach development
- Use descriptive commit messages following conventional commits
- Keep quantum approach implementations in separate commits

## Deployment Context

### Gefion Supercomputer Integration
- **Target Platform**: NVIDIA Gefion AI Supercomputer (Denmark)
- **Hardware**: 1,528 NVIDIA H100 Tensor Core GPUs with Quantum-2 InfiniBand networking
- **Quantum Framework**: NVIDIA CUDA-Q platform for hybrid quantum-classical computing
- **Performance**: Exaflop-class performance, ranked 7th globally for storage systems
- **Environment**: 100% renewable energy, operated by Danish Center for AI Innovation (DCAI)

### Quantum Innovation Challenge 2025 (LSQI)
- **Competition**: Global quantum algorithm competition for pharmaceutical innovation
- **Focus**: Determining optimal drug dosing in early-stage clinical trials
- **Repository**: All code will be open-sourced under Apache/MIT license
- **Evaluation**: Top 5 finalists present at European Quantum Technologies Conference 2025
- **Partners**: Bio Innovation Institute, Molecular Quantum Solutions, Novo Nordisk A/S, Roche

## Quantum-Only Development Philosophy
- **CRITICAL**: Pure quantum approaches required - NO fallback to classical methods
- **Exception**: Hybrid quantum-classical algorithms allowed, but quantum portion must be operational
- **Simulation**: Use CUDA-Q for quantum circuit simulation on Gefion's H100 GPUs
- **Target**: Demonstrate quantum advantage for PK/PD modeling at scale

## Noise Simulation Requirements
- **Implementation**: Include noise simulation as OPTIONAL extra feature
- **Isolation**: Noise simulation must NOT interfere with core quantum algorithms
- **Purpose**: Comparison studies between ideal and noisy quantum circuits
- **Integration**: Use PennyLane noise channels for realistic NISQ device modeling

## Synthetic Data Strategy
- **Requirement**: Synthetic data generation mandatory for comprehensive testing
- **Coverage**: Generate data for all population scenarios (weight ranges, concomitant meds)
- **Validation**: Synthetic datasets must match statistical properties of EstData.csv
- **Testing**: Use synthetic data for unit tests, real data for integration tests

## Documentation and Visualization

### Automatic Documentation Generation
- **Tools**: Use Sphinx with autodoc for automatic docstring documentation
- **Standards**: All quantum functions require comprehensive docstrings
- **Format**: Google-style docstrings with parameter types and quantum circuit descriptions
- **Generation**: `sphinx-build -b html docs/ docs/_build/html`

### Quantum Circuit Visualization
- **Requirement**: Auto-generate and store quantum circuit diagrams
- **Storage**: Save circuit diagrams to `docs/circuits/` directory
- **Format**: Both SVG (for docs) and PNG (for presentations) formats
- **Tools**: Use PennyLane's `qml.draw()` and CUDA-Q visualization tools
- **Naming**: `{approach}_{circuit_name}_{n_qubits}q_{n_layers}l.{ext}`

## Logging and Monitoring
- Use `QuantumPKPDLogger` for structured experiment logging
- Log all hyperparameter optimization results
- Track quantum circuit performance metrics
- Save optimization results to `logs/experiments/`
- **Gefion Integration**: Log GPU utilization and quantum simulation performance
- **Challenge Reporting**: Generate standardized reports for LSQI competition evaluation