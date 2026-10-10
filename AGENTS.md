# AGENTS.md - Digital Homeostasis

## Overview

This is a bio-inspired adaptive ML system for online learning under concept drift. The codebase implements homeostatic control with multiple adaptation strategies.

## Quick Commands

```bash
# Run all tests
cd "C:\Users\pator\Desktop\Digital Homeostasis"
python -m pytest tests/test_all.py -v

# Quick smoke test
python quick_run.py

# Run experiments
python experiments/run_experiments.py

# With plots
python experiments/run_experiments.py --no-plot false
```

## Code Architecture

- `system_architecture/`: Core homeostatic system components
  - `prediction_model.py`: Online SGD with epoch-aware episodic memory, drift notification, filtered replay
  - `performance_monitor.py`: Tracks metrics, detects drift
  - `homeostasis_controller.py`: PID-based regulation, plasticity mode, cooldowns
  - `adaptive_mechanisms.py`: Executes FAST/HORMONAL/STRUCTURAL adaptation strategies
  - `model_self_regulation.py`: Orchestrates full step pipeline
  - `input_data.py`: Streaming generator with drift types (regular/sudden/gradual/recurring)
- `base_model/`: Baseline system for comparison
- `evaluation/`: Metrics and evaluators
- `experiments/`: Multi-seed experiment runners
- `visualization/`: Plotting utilities
- `tests/test_all.py`: Comprehensive test suite (29 tests)

## Key Concepts

**Epochs**: When drift is detected, `notify_drift()` increments epoch. Memory samples tagged by epoch; replay can be filtered to `min_epoch=current_epoch` to avoid contamination from old concepts.

**Plasticity mode**: Active for 100 steps after drift - suppresses target self-tuning, uses aggressive learning rates, allows structural adaptation earlier if enough post-drift samples available.

**Actions**:
- NO_ACTION: Homeostasis score high, stable
- FAST_ADAPT: Light online updates + prioritized replay
- HORMONAL_ADAPT: Multiple rapid updates + error marking
- STRUCTURAL_ADAPT: Memory rebuild restricted to current epoch (post-drift samples) when threshold met

## Development Guidelines

- Maintain backward compatibility with existing test expectations
- When modifying controller logic, preserve plasticity/drift handling behavior
- Epoch-aware operations: always consider min_epoch when rebuilding/replaying after drift
- Tests run deterministically with seeds - be careful not to break seeded behavior
- Follow existing code style (minimal comments as requested by conventions)

## Testing

All 29 tests must pass. Tests check accuracy before/after drift, drift detection, recovery speed, adaptation triggering, and homeostasis scores across all drift types.

```bash
python -m pytest tests/test_all.py -v
```
