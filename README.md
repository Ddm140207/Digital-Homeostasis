# Digital Homeostasis: A Bio-Inspired Adaptive ML System

**Digital Homeostasis** is a bio-inspired machine learning framework that maintains model stability and performance under concept drift through homeostatic control, adaptive mechanisms, and self-regulation.

## Overview

Inspired by biological homeostasis, this system dynamically monitors model performance, detects distributional shifts, and triggers appropriate adaptive responses (from fast online updates to structural adaptations) to preserve accuracy while minimizing unnecessary interventions.

Key features:
- Bio-inspired homeostasis controller with PID regulation and plasticity modes
- Multi-timescale adaptation (fast/hormonal/structural responses)
- Epoch-aware episodic memory to prevent catastrophic forgetting under drift
- Drift detection and responsive plasticity windows
- Comprehensive evaluation across multiple drift types (regular, sudden, gradual, recurring)

## Architecture

```text
system_architecture/
  ├── homeostasis_controller.py  # PID-based homeostatic regulation
  ├── model_self_regulation.py   # Orchestrates monitoring, control, adaptation
  ├── adaptive_mechanisms.py     # Executes adaptation strategies
  ├── performance_monitor.py    # Metrics tracking and drift detection
  ├── prediction_model.py       # Online SGD with epoch-aware memory
  ├── input_data.py             # Streaming data generator with drift types
  └── ...

base_model/
  ├── ml_base_model.py           # Baseline online learning model
  └── base_runner.py             # Baseline system wrapper
```

## Quick Start

### Install Dependencies

```bash
pip install numpy scipy scikit-learn pytest matplotlib pyyaml joblib
```

### Run Tests

```bash
cd "C:\Users\pator\Desktop\Digital Homeostasis"
python -m pytest tests/test_all.py -v
```

All 29 tests cover regular, sudden, gradual, and recurring drift scenarios.

### Run Experiments

```bash
# Run multi-seed experiments across all drift types
python experiments/run_experiments.py

# Run with custom parameters
python experiments/run_experiments.py --seeds 5 --steps 800 --warmup 150

# Run a single drift type
python experiments/run_experiments.py --drift sudden
```

### Quick Smoke Test

```bash
python quick_run.py  # Runs a quick comparison for one drift type
```

## How It Works

1. **Prediction**: Online SGD model makes predictions on streaming samples
2. **Monitoring**: PerformanceMonitor tracks error rate, confidence, entropy, and detects drift
3. **Control**: HomeostasisController computes homeostasis score (H) and selects action
   - `NO_ACTION`: H>0.70 and healthy
   - `FAST_ADAPT`: minor adjustments needed
   - `HORMONAL_ADAPT`: moderate degradation
   - `STRUCTURAL_ADAPT`: severe degradation/recovery from drift
4. **Plasticity**: On drift detection, enters 100-step plasticity window for rapid adaptation
5. **Epoch-Aware Memory**: Replay prioritizes post-drift (current epoch) samples to avoid old-concept contamination
6. **Self-Regulation**: ModelSelfRegulation orchestrates the full pipeline

## Key Design Principles

- **Stability-plasticity tradeoff**: Controlled by plasticity windows and cooldowns
- **Drift-aware adaptation**: Escalates response intensity when drift is detected
- **Epoch separation**: Prevents interference between concepts
- **Conservative by default**: Triggers structural changes only when necessary

## Project Status

All tests pass (29/29). The system successfully detects drift, recovers faster than baseline in sudden drift cases, and maintains above-chance performance across all drift types.

## Citation

If you use this work, please cite appropriately. This is an academic/research prototype exploring bio-inspired approaches to continual learning under distribution shift.
