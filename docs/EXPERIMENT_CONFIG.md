# Experiment Configuration Guide

This document explains the experiment configuration system for SR-FedOpt.

## Overview

The experiment system supports two configuration styles:

1. **Legacy style**: Explicit list of experiment dictionaries (backward compatible)
2. **Compact style with seeds**: Automatic expansion from concise specifications (recommended)

## Compact Configuration with Seeds

### Basic Structure

```json
{
  "schedule_name": [
    {
      "dataset": ["mnist"],
      "server_optimizer": ["fedavg", "fedadam", "sr_fedadam"],
      "client_update_noise_std": [0.0, 0.01, 0.05],
      "seeds": [42, 43, 44, 45, 46],
      "base_log_path": ["results/my_experiment/"],
      ... (other hyperparameters)
    }
  ]
}
```

### Automatic Expansion

When a configuration includes a `seeds` key, the system automatically:

1. **Generates all parameter combinations** (Cartesian product of all list parameters)
2. **Creates one experiment per seed** for each combination
3. **Auto-generates log paths** following the pattern:
   - **Without noise**: `{base_log_path}/{method}/run{seed}/`
   - **With noise**: `{base_log_path}/sigma{noise}/{method}/run{seed}/`

### Example: Compare SR Schedule

**Input** (compact):
```json
{
  "compare_sr": [
    {
      "dataset": ["mnist"],
      "net": ["logistic"],
      "iterations": [200],
      "server_optimizer": ["fedavg", "sr_fedadam", "fedadam"],
      "seeds": [42, 43, 44, 45, 46],
      "base_log_path": ["results/compare/"],
      ... (other parameters)
    }
  ]
}
```

**Automatically expands to** 15 experiments:
- 3 methods × 5 seeds = 15 runs
- Log paths:
  - `results/compare/fedavg/run42/`
  - `results/compare/fedavg/run43/`
  - ...
  - `results/compare/sr_fedadam/run42/`
  - ...
  - `results/compare/fedadam/run46/`

### Example: Noise Robustness Study

**Input** (compact):
```json
{
  "compare_sr_noise": [
    {
      "dataset": ["mnist"],
      "server_optimizer": ["fedavg", "sr_fedadam", "fedadam"],
      "client_update_noise_std": [0.01, 0.05, 0.1],
      "seeds": [42, 43, 44],
      "base_log_path": ["results/compare_noise/"],
      ... (other parameters)
    }
  ]
}
```

**Automatically expands to** 27 experiments:
- 3 methods × 3 noise levels × 3 seeds = 27 runs
- Log paths:
  - `results/compare_noise/sigma0p01/fedavg/run42/`
  - `results/compare_noise/sigma0p01/fedavg/run43/`
  - ...
  - `results/compare_noise/sigma0p05/sr_fedadam/run42/`
  - ...
  - `results/compare_noise/sigma0p1/fedadam/run44/`

### Noise Path Formatting

Noise values in paths have periods replaced with 'p':
- `0.01` → `sigma0p01`
- `0.05` → `sigma0p05`
- `0.1` → `sigma0p1`

## Usage

### Running Experiments

The CLI interface remains unchanged:

```bash
# Run all experiments in a schedule
python federated_learning.py --schedule compare_sr

# Run a subset (by index after expansion)
python federated_learning.py --schedule compare_sr --start 0 --end 5

# Run in reverse order
python federated_learning.py --schedule compare_sr --reverse_order True
```

### Slicing Examples

For a schedule that expands to 15 experiments:

```bash
# First 5 experiments (seeds 42-46 for fedavg)
python federated_learning.py --schedule compare_sr --start 0 --end 5

# Next 5 experiments (seeds 42-46 for sr_fedadam)
python federated_learning.py --schedule compare_sr --start 5 --end 10

# Last 5 experiments (seeds 42-46 for fedadam)
python federated_learning.py --schedule compare_sr --start 10 --end 15
```

## Advanced Features

### Custom Log Paths

You can still manually specify `log_path` to override auto-generation:

```json
{
  "dataset": ["mnist"],
  "seeds": [42, 43],
  "log_path": ["results/custom/path/"]
}
```

### Multi-Dataset Sweeps

Combine multiple datasets with methods and seeds:

```json
{
  "dataset": ["mnist", "cifar10"],
  "server_optimizer": ["fedavg", "sr_fedadam"],
  "seeds": [42, 43, 44],
  "base_log_path": ["results/multi_dataset/"]
}
```

This expands to 12 experiments (2 datasets × 2 methods × 3 seeds).

### Hyperparameter Grids

Search over hyperparameters while maintaining reproducibility:

```json
{
  "dataset": ["mnist"],
  "server_optimizer": ["sr_fedadam"],
  "server_lr": [0.1, 1.0, 10.0],
  "server_beta1": [0.9, 0.95],
  "seeds": [42, 43, 44, 45, 46],
  "base_log_path": ["results/hp_search/"]
}
```

Expands to 30 experiments (3 LRs × 2 β₁ × 5 seeds).

## Legacy Configuration (Still Supported)

You can still use explicit experiment lists without the `seeds` key:

```json
{
  "legacy_schedule": [
    {
      "dataset": ["mnist"],
      "server_optimizer": ["fedavg"],
      "seed": [42],
      "log_path": ["results/legacy/run1/"]
    },
    {
      "dataset": ["mnist"],
      "server_optimizer": ["fedavg"],
      "seed": [43],
      "log_path": ["results/legacy/run2/"]
    }
  ]
}
```

## Best Practices

### For Journal Papers

1. **Always specify seeds explicitly** for reproducibility
2. **Use descriptive base_log_path** that reflects the experiment purpose
3. **Document your seed choices** (e.g., consecutive integers starting from 42)
4. **Use 5+ seeds** for statistical validity

### For Quick Testing

1. **Use fewer seeds** (e.g., `[42, 43]`) for rapid iteration
2. **Reduce iterations** for smoke tests
3. **Test with one method first** before running full sweeps

### For Large Sweeps

1. **Use compact notation** to reduce JSON file size
2. **Run in batches** with `--start` and `--end`
3. **Parallelize across machines** by distributing index ranges

## Troubleshooting

### Problem: Too many experiments generated

**Solution**: Use `--start` and `--end` to run subsets, or reduce the number of seeds/methods/noise levels.

### Problem: Log paths colliding

**Solution**: Ensure unique combinations of (dataset, noise, method, seed). The auto-generation handles this automatically.

### Problem: Need custom path structure

**Solution**: Either manually specify `log_path` in each experiment, or modify `_expand_with_seeds()` in `experiment_manager.py`.

## Implementation Details

The expansion logic is in [experiment_manager.py](../experiment_manager.py):

- `get_all_hp_combinations()`: Detects `seeds` key and dispatches to expansion
- `_expand_with_seeds()`: Performs Cartesian product and path generation
- Backward compatible with legacy configurations (no `seeds` key)

---

For more examples, see [federated_learning.json](../federated_learning.json).
