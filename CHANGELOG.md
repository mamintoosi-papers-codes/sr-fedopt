# CHANGELOG

All notable changes made to this repository during the SR-FedAdam integration and tooling update.

## Summary (concise)

- Implement SR-FedAdam (server-side Stein-rule shrinkage) in `distributed_training_utils.py`.
- Add server-side hyperparameters and defaults in `default_hyperparameters.py`.
- Save server optimizer metadata with experiments via `Experiment.prepare()`.
- Add visualization tools: `tools/visualize_results.py` and `tools/plot_summary_by_bs.py`.
- Add `smoke_run.py` (temporary schedule backup/restore) and `compare_sr` schedule in `federated_learning.json`.
- Update `README.md` with a short description and pointers to tools.
- Add client-update Gaussian noise option, noise-aware schedule `compare_sr_noise`, and visualizer support (noise-tagged plots/CSVs).
- Expand `compare_sr` to explicit multi-run (5x seeds per method) with stable colors and mean/std shading in plots.
- **NEW**: Refactor experiment configuration system to support compact, reproducible specifications with automatic expansion from seeds.

## Recent Changes (Experiment Configuration Refactoring)

### Motivation
- Previous approach required manual duplication of 15-27 experiment entries
- Hard to maintain, error-prone, not scalable for journal-level sweeps
- Difficult to add new methods, datasets, or noise levels

### What Changed

1. **New Configuration Format** (recommended):
   ```json
   {
     "dataset": ["mnist"],
     "server_optimizer": ["fedavg", "sr_fedadam", "fedadam"],
     "client_update_noise_std": [0.01, 0.05, 0.1],
     "seeds": [42, 43, 44, 45, 46],
     "base_log_path": ["results/experiment/"]
   }
   ```

2. **Automatic Expansion**:
   - System detects `seeds` key and automatically generates all combinations
   - Auto-generates structured log paths: `{base}/{noise}/{method}/run{seed}/`
   - Reduces 27-line configs to single compact specifications

3. **Backward Compatibility**:
   - Legacy configs without `seeds` key still work
   - No changes to CLI interface or execution logic

### Files Modified

- **experiment_manager.py**:
  - Enhanced `get_all_hp_combinations()` to detect and dispatch seed-based expansion
  - Added `_expand_with_seeds()` for automatic experiment generation with smart path formatting

- **federated_learning.json**:
  - Refactored `compare_sr` from 15 explicit entries to 1 compact spec
  - Refactored `compare_sr_noise` from 27 explicit entries to 1 compact spec
  - Maintained identical experimental outcomes (same seeds, paths, parameters)

- **docs/EXPERIMENT_CONFIG.md** (new):
  - Comprehensive guide to compact configuration syntax
  - Usage examples for common scenarios
  - Best practices for reproducible research
  - Troubleshooting guide

### Benefits

✅ **Reproducibility**: Explicit seed lists, no hidden randomness  
✅ **Maintainability**: Change one line to add a method across all noise levels  
✅ **Readability**: See experiment structure at a glance  
✅ **Scalability**: Easy to add datasets, methods, or hyperparameter sweeps  
✅ **Professional**: Clean configuration suitable for publication

### Migration Guide

Old style (manual):
```json
[
  {"server_optimizer": ["fedavg"], "seed": [42], "log_path": ["results/fedavg/run42/"]},
  {"server_optimizer": ["fedavg"], "seed": [43], "log_path": ["results/fedavg/run43/"]},
  ...
]
```

New style (compact):
```json
[
  {
    "server_optimizer": ["fedavg"],
    "seeds": [42, 43],
    "base_log_path": ["results/"]
  }
]
```

## Details

- SR-FedAdam
  - Server maintains first/second moments and optional EMA of inter-client variance.
  - Computes per-block or global inter-client variance and applies positive-part Stein shrinkage.
  - Applies FedAdam-like scaling (divide by sqrt(v)+eps) before updating server weights.
  - Logs shrinkage statistics (`sr_alpha_mean`, `sr_alpha_frac_clipped`, `sr_sigma_mean`) to experiment results.

- Tooling
  - `tools/visualize_results.py`: reads `results/*.npz` files, produces per-run accuracy curves and barplots, now groups by noise level, uses fixed colors per method, mean±std shading, and exports per-noise CSVs (`*_accuracy_curves.csv`, `*_statistics_summary.csv`).
  - `tools/plot_summary_by_bs.py`: builds a sweep-style summary and saves plots grouped by batch-size and dataset.

- Experiments
  - `smoke_run.py` provides a safe one-shot smoke test (backs up and restores `federated_learning.json`).
  - `federated_learning.json` includes `compare_sr` expanded to 5 seeds per method (distinct log paths) and a noise stress-test `compare_sr_noise` sweeping `client_update_noise_std` ∈ {0.01, 0.05, 0.1} with 3 seeds per method.

## How to reproduce

1. Run compare schedule (now generates 15 experiments from compact spec):

```bash
python federated_learning.py --schedule compare_sr
```

2. Run noise robustness test (generates 27 experiments):

```bash
python federated_learning.py --schedule compare_sr_noise
```

3. Visualize results:

```bash
python tools/visualize_results.py
python tools/plot_summary_by_bs.py
```

## Notes / Next steps

- Experiment configuration system ready for large-scale journal experiments
- Easy to extend to multiple datasets (MNIST, CIFAR-10, etc.)
- Consider adding statistical significance tests (t-test) to visualization tools
- Optionally implement deeper per-round shrinkage logging for detailed analysis
