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

1. Run compare schedule:

```bash
python federated_learning.py --schedule compare_sr --start 0 --end 3
```

2. Visualize results:

```bash
python tools/visualize_results.py
python tools/plot_summary_by_bs.py
```

## Notes / Next steps

- Optionally implement a server-side vanilla FedAdam aggregator for an additional baseline.
- Consider adding per-round detailed logging of shrinkage factors into `.npz` for richer analysis.
