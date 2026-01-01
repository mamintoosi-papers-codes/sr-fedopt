# Federated Learning Simulator

Simulate Federated Learning with compressed communication on a large number of Clients.

Recreate experiments described in [*Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). Robust and Communication-Efficient Federated Learning from Non-IID Data. arXiv preprint arXiv:1903.02891.*](https://arxiv.org/abs/1903.02891)

## Quick Start on Colab

Open in Colab (runs `main.ipynb`):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamintoosi-papers-codes/SR-Adam/blob/main/main.ipynb)

## Quick Start

```bash
# Run a predefined experiment schedule
python federated_learning.py --schedule compare_sr

# Visualize results
python tools/visualize_results.py
```

## What's New

✨ **Refactored Experiment Configuration System**
- **Compact, reproducible specifications** with automatic expansion from seeds
- **Professional-grade** configuration suitable for journal papers
- **Backward compatible** with legacy configs

Example compact configuration:
```json
{
  "dataset": ["mnist"],
  "server_optimizer": ["fedavg", "sr_fedadam", "fedadam"],
  "client_update_noise_std": [0.0, 0.01, 0.05],
  "seeds": [42, 43, 44, 45, 46],
  "base_log_path": ["results/experiment/"]
}
```

This **automatically generates** all combinations with structured log paths:
- `results/experiment/fedavg/run42/`
- `results/experiment/sigma0p01/sr_fedadam/run43/`
- etc.

📚 **Documentation**:
- [Experiment Configuration Guide](docs/EXPERIMENT_CONFIG.md) — Complete configuration system documentation
- [Example Configs](examples/) — Ready-to-use experiment templates
- [CHANGELOG.md](CHANGELOG.md) — Recent changes and implementation details


## Usage
First, set environment variable 'TRAINING_DATA' to point to the directory where you want your training data to be stored. MNIST, FASHION-MNIST and CIFAR10 will download automatically. 

`python federated_learning.py`

will run the Federated Learning experiment specified in `federated_learning.json`.

## What this repo is
Lightweight federated-learning simulator for experimenting with compressed communication and server-side aggregation rules.

Quick highlights:
- Run experiments defined in `federated_learning.json` with `python federated_learning.py`.
- New experimental server optimizer: SR-FedAdam (server-side Stein-rule shrinkage).
- Lightweight plotting tools in `tools/` to visualize `.npz` results.
- **Compact experiment configuration** with automatic seed expansion and path generation.

SR-FedAdam is applied after aggregation and before the server broadcasts updates to clients.

You can specify:

### Task
- `"dataset"` : Choose from `["mnist", "cifar10", "kws", "fashionmnist"]`
- `"net"` : Choose from `["logistic", "lstm", "cnn", "vgg11", "vgg11s"]`

### Federated Learning Environment

- `"n_clients"` : Number of Clients
- `"classes_per_client"` : Number of different Classes every Client holds in it's local data
- `"participation_rate"` : Fraction of Clients which participate in every Communication Round
- `"batch_size"` : Batch-size used by the Clients
- `"balancedness"` : Default 1.0, if <1.0 data will be more concentrated on some clients
- `"iterations"` : Total number of training iterations
- `"momentum"` : Momentum used during training on the clients

### Server Optimizer (New)

- `"server_optimizer"` : Choose from `["fedavg", "sr_fedadam", "fedadam"]`
- `"server_lr"` : Server learning rate (default: 1.0)
- `"server_beta1"` : First moment decay (default: 0.9)
- `"server_beta2"` : Second moment decay (default: 0.999)
- `"shrinkage_mode"` : For SR-FedAdam: `["global", "per-layer"]`
- `"shrinkage_scope"` : For SR-FedAdam: `["all", "conv_only"]`
- `"client_update_noise_std"` : Gaussian noise std added to client updates (default: 0.0)

### Experiment Organization (New)

- `"seeds"` : List of random seeds (e.g., `[42, 43, 44, 45, 46]`)
- `"base_log_path"` : Base directory for auto-generated log paths

When `seeds` is specified, the system automatically:
1. Generates one experiment per seed for each parameter combination
2. Creates structured log paths: `{base_log_path}/[sigma{noise}/]{method}/run{seed}/`

See [docs/EXPERIMENT_CONFIG.md](docs/EXPERIMENT_CONFIG.md) for complete documentation.

### Compression Method

- `"compression"` : Choose from `[["none", {}], ["fedavg", {"n" : ?}], ["signsgd", {"lr" : ?}], ["stc_updown", [{"p_up" : ?, "p_down" : ?}]], ["stc_up", {"p_up" : ?}], ["dgc_updown", [{"p_up" : ?, "p_down" : ?}]], ["dgc_up", {"p_up" : ?}] ]`

### Logging 
- `"log_frequency"` : Number of communication rounds after which results are logged and saved to disk
- `"log_path"` : e.g. "results/experiment1/" (auto-generated when using `seeds` and `base_log_path`)

Run multiple experiments by listing different configurations.

## Options
- `--schedule` : specify which batch of experiments to run, defaults to "main"

## Utilities: plotting & analysis
I added lightweight plotting utilities (adapted from a previous Taylor prototype) under `tools/`:

- `tools/visualize_results.py` — read `results/*.npz` experiment outputs and produce per-run accuracy curves and dataset-level barplots.
- `tools/plot_summary_by_bs.py` — build a sweep-like summary (dataset/method/batch-size) from saved `.npz` runs and produce bar/line plots grouped by batch-size and dataset.

Usage examples:
```
python tools/visualize_results.py
python tools/plot_summary_by_bs.py
```

Outputs are written to `results/plots`, `results/plots_by_bs`, and `results/plots_by_dataset` respectively.

## Quick smoke run
To run a small experiment and test SR-FedAdam locally edit `federated_learning.json` or override hyperparameters, then run:
```
python federated_learning.py --schedule main --start 0 --end 1
```

or

```
python federated_learning.py --schedule compare_sr --start 0 --end 15
```

If you want a dedicated single-round smoke test script, ask me and I'll add `smoke_run.py`.

## Citation 
[Paper](https://arxiv.org/abs/1903.02891)

Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). Robust and Communication-Efficient Federated Learning from Non-IID Data. arXiv preprint arXiv:1903.02891.
