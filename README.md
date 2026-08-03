# SR-FedOpt: Stein-Rule Federated Optimization

A federated learning simulator extending [felisat/federated-learning](https://github.com/felisat/federated-learning) with **SR-FedAdam** — a server-side optimizer that applies Stein-rule shrinkage to aggregated client updates.

> **Paper context:** This repo implements the federated variant of Stein-Rule Adam (SR-Adam). A separate repository handles the Taylor-expansion compression experiments. This repo is dedicated to the FedOpt paradigm: server-side variance-aware shrinkage without modifying client behavior.

---

## What is SR-FedAdam?

In each communication round, the server receives noisy client updates Δ⁽ᵏ⁾ that carry both signal and inter-client heterogeneity noise. SR-FedAdam applies a **positive-part James–Stein shrinkage** to the aggregated update before the adaptive scaling step:

```
σ²_t  = (1/K) Σ_k ‖Δ⁽ᵏ⁾ − Δ‖²           (inter-client variance)

α_t   = [1 − (d−2)σ²_t / (‖Δ_t − m_t‖² + ε)]₊     (shrinkage factor, clamped to [0,1])

Δ̃_t  = m_t + α_t (Δ_t − m_t)              (shrunk update)

θ_{t+1} = θ_t − η · Δ̃_t / (√v_t + ε)     (FedAdam-style parameter update)
```

where m_t, v_t are FedAdam first/second moments maintained at the server.

| SR-Adam concept | FL analogue |
|---|---|
| Noisy gradient g_t | Aggregated client update Δ_t |
| Noise source | Inter-client heterogeneity |
| Restricted estimator m_t | Server momentum (FedAdam 1st moment) |
| Dimension d | Parameter block size |
| σ² | Inter-client variance of updates |

---

## Repository Structure

```
sr-fedopt/
├── federated_learning.py        # Main entry point: runs all experiments
├── federated_learning.json      # Experiment schedules (compact seed-aware format)
├── distributed_training_utils.py # Client, Server classes; SR-FedAdam implementation
├── default_hyperparameters.py   # Default HP dict + get_hp()
├── experiment_manager.py        # Experiment tracking + compact config expansion
├── data_utils.py                # Data loading + FL data splitting
├── neural_nets.py               # Models: logistic, cnn, simple_cnn, lstm, vgg11[s]
├── compression_utils.py         # Gradient compression: DGC, STC, signSGD
├── visualize.py                 # Wrapper for visualization (Colab-safe)
├── smoke_run.py                 # One-round smoke test
├── test_expansion.py            # Verify compact config expansion
├── test_paths.py                # Verify generated log paths
├── tools/
│   ├── visualize_results.py     # Accuracy curves, barplots, CSV export
│   └── plot_summary_by_bs.py    # Batch-size sweep summary plots
├── examples/
│   ├── federated_learning_examples.json  # Ready-to-use schedule templates
│   └── README.md
├── docs/
│   ├── EXPERIMENT_CONFIG.md     # Compact config system guide
│   └── COLAB_GUIDE.md           # Google Colab setup guide
├── run_on_colab.ipynb           # Colab notebook
├── requirements.txt
├── CHANGELOG.md                 # Implementation history
└── TO-Do.md                     # Algorithm design spec for SR-FedAdam
```

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run a quick smoke test (1 round, MNIST, SR-FedAdam)

```bash
python smoke_run.py
```

### Run the main comparison schedule

```bash
python federated_learning.py --schedule main
```

This expands to experiments comparing **FedAvg**, **SR-FedAdam**, and **FedAdam** on MNIST/FashionMNIST (logistic regression) and CIFAR-10 (simple_cnn) across multiple noise levels and 5 seeds.

### Visualize results

```bash
python tools/visualize_results.py
```

Outputs plots and CSV files to `results/plots/`.

---

## Running on an RTX 3090 System

The experiments are designed for GPU acceleration. On a machine with an NVIDIA RTX 3090:

### 1. Set up the environment

```bash
# Create a conda environment (Python 3.10 recommended)
conda create -n srfedopt python=3.10 -y
conda activate srfedopt

# Install PyTorch with CUDA 12 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify GPU is detected

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
# Expected: CUDA: True | NVIDIA GeForce RTX 3090
```

### 3. Run experiments

**Quick test** (MNIST only, 2 seeds, ~5 minutes):
```bash
python federated_learning.py --schedule main --start 0 --end 10
```

**Full main schedule** (MNIST + FashionMNIST + CIFAR-10, 5 seeds per method, ~2–4 hours on RTX 3090):
```bash
python federated_learning.py --schedule main
```

**Run methods in parallel** (3 separate terminals):
```bash
# Terminal 1 — FedAvg runs
python federated_learning.py --schedule main --start 0 --end 40

# Terminal 2 — SR-FedAdam runs
python federated_learning.py --schedule main --start 40 --end 80

# Terminal 3 — FedAdam runs
python federated_learning.py --schedule main --start 80 --end 120
```

> **Tip:** Use `--start N` to resume from a specific experiment if a run is interrupted. Results are saved incrementally to `results/`.

### 4. Check progress

```bash
find results/ -name "*.npz" | wc -l  # number of completed experiments
```

### 5. Generate plots

```bash
python tools/visualize_results.py
# Outputs: results/plots/*.png  and  results/plots/*.csv
```

---

## Experiment Configuration

Schedules are defined in `federated_learning.json`. The compact seed-aware format auto-generates all experiment combinations:

```json
{
  "my_schedule": [
    {
      "dataset": ["mnist"],
      "net": ["logistic"],
      "iterations": [200],
      "n_clients": [50],
      "participation_rate": [0.5],
      "classes_per_client": [10],
      "batch_size": [1],
      "balancedness": [1.0],
      "momentum": [0.0],
      "compression": [["none", {}]],
      "log_frequency": [10],
      "optimizer": ["SGD"],
      "server_optimizer": ["fedavg", "sr_fedadam", "fedadam"],
      "server_lr": [1.0],
      "server_beta1": [0.9],
      "server_beta2": [0.999],
      "server_eps": [1e-8],
      "shrinkage_mode": ["global"],
      "shrinkage_scope": ["all"],
      "sigma_source": ["inter_client"],
      "client_update_noise_std": [0.0, 0.05],
      "base_log_path": ["results/my_schedule/"],
      "seeds": [42, 43, 44, 45, 46]
    }
  ]
}
```

This expands to **30 experiments** (3 methods × 2 noise levels × 5 seeds) with auto-generated log paths like `results/my_schedule/mnist/sigma0p05/sr_fedadam/run42/`.

See [docs/EXPERIMENT_CONFIG.md](docs/EXPERIMENT_CONFIG.md) for the full guide.

---

## Key Hyperparameters

### Federated Learning

| Parameter | Description | Default |
|---|---|---|
| `n_clients` | Number of clients | — |
| `participation_rate` | Fraction of clients per round | 1.0 |
| `classes_per_client` | Non-IID degree | — |
| `local_iterations` | Local SGD steps per round | 1 |
| `batch_size` | Client batch size | 100 |
| `balancedness` | Data balance across clients (1.0 = balanced) | 1.0 |
| `client_update_noise_std` | Gaussian noise std added to client updates | 0.0 |

### Server Optimizer (SR-FedAdam)

| Parameter | Description | Default |
|---|---|---|
| `server_optimizer` | `fedavg`, `sr_fedadam`, or `fedadam` | `fedavg` |
| `server_lr` | Server learning rate η | 1.0 |
| `server_beta1` | First moment decay β₁ | 0.9 |
| `server_beta2` | Second moment decay β₂ | 0.999 |
| `server_eps` | Adaptive denominator ε | 1e-8 |
| `shrinkage_mode` | `global` (single α over all params) or `per_layer` | `global` |
| `shrinkage_scope` | `all` parameters or `conv_only` (4D tensors) | `all` |
| `sigma_source` | `inter_client` (preferred) or `ema` | `inter_client` |

### Datasets & Models

| `dataset` | `net` | Notes |
|---|---|---|
| `mnist` | `logistic`, `cnn` | 28×28 grayscale |
| `fashionmnist` | `logistic`, `cnn` | 28×28 grayscale |
| `cifar10` | `simple_cnn`, `vgg11s`, `vgg11` | 32×32 RGB |

---

## Compression Methods

In addition to SR-FedAdam, the original DGC/STC gradient compression baselines are available via the `compression` key:

| Value | Description |
|---|---|
| `["none", {}]` | No compression (default) |
| `["dgc_up", {"p_up": 0.001}]` | Deep Gradient Compression (top-0.1% uplink) |
| `["stc_up", {"p_up": 0.001}]` | Sparse Ternary Compression (uplink) |
| `["signsgd", {"lr": 0.01}]` | signSGD with majority vote aggregation |

---

## Citation

This repo extends the simulator from:

> Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). *Robust and Communication-Efficient Federated Learning from Non-IID Data.* arXiv:1903.02891.

If you use SR-FedAdam, please also cite the SR-Adam paper (link TBD).

