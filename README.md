# Federated Learning Simulator

Simulate Federated Learning with compressed communication on a large number of Clients.

Recreate experiments described in [*Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). Robust and Communication-Efficient Federated Learning from Non-IID Data. arXiv preprint arXiv:1903.02891.*](https://arxiv.org/abs/1903.02891)



## Usage
First, set environment variable 'TRAINING_DATA' to point to the directory where you want your training data to be stored. MNIST, FASHION-MNIST and CIFAR10 will download automatically. 

`python federated_learning.py`

will run the Federated Learning experiment specified in  

## What this repo is
Lightweight federated-learning simulator for experimenting with compressed communication and server-side aggregation rules.

Quick highlights:
- Run experiments defined in `federated_learning.json` with `python federated_learning.py`.
- New experimental server optimizer: SR-FedAdam (server-side Stein-rule shrinkage).
- Lightweight plotting tools in `tools/` to visualize `.npz` results.

See `CHANGELOG.md` for a concise list of recent modifications and usage notes.

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

### Compression Method

- `"compression"` : Choose from `[["none", {}], ["fedavg", {"n" : ?}], ["signsgd", {"lr" : ?}], ["stc_updown", [{"p_up" : ?, "p_down" : ?}]], ["stc_up", {"p_up" : ?}], ["dgc_updown", [{"p_up" : ?, "p_down" : ?}]], ["dgc_up", {"p_up" : ?}] ]`

### Logging 
- `"log_frequency"` : Number of communication rounds after which results are logged and saved to disk
- `"log_path"` : e.g. "results/experiment1/"

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
