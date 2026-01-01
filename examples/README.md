# Example Experiment Configurations

This directory contains example experiment configurations demonstrating various use cases for SR-FedOpt.

## Files

- **[federated_learning_examples.json](federated_learning_examples.json)**: Comprehensive examples of different experiment types

## Example Schedules

### 1. Basic Test (`basic_test`)
**Purpose**: Single-run experiment using legacy format  
**Expands to**: 1 experiment  
**Use case**: Testing single configuration without seed expansion

```bash
python federated_learning.py --schedule basic_test
```

### 2. Compare Methods (`compare_methods`)
**Purpose**: Compare FedAvg, SR-FedAdam, and FedAdam  
**Expands to**: 15 experiments (3 methods × 5 seeds)  
**Use case**: Standard baseline comparison for papers

```bash
python federated_learning.py --schedule compare_methods
```

### 3. Noise Robustness (`noise_robustness`)
**Purpose**: Test robustness under client update noise  
**Expands to**: 36 experiments (3 methods × 4 noise levels × 3 seeds)  
**Use case**: Evaluating SR-FedAdam's advantage under noise

```bash
python federated_learning.py --schedule noise_robustness
```

### 4. Hyperparameter Search (`hp_search`)
**Purpose**: Grid search over server learning rate and beta1  
**Expands to**: 30 experiments (3 LRs × 2 β₁ × 5 seeds)  
**Use case**: Finding optimal hyperparameters for SR-FedAdam

```bash
python federated_learning.py --schedule hp_search
```

### 5. Multi-Dataset (`multi_dataset`)
**Purpose**: Compare methods across MNIST and CIFAR-10  
**Expands to**: 18 experiments (2 datasets × 3 methods × 3 seeds)  
**Use case**: Demonstrating generalization across datasets

```bash
python federated_learning.py --schedule multi_dataset
```

### 6. Quick Test (`quick_test`)
**Purpose**: Fast iteration with reduced iterations and seeds  
**Expands to**: 4 experiments (2 methods × 2 seeds)  
**Use case**: Rapid prototyping and debugging

```bash
python federated_learning.py --schedule quick_test
```

## Using These Examples

### Copy to Main Config

To use an example schedule, copy it to `federated_learning.json`:

```bash
# On Linux/Mac
cp examples/federated_learning_examples.json federated_learning.json

# On Windows
copy examples\federated_learning_examples.json federated_learning.json
```

### Run Specific Examples

You can also point to the examples file directly by modifying the schedule loader in `federated_learning.py`, or copy individual schedules.

### Modify for Your Needs

Each example can be customized:

- **Add more seeds**: `"seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50]`
- **Change datasets**: `"dataset": ["mnist", "cifar10", "fashionmnist"]`
- **Adjust noise levels**: `"client_update_noise_std": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]`
- **Add methods**: `"server_optimizer": ["fedavg", "sr_fedadam", "fedadam", "your_method"]`

## Expansion Calculator

To see how many experiments a schedule will generate:

```python
import json
import experiment_manager as xpm

with open('examples/federated_learning_examples.json') as f:
    schedule = json.load(f)['your_schedule_name']

expanded = [hp for spec in schedule for hp in xpm.get_all_hp_combinations(spec)]
print(f"This schedule expands to {len(expanded)} experiments")
```

Or use the test script:

```bash
python test_expansion.py
```

## Best Practices

### For Journal Papers
- Use ≥5 seeds for statistical validity
- Include both clean and noisy conditions
- Document all hyperparameters explicitly
- Use descriptive `base_log_path` names

### For Conference Papers
- 3 seeds minimum
- Focus on key methods (FedAvg, SR-FedAdam)
- One or two datasets
- Clear noise sweep (0.0, 0.05, 0.1)

### For Quick Experiments
- 2 seeds for initial testing
- Reduce iterations (50-100)
- Fewer clients (10-20)
- Single dataset

## Batch Execution

Run large experiments in parallel by slicing:

```bash
# Terminal 1: First third
python federated_learning.py --schedule hp_search --start 0 --end 10

# Terminal 2: Second third
python federated_learning.py --schedule hp_search --start 10 --end 20

# Terminal 3: Last third
python federated_learning.py --schedule hp_search --start 20 --end 30
```

## See Also

- [Experiment Configuration Guide](../docs/EXPERIMENT_CONFIG.md): Complete documentation
- [Main README](../README.md): Project overview
- [CHANGELOG](../CHANGELOG.md): Recent changes

---

For more details on the compact configuration syntax, see [docs/EXPERIMENT_CONFIG.md](../docs/EXPERIMENT_CONFIG.md).
