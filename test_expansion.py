"""
Quick test to verify the experiment expansion logic works correctly.
"""
import json
import experiment_manager as xpm

# Load schedules
with open('federated_learning.json') as f:
    schedules = json.load(f)

# Test compare_sr expansion
print("=" * 80)
print("Testing compare_sr expansion:")
print("=" * 80)
compare_sr = schedules['compare_sr']
print(f"\nInput: {len(compare_sr)} schedule spec(s)")

expanded = [hp for spec in compare_sr for hp in xpm.get_all_hp_combinations(spec)]
print(f"Expanded to: {len(expanded)} experiments\n")

# Show first 3 and last 3 experiments
print("First 3 experiments:")
for i, exp in enumerate(expanded[:3]):
    print(f"  {i}: dataset={exp['dataset']}, seed={exp['seed']}, method={exp['server_optimizer']}")
    print(f"      path={exp['log_path']}")

print("\n...")
print("\nLast 3 experiments:")
for i, exp in enumerate(expanded[-3:], start=len(expanded)-3):
    print(f"  {i}: dataset={exp['dataset']}, seed={exp['seed']}, method={exp['server_optimizer']}")
    print(f"      path={exp['log_path']}")

# Test multi_dataset_noise expansion
print("\n" + "=" * 80)
print("Testing multi_dataset_noise expansion:")
print("=" * 80)
multi_noise = schedules['multi_dataset_noise']
print(f"\nInput: {len(multi_noise)} schedule spec(s)")

expanded_multi = [hp for spec in multi_noise for hp in xpm.get_all_hp_combinations(spec)]
print(f"Expanded to: {len(expanded_multi)} experiments\n")

# Group by dataset and noise
by_dataset = {}
for exp in expanded_multi:
    dataset = exp['dataset']
    noise = exp['client_update_noise_std']
    key = (dataset, noise)
    if key not in by_dataset:
        by_dataset[key] = []
    by_dataset[key].append(exp)

print(f"Experiments by dataset and noise level:")
for (dataset, noise), exps in sorted(by_dataset.items()):
    methods = {}
    for exp in exps:
        method = exp['server_optimizer']
        if method not in methods:
            methods[method] = []
        methods[method].append(exp['seed'])
    
    print(f"\n  {dataset} + σ = {noise}:")
    for method, seeds in sorted(methods.items()):
        print(f"    {method}: {len(seeds)} runs with seeds {sorted(seeds)}")

# Show sample paths for each dataset
print("\n" + "=" * 80)
print("Sample log paths by dataset and noise:")
print("=" * 80)
for dataset in ['mnist', 'fashionmnist', 'cifar10']:
    print(f"\n{dataset.upper()}:")
    dataset_exps = [e for e in expanded_multi if e['dataset'] == dataset]
    if dataset_exps:
        # Show one clean and one noisy example
        clean_exp = next((e for e in dataset_exps if e['client_update_noise_std'] == 0.0), None)
        noisy_exp = next((e for e in dataset_exps if e['client_update_noise_std'] > 0.0), None)
        
        if clean_exp:
            print(f"  Clean: {clean_exp['log_path']}")
        if noisy_exp:
            noise_val = noisy_exp['client_update_noise_std']
            print(f"  Noise σ={noise_val}: {noisy_exp['log_path']}")

print("\n" + "=" * 80)
print("Path Structure:")
print("=" * 80)
print("  Clean:  base_path/dataset/method/run{seed}/")
print("  Noisy:  base_path/dataset/sigma{noise}/method/run{seed}/")
print("\n✓ Expansion test completed successfully!")
print("=" * 80)
