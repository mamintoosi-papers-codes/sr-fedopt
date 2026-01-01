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
    print(f"  {i}: seed={exp['seed']}, method={exp['server_optimizer']}, path={exp['log_path']}")

print("\n...")
print("\nLast 3 experiments:")
for i, exp in enumerate(expanded[-3:], start=len(expanded)-3):
    print(f"  {i}: seed={exp['seed']}, method={exp['server_optimizer']}, path={exp['log_path']}")

# Test compare_sr_noise expansion
print("\n" + "=" * 80)
print("Testing compare_sr_noise expansion:")
print("=" * 80)
compare_noise = schedules['compare_sr_noise']
print(f"\nInput: {len(compare_noise)} schedule spec(s)")

expanded_noise = [hp for spec in compare_noise for hp in xpm.get_all_hp_combinations(spec)]
print(f"Expanded to: {len(expanded_noise)} experiments\n")

# Group by noise level
by_noise = {}
for exp in expanded_noise:
    noise = exp['client_update_noise_std']
    if noise not in by_noise:
        by_noise[noise] = []
    by_noise[noise].append(exp)

print(f"Experiments by noise level:")
for noise, exps in sorted(by_noise.items()):
    methods = {}
    for exp in exps:
        method = exp['server_optimizer']
        if method not in methods:
            methods[method] = []
        methods[method].append(exp['seed'])
    
    print(f"\n  σ = {noise}:")
    for method, seeds in sorted(methods.items()):
        print(f"    {method}: {len(seeds)} runs with seeds {sorted(seeds)}")

# Show sample paths for each noise level
print("\n" + "=" * 80)
print("Sample log paths by noise level:")
print("=" * 80)
for noise in sorted(by_noise.keys()):
    print(f"\nσ = {noise}:")
    sample_exp = by_noise[noise][0]
    print(f"  Example: {sample_exp['log_path']}")
    print(f"  Pattern: results/compare_noise/sigma{str(noise).replace('.', 'p')}/{{method}}/run{{seed}}/")

print("\n" + "=" * 80)
print("✓ Expansion test completed successfully!")
print("=" * 80)
