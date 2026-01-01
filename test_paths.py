import json
import experiment_manager as xpm

hp = json.load(open('federated_learning.json'))['main'][0]
exps = xpm.get_all_hp_combinations(hp)

print(f'Total: {len(exps)} experiments')
print('\nFirst 6 paths (showing different combinations):')
for i, e in enumerate(exps[:6]):
    print(f"  {i+1}. dataset={e['dataset']}, noise={e.get('client_update_noise_std',0)}, method={e['server_optimizer']}")
    print(f"     -> {e['log_path']}")
