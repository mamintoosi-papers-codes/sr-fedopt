import json
import subprocess
import sys
from pathlib import Path
import shutil

FILENAME = Path('federated_learning.json')
BACKUP = Path('federated_learning.json.bak')

smoke_schedule = {
    "smoke": [
        {
            "dataset": ["mnist"],
            "net": ["logistic"],
            "iterations": [10],
            "n_clients": [10],
            "participation_rate": [0.5],
            "classes_per_client": [10],
            "batch_size": [1],
            "balancedness": [1.0],
            "momentum": [0.0],
            "compression": [["none", {}]],
            "log_frequency": [1],
            "log_path": ["results/smoke/"],
            # server-side test of SR-FedAdam
            "server_optimizer": ["sr_fedadam"],
            "server_beta1": [0.9],
            "server_beta2": [0.999],
            "server_eps": [1e-8],
            "server_lr": [1.0],
            "shrinkage_mode": ["global"],
            "shrinkage_scope": ["all"],
            "sigma_source": ["inter_client"]
        }
    ]
}


def backup_original():
    if FILENAME.exists():
        shutil.copy2(FILENAME, BACKUP)


def restore_original():
    if BACKUP.exists():
        shutil.move(str(BACKUP), str(FILENAME))


def write_smoke_to_main_json():
    # write smoke schedule to federated_learning.json (overwrites)
    with open(FILENAME, 'w') as f:
        json.dump(smoke_schedule, f, indent=2)


def run_smoke():
    try:
        backup_original()
        write_smoke_to_main_json()
        cmd = [sys.executable, 'federated_learning.py', '--schedule', 'smoke', '--start', '0', '--end', '1']
        print('Running smoke command:', ' '.join(cmd))
        p = subprocess.Popen(cmd)
        p.wait()
        print('Smoke run finished with return code', p.returncode)
    finally:
        restore_original()


if __name__ == '__main__':
    run_smoke()
