import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os
import pandas as pd

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_npz_files(root: Path):
    files = list(root.rglob("*.npz"))
    entries = []
    for f in files:
        try:
            data = dict(np.load(f, allow_pickle=True))
        except Exception:
            continue

        # extract hyperparameters if present
        hp = None
        if 'hyperparameters' in data:
            try:
                hp = data['hyperparameters'].item()
            except Exception:
                hp = data['hyperparameters']

        # try to infer dataset and method
        dataset = None
        method = None
        if isinstance(hp, dict):
            dataset = hp.get('dataset')
            method = hp.get('server_optimizer') or hp.get('optimizer')

        # fallback to filename parsing
        m = re.match(r"(.*?)_(.*?)_.*", f.stem)
        if not dataset:
            if m:
                dataset = m.group(1)
            else:
                dataset = 'unknown'
        if not method:
            if m:
                method = m.group(2)
            else:
                method = 'unknown'

        entries.append({'path': f, 'data': data, 'hp': hp, 'dataset': dataset, 'method': method})

    return entries


def plot_per_run_curves(entries):
    # group by dataset+method
    groups = {}
    for e in entries:
        key = (e['dataset'], e['method'])
        groups.setdefault(key, []).append(e)

    for (dataset, method), runs in groups.items():
        plt.figure(figsize=(6, 4))
        for r in runs:
            data = r['data']
            if 'accuracy_test' in data:
                acc = np.array(data['accuracy_test'])
                plt.plot(acc, alpha=0.6)
        plt.title(f"{dataset} - {method}")
        plt.xlabel("Communication Round")
        plt.ylabel("Test Accuracy")
        plt.grid(True)
        out = OUT_DIR / f"{dataset}_{method}_accuracy_curve.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def plot_summary_bar(entries):
    # collect final accuracy per run
    rows = []
    for e in entries:
        data = e['data']
        if 'accuracy_test' in data:
            try:
                final = float(np.array(data['accuracy_test'])[-1])
            except Exception:
                continue
            rows.append({'dataset': e['dataset'], 'method': e['method'], 'acc_final': final})

    if not rows:
        print("[WARN] No accuracy_test arrays found in results.")
        return

    df = pd.DataFrame(rows)
    for dataset in df['dataset'].unique():
        sub = df[df['dataset'] == dataset]
        agg = sub.groupby('method')['acc_final'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(8, 4))
        plt.bar(agg['method'], agg['mean'], yerr=agg['std'].fillna(0.0))
        plt.title(f"{dataset} - Final Accuracy")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        out = OUT_DIR / f"{dataset}_accuracy_barplot.png"
        plt.savefig(out)
        plt.close()


def main():
    entries = load_npz_files(RESULTS_DIR)
    if not entries:
        print("No .npz results found under results/ - run experiments first or point to another directory.")
        return
    plot_per_run_curves(entries)
    plot_summary_bar(entries)
    print("Plots saved to:", OUT_DIR)


if __name__ == '__main__':
    main()
