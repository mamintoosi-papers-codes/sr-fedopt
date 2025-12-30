"""
Create sweep-like summary (dataset, method, batch_size, acc_mean, acc_std)
by scanning `results/` for saved experiment npz files and produce the same
plots as a typical sweep-summary pipeline.

This adapts ideas from external_taylor/plot_summary_by_bs.py but builds the
summary CSV from existing npz experiment files in this repo.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
OUT_DIR_BS = RESULTS_DIR / "plots_by_bs"
OUT_DIR_DATASET = RESULTS_DIR / "plots_by_dataset"
OUT_DIR_BS.mkdir(parents=True, exist_ok=True)
OUT_DIR_DATASET.mkdir(parents=True, exist_ok=True)


def gather_runs(root: Path):
    files = list(root.rglob("*.npz"))
    rows = []
    for f in files:
        try:
            data = dict(np.load(f, allow_pickle=True))
        except Exception:
            continue

        hp = None
        if 'hyperparameters' in data:
            try:
                hp = data['hyperparameters'].item()
            except Exception:
                hp = data['hyperparameters']

        dataset = None
        method = None
        bs = None
        if isinstance(hp, dict):
            dataset = hp.get('dataset')
            method = hp.get('server_optimizer') or hp.get('optimizer') or hp.get('method')
            bs = hp.get('batch_size')

        # fallback parsing from filename
        if not dataset or not method or bs is None:
            # attempt to parse name tokens
            parts = f.stem.split("_")
            if not dataset and len(parts) > 0:
                dataset = parts[0]
            if not method and len(parts) > 1:
                method = parts[1]
            # find bs token
            for p in parts:
                if p.startswith('bs'):
                    try:
                        bs = int(p[2:])
                    except Exception:
                        pass

        if dataset is None:
            dataset = 'unknown'
        if method is None:
            method = 'unknown'
        try:
            accs = np.array(data.get('accuracy_test', []))
            if accs.size == 0:
                continue
            acc_final = float(accs[-1])
        except Exception:
            continue

        rows.append({'name': f.stem, 'dataset': dataset, 'method': method, 'batch_size': bs, 'acc_final': acc_final})

    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame):
    if df.empty:
        print("No runs found to summarize.")
        return

    # compute mean/std per (dataset, method, batch_size)
    grouped = df.groupby(['dataset', 'method', 'batch_size'])['acc_final'].agg(['mean', 'std']).reset_index()

    for dataset in grouped['dataset'].unique():
        sub = grouped[grouped['dataset'] == dataset]
        for bs in sorted(sub['batch_size'].unique()):
            sub_bs = sub[sub['batch_size'] == bs]
            if sub_bs.empty:
                continue
            plt.figure(figsize=(8, 5))
            plt.bar(sub_bs['method'], sub_bs['mean'], yerr=sub_bs['std'].fillna(0.0))
            plt.title(f"{dataset} - Batch Size {bs}")
            plt.ylabel("Final Accuracy")
            plt.xticks(rotation=30)
            plt.tight_layout()
            out = OUT_DIR_BS / f"{dataset}_bs{bs}_bar.png"
            plt.savefig(out)
            plt.close()

    # line plots aggregated across batch sizes
    for dataset in grouped['dataset'].unique():
        sub = grouped[grouped['dataset'] == dataset]
        methods = [m for m in sorted(sub['method'].unique()) if m != 'none']
        if not methods:
            continue
        plt.figure(figsize=(9, 6))
        for method in methods:
            meth = sub[sub['method'] == method].sort_values('batch_size')
            x = meth['batch_size'].tolist()
            y = meth['mean'].tolist()
            plt.plot(x, y, marker='o', label=method)
        plt.title(f"{dataset} - Accuracy vs Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Final Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out = OUT_DIR_DATASET / f"{dataset}_line.png"
        plt.savefig(out)
        plt.close()


def main():
    df = gather_runs(RESULTS_DIR)
    if df.empty:
        print("No experiment .npz files found under results/. Run experiments first.")
        return
    make_plots(df)
    print("Plots saved to:")
    print(" -", OUT_DIR_BS)
    print(" -", OUT_DIR_DATASET)


if __name__ == '__main__':
    main()
