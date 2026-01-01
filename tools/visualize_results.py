import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os
import pandas as pd

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed color mapping for consistent colors across all plots
METHOD_COLORS = {
    'FedAvg': '#1f77b4',      # blue
    'SR-FedAdam': '#ff7f0e',  # orange
    'FedAdam': '#2ca02c'      # green
}


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
        noise = None
        if isinstance(hp, dict):
            dataset = hp.get('dataset')
            method = hp.get('server_optimizer') or hp.get('optimizer')
            noise = hp.get('client_update_noise_std')
            # normalize method label (handle lists and legacy 'none')
            if isinstance(method, (list, tuple)) and len(method) > 0:
                method = method[0]
            if isinstance(method, str):
                m_low = method.lower()
                if m_low in ('none', 'fedavg'):
                    method = 'FedAvg'
                elif m_low in ('sr_fedadam', 'sr-fedadam'):
                    method = 'SR-FedAdam'
                elif m_low in ('fedadam', 'fed_adam'):
                    method = 'FedAdam'

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
        # attach noise info
        entries[-1]['noise'] = noise if noise is not None else 'clean'

    return entries


def plot_per_run_curves(entries):
    # group by dataset+method
    groups = {}
    for e in entries:
        key = (e['dataset'], e.get('noise', 'clean'), e['method'])
        groups.setdefault(key, []).append(e)

    for (dataset, noise, method), runs in groups.items():
        plt.figure(figsize=(6, 4))
        for r in runs:
            data = r['data']
            if 'accuracy_test' in data:
                acc = np.array(data['accuracy_test'])
                plt.plot(acc, alpha=0.6)
        plt.title(f"{dataset} ({noise}) - {method}")
        plt.xlabel("Communication Round")
        plt.ylabel("Test Accuracy")
        plt.grid(True)
        out_noise = str(noise).replace('.', 'p')
        out = OUT_DIR / f"{dataset}_{out_noise}_{method}_accuracy_curve.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def plot_combined_curves(entries):
    """Plot all methods for each dataset/noise in a single figure with mean and std."""
    # group by (dataset, noise) first, then method
    dataset_groups = {}
    for e in entries:
        dataset_groups.setdefault((e['dataset'], e.get('noise', 'clean')), {}).setdefault(e['method'], []).append(e)
    
    for (dataset, noise), method_dict in dataset_groups.items():
        plt.figure(figsize=(10, 6))
        for method, runs in method_dict.items():
            # collect all runs for this method
            all_curves = []
            for r in runs:
                data = r['data']
                if 'accuracy_test' in data:
                    acc = np.array(data['accuracy_test'])
                    all_curves.append(acc)
            
            if all_curves:
                # pad curves to same length
                max_len = max(len(c) for c in all_curves)
                padded = []
                for c in all_curves:
                    if len(c) < max_len:
                        padded.append(np.concatenate([c, [np.nan] * (max_len - len(c))]))
                    else:
                        padded.append(c)
                
                curves_array = np.array(padded)
                mean_curve = np.nanmean(curves_array, axis=0)
                std_curve = np.nanstd(curves_array, axis=0)
                
                rounds = np.arange(len(mean_curve))
                color = METHOD_COLORS.get(method, '#333333')
                
                # plot mean line
                plt.plot(rounds, mean_curve, label=method, color=color, linewidth=2)
                # plot shaded std region
                plt.fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve, 
                                color=color, alpha=0.2)
            out_noise = str(noise).replace('.', 'p')
            plt.title(f"{dataset} ({noise}) - All Methods (Mean ± Std)")
        plt.xlabel("Communication Round")
        plt.ylabel("Test Accuracy")
        plt.legend()
        plt.grid(True)
        out = OUT_DIR / f"{dataset}_{out_noise}_all_methods_combined.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"Combined plot saved: {out}")


def export_data_to_csv(entries):
    """Export accuracy curves and statistics to CSV files."""
    # group by (dataset, noise)
    dataset_groups = {}
    for e in entries:
        dataset_groups.setdefault((e['dataset'], e.get('noise', 'clean')), {}).setdefault(e['method'], []).append(e)
    
    for (dataset, noise), method_dict in dataset_groups.items():
        # === Export individual run curves ===
        all_data = {}
        max_rounds = 0
        
        for method, runs in method_dict.items():
            for run_idx, r in enumerate(runs):
                data = r['data']
                if 'accuracy_test' in data:
                    acc = np.array(data['accuracy_test'])
                    col_name = f"{method}_run{run_idx+1}" if len(runs) > 1 else method
                    all_data[col_name] = acc
                    max_rounds = max(max_rounds, len(acc))
        
        if all_data:
            # create dataframe with round numbers
            df_dict = {'round': list(range(max_rounds))}
            for col_name, values in all_data.items():
                # pad with NaN if needed
                padded = list(values) + [np.nan] * (max_rounds - len(values))
                df_dict[col_name] = padded
            
            df = pd.DataFrame(df_dict)
            noise_tag = str(noise).replace('.', 'p')
            csv_path = OUT_DIR / f"{dataset}_{noise_tag}_accuracy_curves.csv"
            df.to_csv(csv_path, index=False)
            print(f"CSV data saved: {csv_path}")
        
        # === Export statistics summary ===
        stats_rows = []
        for method, runs in method_dict.items():
            final_accs = []
            runtimes = []
            for r in runs:
                data = r['data']
                if 'accuracy_test' in data:
                    acc = np.array(data['accuracy_test'])
                    final_accs.append(float(acc[-1]))
                # collect total runtime if available
                if 'total_time' in data:
                    try:
                        runtimes.append(float(data['total_time'][ -1 ]))
                    except Exception:
                        try:
                            runtimes.append(float(data['total_time']))
                        except Exception:
                            pass
            
            if final_accs:
                row = {
                    'method': method,
                    'mean_accuracy': np.mean(final_accs),
                    'std_accuracy': np.std(final_accs),
                    'best_accuracy': np.max(final_accs),
                    'worst_accuracy': np.min(final_accs),
                    'num_runs': len(final_accs)
                }
                if runtimes:
                    row.update({'mean_runtime_s': float(np.mean(runtimes)), 'std_runtime_s': float(np.std(runtimes))})
                stats_rows.append(row)
        
        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            noise_tag = str(noise).replace('.', 'p')
            stats_path = OUT_DIR / f"{dataset}_{noise_tag}_statistics_summary.csv"
            stats_df.to_csv(stats_path, index=False, float_format='%.4f')
            print(f"Statistics summary saved: {stats_path}")


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
            rows.append({'dataset': e['dataset'], 'noise': e.get('noise', 'clean'), 'method': e['method'], 'acc_final': final})

    if not rows:
        print("[WARN] No accuracy_test arrays found in results.")
        return

    df = pd.DataFrame(rows)
    for (dataset, noise), sub in df.groupby(['dataset', 'noise']):
        agg = sub.groupby('method')['acc_final'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(8, 4))
        # use fixed colors for each method
        colors = [METHOD_COLORS.get(m, '#333333') for m in agg['method']]
        plt.bar(agg['method'], agg['mean'], yerr=agg['std'].fillna(0.0), color=colors, alpha=0.8)
        plt.title(f"{dataset} ({noise}) - Final Accuracy (Mean ± Std)")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        noise_tag = str(noise).replace('.', 'p')
        out = OUT_DIR / f"{dataset}_{noise_tag}_accuracy_barplot.png"
        plt.savefig(out)
        plt.close()

        # Also plot mean runtimes per method if available
        runtime_rows = []
        for e in entries:
            d = e['dataset']
            n = e.get('noise', 'clean')
            if d == dataset and n == noise:
                data = e['data']
                method = e['method']
                if 'total_time' in data:
                    try:
                        rt = float(data['total_time'][-1])
                    except Exception:
                        try:
                            rt = float(data['total_time'])
                        except Exception:
                            continue
                    runtime_rows.append({'method': method, 'runtime': rt})

        if runtime_rows:
            rdf = pd.DataFrame(runtime_rows)
            ragg = rdf.groupby('method')['runtime'].agg(['mean', 'std']).reset_index()
            plt.figure(figsize=(8, 4))
            colors = [METHOD_COLORS.get(m, '#333333') for m in ragg['method']]
            plt.bar(ragg['method'], ragg['mean'], yerr=ragg['std'].fillna(0.0), color=colors, alpha=0.8)
            plt.title(f"{dataset} ({noise}) - Mean Runtime (s)")
            plt.ylabel("Seconds")
            plt.xticks(rotation=45)
            plt.tight_layout()
            out_rt = OUT_DIR / f"{dataset}_{noise_tag}_runtime_barplot.png"
            plt.savefig(out_rt)
            plt.close()


def main():
    entries = load_npz_files(RESULTS_DIR)
    if not entries:
        print("No .npz results found under results/ - run experiments first or point to another directory.")
        return
    plot_per_run_curves(entries)
    plot_combined_curves(entries)
    plot_summary_bar(entries)
    export_data_to_csv(entries)
    print("All plots and CSV files saved to:", OUT_DIR)


if __name__ == '__main__':
    main()
