#!/usr/bin/env python3
"""
Analyze threshold calibration results.
Merges all grid cell CSVs and determines optimal cPHS thresholds.

Usage:
    python analyze_thresholds.py --results_dir results/ --output threshold_analysis.csv
"""

import argparse
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def analyze_grid_cell(df, n, k):
    """Analyze one grid cell: find threshold that best separates accurate from inaccurate."""
    
    results = []
    
    for epsilon in [0.01, 0.1, 0.25, 0.5, 2/3]:
        # Label trees as accurate/inaccurate based on RF distance
        df_valid = df.dropna(subset=['cPHS', 'rf_norm'])
        df_valid = df_valid.copy()
        df_valid['accurate'] = df_valid['rf_norm'] <= epsilon
        
        n_accurate = df_valid['accurate'].sum()
        n_inaccurate = (~df_valid['accurate']).sum()
        
        if n_accurate < 5 or n_inaccurate < 5:
            continue
        
        # Find optimal threshold by maximizing balanced accuracy
        cphs_vals = np.sort(df_valid['cPHS'].unique())
        best_t, best_ba = None, 0
        
        for t in cphs_vals:
            # Accept if cPHS >= t (tree passes)
            predicted_accurate = df_valid['cPHS'] >= t
            
            tp = (predicted_accurate & df_valid['accurate']).sum()
            tn = (~predicted_accurate & ~df_valid['accurate']).sum()
            fp = (predicted_accurate & ~df_valid['accurate']).sum()
            fn = (~predicted_accurate & df_valid['accurate']).sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ba = (sensitivity + specificity) / 2
            
            if ba > best_ba:
                best_ba = ba
                best_t = t
        
        # Also check performance at standard thresholds
        for fixed_t in [0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]:
            predicted = df_valid['cPHS'] >= fixed_t
            tp = (predicted & df_valid['accurate']).sum()
            tn = (~predicted & ~df_valid['accurate']).sum()
            fp = (predicted & ~df_valid['accurate']).sum()
            fn = (~predicted & df_valid['accurate']).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ba_fixed = (sens + spec) / 2
            
            results.append({
                'n': n, 'k': k, 'epsilon': epsilon,
                'threshold_type': f'fixed_{fixed_t}',
                'threshold': fixed_t,
                'balanced_accuracy': ba_fixed,
                'sensitivity': sens, 'specificity': spec,
                'n_accurate': n_accurate, 'n_inaccurate': n_inaccurate,
            })
        
        results.append({
            'n': n, 'k': k, 'epsilon': epsilon,
            'threshold_type': 'optimal',
            'threshold': best_t,
            'balanced_accuracy': best_ba,
            'sensitivity': np.nan, 'specificity': np.nan,
            'n_accurate': n_accurate, 'n_inaccurate': n_inaccurate,
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--output', type=str, default='threshold_analysis.csv')
    args = parser.parse_args()
    
    # Load all result files
    files = sorted(glob.glob(f"{args.results_dir}/threshold_n*_k[0-9]*.csv"))
    files = [f for f in files if "rho" not in f and "quick" not in f]
    print(f"Found {len(files)} result files")
    
    all_analysis = []
    
    for f in files:
        df = pd.read_csv(f)
        n = df['n'].iloc[0]
        k = df['k'].iloc[0]
        n_reps = df['rep'].nunique()
        n_rows = len(df)
        
        print(f"\n  {f}: n={n}, k={k}, {n_reps} reps, {n_rows} rows")
        print(f"    RF distance: mean={df.rf_norm.mean():.3f}, "
              f"median={df.rf_norm.median():.3f}")
        print(f"    cPHS: median={df.cPHS.median():.2e}, "
              f"mean log10={np.log10(np.clip(df.cPHS.dropna(), 1e-16, None)).mean():.1f}")
        
        results = analyze_grid_cell(df, n, k)
        all_analysis.extend(results)
    
    analysis_df = pd.DataFrame(all_analysis)
    analysis_df.to_csv(args.output, index=False)
    print(f"\n=== Analysis saved to {args.output} ===")
    
    # Print summary
    print(f"\n=== THRESHOLD SUMMARY ===")
    print(f"{'n':>6s} {'k':>4s} {'ε':>6s} {'Optimal t':>12s} {'BA(opt)':>8s} "
          f"{'BA(0.05)':>9s} {'BA(1e-3)':>9s} {'BA(1e-5)':>9s}")
    print("-" * 75)
    
    for (n, k, eps), grp in analysis_df.groupby(['n', 'k', 'epsilon']):
        opt = grp[grp.threshold_type == 'optimal']
        def getba(tt):
            sub = grp[grp.threshold_type == tt]
            return f"{sub.balanced_accuracy.values[0]:.3f}" if len(sub) else "N/A"
        
        opt_t = f"{opt.threshold.values[0]:.2e}" if len(opt) else "N/A"
        opt_ba = f"{opt.balanced_accuracy.values[0]:.3f}" if len(opt) else "N/A"
        
        print(f"{n:6d} {k:4d} {eps:6.2f} {opt_t:>12s} {opt_ba:>8s} "
              f"{getba('fixed_0.05'):>9s} {getba('fixed_0.01'):>9s} {getba('fixed_0.001'):>9s} {getba('fixed_0.0001'):>9s} {getba('fixed_1e-05'):>9s} {getba('fixed_1e-06'):>9s}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, eps in zip(axes, [0.1, 0.25, 0.5]):
        sub = analysis_df[(analysis_df.epsilon == eps) & 
                          (analysis_df.threshold_type == 'optimal')]
        for k_val in sorted(sub.k.unique()):
            ksub = sub[sub.k == k_val]
            ax.plot(ksub.n, ksub.threshold, 'o-', label=f'k={k_val}')
        
        ax.set_xlabel('n (leaves)')
        ax.set_ylabel('Optimal threshold')
        ax.set_yscale('log')
        ax.set_title(f'ε = {eps}')
        ax.legend()
    
    plt.suptitle('Optimal cPHS threshold by simulation parameters', fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output.replace('.csv', '.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {args.output.replace('.csv', '.png')}")


if __name__ == '__main__':
    main()
