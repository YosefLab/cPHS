# figures 1-5 of section 4, from the csv files of run_simulation.py (see README.md)
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import balanced_accuracy_score, precision_recall_curve, roc_auc_score, roc_curve

# columns that identify a realization
keys = ["n", "subsampling_ratio", "k", "m", "rho", "seed"]

# cutoff values eps of the hypotheses (3)
eps_fig1 = 1/4 # d_RF, figure 1
eps_rf = 1/7 # d_RF, figures 2-4
eps_pars = 0.01 # d_P, figure 2 (right)
eps_list = [0.01, 0.04, 0.07, 0.1, 0.13, 1/7, 0.16, 1/6, 0.19, 0.2, 0.22, 0.25, 0.28, 1/3, 0.5] # right panels of figures 3-4

# test statistics. sign is 1 if larger values mean a more accurate tree, and -1 otherwise
stats = {
    "cPHS": {"label": "cPHS", "color": "blue", "marker": "o", "sign": 1},
    "parsimony": {"label": "parsimony", "color": "orange", "marker": "*", "sign": -1},
    "parsimony_pvalue": {"label": "parsimony tail-prob", "color": "teal", "marker": "X", "sign": 1},
    "likelihood": {"label": "likelihood", "color": "brown", "marker": "s", "sign": 1},
}

grid = np.linspace(0, 1, 201) # x-axis of the ROC and precision-recall curves

plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.color": "#b0b0b0",
                     "lines.linewidth": 3, "lines.markersize": 10})


def load_results(csv_files, cphs_column):
    # csv files (or globs) of run_simulation.py, e.g. of several jobs with the same setting
    file_names = []
    for pattern in csv_files:
        file_names += glob.glob(pattern)
    file_names = sorted(set(file_names))
    if len(file_names) == 0:
        raise SystemExit("no files match {}".format(csv_files))

    df = pd.concat([pd.read_csv(file_name) for file_name in file_names], ignore_index=True)
    df = df.drop_duplicates(keys + ["tree"])
    df["cPHS"] = df[cphs_column]

    # only k may vary
    for col in ["n", "subsampling_ratio", "m", "rho"]:
        if df[col].nunique() > 1:
            raise SystemExit("several values of {} in {}".format(col, file_names))

    print("loaded {} file(s), realizations per k: {}".format(len(file_names), df.groupby("k").seed.nunique().to_dict()))
    return df


def get_reconstructed(df):
    # the reconstructed trees, with their RF and parsimony distances to the GT tree of the same realization
    df_gt = df[df.tree == "gt"].set_index(keys)[["parsimony"]].rename(columns={"parsimony": "parsimony_gt"})
    df_rec = df[df.tree != "gt"].join(df_gt, on=keys)
    df_rec["d_rf"] = df_rec["rf"]
    df_rec["d_pars"] = np.maximum(0, df_rec["parsimony"] / df_rec["parsimony_gt"] - 1)
    return df_rec


def get_labels_and_scores(df_rec, stat, distance, eps):
    # labels: 1 for accurate trees (H0: distance <= eps) and 0 for inaccurate trees (H1)
    # scores: the test statistic, with the sign such that larger scores mean more accurate trees
    labels = (df_rec[distance] <= eps).to_numpy(int)
    scores = stats[stat]["sign"] * df_rec[stat].to_numpy(float)
    return labels, scores


def bootstrap(func, labels, scores, n_boot):
    # func(labels, scores) and its 95% confidence interval, by resampling the trees.
    # it is undefined (nan) if all the trees are accurate, or all are inaccurate
    n_trees = len(labels)
    if labels.sum() == 0 or labels.sum() == n_trees:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(0)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_trees, n_trees)
        if 0 < labels[idx].sum() < n_trees:
            values.append(func(labels[idx], scores[idx]))
    ci_low, ci_high = np.percentile(values, [2.5, 97.5], axis=0)
    return func(labels, scores), ci_low, ci_high


def best_balanced_accuracy(labels, scores):
    # balanced accuracy with the optimal threshold
    fpr, tpr, _ = roc_curve(labels, scores)
    return np.max((tpr + 1 - fpr) / 2)


def roc_on_grid(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return np.interp(grid, fpr, tpr)


def precision_recall_on_grid(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return np.interp(grid, recall[::-1], precision[::-1])


def plot_with_ci(ax, x, values, ci_low, ci_high, stat, marker=None, linestyle="-"):
    ax.plot(x, values, color=stats[stat]["color"], marker=marker, linestyle=linestyle, label=stats[stat]["label"])
    ax.fill_between(x, ci_low, ci_high, color=stats[stat]["color"], alpha=0.4)


def save_figure(fig, outdir, name, table=None):
    # saves the figure, and the plotted numbers if given
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=300)
    plt.close(fig)
    if table is not None:
        table.to_csv(os.path.join(outdir, name + "_data.csv"), index=False)
    print("saved {}".format(name))


def transform_for_kde(stat, values):
    # the transformations of figure 1 in the paper
    if stat == "parsimony_pvalue":
        return np.log(-np.log(np.clip(values, 1e-300, 1 - 1e-16)))
    elif stat == "likelihood":
        return -values
    elif stat == "cPHS":
        return 1 - np.log(values)
    return values


def plot_fig1(df, args):
    # KDEs of the test statistics for the GT trees and for the accurate (H0) and inaccurate (H1)
    # reconstructed trees, with d = d_RF and eps = 1/4
    df = df[df.k == args.k_ref]
    df_rec = get_reconstructed(df)
    groups = [df[df.tree == "gt"], df_rec[df_rec.d_rf <= eps_fig1], df_rec[df_rec.d_rf > eps_fig1]]
    group_labels = ["ground-truth trees", r"accurate reconstructions ($\mathcal{H}_0$)",
                    r"inaccurate reconstructions ($\mathcal{H}_1$)"]
    group_colors = ["blue", "olive", "magenta"]
    group_linestyles = ["--", "-", "-"]
    titles = [r"(a) parsimony, $M(\mathcal{T}^S)$",
              r"(b) parsimony tail probability, $\log(-\log p(\mathcal{T}^S))$",
              r"(c) negative log-likelihood, $-\log L(\mathcal{T}^S)$",
              r"(d) PHS test statistic, $1-\log(\mathrm{cPHS}(\mathcal{T}^S))$"]

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax, stat, title in zip(axs.flat, ["parsimony", "parsimony_pvalue", "likelihood", "cPHS"], titles):
        values = [transform_for_kde(stat, df_group[stat].to_numpy(float)) for df_group in groups]
        all_values = np.concatenate(values)
        margin = 0.1 * (all_values.max() - all_values.min() + 1e-12)
        x = np.linspace(all_values.min() - margin, all_values.max() + margin, 400)
        for i in range(len(groups)):
            if len(values[i]) > 1 and np.std(values[i]) > 0: # otherwise there is no KDE
                ax.plot(x, gaussian_kde(values[i])(x), color=group_colors[i], linestyle=group_linestyles[i],
                        lw=4, label=group_labels[i])
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("KDE")
    axs[1, 1].legend(fontsize=10)
    save_figure(fig, args.outdir, "fig1")


def plot_fig2(df, args):
    # ROC (top) and precision-recall (bottom) curves, with d = d_RF and eps = 1/7 (left),
    # and with d = d_P and eps = 0.01 (right)
    df_rec = get_reconstructed(df[df.k == args.k_ref])
    distances = [("d_rf", eps_rf, r"RF distance, $\epsilon=1/7$"),
                 ("d_pars", eps_pars, r"parsimony distance, $\epsilon=0.01$")]
    curves = [(roc_on_grid, "ROC", "false positive rate", "true positive rate"),
              (precision_recall_on_grid, "Precision-Recall", "recall", "precision")]

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for row in range(2):
        curve_func, curve_name, xlabel, ylabel = curves[row]
        for col in range(2):
            distance, eps, distance_name = distances[col]
            ax = axs[row, col]
            for stat in stats:
                labels, scores = get_labels_and_scores(df_rec, stat, distance, eps)
                curve, ci_low, ci_high = bootstrap(curve_func, labels, scores, args.n_boot)
                if np.ndim(curve) > 0: # the curve is nan if all trees are accurate, or all are inaccurate
                    plot_with_ci(ax, grid, curve, ci_low, ci_high, stat)

            # random guessing
            if row == 0:
                ax.plot(grid, grid, "k--", lw=2)
            else:
                ax.plot(grid, np.full_like(grid, labels.mean()), "k--", lw=2)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title("({}) {} curve, {}".format("abcd"[2*row + col], curve_name, distance_name))
    axs[0, 0].legend(fontsize=10)
    axs[0, 1].legend(fontsize=10)
    save_figure(fig, args.outdir, "fig2")


def get_sweep_points(df, k_ref):
    # the points of the left panels of figures 3-4 (varying k, with eps = 1/7) and of the right
    # panels (k = k_ref, varying eps), as tuples (x, reconstructed trees, eps)
    points_k = []
    for k in sorted(df.k.unique()):
        points_k.append((k, get_reconstructed(df[df.k == k]), eps_rf))
    df_rec = get_reconstructed(df[df.k == k_ref])
    points_eps = []
    for eps in eps_list:
        points_eps.append((eps, df_rec, eps))
    return points_k, points_eps


def sweep(func, points, n_boot):
    # func (AUC or balanced accuracy) of each test statistic at each point, with d = d_RF
    table = []
    for x, df_rec, eps in points:
        for stat in stats:
            labels, scores = get_labels_and_scores(df_rec, stat, "d_rf", eps)
            value, ci_low, ci_high = bootstrap(func, labels, scores, n_boot)
            table.append({"x": x, "stat": stat, "value": value, "ci_low": ci_low, "ci_high": ci_high,
                          "accurate_fraction": labels.mean()})
    return pd.DataFrame(table)


def plot_sweep(ax, table, linestyle="-"):
    for stat in stats:
        t = table[table.stat == stat]
        plot_with_ci(ax, t.x, t.value, t.ci_low, t.ci_high, stat, marker=stats[stat]["marker"], linestyle=linestyle)


def plot_fig3(df, args):
    # AUC as a function of k (left, eps = 1/7) and of eps (right, k = k_ref)
    points_k, points_eps = get_sweep_points(df, args.k_ref)
    table_k = sweep(roc_auc_score, points_k, args.n_boot)
    table_eps = sweep(roc_auc_score, points_eps, args.n_boot)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_sweep(axs[0], table_k)
    plot_sweep(axs[1], table_eps)
    axs[0].set_xlabel("sequence length $k$")
    axs[0].set_ylabel("area under the curve (AUC)")
    axs[0].set_title(r"$\epsilon=1/7$")
    axs[1].set_xlabel(r"cutoff value $\epsilon$")
    axs[1].set_ylabel("area under the curve (AUC)")
    axs[1].set_title("$k={}$".format(args.k_ref))
    axs[0].legend(fontsize=10)
    save_figure(fig, args.outdir, "fig3", pd.concat([table_k.assign(panel="vs_k"), table_eps.assign(panel="vs_eps")]))


def plot_fig4(df, args):
    # balanced accuracy as a function of k (left, eps = 1/7) and of eps (right, k = k_ref), with the
    # optimal threshold of each test statistic. on the left also cPHS with the fixed threshold t
    points_k, points_eps = get_sweep_points(df, args.k_ref)
    table_k = sweep(best_balanced_accuracy, points_k, args.n_boot)
    table_eps = sweep(best_balanced_accuracy, points_eps, args.n_boot)

    # cPHS with the fixed threshold: a tree is accepted as accurate iff cPHS >= t
    def balanced_accuracy_fixed_threshold(labels, scores):
        return balanced_accuracy_score(labels, (scores >= args.cphs_threshold).astype(int))

    table_fixed = []
    for k, df_rec, eps in points_k:
        labels, scores = get_labels_and_scores(df_rec, "cPHS", "d_rf", eps)
        value, ci_low, ci_high = bootstrap(balanced_accuracy_fixed_threshold, labels, scores, args.n_boot)
        table_fixed.append({"x": k, "stat": "cPHS_t={:g}".format(args.cphs_threshold), "value": value,
                            "ci_low": ci_low, "ci_high": ci_high})
    table_fixed = pd.DataFrame(table_fixed)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    axs[0].plot(table_fixed.x, table_fixed.value, color="green", marker="P", label="cPHS (fixed threshold)")
    axs[0].fill_between(table_fixed.x, table_fixed.ci_low, table_fixed.ci_high, color="green", alpha=0.4)
    plot_sweep(axs[0], table_k, linestyle="--")
    plot_sweep(axs[1], table_eps, linestyle="--")
    axs[0].set_xlabel("sequence length $k$")
    axs[0].set_ylabel("balanced accuracy")
    axs[0].set_title(r"$\epsilon=1/7$, fixed threshold $t={:g}$".format(args.cphs_threshold))
    axs[1].set_xlabel(r"cutoff value $\epsilon$")
    axs[1].set_ylabel("balanced accuracy")
    axs[1].set_title("$k={}$".format(args.k_ref))
    axs[0].legend(fontsize=10)
    table = pd.concat([table_fixed.assign(panel="vs_k"), table_k.assign(panel="vs_k"), table_eps.assign(panel="vs_eps")])
    save_figure(fig, args.outdir, "fig4", table)


def gt_is_best(df):
    # for each realization and test statistic: does the GT tree get the best score, i.e. no reconstructed
    # tree scores strictly better? (ties count as the GT tree scoring best)
    table = []
    for _, df_realization in df.groupby(keys):
        df_gt = df_realization[df_realization.tree == "gt"]
        df_rec = df_realization[df_realization.tree != "gt"]
        if len(df_gt) != 1 or len(df_rec) == 0:
            continue
        for stat in stats:
            sign = stats[stat]["sign"]
            best = sign * df_gt[stat].iloc[0] >= (sign * df_rec[stat]).max()
            table.append({"k": df_gt.k.iloc[0], "stat": stat, "best": best})

    # fraction of the realizations, with a 95% confidence interval (normal approximation)
    table = pd.DataFrame(table).groupby(["k", "stat"]).best.agg(["mean", "count"]).reset_index()
    table["ci"] = 1.96 * np.sqrt(table["mean"] * (1 - table["mean"]) / table["count"])
    return table


def plot_fig5(dfs, args):
    # fraction of the realizations in which the GT tree gets the best score, as a function of k.
    # dfs is a list of (panel name, results), one for each panel
    fig, axs = plt.subplots(1, len(dfs), figsize=(6.5 * len(dfs), 4.8), squeeze=False)
    tables = []
    for i in range(len(dfs)):
        panel, df = dfs[i]
        ax = axs[0, i]
        table = gt_is_best(df)
        for stat in stats:
            t = table[table.stat == stat]
            yerr = [t["mean"] - np.maximum(0, t["mean"] - t["ci"]), np.minimum(1, t["mean"] + t["ci"]) - t["mean"]] # clipped to [0, 1]
            ax.errorbar(t.k, t["mean"], yerr=yerr, color=stats[stat]["color"], marker=stats[stat]["marker"],
                        capsize=8, label=stats[stat]["label"])
        ax.set_xlabel("sequence length $k$")
        ax.set_ylabel("percentage of GT-trees\ngetting the best score")
        ax.set_title("$n={}$, $m={}$".format(df.n.iloc[0], df.m.iloc[0]))
        tables.append(table.assign(panel=panel))
    axs[0, 0].legend(fontsize=10)
    save_figure(fig, args.outdir, "fig5", pd.concat(tables))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="csv file(s) of run_simulation.py, or globs")
    parser.add_argument("--csv-large", nargs="+", help="csv file(s) for the right panel of figure 5")
    parser.add_argument("--k-ref", type=int, default=30, help="k of figures 1-2 and of the right panels of figures 3-4")
    parser.add_argument("--cphs-threshold", type=float, default=1e-3, help="fixed cPHS threshold of figure 4")
    parser.add_argument("--cphs-column", default="cPHS", choices=["cPHS", "cPHS_A3"],
                        help="cPHS of Eq. (13), or its variant of Appendix A.3")
    parser.add_argument("--n-boot", type=int, default=200, help="bootstrap samples for the confidence intervals")
    parser.add_argument("--figs", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--outdir", default="output/figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_results(args.csv, args.cphs_column)
    if args.k_ref not in df.k.values and any(fig in args.figs for fig in [1, 2, 3, 4]):
        raise SystemExit("--k-ref {} is not among the simulated values of k".format(args.k_ref))

    if 1 in args.figs:
        plot_fig1(df, args)
    if 2 in args.figs:
        plot_fig2(df, args)
    if 3 in args.figs:
        plot_fig3(df, args)
    if 4 in args.figs:
        plot_fig4(df, args)
    if 5 in args.figs:
        dfs = [("main", df)]
        if args.csv_large:
            dfs.append(("large", load_results(args.csv_large, args.cphs_column)))
        plot_fig5(dfs, args)


if __name__ == "__main__":
    main()
