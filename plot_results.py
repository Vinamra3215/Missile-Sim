"""
Plot training curves (mean +/- std over seeds) and the evaluation table.

    python plot_results.py                      # reads results/*/progress.csv and results/eval.json
    python plot_results.py --out plots

Runs are grouped by name prefix: results/ppo_s1, ppo_s2, ... -> "PPO".
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"PPO": "#1f9bd1", "DQN": "#e8743b", "STEER": "#7a7a7a", "RANDOM": "#bbbbbb"}


def load_runs(results_dir):
    groups = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "progress.csv"))):
        name = os.path.basename(os.path.dirname(path))
        algo = re.sub(r"_s\d+$", "", name).upper()
        df = pd.read_csv(path)
        if len(df) > 1:
            groups.setdefault(algo, []).append(df)
    return groups


def band(ax, runs, col, label, color, smooth=5):
    grid = np.linspace(0, min(r["global_step"].max() for r in runs), 300)
    ys = []
    for r in runs:
        y = r[col].astype(float).interpolate().bfill().rolling(smooth, min_periods=1).mean()
        ys.append(np.interp(grid, r["global_step"], y))
    ys = np.array(ys)
    m, s = ys.mean(0), ys.std(0)
    ax.plot(grid / 1e6, m, color=color, label=f"{label} ({len(runs)} seed{'s' if len(runs) > 1 else ''})", lw=2)
    if len(runs) > 1:
        ax.fill_between(grid / 1e6, m - s, m + s, color=color, alpha=0.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="plots")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    groups = load_runs(args.results)
    if not groups:
        print("no runs found in", args.results)

    panels = [("success", "Success rate (all maps)"), ("level", "Curriculum level reached"),
              ("return", "Episode return"), ("crash", "Crash rate")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.flat, panels):
        for algo, runs in groups.items():
            band(ax, runs, col, algo, COLORS.get(algo, None), smooth=1 if col == "level" else 5)
        ax.set_title(title)
        ax.set_xlabel("environment steps (millions)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("MissileSim 3D - training curves (mean ± std over seeds)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "training_curves.png"), dpi=150)
    print("saved", os.path.join(args.out, "training_curves.png"))

    # wall-clock view
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, runs in groups.items():
        for i, r in enumerate(runs):
            ax.plot(r["time_s"] / 60, r["level"] + r["success_top"].fillna(0) * 0.1,
                    color=COLORS.get(algo), alpha=0.8, label=algo if i == 0 else None)
    ax.set_xlabel("wall-clock minutes")
    ax.set_ylabel("curriculum level (+0.1 x success at level)")
    ax.set_title("Progress vs wall-clock time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "progress_vs_time.png"), dpi=150)

    # evaluation bar chart
    ev_path = os.path.join(args.results, "eval.json")
    if os.path.exists(ev_path):
        ev = json.load(open(ev_path))
        levels = [str(l) for l in ev["levels"]]
        res = ev["results"]
        names = list(res)
        x = np.arange(len(levels))
        w = 0.8 / max(len(names), 1)
        fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(levels) * max(1, len(names) / 3), 4.5))
        for i, n in enumerate(names):
            vals = [res[n][l]["success"] * 100 for l in levels]
            key = os.path.basename(n).split("_")[0].upper()
            bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=os.path.basename(n),
                          color=COLORS.get(key), alpha=0.6 + 0.4 * (i % 2))
            ax.bar_label(bars, fmt="%.0f", fontsize=8)
        ax.set_xticks(x, [f"level {l}" for l in levels])
        ax.set_ylabel("success rate on held-out maps (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Evaluation: {ev['episodes']} unseen maps per level")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "evaluation.png"), dpi=150)
        print("saved", os.path.join(args.out, "evaluation.png"))


if __name__ == "__main__":
    main()
