"""Comparison figures for the report: baseline (results_baseline/) vs
improved (results/).   Run from the repository root:  python report/make_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE, IMP, OUT = ROOT / "results_baseline", ROOT / "results", ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
GOAL = np.array([40.0, 30.0, 2500.0])
C_BASE, C_IMP = "#9a9a9a", "#0b6e8a"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(folder, task):
    sub = {1: "task1_static_ibvs/task1", 2: "task2_moving_target/task2",
           3: "task3_rrt/task3", 4: "task4_rrt_star/task4"}[task]
    return np.load(folder / f"{sub}_data.npz")


fig, axs = plt.subplots(2, 2, figsize=(8, 5.6))
axs = axs.ravel()

# Task 1: error on a log scale
ax = axs[0]
for folder, c, lab in ((BASE, C_BASE, "baseline"), (IMP, C_IMP, "adaptive gain")):
    d = load(folder, 1)
    ax.semilogy(d["t"], np.maximum(d["errors"], 1e-3), color=c, lw=1.6, label=lab)
ax.axhline(10, color="k", lw=0.7, ls=":")
ax.set(title="Task 1: feature error", xlabel="time (s)", ylabel="|s - s*| (px)")
ax.legend(frameon=False)

# Task 2: tracking error, compensated runs
ax = axs[1]
for folder, c, lab in ((BASE, C_BASE, "compensated (baseline)"),
                       (IMP, C_IMP, "+ prediction, adaptive gain")):
    d = load(folder, 2)
    ax.plot(d["compensated_t"], d["compensated_xy_errors"], color=c, lw=1.4, label=lab)
ax.set(title="Task 2: tracking error", xlabel="time (s)", ylabel="centroid error (px)", ylim=(0, 40))
ax.legend(frameon=False)

# Tasks 3-4: error to s* during execution
for ax, task in ((axs[2], 3), (axs[3], 4)):
    for folder, c, lab in ((BASE, C_BASE, "baseline"), (IMP, C_IMP, "shortcut + final IBVS")):
        d = load(folder, task)
        err = np.linalg.norm(d["executed_features"] - GOAL, axis=1)
        ax.plot(np.arange(len(err)) * 0.1, err, color=c, lw=1.4, label=lab)
    ax.set(title=f"Task {task}: error to s* while moving", xlabel="time (s)", ylabel="|s - s*| (px)")
    ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(OUT / "comparison.pdf")
print("saved", OUT / "comparison.pdf")

# Image-plane paths of Tasks 3-4, baseline vs improved
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, task in ((axs[0], 3), (axs[1], 4)):
    for folder, c, lab in ((BASE, C_BASE, "baseline"), (IMP, C_IMP, "improved")):
        d = load(folder, task)
        f = d["executed_features"]
        ax.plot(f[:, 0], f[:, 1], color=c, lw=1.6, label=lab)
    ax.plot(*load(BASE, task)["executed_features"][0, :2], "ks", ms=6, label="start")
    ax.plot(*GOAL[:2], "*", color="#2a9d3c", ms=13, label="s*")
    ax.set(xlim=(-320, 320), ylim=(240, -240), xlabel="x (px)", ylabel="y (px)",
           title=f"Task {task}: executed path in the image")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "paths.pdf")
print("saved", OUT / "paths.pdf")
