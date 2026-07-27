"""
plot_training.py
================
Lee los CSVs de entrenamiento en src/results/ y genera una imagen
con las curvas de reward y de aprendizaje (explained variance).

Salida: src/results/training_curves.png
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # sin display; funciona en servidor/pipeline
import matplotlib.pyplot as plt

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_SRC_DIR, "results")


def _load(name: str):
    path = os.path.join(_RESULTS_DIR, f"logs_{name}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] No se encontró {path}, se omite.")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def plot_training():
    data = {"RL": _load("rl"), "MPC+RL": _load("mpcrl")}
    available = {k: v for k, v in data.items() if v is not None}

    if not available:
        print("No se encontraron logs de entrenamiento en results/. Saltando plots.")
        return

    n_cols = len(available)
    fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 10), squeeze=False)
    fig.suptitle("Curvas de entrenamiento", fontsize=14, fontweight="bold")

    COLORS_REW = {"RL": "steelblue",    "MPC+RL": "darkorange"}
    COLORS_EV  = {"RL": "mediumseagreen", "MPC+RL": "tomato"}

    X     = "time/total_timesteps"
    REW   = "rollout/ep_rew_mean"
    EV    = "train/explained_variance"

    for col, (name, df) in enumerate(available.items()):

        # ── Fila 0: Reward ──────────────────────────────────────────────────
        ax = axes[0, col]
        if X in df.columns and REW in df.columns:
            s = df[[X, REW]].dropna()
            ax.plot(s[X], s[REW], color=COLORS_REW[name], linewidth=1.5)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(f"{name} — Reward medio por episodio")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Reward medio (diferencial)")
        ax.grid(True, alpha=0.3)

        # ── Fila 1: Explained Variance (proxy del aprendizaje) ──────────────
        ax = axes[1, col]
        if X in df.columns and EV in df.columns:
            s = df[[X, EV]].dropna()
            ax.plot(s[X], s[EV], color=COLORS_EV[name], linewidth=1.5)
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, label="máximo teórico")
            ax.set_ylim(-0.15, 1.1)
        ax.set_title(f"{name} — Explained Variance (convergencia)")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Explained Variance [0 → 1]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_RESULTS_DIR, "training_curves.png")
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("Gráficas guardadas en: src/results/training_curves.png")
    plt.close()


if __name__ == "__main__":
    plot_training()
