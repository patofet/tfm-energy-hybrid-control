"""
plot_learning_evolution.py
==========================
Genera la corba d'evolució de l'aprenentatge de l'agent PPO amb banda
d'interval de confiança del 95% (IC 95%), a partir dels logs CSV
generats per train_multiseed_rl.py.

Equivalent visual de les Figures 34 i 37 del document de referència taes.pdf
(corbes d'aprenentatge GPOMDP), però per al nostre agent PPO sobre la BESS.

Sortida:
    results/learning_evolution_rl.png   — figura per al LaTeX
    results/learning_evolution_rl.pdf   — versió vectorial (opcional)

Ús:
    cd src
    python plot_learning_evolution.py
    python plot_learning_evolution.py --smooth 15 --no-pdf
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SRC_DIR, "results")

# 1 episodi = 1 dia = 1440 passos (granularitat 1 min)
STEPS_PER_EPISODE = 1440


def load_seed_logs(pattern):
    """Carrega tots els CSVs que coincideixen amb el patró i retorna llista de DataFrames."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No s'han trobat fitxers amb el patró: {pattern}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Normalitzem noms de columnes per si difereixen lleugerament
        df.columns = [c.strip() for c in df.columns]
        if "time/total_timesteps" not in df.columns or "rollout/ep_rew_mean" not in df.columns:
            print(f"  Avís: {os.path.basename(f)} no té les columnes esperades, saltant.")
            continue
        df = df[["time/total_timesteps", "rollout/ep_rew_mean"]].dropna()
        dfs.append(df)
    print(f"  {len(dfs)} fitxers de seed carregats.")
    return dfs


def interpolate_to_common_axis(dfs, n_points=300):
    """
    Interpola totes les runs a un eix comú d'episodis d'entrenament.
    Retorna (x_common, matrix) on matrix.shape = (n_runs, n_points).
    """
    # Convertim timesteps → episodis
    all_x = [df["time/total_timesteps"].values / STEPS_PER_EPISODE for df in dfs]
    all_y = [df["rollout/ep_rew_mean"].values for df in dfs]

    x_min = max(x[0]  for x in all_x)
    x_max = min(x[-1] for x in all_x)
    x_common = np.linspace(x_min, x_max, n_points)

    matrix = np.array([
        np.interp(x_common, x, y) for x, y in zip(all_x, all_y)
    ])
    return x_common, matrix


def rolling_smooth(arr, window):
    """Suavitzat per finestra mòbil (edge-aware)."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def plot_learning_evolution(
    x, mean, ci_low, ci_high, n_runs, out_path, smooth_window=10, save_pdf=True
):
    fig, ax = plt.subplots(figsize=(7, 4))

    # Suavitzem per visualització (la IC ja reflecteix variabilitat real)
    mean_s   = rolling_smooth(mean,    smooth_window)
    ci_low_s = rolling_smooth(ci_low,  smooth_window)
    ci_hi_s  = rolling_smooth(ci_high, smooth_window)

    ax.fill_between(x, ci_low_s, ci_hi_s,
                    alpha=0.25, color="steelblue", label="IC 95%")
    ax.plot(x, mean_s, color="steelblue", linewidth=1.8,
            label=f"Mitjana ({n_runs} runs)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("Episodis d'entrenament (dies)", fontsize=11)
    ax.set_ylabel("Recompensa mitjana per episodi", fontsize=11)
    ax.set_title(
        f"Evolució de l'aprenentatge — Agent PPO\n"
        f"(Mitjana de {n_runs} execucions independents)",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Figura guardada: {out_path}")

    if save_pdf:
        pdf_path = out_path.replace(".png", ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  Figura guardada: {pdf_path}")

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", type=int, default=12,
                        help="Finestra de suavitzat per a la visualització (default: 12)")
    parser.add_argument("--n-points", type=int, default=300,
                        help="Punts en l'eix comú d'interpolació (default: 300)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="No genera versió PDF")
    parser.add_argument("--pattern", type=str,
                        default=None,
                        help="Patró glob per als CSVs (default: results/logs_rl_seed_*.csv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pattern = args.pattern or os.path.join(RESULTS_DIR, "logs_rl_seed_*.csv")
    print(f"Buscant logs: {pattern}")
    dfs = load_seed_logs(pattern)

    if len(dfs) < 2:
        print("ERROR: Es necessiten almenys 2 runs per calcular l'IC 95%.")
        sys.exit(1)

    print(f"Interpolant {len(dfs)} runs a {args.n_points} punts comuns...")
    x_common, matrix = interpolate_to_common_axis(dfs, n_points=args.n_points)

    n = matrix.shape[0]
    mean    = matrix.mean(axis=0)
    std     = matrix.std(axis=0, ddof=1)
    # IC 95%: mean ± 1.96 * std / sqrt(n)
    margin  = 1.96 * std / np.sqrt(n)
    ci_low  = mean - margin
    ci_high = mean + margin

    out_path = os.path.join(RESULTS_DIR, "learning_evolution_rl.png")
    print(f"Generant figura amb IC 95% (n={n}, smooth={args.smooth})...")
    plot_learning_evolution(
        x_common, mean, ci_low, ci_high,
        n_runs=n,
        out_path=out_path,
        smooth_window=args.smooth,
        save_pdf=not args.no_pdf,
    )

    # Estadístiques finals
    print(f"\n  Episodis d'entrenament: {x_common[0]:.0f} – {x_common[-1]:.0f}")
    print(f"  Recompensa final: {mean[-1]:.3f} ± {margin[-1]:.3f} (IC 95%)")
    print(f"  Millor recompensa: {mean.max():.3f}")
    print("\nFet!")
