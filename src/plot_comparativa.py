"""
Genera gràfiques de comparativa de controladors BESS.
  - Per cada controlador: SoC / Potència bateria / Generació+Demanda
  - Per cada grup de controladors: comparativa
  - Gràfica "millors" amb els controladors destacats

Modes:
  --mode day   → mostra un dia concret (--day-idx N)
  --mode avg   → mitjana de tots els dies (per defecte)

Flags addicionals:
  --interactive → genera HTMLs interactius amb Plotly (a més dels PNGs)
                  Pots clicar a la llegenda per mostrar/ocultar cada controlador

Ús:
    python plot_comparativa.py [--mode avg|day] [--interactive] [--skip-individual]
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_ROOT_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from params import P_MAX_KW, DT_H

MINS_IN_DAY  = 1440
RESULTS_CSV  = os.path.join(_SRC_DIR, "results", "resultados_comparativa.csv")
PLOTS_DIR    = os.path.join(_SRC_DIR, "results", "plots")

# ── Etiquetes ─────────────────────────────────────────────────────────────────
LABELS = {
    "mpc":           "MPC(H=1)",
    "mpch":          "MPC(1h)",
    "rl":            "RL",
    "mpc_30m":       "MPC(30m)",
    "mpc_90m":       "MPC(90m)",
    "mpc_2h":        "MPC(2h)",
    "mpc_3h":        "MPC(3h)",
    "mpc_4h":        "MPC(4h)",
    "mpc_6h":        "MPC(6h)",
    "mpc_12h":       "MPC(12h)",
    "oracle":        "MPC(1d)",
    "mph_30m":       "MPC/h(30m)",
    "mph_1h":        "MPC/h(1h)",
    "mph_90m":       "MPC/h(90m)",
    "mph_2h":        "MPC/h(2h)",
    "mph_3h":        "MPC/h(3h)",
    "mph_4h":        "MPC/h(4h)",
    "mph_6h":        "MPC/h(6h)",
    "mph_12h":       "MPC/h(12h)",
    "mph_1d":        "MPC/h(1d)",
    "mpcrl_30min":   "MPC+RL(30m)",
    "mpcrl_h":       "MPC+RL(1h)",
    "mpcrl_90min":   "MPC+RL(90m)",
    "mpcrl_2h":      "MPC+RL(2h)",
    "mpcrl_3h":      "MPC+RL(3h)",
    "mpcrl_4h":      "MPC+RL(4h)",
    "mpcrl_6h":      "MPC+RL(6h)",
    "mpcrl_12h":     "MPC+RL(12h)",
    "mpcrl_d":       "MPC+RL(D)",
    "mpcrl_hm_h":    "MPC+RL/h(1h)",
    "mpcrl_hm_90min":"MPC+RL/h(90m)",
    "mpcrl_hm_2h":   "MPC+RL/h(2h)",
    "mpcrl_hm_3h":   "MPC+RL/h(3h)",
    "mpcrl_hm_4h":   "MPC+RL/h(4h)",
    "mpcrl_hm_6h":   "MPC+RL/h(6h)",
    "mpcrl_hm_12h":  "MPC+RL/h(12h)",
    "mpcrl_hm_d":    "MPC+RL/h(D)",
}

# Àlies de nom de fitxer per a claus internes que no coincideixen amb la convenció
FILE_ALIAS = {
    "oracle": "mpc_1d",
    "mpch":   "mpc_1h",
}

# ── Grups ─────────────────────────────────────────────────────────────────────
GROUPS = {
    "01_baselines": {
        "title":  "Baselines",
        "keys":   ["mpc", "mpch", "rl"],
        "colors": ["#888888", "#2171b5", "#e6550d"],
    },
    "02_mpc_min": {
        "title": "MPC sweep · solve/minut",
        "keys":  ["mpc_30m", "mpch", "mpc_90m", "mpc_2h", "mpc_3h", "mpc_4h", "mpc_6h", "mpc_12h", "oracle"],
        "cmap":  "viridis",
    },
    "03_mpc_h": {
        "title": "MPC sweep · solve/hora",
        "keys":  ["mph_30m", "mph_1h", "mph_90m", "mph_2h", "mph_3h", "mph_4h", "mph_6h", "mph_12h", "mph_1d"],
        "cmap":  "plasma",
    },
    "04_mpcrl_min": {
        "title": "MPC+RL sweep · tracker/minut",
        "keys":  ["mpcrl_30min", "mpcrl_h", "mpcrl_90min", "mpcrl_2h", "mpcrl_3h", "mpcrl_4h", "mpcrl_6h", "mpcrl_12h", "mpcrl_d"],
        "cmap":  "cool",
    },
    "05_mpcrl_h": {
        "title": "MPC+RL sweep · tracker/hora",
        "keys":  ["mpcrl_hm_h", "mpcrl_hm_90min", "mpcrl_hm_2h", "mpcrl_hm_3h", "mpcrl_hm_4h", "mpcrl_hm_6h", "mpcrl_hm_12h", "mpcrl_hm_d"],
        "cmap":  "autumn",
    },
    "06_best": {
        "title":  "Millors controladors (comparativa global)",
        "keys":   ["mpc", "mpch", "mpc_12h", "oracle", "rl", "mpcrl_h", "mpcrl_2h", "mph_12h", "mpcrl_hm_h"],
        "colors": ["#888888", "#2171b5", "#08519c", "#000000",
                   "#e6550d", "#31a354", "#006d2c",
                   "#9e9ac8", "#c51b8a"],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def hours():
    return np.arange(MINS_IN_DAY) / 60.0


def smooth(arr, w):
    if w <= 1:
        return arr
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values


def get_colors(group_def, n):
    if "colors" in group_def:
        return group_def["colors"][:n]
    cmap = plt.colormaps[group_def["cmap"]].resampled(n)
    return [cmap(i) for i in range(n)]


def bat_power_kw(df, key):
    """Potència real de la bateria en kW des de e_real o action."""
    ecol = f"e_real_{key}"
    acol = f"{key}_action"
    if ecol in df.columns:
        return df[ecol].values / DT_H
    elif acol in df.columns:
        return df[acol].values * P_MAX_KW
    return np.zeros(len(df))


def available(df, keys):
    return [k for k in keys if f"soc_{k}" in df.columns]


def compute_daily_avg(df):
    """
    Agrupa per minut del dia (0–1439) i calcula mitjana i desv.estàndard.
    Retorna (avg_df, std_df) amb índex 0..1439.
    """
    df = df.copy()
    df["_min"] = df["step"] % MINS_IN_DAY
    avg = df.groupby("_min").mean(numeric_only=True)
    std = df.groupby("_min").std(numeric_only=True)
    return avg, std


# ── Funció de plot ─────────────────────────────────────────────────────────────

def make_plot(data_mean, data_std, keys, title, save_path,
              colors, smooth_w=5, figsize=(11, 8), show_std=True):
    """
    Crea figura 3-panells.

    data_mean: DataFrame amb columnes soc_{k}, e_real_{k}, gen_kw, dem_esc_kw, dem_cas_kw
    data_std:  DataFrame amb les mateixes columnes (o None si no hi ha banda)
    show_std:  si True, dibuixa banda ± 1 desv.est. al SoC
    """
    keys = available(data_mean, keys)
    if not keys:
        print(f"  [SKIP] Cap clau disponible: {title}")
        return

    h = hours()
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1.5]})
    ax_soc, ax_bat, ax_gen = axes

    # ── SoC ──────────────────────────────────────────────────────────────────
    for key, col in zip(keys, colors[:len(keys)]):
        soc_m = smooth(data_mean[f"soc_{key}"].values, smooth_w)
        ax_soc.plot(h, soc_m, color=col, label=LABELS.get(key, key), linewidth=1.3)
        if show_std and data_std is not None and f"soc_{key}" in data_std.columns:
            soc_s = smooth(data_std[f"soc_{key}"].values, smooth_w)
            ax_soc.fill_between(h,
                                np.clip(soc_m - soc_s, 0, 1),
                                np.clip(soc_m + soc_s, 0, 1),
                                color=col, alpha=0.12)

    ax_soc.set_ylabel("SoC")
    ax_soc.set_ylim(-0.03, 1.03)
    ncol = max(1, len(keys) // 5)
    ax_soc.legend(loc="upper right", fontsize=8, ncol=ncol)
    ax_soc.grid(True, alpha=0.3)
    ax_soc.set_title(title, fontsize=10, fontweight="bold")

    # ── Potència bateria ─────────────────────────────────────────────────────
    for key, col in zip(keys, colors[:len(keys)]):
        p = smooth(bat_power_kw(data_mean, key), smooth_w)
        ax_bat.plot(h, p, color=col, linewidth=1.0, alpha=0.85)
        if show_std and data_std is not None:
            ecol = f"e_real_{key}"
            acol = f"{key}_action"
            std_col = ecol if ecol in data_std.columns else (acol if acol in data_std.columns else None)
            if std_col:
                scale = 1.0 / DT_H if "e_real" in std_col else P_MAX_KW
                ps = smooth(data_std[std_col].values * scale, smooth_w)
                ax_bat.fill_between(h, p - ps, p + ps, color=col, alpha=0.10)

    ax_bat.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_bat.set_ylabel("Potència bat. (kW)")
    ax_bat.grid(True, alpha=0.3)

    # ── Generació / Demanda ──────────────────────────────────────────────────
    gen = smooth(data_mean["gen_kw"].values, smooth_w)
    dem = smooth((data_mean["dem_esc_kw"] + data_mean["dem_cas_kw"]).values, smooth_w)
    ax_gen.plot(h, gen, color="#2ca02c", label="Generació solar (kW)", linewidth=1.5)
    ax_gen.plot(h, dem, color="#555555", linestyle="--", label="Demanda total (kW)", linewidth=1.5)
    if show_std and data_std is not None:
        gen_s = smooth(data_std["gen_kw"].values, smooth_w)
        dem_s = smooth((data_std["dem_esc_kw"] + data_std["dem_cas_kw"]).values, smooth_w)
        ax_gen.fill_between(h, np.clip(gen - gen_s, 0, None), gen + gen_s,
                            color="#2ca02c", alpha=0.15)
        ax_gen.fill_between(h, np.clip(dem - dem_s, 0, None), dem + dem_s,
                            color="#555555", alpha=0.10)

    ax_gen.set_ylabel("Generació / Demanda\n(kW)")
    ax_gen.set_xlabel("Hora del dia (h)")
    ax_gen.legend(fontsize=8)
    ax_gen.grid(True, alpha=0.3)
    ax_gen.set_xlim(0, 24)
    ax_gen.set_xticks(range(0, 25, 3))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.relpath(save_path, _SRC_DIR)}")


# ── Gràfica interactiva Plotly ────────────────────────────────────────────────

def make_interactive_plot(mean_df, keys, title, save_path, colors, smooth_w=5):
    """
    Genera un HTML interactiu amb Plotly.
    Clicant a la llegenda es mostra/amaga cada controlador simultàniament
    als panells de SoC i Potència bateria.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"  [SKIP] Error important plotly: {e}")
        return

    keys = available(mean_df, keys)
    if not keys:
        return

    h = hours().tolist()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.38, 0.38, 0.24],
        subplot_titles=["SoC", "Potència bat. (kW)", "Generació / Demanda (kW)"],
    )

    # ── Controladors (panels 1 i 2) ──
    for key, col in zip(keys, colors[:len(keys)]):
        label = LABELS.get(key, key)
        hex_col = col if isinstance(col, str) else f"rgba({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)},1)"

        soc = smooth(mean_df[f"soc_{key}"].values, smooth_w).tolist()
        p   = smooth(bat_power_kw(mean_df, key), smooth_w).tolist()

        fig.add_trace(go.Scatter(
            x=h, y=soc, name=label,
            line=dict(color=hex_col, width=1.5),
            legendgroup=key, showlegend=True,
            hovertemplate=f"<b>{label}</b><br>Hora: %{{x:.1f}}h<br>SoC: %{{y:.3f}}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=h, y=p, name=label,
            line=dict(color=hex_col, width=1.2),
            legendgroup=key, showlegend=False,
            hovertemplate=f"<b>{label}</b><br>Hora: %{{x:.1f}}h<br>Potència: %{{y:.1f}} kW<extra></extra>",
        ), row=2, col=1)

    # ── Línia zero panell 2 ──
    fig.add_hline(y=0, row=2, col=1, line=dict(color="black", width=0.8, dash="dot"))

    # ── Generació / Demanda (panel 3, no togglable per controlador) ──
    gen = smooth(mean_df["gen_kw"].values, smooth_w).tolist()
    dem = smooth((mean_df["dem_esc_kw"] + mean_df["dem_cas_kw"]).values, smooth_w).tolist()

    fig.add_trace(go.Scatter(
        x=h, y=gen, name="Generació solar",
        line=dict(color="#2ca02c", width=2),
        hovertemplate="Generació: %{y:.1f} kW<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=h, y=dem, name="Demanda total",
        line=dict(color="#555555", width=2, dash="dash"),
        hovertemplate="Demanda: %{y:.1f} kW<extra></extra>",
    ), row=3, col=1)

    # ── Layout ──
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=750,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            font=dict(size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",  # doble clic per aïllar un controlador
        ),
        margin=dict(l=60, r=180, t=60, b=50),
    )
    fig.update_xaxes(title_text="Hora del dia (h)", row=3, col=1,
                     range=[0, 24], tickvals=list(range(0, 25, 3)))
    fig.update_yaxes(title_text="SoC", range=[-0.03, 1.03], row=1, col=1)
    fig.update_yaxes(title_text="Potència (kW)", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"  ✓ {os.path.relpath(save_path, _SRC_DIR)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",            choices=["avg", "day"], default="avg",
                        help="'avg' = mitjana de tots els dies (default); 'day' = un dia concret")
    parser.add_argument("--day-idx",         type=int, default=0,
                        help="[mode=day] Índex del dia (default: 0)")
    parser.add_argument("--smooth",          type=int, default=5,
                        help="Finestra de suavitzat en minuts (default: 5)")
    parser.add_argument("--std",             action="store_true",
                        help="[mode=avg] Dibuixa la banda ±1 desv.est. (desactivat per defecte)")
    parser.add_argument("--skip-individual", action="store_true",
                        help="Salta les gràfiques individuals per controlador")
    parser.add_argument("--interactive",     action="store_true",
                        help="Genera HTMLs interactius amb Plotly (a més dels PNGs)")
    args = parser.parse_args()

    print(f"Carregant {RESULTS_CSV} ...")
    df = pd.read_csv(RESULTS_CSV)
    total_days = len(df) // MINS_IN_DAY
    print(f"  {len(df)} files · {total_days} dies simulats")

    # ── Preparar dades ────────────────────────────────────────────────────────
    if args.mode == "avg":
        print(f"\nMode: MITJANA ({total_days} dies){' ± 1 desv.est.' if args.std else ''}")
        mean_df, std_df = compute_daily_avg(df)
        subtitle = f"Mitjana diària ({total_days} dies)"
        show_std = args.std
        plots_subdir = "avg"
    else:
        day_idx = args.day_idx % total_days
        start   = day_idx * MINS_IN_DAY
        mean_df = df.iloc[start: start + MINS_IN_DAY].reset_index(drop=True)
        # Afegim columna _min com a índex per compatibilitat
        mean_df.index = range(MINS_IN_DAY)
        std_df  = None
        show_std = False
        subtitle = f"Dia {day_idx}"
        plots_subdir = f"day_{day_idx}"
        print(f"\nMode: DIA {day_idx} (files {start}–{start + MINS_IN_DAY - 1})")

    out_dir = os.path.join(PLOTS_DIR, plots_subdir)

    # ── 1. Gràfiques individuals ──────────────────────────────────────────────
    if not args.skip_individual:
        print("\n[1/3] Gràfiques individuals ...")
        ind_dir = os.path.join(out_dir, "individual")
        for key, label in LABELS.items():
            if f"soc_{key}" not in mean_df.columns:
                continue
            fname = FILE_ALIAS.get(key, key)
            make_plot(mean_df, std_df, [key],
                      f"{label} · {subtitle}",
                      os.path.join(ind_dir, f"{fname}.png"),
                      colors=["#2171b5"],
                      smooth_w=args.smooth, figsize=(10, 7),
                      show_std=show_std)

    # ── 2. Gràfiques per grup ─────────────────────────────────────────────────
    print("\n[2/3] Gràfiques per grup ...")
    grp_dir = os.path.join(out_dir, "grups")
    for grp_name, grp in GROUPS.items():
        keys   = grp["keys"]
        colors = get_colors(grp, len(keys))
        title  = f"{grp['title']} · {subtitle}"
        make_plot(mean_df, std_df, keys, title,
                  os.path.join(grp_dir, f"{grp_name}.png"),
                  colors=colors, smooth_w=args.smooth,
                  show_std=show_std)
        if args.interactive:
            make_interactive_plot(mean_df, keys, title,
                                  os.path.join(grp_dir, f"{grp_name}.html"),
                                  colors=colors, smooth_w=args.smooth)

    # ── 3. Gràfica global (tots els controladors) ─────────────────────────────
    print("\n[3/3] Gràfica global ...")
    all_keys   = [k for k in LABELS if f"soc_{k}" in mean_df.columns]
    cmap_all   = plt.colormaps["tab20"].resampled(len(all_keys))
    colors_all = [cmap_all(i) for i in range(len(all_keys))]
    title_all  = f"Tots els controladors · {subtitle}"
    make_plot(mean_df, std_df, all_keys, title_all,
              os.path.join(out_dir, "tots_els_controladors.png"),
              colors=colors_all, smooth_w=args.smooth,
              figsize=(14, 10), show_std=False)
    if args.interactive:
        make_interactive_plot(mean_df, all_keys, title_all,
                              os.path.join(out_dir, "tots_els_controladors.html"),
                              colors=colors_all, smooth_w=args.smooth)

    print(f"\n✓ Gràfiques guardades a: {os.path.relpath(out_dir, _SRC_DIR)}/")


if __name__ == "__main__":
    main()
