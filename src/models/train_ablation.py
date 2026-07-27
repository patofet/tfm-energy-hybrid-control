"""
train_ablation.py
=================
Estudi d'ablació: compara l'efecte de la funció d'activació i el paràmetre GAE-lambda
sobre la convergència de PPO en l'agent RL pur (BessEnv 1-minut/pas).

Ús:
    cd src
    python models/train_ablation.py                        # totes les combinacions
    python models/train_ablation.py --activation LeakyReLU --gae-lambda 0.98
    python models/train_ablation.py --timesteps 500000     # runs ràpids per a test

Sortida: src/results/logs_ablation_{nom}.csv + src/results/ablation_summary.png
"""
import os
import sys
import glob
import argparse
import itertools

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger

_SRC_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_ROOT_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import calcular_economia, actualizar_dinamica_bateria, filtro_seguridad
from params import (
    SOC_INIT, SOC_MIN, SOC_MAX, RESULTS_DIR, SCHOOL_PEAK_GEN_KW, MINS_IN_DAY,
    STEPS_PER_DAY, P_MAX_KW, OBS_DEM_ESC_MAX_KW, OBS_DEM_CAS_MAX_KW, OBS_PRICE_MAX_EUR,
)

EPISODE_LEN         = STEPS_PER_DAY
DEG_PENALTY_MULT    = 8.0
FILTER_PENALTY_COEF = 0.05
REWARD_SCALE        = 5.0

ACTIVATIONS = {
    "ReLU":      nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU":       nn.ELU,
    "Tanh":      nn.Tanh,
}

LAMBDA_VALUES = {
    "lambda098": 0.98,
    "lambda100": 1.00,
}


class BessEnv(gym.Env):
    def __init__(self, data_df):
        super().__init__()
        self.df = data_df
        self.n_steps = len(self.df)
        self.current_step = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.soc_actual = SOC_INIT

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(0, self.n_steps - STEPS_PER_DAY - 1)
        if max_start > 0:
            max_day = max_start // STEPS_PER_DAY
            day_idx = self.np_random.integers(0, max_day + 1)
            self.current_step = day_idx * STEPS_PER_DAY
        else:
            self.current_step = 0
        self.soc_actual = np.clip(
            SOC_INIT + self.np_random.uniform(-0.2, 0.2), SOC_MIN, SOC_MAX
        )
        self._step_in_episode = 0
        return self._get_obs(), {}

    def _get_obs(self):
        m = min(self.current_step, self.n_steps - 1)
        fila = self.df.iloc[m]
        minuto_del_dia = fila["Time_Min"] % float(MINS_IN_DAY)
        angulo = 2.0 * np.pi * minuto_del_dia / float(MINS_IN_DAY)
        obs = np.array([
            self.soc_actual,
            fila["Gen_Escuela_kW"]  / SCHOOL_PEAK_GEN_KW,
            fila["Dem_Escuela_kW"]  / OBS_DEM_ESC_MAX_KW,
            fila["Dem_Casas_kW"]    / OBS_DEM_CAS_MAX_KW,
            fila["Precio_Compra"]   / OBS_PRICE_MAX_EUR,
            fila["Precio_Venta"]    / OBS_PRICE_MAX_EUR,
            np.sin(angulo),
            np.cos(angulo),
        ], dtype=np.float32)
        np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
        return obs

    def step(self, action):
        action_val = float(action[0])
        if abs(action_val) < 0.1:
            action_val = 0.0
        p_desired_kw = action_val * P_MAX_KW
        p_safe_kw, clipping_amount = filtro_seguridad(p_desired_kw, self.soc_actual)
        self.soc_actual, p_bat_kw, energia_real_kwh = actualizar_dinamica_bateria(
            p_safe_kw, self.soc_actual
        )
        fila = self.df.iloc[min(self.current_step, self.n_steps - 1)]
        beneficio, coste_deg = calcular_economia(
            dem_esc_kw=fila["Dem_Escuela_kW"],
            gen_kw=fila["Gen_Escuela_kW"],
            p_bat_kw=p_bat_kw,
            dem_cas_kw=fila["Dem_Casas_kW"],
            precio_c=fila["Precio_Compra"],
            precio_v=fila["Precio_Venta"],
            soc_actual=self.soc_actual,
            energia_real_kwh=energia_real_kwh,
        )
        reward = beneficio - coste_deg * DEG_PENALTY_MULT - FILTER_PENALTY_COEF * clipping_amount
        self.current_step += 1
        self._step_in_episode += 1
        terminated = bool(self.current_step >= self.n_steps - 1)
        truncated  = bool(self._step_in_episode >= EPISODE_LEN)
        return self._get_obs(), float(reward) / REWARD_SCALE, terminated, truncated, {}


def cargar_dataset(data_dir, split="train", train_ratio=0.8):
    archivos = sorted(glob.glob(os.path.join(data_dir, "*_datos_cornella.csv")))
    if not archivos:
        raise ValueError(f"No se encontraron archivos en {data_dir}")
    n_train = int(len(archivos) * train_ratio)
    archivos = archivos[:n_train] if split == "train" else archivos[n_train:]
    print(f"  Dataset [{split}]: {len(archivos)} días")
    return pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)


def run_experiment(name, activation_fn, gae_lambda, timesteps, df_train, results_dir):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Activation: {activation_fn.__name__}, gae_lambda: {gae_lambda}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}")

    env = BessEnv(df_train)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=EPISODE_LEN * 4,
        batch_size=288,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=gae_lambda,
        clip_range=0.2,
        ent_coef=0.01,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs={
            "activation_fn": activation_fn,
            "net_arch": {"pi": [64], "vf": [64, 64]},
        },
    )

    log_path = os.path.join(results_dir, f"logs_ablation_{name}.csv")
    model.set_logger(Logger(folder=None, output_formats=[
        CSVOutputFormat(log_path),
    ]))

    model.learn(total_timesteps=timesteps)
    model.save(os.path.join(results_dir, f"model_ablation_{name}"))

    print(f"  Guardat: {log_path}")
    return log_path


def plot_ablation_summary(results_dir, experiment_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Estudi d'ablació: activació i lambda decay", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors
    x_col = "time/total_timesteps"
    y_col = "rollout/ep_rew_mean"

    for ax, group_by, title in [
        (axes[0], "activation", "Efecte de la funció d'activació\n(gae_lambda=0.98)"),
        (axes[1], "lambda",     "Efecte del lambda decay\n(activació=ReLU)"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Recompensa mitjana per episodi")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

    for i, name in enumerate(experiment_names):
        log_path = os.path.join(results_dir, f"logs_ablation_{name}.csv")
        if not os.path.exists(log_path):
            continue
        df = pd.read_csv(log_path)
        df.columns = df.columns.str.strip()
        if x_col not in df.columns or y_col not in df.columns:
            continue
        s = df[[x_col, y_col]].dropna()

        parts = name.split("_")
        act = parts[0]
        lam = parts[1] if len(parts) > 1 else ""

        lam_labels = {"lambda098": "λ=0.98 (GAE)", "lambda100": "λ=1.00 (MC)"}
        if lam == "lambda098":
            axes[0].plot(s[x_col], s[y_col], label=act, color=colors[i % 10], linewidth=1.5)
        if act == "ReLU":
            axes[1].plot(s[x_col], s[y_col], label=lam_labels.get(lam, lam),
                         color=colors[i % 10], linewidth=1.5)

    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(results_dir, "ablation_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGràfica d'ablació guardada: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Estudi d'ablació PPO: activació i lambda")
    parser.add_argument("--activation", choices=list(ACTIVATIONS.keys()),
                        help="Executa únicament aquesta activació (per defecte: totes)")
    parser.add_argument("--gae-lambda", type=float,
                        help="Executa únicament amb aquest lambda (per defecte: tots)")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Timesteps per experiment (default: 500000)")
    args = parser.parse_args()

    results_dir = os.path.join(_SRC_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("Carregant dataset d'entrenament...")
    df_train = cargar_dataset(RESULTS_DIR)

    activations_to_run = {args.activation: ACTIVATIONS[args.activation]} \
        if args.activation else ACTIVATIONS

    lambdas_to_run = {}
    if args.gae_lambda is not None:
        key = f"lambda{str(args.gae_lambda).replace('.','')}"
        lambdas_to_run = {key: args.gae_lambda}
    else:
        lambdas_to_run = LAMBDA_VALUES

    experiment_names = []
    for act_name, act_fn in activations_to_run.items():
        for lam_name, lam_val in lambdas_to_run.items():
            exp_name = f"{act_name}_{lam_name}"
            experiment_names.append(exp_name)
            run_experiment(exp_name, act_fn, lam_val, args.timesteps, df_train, results_dir)

    print("\nGenerant gràfica de resum...")
    plot_ablation_summary(results_dir, experiment_names)
    print("\n✓ Estudi d'ablació finalitzat.")


if __name__ == "__main__":
    main()
