"""
train_ablation_mpcrl.py
=======================
Estudi d'ablació sobre el model MPC+RL (macro-step 60 min/pas):
compara funcions d'activació i GAE-lambda sobre la convergència del PPO.

Ús:
    cd src
    python models/train_ablation_mpcrl.py                        # totes les combinacions
    python models/train_ablation_mpcrl.py --activation LeakyReLU --gae-lambda 0.95
    python models/train_ablation_mpcrl.py --timesteps 200000     # runs ràpids per a test

Sortida: src/results/logs_ablation_mpcrl_{nom}.csv + src/results/ablation_mpcrl_summary.png
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, Logger

_SRC_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_ROOT_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import calcular_economia, actualizar_dinamica_bateria, filtro_seguridad
from params import (
    SOC_INIT, SOC_MIN, SOC_MAX, RESULTS_DIR, SCHOOL_PEAK_GEN_KW,
    MINS_IN_DAY, STEPS_PER_DAY, E_MAX_KWH, DT_H, P_MAX_KW,
    OBS_DEM_ESC_MAX_KW, OBS_DEM_CAS_MAX_KW, OBS_PRICE_MAX_EUR,
)

MACRO_STEP_MIN      = 60
EPISODE_LEN         = int(MINS_IN_DAY / MACRO_STEP_MIN)   # 24 macro-passos/dia
DEG_PENALTY_MULT    = 3.0
FILTER_PENALTY_COEF = 0.05

ACTIVATIONS = {
    "ReLU":      nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU":       nn.ELU,
    "Tanh":      nn.Tanh,
}

LAMBDA_VALUES = {
    "lambda095": 0.95,   # baseline del model MPC+RL
    "lambda100": 1.00,   # Monte Carlo pur, sense decay
}


class BessEnv(gym.Env):
    """Entorn MPC+RL: l'acció és el SoC objectiu (macro-step de 60 min)."""

    def __init__(self, data_df):
        super().__init__()
        self.n_steps = len(data_df)

        self.action_space      = spaces.Box(low=SOC_MIN, high=SOC_MAX, shape=(1,), dtype=np.float32)
        low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._gen      = data_df["Gen_Escuela_kW"].to_numpy(dtype=np.float64)
        self._dem_esc  = data_df["Dem_Escuela_kW"].to_numpy(dtype=np.float64)
        self._dem_cas  = data_df["Dem_Casas_kW"].to_numpy(dtype=np.float64)
        self._precio_c = data_df["Precio_Compra"].to_numpy(dtype=np.float64)
        self._precio_v = data_df["Precio_Venta"].to_numpy(dtype=np.float64)
        self._time_min = data_df["Time_Min"].to_numpy(dtype=np.float64)

        bal_esc    = self._dem_esc - self._gen
        bal_cas    = self._dem_cas.copy()
        mask       = bal_esc < 0
        compartido = np.where(mask, np.minimum(-bal_esc, bal_cas), 0.0)
        bal_esc   += compartido
        bal_cas   -= compartido
        kwh_esc    = bal_esc * DT_H
        kwh_cas    = bal_cas * DT_H
        self._ben_sin_bat = np.where(
            kwh_esc > 0,
            -kwh_esc * self._precio_c - kwh_cas * self._precio_c,
            np.abs(kwh_esc) * self._precio_v - kwh_cas * self._precio_c,
        )

        self.soc_actual   = SOC_INIT
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(0, self.n_steps - STEPS_PER_DAY - 1)
        if max_start > 0:
            max_day = max_start // STEPS_PER_DAY
            day_idx = self.np_random.integers(0, max_day + 1)
            self.current_step = day_idx * STEPS_PER_DAY
        else:
            self.current_step = 0
        self.soc_actual       = SOC_INIT
        self._step_in_episode = 0
        return self._get_obs(), {}

    def _get_obs(self):
        m = min(self.current_step, self.n_steps - 1)
        minuto_del_dia = self._time_min[m] % float(MINS_IN_DAY)
        angulo = 2.0 * np.pi * minuto_del_dia / float(MINS_IN_DAY)
        obs = np.array([
            self.soc_actual,
            self._gen[m]      / SCHOOL_PEAK_GEN_KW,
            self._dem_esc[m]  / OBS_DEM_ESC_MAX_KW,
            self._dem_cas[m]  / OBS_DEM_CAS_MAX_KW,
            self._precio_c[m] / OBS_PRICE_MAX_EUR,
            self._precio_v[m] / OBS_PRICE_MAX_EUR,
            np.sin(angulo),
            np.cos(angulo),
        ], dtype=np.float32)
        np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
        return obs

    def step(self, action):
        soc_ref      = float(action[0])
        total_reward = 0.0

        for i in range(MACRO_STEP_MIN):
            if self.current_step >= self.n_steps - 1:
                break
            minutos_restants = max(MACRO_STEP_MIN - i, 1)
            p_req_kw = (soc_ref - self.soc_actual) * E_MAX_KWH / (minutos_restants * DT_H)
            p_safe_kw, clip_amt = filtro_seguridad(p_req_kw, self.soc_actual)
            self.soc_actual, p_bat_kw, energia_real_kwh = actualizar_dinamica_bateria(
                p_safe_kw, self.soc_actual
            )
            idx = self.current_step
            beneficio, coste_deg = calcular_economia(
                dem_esc_kw=self._dem_esc[idx], gen_kw=self._gen[idx],
                p_bat_kw=p_bat_kw, dem_cas_kw=self._dem_cas[idx],
                precio_c=self._precio_c[idx], precio_v=self._precio_v[idx],
                soc_actual=self.soc_actual, energia_real_kwh=energia_real_kwh,
            )
            total_reward += (beneficio - self._ben_sin_bat[idx]) \
                            - coste_deg * DEG_PENALTY_MULT \
                            - FILTER_PENALTY_COEF * clip_amt
            self.current_step += 1

        self._step_in_episode += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = self._step_in_episode >= EPISODE_LEN
        return self._get_obs(), float(total_reward), terminated, truncated, {}


def cargar_dataset(data_dir, split="train", train_ratio=0.8):
    archivos = sorted(glob.glob(os.path.join(data_dir, "*_datos_cornella.csv")))
    if not archivos:
        raise ValueError(f"No s'han trobat arxius a {data_dir}")
    n_train = int(len(archivos) * train_ratio)
    archivos = archivos[:n_train] if split == "train" else archivos[n_train:]
    print(f"  Dataset [{split}]: {len(archivos)} dies")
    return pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)


def run_experiment(name, activation_fn, gae_lambda, timesteps, df_train, results_dir):
    print(f"\n{'='*60}")
    print(f"  Experiment MPC+RL: {name}")
    print(f"  Activation: {activation_fn.__name__}, gae_lambda: {gae_lambda}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}")

    env = BessEnv(df_train)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=lambda p: 0.001 * p,
        n_steps=EPISODE_LEN * 80,
        batch_size=128,
        n_epochs=20,
        gamma=0.99413,
        gae_lambda=gae_lambda,
        clip_range=0.2,
        ent_coef=0.007538,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs={
            "activation_fn": activation_fn,
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        },
    )

    log_path = os.path.join(results_dir, f"logs_ablation_mpcrl_{name}.csv")
    model.set_logger(Logger(folder=None, output_formats=[CSVOutputFormat(log_path)]))
    model.learn(total_timesteps=timesteps)
    model.save(os.path.join(results_dir, f"model_ablation_mpcrl_{name}"))

    print(f"  Guardat: {log_path}")
    return log_path


def plot_summary(results_dir, experiment_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Ablació MPC+RL: activació i GAE-λ (macro-step 60 min)", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors
    x_col  = "time/total_timesteps"
    y_col  = "rollout/ep_rew_mean"

    lam_labels = {"lambda095": "λ=0.95 (baseline)", "lambda100": "λ=1.00 (MC)"}

    axes[0].set_title("Efecte de la funció d'activació (λ=0.95)")
    axes[1].set_title("Efecte del GAE-λ (activació ReLU)")
    for ax in axes:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Recompensa mitjana per episodi")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

    for i, name in enumerate(experiment_names):
        log_path = os.path.join(results_dir, f"logs_ablation_mpcrl_{name}.csv")
        if not os.path.exists(log_path):
            continue
        df = pd.read_csv(log_path)
        df.columns = df.columns.str.strip()
        if x_col not in df.columns or y_col not in df.columns:
            continue
        s = df[[x_col, y_col]].dropna()
        parts = name.split("_")
        act, lam = parts[0], parts[1] if len(parts) > 1 else ""

        if lam == "lambda095":
            axes[0].plot(s[x_col], s[y_col], label=act, color=colors[i % 10], linewidth=1.5)
        if act == "ReLU":
            axes[1].plot(s[x_col], s[y_col], label=lam_labels.get(lam, lam),
                         color=colors[i % 10], linewidth=1.5)

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    out = os.path.join(results_dir, "ablation_mpcrl_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGràfica guardada: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Ablació MPC+RL: activació i GAE-lambda")
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

    if args.gae_lambda is not None:
        key = f"lambda{str(args.gae_lambda).replace('.', '')}"
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
    plot_summary(results_dir, experiment_names)
    print("\n✓ Ablació MPC+RL finalitzada.")


if __name__ == "__main__":
    main()
