"""
train_multiseed_rl.py
=====================
Entrena l'agent PPO (RL pur) N vegades amb llavors aleatòries distintes.
Genera N fitxers  results/logs_rl_seed_{seed}.csv  per calcular
la corba d'aprenentatge amb interval de confiança del 95%.

Ús:
    cd src
    python models/train_multiseed_rl.py               # 5 seeds per defecte
    python models/train_multiseed_rl.py --n-seeds 10  # 10 seeds
    python models/train_multiseed_rl.py --timesteps 500000  # runs ràpids
    python models/train_multiseed_rl.py --seeds 0 1 2 3 4   # seeds explícites
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
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, Logger

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
    print(f"  Dataset [{split}]: {len(archivos)} dies")
    return pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)


def train_one_seed(seed, df_train, results_dir, timesteps):
    print(f"\n[Seed {seed}] Iniciant entrenament ({timesteps:,} timesteps)...")
    np.random.seed(seed)

    env = BessEnv(df_train)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        n_steps=EPISODE_LEN * 4,
        batch_size=288,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.01,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs={"activation_fn": nn.ReLU, "net_arch": {"pi": [64], "vf": [64, 64]}},
    )

    csv_path = os.path.join(results_dir, f"logs_rl_seed_{seed}.csv")
    model.set_logger(Logger(folder=None, output_formats=[
        CSVOutputFormat(csv_path),
    ]))

    model.learn(total_timesteps=timesteps)
    print(f"[Seed {seed}] Fet. Log guardat a {csv_path}")
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena PPO amb N seeds per obtenir IC 95%")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Nombre de seeds a entrenar (default: 5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Llista explícita de seeds (substitueix --n-seeds)")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Total timesteps per run (default: 2_000_000)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Salta seeds que ja tinguin un CSV de log existent")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))

    results_dir = os.path.join(_SRC_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Carregant dataset d'entrenament...")
    df_train = cargar_dataset(RESULTS_DIR)

    print(f"\n{'='*55}")
    print(f"  Multi-seed PPO | seeds={seeds} | {args.timesteps:,} timesteps/run")
    print(f"{'='*55}")

    generated = []
    for seed in seeds:
        csv_path = os.path.join(results_dir, f"logs_rl_seed_{seed}.csv")
        if args.skip_existing and os.path.exists(csv_path):
            print(f"[Seed {seed}] Ja existeix {csv_path}, saltant.")
            generated.append(csv_path)
            continue
        path = train_one_seed(seed, df_train, results_dir, args.timesteps)
        generated.append(path)

    print(f"\n{'='*55}")
    print(f"  Entrenament completat. {len(generated)} logs generats.")
    print(f"  Ara executa:  python plot_learning_evolution.py")
    print(f"{'='*55}")
