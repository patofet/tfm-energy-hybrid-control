import os
import sys
import glob
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger

_SRC_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_ROOT_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import calcular_economia, actualizar_dinamica_bateria, filtro_seguridad
from params import (
    SOC_INIT, SOC_MIN, SOC_MAX, RESULTS_DIR, SCHOOL_PEAK_GEN_KW,
    GRAN_MIN, MINS_IN_DAY, STEPS_PER_DAY, E_MAX_KWH, DT_H, P_MAX_KW,
    OBS_DEM_ESC_MAX_KW, OBS_DEM_CAS_MAX_KW, OBS_PRICE_MAX_EUR, LOCAL_LOGS_DIR,
)

MACRO_STEP_MIN      = 60
EPISODE_LEN         = int(MINS_IN_DAY / MACRO_STEP_MIN)  # 24 macro-pasos por día
DEG_PENALTY_MULT    = 3.0
FILTER_PENALTY_COEF = 0.05
REWARD_SCALE        = 1.0   # reward ya es marginal (~±0.5€/macro-paso)


class BessEnv(gym.Env):
    def __init__(self, data_df, macro_step_min=MACRO_STEP_MIN, mpc_solve_interval_min: int = 1):
        super().__init__()
        self._macro_step_min    = macro_step_min
        self._mpc_solve_interval = mpc_solve_interval_min
        self._episode_len        = int(MINS_IN_DAY / macro_step_min)
        self.n_steps = len(data_df)

        self.action_space = spaces.Box(low=SOC_MIN, high=SOC_MAX, shape=(1,), dtype=np.float32)
        low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Pre-convertir a numpy para evitar df.iloc en el micro-loop (pandas es lento)
        self._gen      = data_df["Gen_Escuela_kW"].to_numpy(dtype=np.float64)
        self._dem_esc  = data_df["Dem_Escuela_kW"].to_numpy(dtype=np.float64)
        self._dem_cas  = data_df["Dem_Casas_kW"].to_numpy(dtype=np.float64)
        self._precio_c = data_df["Precio_Compra"].to_numpy(dtype=np.float64)
        self._precio_v = data_df["Precio_Venta"].to_numpy(dtype=np.float64)
        self._time_min = data_df["Time_Min"].to_numpy(dtype=np.float64)

        # Precalcular beneficio_sin_bateria vectorialmente (no depende del SoC)
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
        self.soc_actual = SOC_INIT
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
        soc_ref = float(action[0])

        total_reward    = 0.0
        total_beneficio = 0.0
        total_coste_deg = 0.0

        # Micro-loop: macro_step_min minutos por macro-paso
        p_req_kw = 0.0
        for i in range(self._macro_step_min):
            if self.current_step >= self.n_steps - 1:
                break

            # Recomputa p_req cada mpc_solve_interval minuts (simula tracker hourly)
            if i % self._mpc_solve_interval == 0:
                minutos_restantes = max(self._macro_step_min - i, 1)
                tiempo_restante_h = minutos_restantes * DT_H
                energia_req_kwh   = (soc_ref - self.soc_actual) * E_MAX_KWH
                p_req_kw          = energia_req_kwh / tiempo_restante_h

            p_safe_kw, clipping_amount = filtro_seguridad(p_req_kw, self.soc_actual)
            self.soc_actual, p_bat_kw, energia_real_kwh = actualizar_dinamica_bateria(
                p_safe_kw, self.soc_actual
            )
            idx = self.current_step
            beneficio, coste_deg = calcular_economia(
                dem_esc_kw=self._dem_esc[idx],
                gen_kw=self._gen[idx],
                p_bat_kw=p_bat_kw,
                dem_cas_kw=self._dem_cas[idx],
                precio_c=self._precio_c[idx],
                precio_v=self._precio_v[idx],
                soc_actual=self.soc_actual,
                energia_real_kwh=energia_real_kwh,
            )
            reward = (beneficio - self._ben_sin_bat[idx]) - coste_deg * DEG_PENALTY_MULT - FILTER_PENALTY_COEF * clipping_amount

            total_reward    += reward
            total_beneficio += beneficio
            total_coste_deg += coste_deg
            self.current_step += 1

        self._step_in_episode += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = self._step_in_episode >= self._episode_len

        info = {
            "soc":               self.soc_actual,
            "beneficio":         total_beneficio,
            "coste_degradacion": total_coste_deg,
        }
        return self._get_obs(), float(total_reward), terminated, truncated, info


def cargar_dataset(data_dir, split="train", train_ratio=0.8):
    archivos = sorted(glob.glob(os.path.join(data_dir, "*_datos_cornella.csv")))
    if not archivos:
        raise ValueError(f"No se encontraron archivos en {data_dir}")
    n_total = len(archivos)
    n_train = int(n_total * train_ratio)
    archivos = archivos[:n_train] if split == "train" else archivos[n_train:]
    print(f"Dataset [{split}]: {len(archivos)} días (de {n_total} totales)")
    return pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)


class LiveOutputFormat(HumanOutputFormat):
    """Muestra métricas clave en una línea que se actualiza en sitio."""
    def __init__(self):
        super().__init__(sys.stdout)

    def write(self, key_values, _key_excluded, _step=0):
        kv = dict(key_values)
        parts = []
        if "time/total_timesteps" in kv:
            parts.append(f"steps={int(kv['time/total_timesteps']):>9,}")
        if "rollout/ep_rew_mean" in kv:
            parts.append(f"reward={float(kv['rollout/ep_rew_mean']):>8.3f}")
        if "train/explained_variance" in kv:
            parts.append(f"ev={float(kv['train/explained_variance']):>6.3f}")
        if "time/fps" in kv:
            parts.append(f"fps={int(kv['time/fps']):>5}")
        sys.stdout.write("\r  " + "  |  ".join(parts) + "   ")
        sys.stdout.flush()


if __name__ == "__main__":
    print("Pre-procesando datos...")
    df_train = cargar_dataset(RESULTS_DIR)

    print("Creando Entorno BESS jerárquico (MPC+RL)...")
    env = BessEnv(df_train)

    print("Instanciando Agente PPO (Macro-Step)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lambda p: 0.001 * p,
        n_steps=EPISODE_LEN * 80,
        batch_size=128,
        n_epochs=20,
        gamma=0.99413,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.007538,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs={"activation_fn": nn.ReLU, "net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )

    os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)
    model.set_logger(Logger(folder=None, output_formats=[
        LiveOutputFormat(),
        CSVOutputFormat(os.path.join(LOCAL_LOGS_DIR, "logs_mpcrl.csv")),
    ]))

    ruta_guardado = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_mpc_rl_bess")
    timesteps_totales = 2_000_000
    print(f"Comenzando entrenamiento por {timesteps_totales} timesteps...")
    model.learn(total_timesteps=timesteps_totales)

    model.save(ruta_guardado)
    print(f"Modelo guardado en: {ruta_guardado}.zip")
