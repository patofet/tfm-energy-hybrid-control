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

from utils import calcular_economia, calcular_beneficio_sin_bateria, actualizar_dinamica_bateria, filtro_seguridad

from params import (
    SOC_INIT, SOC_MIN, SOC_MAX, RESULTS_DIR, SCHOOL_PEAK_GEN_KW, MINS_IN_DAY,
    STEPS_PER_DAY, P_MAX_KW, OBS_DEM_ESC_MAX_KW, OBS_DEM_CAS_MAX_KW, OBS_PRICE_MAX_EUR, LOCAL_LOGS_DIR,
)

# ── Constantes de entrenamiento ──
EPISODE_LEN         = STEPS_PER_DAY  # 1 episodio = 1 día = 1440 pasos (1 min cada uno)
DEG_PENALTY_MULT    = 8.0            # Multiplicador de degradación solo en entrenamiento
FILTER_PENALTY_COEF = 0.05           # Penalización ligera por pedir acción fuera de límites
REWARD_SCALE        = 1.0            # reward relativa (~±0.01€/paso) no necesita escala extra


class BessEnv(gym.Env):
    def __init__(self, data_df):
        super(BessEnv, self).__init__()

        self.df = data_df
        self.n_steps = len(self.df)
        self.current_step = 0

        # Acción: [-1, 1] = fracción de P_MAX_KW (carga/descarga directa)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Todas las obs normalizadas a [-1, 1] para mejor convergencia de PPO
        low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.soc_actual = SOC_INIT

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Arrancar en un día al azar para diversidad de exploración
        max_start = max(0, self.n_steps - STEPS_PER_DAY - 1)
        if max_start > 0:
            max_day = max_start // STEPS_PER_DAY
            day_idx = self.np_random.integers(0, max_day + 1)
            self.current_step = day_idx * STEPS_PER_DAY
        else:
            self.current_step = 0

        # SoC inicial con algo de ruido para mayor robustez
        self.soc_actual = np.clip(
            SOC_INIT + self.np_random.uniform(-0.2, 0.2),
            SOC_MIN, SOC_MAX
        )

        self._step_in_episode = 0
        return self._get_obs(), {}

    def _get_obs(self):
        m = min(self.current_step, self.n_steps - 1)
        fila = self.df.iloc[m]

        # Hora del día codificada cíclicamente (sin/cos)
        minuto_del_dia = fila["Time_Min"] % float(MINS_IN_DAY)
        angulo = 2.0 * np.pi * minuto_del_dia / float(MINS_IN_DAY)

        # Todas las features normalizadas a [0,1] (sin/cos ya están en [-1,1])
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
        # Guardia: si algún valor aberrante del CSV produce NaN/Inf, neutralizarlo
        np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
        return obs

    def step(self, action):
        # ── 1. DECODIFICAR ACCIÓN ──
        # action ∈ [-1, 1] → potencia deseada en kW
        action_val = float(action[0])
        if abs(action_val) < 0.1:
            action_val = 0.0
            
        p_desired_kw = action_val * P_MAX_KW
        # ── 2. FILTRO DE SEGURIDAD (Safety Shield) ──
        # Recorta la potencia a los límites físicos de la batería (numpy puro)
        p_safe_kw, clipping_amount = filtro_seguridad(p_desired_kw, self.soc_actual)
        
        # ── 3. APLICAR DINÁMICA DE BATERÍA ──
        self.soc_actual, p_bat_kw, energia_real_kwh = actualizar_dinamica_bateria(
            p_safe_kw, self.soc_actual
        )
        
        # ── 4. CÁLCULO ECONÓMICO ──
        fila = self.df.iloc[min(self.current_step, self.n_steps - 1)]
        
        beneficio, coste_deg = calcular_economia(
            dem_esc_kw=fila["Dem_Escuela_kW"],
            gen_kw=fila["Gen_Escuela_kW"],
            p_bat_kw=p_bat_kw,
            dem_cas_kw=fila["Dem_Casas_kW"],
            precio_c=fila["Precio_Compra"],
            precio_v=fila["Precio_Venta"],
            soc_actual=self.soc_actual,
            energia_real_kwh=energia_real_kwh
        )
        
        # ── 5. RECOMPENSA (relativa al baseline sense bateria) ──
        ben_sin_bat = calcular_beneficio_sin_bateria(
            dem_esc_kw=fila["Dem_Escuela_kW"],
            gen_kw=fila["Gen_Escuela_kW"],
            dem_cas_kw=fila["Dem_Casas_kW"],
            precio_c=fila["Precio_Compra"],
            precio_v=fila["Precio_Venta"],
        )
        reward = (beneficio - ben_sin_bat) - coste_deg * DEG_PENALTY_MULT - FILTER_PENALTY_COEF * clipping_amount
        
        # ── 6. AVANZAR PASO ──
        self.current_step += 1
        self._step_in_episode += 1

        terminated = bool(self.current_step >= self.n_steps - 1)
        truncated = bool(self._step_in_episode >= EPISODE_LEN)

        info = {
            "soc": self.soc_actual,
            "beneficio": beneficio,
            "coste_degradacion": coste_deg,
            "clipping": clipping_amount,
            "p_desired_kw": p_desired_kw,
            "p_safe_kw": p_safe_kw,
        }
        return self._get_obs(), float(reward) / REWARD_SCALE, terminated, truncated, info


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


def cargar_dataset(data_dir, split="train", train_ratio=0.8):
    archivos = sorted(glob.glob(os.path.join(data_dir, "*_datos_cornella.csv")))
    if not archivos:
        raise ValueError(f"No se encontraron archivos en {data_dir}")

    n_total = len(archivos)
    n_train = int(n_total * train_ratio)

    if split == "train":
        archivos = archivos[:n_train]
    elif split == "test":
        archivos = archivos[n_train:]

    print(f"Dataset [{split}]: {len(archivos)} días (de {n_total} totales)")

    dfs = [pd.read_csv(f) for f in archivos]
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    print("Pre-procesando datos...")
    df_train = cargar_dataset(RESULTS_DIR)

    print("Creando Entorno BESS (Acción Directa + Safety Shield)...")
    env = BessEnv(df_train)

    print("Instanciando Agente PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
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

    os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)
    model.set_logger(Logger(folder=None, output_formats=[
        LiveOutputFormat(),
        CSVOutputFormat(os.path.join(LOCAL_LOGS_DIR, "logs_rl.csv")),
    ]))

    ruta_guardado = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_rl_bess")
    timesteps_totales = 2_000_000
    print(f"Comenzando Entrenamiento por {timesteps_totales} timesteps...")
    model.learn(total_timesteps=timesteps_totales)

    model.save(ruta_guardado)
    print(f"Modelo guardado exitosamente en: {ruta_guardado}.zip")
