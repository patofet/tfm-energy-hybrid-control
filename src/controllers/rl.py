import os
import numpy as np
from stable_baselines3 import PPO

from params import (
    MINS_IN_DAY, GRAN_MIN, SCHOOL_PEAK_GEN_KW, OBS_DEM_ESC_MAX_KW, OBS_DEM_CAS_MAX_KW, OBS_PRICE_MAX_EUR,
)


class RLController:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "modelo_rl_bess")
        try:
            self.model = PPO.load(str(model_path).replace(".zip", ""))
            print("RLController: Modelo cargado correctamente.")
        except Exception as e:
            print(f"WARNING: No se pudo cargar el modelo RL ({e}). Usaremos dummy.")
            self.model = None

        self._minuto_actual = 0

    def get_action(self, soc_actual: float, gen_kw: float, dem_esc_kw: float, precio_c: float, precio_v: float, dem_cas_kw: float = 0.0, minuto_del_dia: int = None) -> float:
        if minuto_del_dia is None:
            minuto_del_dia = self._minuto_actual
        angulo = 2.0 * np.pi * minuto_del_dia / float(MINS_IN_DAY)
        obs = np.array([
            float(soc_actual),
            float(gen_kw)     / SCHOOL_PEAK_GEN_KW,
            float(dem_esc_kw) / OBS_DEM_ESC_MAX_KW,
            float(dem_cas_kw) / OBS_DEM_CAS_MAX_KW,
            float(precio_c)   / OBS_PRICE_MAX_EUR,
            float(precio_v)   / OBS_PRICE_MAX_EUR,
            np.sin(angulo),
            np.cos(angulo),
        ], dtype=np.float32)

        if self.model is not None:
            action, _states = self.model.predict(obs, deterministic=True)
            action_val = float(action[0])
        else:
            action_val = 0.0

        if abs(action_val) < 0.1:
            action_val = 0.0

        self._minuto_actual = (self._minuto_actual + int(GRAN_MIN)) % MINS_IN_DAY
        return float(action_val)
