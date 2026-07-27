import os
import numpy as np
from stable_baselines3 import PPO

from params import (
    GRAN_MIN, SOC_MIN, SOC_MAX, MINS_IN_DAY,
    SCHOOL_PEAK_GEN_KW, OBS_DEM_ESC_MAX_KW, OBS_DEM_CAS_MAX_KW, OBS_PRICE_MAX_EUR,
)
from controllers.mpc_rl import MPCTracker


class MPCRLDailyController:
    """
    Variant de MPC+RL amb macro-pas diari.

    El PPO decideix 1 target SoC per dia (a les 00:00).
    El MPCTracker (H=60 min) persegueix aquest target al llarg del dia
    amb control de receding-horizon.
    """

    def __init__(self, model_path=None, mpc_horizon: int = MINS_IN_DAY, mpc_solve_interval_min: int = 1):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "models", "modelo_mpc_rl_bess_daily"
            )
        try:
            self.model = PPO.load(str(model_path).replace(".zip", ""))
            print(f"MPCRLDailyController(mpc_interval={mpc_solve_interval_min}min): modelo cargado correctamente.")
        except Exception as e:
            print(f"WARNING: No se pudo cargar el modelo MPC+RL Daily ({e}). Usaremos dummy.")
            self.model = None

        self._mpc_solve_interval = mpc_solve_interval_min
        self._mpc_call_count     = 0
        self._last_mpc_action    = 0.0
        self._minuto_actual      = 0
        self.soc_ref_current     = (SOC_MIN + SOC_MAX) / 2.0
        self.mpc_tracker         = MPCTracker(horizon=mpc_horizon)

    @property
    def horizon(self):
        return self.mpc_tracker.H

    def get_action(
        self,
        soc_actual: float,
        arr_gen: np.ndarray,
        arr_dem_esc: np.ndarray,
        arr_precio_c: np.ndarray,
        arr_precio_v: np.ndarray,
        dem_cas_kw: float = 0.0,
        minuto_del_dia: int = None,
    ) -> float:
        gen_kw        = float(arr_gen[0])
        dem_esc_kw    = float(arr_dem_esc[0])
        precio_compra = float(arr_precio_c[0])
        precio_venta  = float(arr_precio_v[0])

        minuto_real = minuto_del_dia if minuto_del_dia is not None else self._minuto_actual

        # Macro-step: 1 vegada per dia (a les 00:00) el PPO actualitza el SoC objectiu
        if minuto_real % MINS_IN_DAY == 0 and self.model is not None:
            angulo = 2.0 * np.pi * minuto_real / float(MINS_IN_DAY)
            obs = np.array([
                float(soc_actual),
                gen_kw        / SCHOOL_PEAK_GEN_KW,
                dem_esc_kw    / OBS_DEM_ESC_MAX_KW,
                float(dem_cas_kw) / OBS_DEM_CAS_MAX_KW,
                precio_compra / OBS_PRICE_MAX_EUR,
                precio_venta  / OBS_PRICE_MAX_EUR,
                np.sin(angulo),
                np.cos(angulo),
            ], dtype=np.float32)
            np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
            action, _ = self.model.predict(obs, deterministic=True)
            self.soc_ref_current = float(np.clip(action[0], SOC_MIN, SOC_MAX))

        # Micro-step: MPC receding-horizon amb H=60.
        # target_step = minuts restants fins a final de dia, cap at H.
        minutos_en_dia    = minuto_real % MINS_IN_DAY
        minutos_restantes = max(MINS_IN_DAY - minutos_en_dia, 1)
        target_step       = min(minutos_restantes, self.mpc_tracker.H)

        if self._mpc_call_count % self._mpc_solve_interval == 0:
            self._last_mpc_action = self.mpc_tracker.get_action(
                soc_actual,
                self.soc_ref_current,
                arr_gen,
                arr_dem_esc,
                arr_precio_c,
                arr_precio_v,
                target_step=target_step,
                penalty_lambda=20.0,
            )
        self._mpc_call_count += 1

        self._minuto_actual = (self._minuto_actual + int(GRAN_MIN)) % MINS_IN_DAY
        return float(self._last_mpc_action)
