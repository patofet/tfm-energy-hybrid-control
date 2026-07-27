import os
import numpy as np
import cvxpy as cp
from stable_baselines3 import PPO

from params import (
    GRAN_MIN, SOC_MIN, SOC_MAX, SOC_DEG_MIN, SOC_DEG_MAX, E_MAX_KWH, P_MAX_KW, DT_H, MINS_IN_DAY,
    SCHOOL_PEAK_GEN_KW, OBS_DEM_ESC_MAX_KW, OBS_DEM_CAS_MAX_KW, OBS_PRICE_MAX_EUR,
)


class MPCTracker:
    def __init__(self, horizon=60):
        self.H = horizon
        self._build_problem()

    def _build_problem(self):
        self.P_bat  = cp.Variable(self.H)
        self.E_buy  = cp.Variable(self.H, nonneg=True)
        self.E_sell = cp.Variable(self.H, nonneg=True)
        self.soc    = cp.Variable(self.H + 1)

        self.soc0      = cp.Parameter()
        self.W_sqrt    = cp.Parameter(self.H + 1, nonneg=True)
        self.W_soc_ref = cp.Parameter(self.H + 1)
        self.gen   = cp.Parameter(self.H, nonneg=True)
        self.d_esc = cp.Parameter(self.H, nonneg=True)
        self.p_buy = cp.Parameter(self.H, nonneg=True)
        self.p_sel = cp.Parameter(self.H, nonneg=True)

        constraints = [
            self.soc[0] == self.soc0,
            self.soc[1:] == self.soc[:-1] + self.P_bat * (DT_H / E_MAX_KWH),
            self.P_bat >= -P_MAX_KW,
            self.P_bat <=  P_MAX_KW,
            self.soc >= SOC_MIN,
            self.soc <= SOC_MAX,
            self.E_buy - self.E_sell == (self.d_esc - self.gen + self.P_bat) * DT_H,
        ]

        # Coste: transacciones + penalización cuadrática por desviación del SoC objetivo
        # W_sqrt y W_soc_ref se precalculan en Python para cumplir DPP
        cost = (self.p_buy @ self.E_buy - self.p_sel @ self.E_sell + cp.sum_squares(cp.multiply(self.W_sqrt, self.soc) - self.W_soc_ref))

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def get_action(self, soc_actual: float, soc_ref: float, arr_gen, arr_desc, arr_pbuy, arr_psel, target_step: int, penalty_lambda: float = 1.0) -> float:
        self.soc0.value = float(soc_actual)
        target_step = max(1, min(target_step, self.H))
        weights = np.zeros(self.H + 1)
        weights[target_step] = np.sqrt(float(penalty_lambda))
        self.W_sqrt.value    = weights
        self.W_soc_ref.value = weights * float(soc_ref)
        for param, arr in [(self.gen, arr_gen), (self.d_esc, arr_desc), (self.p_buy, arr_pbuy), (self.p_sel, arr_psel)]:
            val = np.zeros(self.H)
            n = min(self.H, len(arr))
            val[:n] = arr[:n]
            if 0 < n < self.H:
                val[n:] = arr[n - 1]
            param.value = val.astype(float)
        try:
            self.prob.solve(solver=cp.CLARABEL)
            action = float(self.P_bat.value[0]) / P_MAX_KW if self.P_bat.value is not None else 0.0
        except Exception:
            action = 0.0
        return float(action)


class MPCRLController:
    def __init__(self, model_path=None, macro_step_min: int = 60, mpc_solve_interval_min: int = 1):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "modelo_mpc_rl_bess")
        try:
            self.model = PPO.load(str(model_path).replace(".zip", ""))
            print(f"MPCRLController(macro={macro_step_min}min, mpc_interval={mpc_solve_interval_min}min): modelo cargado correctamente.")
        except Exception as e:
            print(f"WARNING: No se pudo cargar el modelo MPC+RL ({e}). Usaremos dummy.")
            self.model = None
        self._macro_step_min       = macro_step_min
        self._mpc_solve_interval   = mpc_solve_interval_min
        self._mpc_call_count       = 0
        self._last_mpc_action      = 0.0
        self._minuto_actual        = 0
        self.soc_ref_current       = (SOC_MIN + SOC_MAX) / 2.0
        self.mpc_tracker           = MPCTracker(horizon=macro_step_min)

    @property
    def horizon(self):
        return self._macro_step_min

    def get_action(self, soc_actual: float, arr_gen: np.ndarray, arr_dem_esc: np.ndarray, arr_precio_c: np.ndarray, arr_precio_v: np.ndarray, dem_cas_kw: float = 0.0, minuto_del_dia: int = None) -> float:
        gen_kw        = float(arr_gen[0])
        dem_esc_kw    = float(arr_dem_esc[0])
        precio_compra = float(arr_precio_c[0])
        precio_venta  = float(arr_precio_v[0])

        minuto_real = minuto_del_dia if minuto_del_dia is not None else self._minuto_actual

        # Macro-step: cada macro_step_min el PPO actualitza el SoC objetivo
        if minuto_real % self._macro_step_min == 0 and self.model is not None:
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

        # Micro-step: MPC receding-horizon (cada mpc_solve_interval minuts)
        if self._mpc_call_count % self._mpc_solve_interval == 0:
            minutos_en_macro  = minuto_real % self._macro_step_min
            minutos_restantes = max(self._macro_step_min - minutos_en_macro, 1)
            target_step       = min(minutos_restantes, self.mpc_tracker.H)
            self._last_mpc_action = self.mpc_tracker.get_action(
                soc_actual, self.soc_ref_current,
                arr_gen, arr_dem_esc, arr_precio_c, arr_precio_v,
                target_step=target_step, penalty_lambda=20.0,
            )
        self._mpc_call_count += 1

        self._minuto_actual = (self._minuto_actual + int(GRAN_MIN)) % MINS_IN_DAY
        return float(self._last_mpc_action)
