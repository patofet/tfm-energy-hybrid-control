import warnings
import numpy as np
import cvxpy as cp

from params import E_MAX_KWH, P_MAX_KW, DT_H, SOC_MIN, SOC_MAX, SOC_DEG_MIN, SOC_DEG_MAX, DEG_COST_EUR_KWH


class MPCHController:
    def __init__(self, horizon: int = 60, solve_interval_min: int = 1):
        self.H = horizon
        self._solve_interval = solve_interval_min
        self._call_count = 0
        self._last_action = 0.0
        self._build_problem()

    def _build_problem(self):
        H = self.H

        # Variables de decisión
        self.P_bat  = cp.Variable(H)
        self.E_buy  = cp.Variable(H, nonneg=True)
        self.E_sell = cp.Variable(H, nonneg=True)

        # Parámetros (se actualizan en cada llamada)
        self.soc0  = cp.Parameter()
        self.gen   = cp.Parameter(H, nonneg=True)
        self.d_esc = cp.Parameter(H, nonneg=True)
        self.p_buy = cp.Parameter(H, nonneg=True)
        self.p_sel = cp.Parameter(H, nonneg=True)

        # Trayectoria del SoC: soc[t] = soc0 + cumsum(P_bat)[t] * DT_H / E_MAX
        soc_traj = self.soc0 + cp.cumsum(self.P_bat) * (DT_H / E_MAX_KWH)

        constraints = [
            self.P_bat >= -P_MAX_KW,
            self.P_bat <=  P_MAX_KW,
            soc_traj   >= SOC_MIN,
            soc_traj   <= SOC_MAX,
            self.E_buy - self.E_sell == cp.multiply(self.d_esc - self.gen + self.P_bat, DT_H), # Balance energético en cada paso
        ]

        objective = cp.Maximize(
            cp.sum(cp.multiply(self.p_sel, self.E_sell) - cp.multiply(self.p_buy, self.E_buy))
            - DEG_COST_EUR_KWH * DT_H * cp.sum(cp.abs(self.P_bat))
        )

        self.prob = cp.Problem(objective, constraints)

    def get_action(self, soc_actual: float, future_gen: np.ndarray, future_dem_esc: np.ndarray, future_p_buy: np.ndarray, future_p_sel: np.ndarray) -> float:
        do_solve = (self._call_count % self._solve_interval == 0)
        self._call_count += 1
        if not do_solve:
            return self._last_action

        H = self.H
        n = len(future_gen)

        def pad(arr, fill_val):
            if n < H:
                return np.concatenate([arr, np.full(H - n, fill_val)])
            return arr[:H]

        fill_gen  = float(future_gen[-1])  if n > 0 else 0.0
        fill_dem  = float(future_dem_esc[-1]) if n > 0 else 0.0
        fill_buy  = float(future_p_buy[-1])   if n > 0 else 0.12
        fill_sel  = float(future_p_sel[-1])   if n > 0 else 0.05

        self.soc0.value  = float(soc_actual)
        self.gen.value   = np.clip(pad(future_gen,     fill_gen),  0.0, None).astype(float)
        self.d_esc.value = np.clip(pad(future_dem_esc, fill_dem),  0.0, None).astype(float)
        self.p_buy.value = np.clip(pad(future_p_buy,   fill_buy),  1e-6, None).astype(float)
        self.p_sel.value = np.clip(pad(future_p_sel,   fill_sel),  1e-6, None).astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.prob.solve(solver=cp.CLARABEL, warm_start=True)
            except cp.error.SolverError:
                pass

            if self.prob.status != "optimal" or self.P_bat.value is None:
                try:
                    self.prob.solve(solver=cp.SCS, warm_start=True)
                except cp.error.SolverError:
                    pass

        if self.prob.status not in ("optimal", "optimal_inaccurate") or self.P_bat.value is None:
            return self._last_action

        self._last_action = float(self.P_bat.value[0] / P_MAX_KW)
        return self._last_action
