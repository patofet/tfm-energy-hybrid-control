import numpy as np
import cvxpy as cp

from params import E_MAX_KWH, P_MAX_KW, DT_H, SOC_MIN, SOC_MAX, DEG_COST_EUR_KWH

class MPCController:
    def __init__(self):
        self._build_problem()

    def _build_problem(self):
        P_bat  = cp.Variable()            # Potencia bateria [-P_MAX_KW, +P_MAX_KW]
        E_buy  = cp.Variable(nonneg=True) # energía comprada a la red [kWh]
        E_sell = cp.Variable(nonneg=True) # energía vendida a la red [kWh]

        # Parámetros: se actualizan en cada llamada a get_action()
        self.soc0  = cp.Parameter()            # SoC actual [0, 1]
        self.gen   = cp.Parameter(nonneg=True) # generación FV [kW]
        self.d_esc = cp.Parameter(nonneg=True) # demanda escuela [kW]
        self.p_buy = cp.Parameter(nonneg=True) # precio compra [€/kWh]
        self.p_sel = cp.Parameter(nonneg=True) # precio venta [€/kWh]

        # SoC después de aplicar la acción
        soc_new = self.soc0 + P_bat * DT_H / E_MAX_KWH

        constraints = [
            P_bat >= -P_MAX_KW, P_bat <= P_MAX_KW, # Límites del inversor
            soc_new >= SOC_MIN, soc_new <= SOC_MAX, # Límites de la batería
            # Balance: lo que compra/vende la escuela este minuto
            E_buy - E_sell == (self.d_esc - self.gen + P_bat) * DT_H,
        ]

        objective = cp.Maximize(
            self.p_sel * E_sell
            - self.p_buy * E_buy
            - DEG_COST_EUR_KWH * DT_H * cp.abs(P_bat)
        )

        self.prob   = cp.Problem(objective, constraints)
        self._P_bat = P_bat

    def get_action(self, soc_actual: float, gen_kw: float, dem_esc_kw: float, precio_compra: float, precio_venta: float) -> float:
        self.soc0.value  = float(soc_actual)
        self.gen.value   = float(gen_kw)
        self.d_esc.value = float(dem_esc_kw)
        self.p_buy.value = float(precio_compra)
        self.p_sel.value = float(precio_venta)

        self.prob.solve(solver=cp.CLARABEL)

        if self._P_bat.value is None:
            return 0.0
        return float(self._P_bat.value / P_MAX_KW)

