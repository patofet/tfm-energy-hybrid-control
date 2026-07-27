import numpy as np
from controllers.mpc_h import MPCHController
from params import MINS_IN_DAY, P_MAX_KW


class MPCOracleController:
    """
    Oracle amb informació perfecta i horitzó complet (H=1440).

    Resol LP(H=1440) a cada minut amb les dades reals futures —
    igual que MPC(H=60) però amb horitzó de 24h.
    Mesura el cost computacional real d'un oracle en temps real.
    """

    def __init__(self):
        self._mpc = MPCHController(horizon=MINS_IN_DAY)

    def get_action(
        self,
        soc_actual: float,
        future_gen:  np.ndarray,
        future_dem:  np.ndarray,
        future_pbuy: np.ndarray,
        future_psel: np.ndarray,
    ) -> float:
        self._mpc.get_action(soc_actual, future_gen, future_dem, future_pbuy, future_psel)
        if self._mpc.P_bat.value is not None:
            return float(self._mpc.P_bat.value[0] / P_MAX_KW)
        return 0.0
