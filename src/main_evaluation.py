import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from params import (SOC_INIT, RESULTS_DIR, P_MAX_KW, DT_H, E_MAX_KWH, MINS_IN_DAY)
from utils import calcular_economia, actualizar_dinamica_bateria, filtro_seguridad
from controllers.mpc        import MPCController
from controllers.mpc_h      import MPCHController
from controllers.mpc_oracle import MPCOracleController
from controllers.rl         import RLController
from controllers.mpc_rl     import MPCRLController
from controllers.mpc_rl_daily import MPCRLDailyController
from models.train_rl import cargar_dataset

MPC_H_HORIZON = 60   # horizonte MPC-H estàndard
_MODELS_DIR   = os.path.join(_SRC_DIR, "models")


def simular(n_days: int = None, skip_oracle: bool = False, start_day: int = 0):
    # ── 1. DADES ──
    df_test = cargar_dataset(RESULTS_DIR, split="test")
    if start_day:
        df_test = df_test.iloc[start_day * 1440:].reset_index(drop=True)
    n_steps = (n_days * 1440) if n_days is not None else len(df_test)
    n_steps = min(n_steps, len(df_test))

    # ── 2. CONTROLADORS ──
    mpc          = MPCController()
    mpch         = MPCHController(horizon=MPC_H_HORIZON)
    oracle       = None if skip_oracle else MPCOracleController()
    rl           = RLController()
    mpcrl_h      = MPCRLController()                                                              # 60 min (horari)
    mpcrl_30min  = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_30min"),  30)   # 30min
    mpcrl_90min  = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_90min"),  90)   # 90min
    mpcrl_2h     = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_2h"),    120)   # 2h
    mpcrl_3h     = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_3h"),    180)   # 3h
    mpcrl_4h     = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_4h"),    240)   # 4h
    mpcrl_6h     = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_6h"),    360)   # 6h
    mpcrl_12h    = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_12h"),   720)   # 12h
    mpcrl_d      = MPCRLDailyController()                                                         # diari

    # MPC sweep pur (sense RL) — solve cada minut
    _mpc_sweep = [
        ("mpc_30m",  MPCHController(horizon=30),   30),
        ("mpc_90m",  MPCHController(horizon=90),   90),
        ("mpc_2h",   MPCHController(horizon=120), 120),
        ("mpc_3h",   MPCHController(horizon=180), 180),
        ("mpc_4h",   MPCHController(horizon=240), 240),
        ("mpc_6h",   MPCHController(horizon=360), 360),
        ("mpc_12h",  MPCHController(horizon=720), 720),
    ]

    # MPC sweep pur — solve cada hora (mateixos horitzons)
    _mpc_sweep_h = [
        ("mph_30m",  MPCHController(horizon=30,            solve_interval_min=60),   30),
        ("mph_1h",   MPCHController(horizon=60,            solve_interval_min=60),   60),
        ("mph_90m",  MPCHController(horizon=90,            solve_interval_min=60),   90),
        ("mph_2h",   MPCHController(horizon=120,           solve_interval_min=60),  120),
        ("mph_3h",   MPCHController(horizon=180,           solve_interval_min=60),  180),
        ("mph_4h",   MPCHController(horizon=240,           solve_interval_min=60),  240),
        ("mph_6h",   MPCHController(horizon=360,           solve_interval_min=60),  360),
        ("mph_12h",  MPCHController(horizon=720,           solve_interval_min=60),  720),
        ("mph_1d",   MPCHController(horizon=MINS_IN_DAY,   solve_interval_min=60), MINS_IN_DAY),
    ]

    # MPC+RL — models entrenats amb tracker horari (mpc_solve_interval=60)
    mpcrl_hm_d     = MPCRLDailyController(mpc_solve_interval_min=60)
    mpcrl_hm_h     = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_hm_1h"),    60, mpc_solve_interval_min=60)
    mpcrl_hm_90min = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_hm_90min"),  90, mpc_solve_interval_min=60)
    mpcrl_hm_2h    = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_hm_2h"),    120, mpc_solve_interval_min=60)
    mpcrl_hm_3h    = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_hm_3h"),    180, mpc_solve_interval_min=60)
    mpcrl_hm_4h    = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_hm_4h"),    240, mpc_solve_interval_min=60)
    mpcrl_hm_6h    = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_hm_6h"),    360, mpc_solve_interval_min=60)
    mpcrl_hm_12h   = MPCRLController(os.path.join(_MODELS_DIR, "modelo_mpc_rl_bess_hm_12h"),   720, mpc_solve_interval_min=60)

    # Llista per iterar MPC+RL horari en el bucle de simulació
    _mpcrl_hm_ctls = [
        ("mpcrl_hm_h",     mpcrl_hm_h),
        ("mpcrl_hm_90min", mpcrl_hm_90min),
        ("mpcrl_hm_2h",    mpcrl_hm_2h),
        ("mpcrl_hm_3h",    mpcrl_hm_3h),
        ("mpcrl_hm_4h",    mpcrl_hm_4h),
        ("mpcrl_hm_6h",    mpcrl_hm_6h),
        ("mpcrl_hm_12h",   mpcrl_hm_12h),
        ("mpcrl_hm_d",     mpcrl_hm_d),
    ]

    _keys = ["mpc","mpch",
             "mpc_30m","mpc_90m","mpc_2h","mpc_3h","mpc_4h","mpc_6h","mpc_12h",
             "mph_30m","mph_1h","mph_90m","mph_2h","mph_3h","mph_4h","mph_6h","mph_12h","mph_1d",
             "rl","mpcrl_h","mpcrl_30min","mpcrl_90min","mpcrl_2h","mpcrl_3h","mpcrl_4h","mpcrl_6h","mpcrl_12h","mpcrl_d",
             "mpcrl_hm_h","mpcrl_hm_90min","mpcrl_hm_2h","mpcrl_hm_3h","mpcrl_hm_4h","mpcrl_hm_6h","mpcrl_hm_12h","mpcrl_hm_d"]
    if not skip_oracle:
        _keys.insert(2, "oracle")
    soc = {k: SOC_INIT for k in _keys}
    registros = []
    t = {k: [] for k in _keys}

    n_ctls = len(_keys)
    print(f"Simulant {n_steps} minuts — {n_ctls} controladors...")

    # ── 3. BUCLE MINUT A MINUT ──
    for step in range(n_steps):
        row = df_test.iloc[step]
        gen_kw         = row["Gen_Escuela_kW"]
        dem_esc_kw     = row["Dem_Escuela_kW"]
        dem_cas_kw     = row["Dem_Casas_kW"]
        precio_c       = row["Precio_Compra"]
        precio_v       = row["Precio_Venta"]
        min_dia        = int(row["Time_Min"]) % 1440

        # Finestres futures per als controladors amb lookahead
        fut60   = df_test.iloc[step: step + MPC_H_HORIZON]
        fut1440 = df_test.iloc[step: step + MINS_IN_DAY]
        fut30   = df_test.iloc[step: step +  30]
        fut90   = df_test.iloc[step: step +  90]
        fut120  = df_test.iloc[step: step + 120]
        fut180  = df_test.iloc[step: step + 180]
        fut240  = df_test.iloc[step: step + 240]
        fut360  = df_test.iloc[step: step + 360]
        fut720  = df_test.iloc[step: step + 720]

        def arr(df, col): return df[col].to_numpy()

        # ── A) DECISIONS ──

        t0 = time.perf_counter()
        a_mpc = mpc.get_action(soc["mpc"], gen_kw, dem_esc_kw, precio_c, precio_v)
        t["mpc"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpch = mpch.get_action(soc["mpch"], arr(fut60,"Gen_Escuela_kW"), arr(fut60,"Dem_Escuela_kW"), arr(fut60,"Precio_Compra"), arr(fut60,"Precio_Venta"))
        t["mpch"].append((time.perf_counter()-t0)*1000)

        if not skip_oracle:
            t0 = time.perf_counter()
            a_oracle = oracle.get_action(soc["oracle"], arr(fut1440,"Gen_Escuela_kW"), arr(fut1440,"Dem_Escuela_kW"), arr(fut1440,"Precio_Compra"), arr(fut1440,"Precio_Venta"))
            t["oracle"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_rl = rl.get_action(soc["rl"], gen_kw, dem_esc_kw, precio_c, precio_v, dem_cas_kw, minuto_del_dia=min_dia)
        t["rl"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpcrl_30min = mpcrl_30min.get_action(soc["mpcrl_30min"], arr(fut30,"Gen_Escuela_kW"), arr(fut30,"Dem_Escuela_kW"), arr(fut30,"Precio_Compra"), arr(fut30,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
        t["mpcrl_30min"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpcrl_h = mpcrl_h.get_action(soc["mpcrl_h"], arr(fut60,"Gen_Escuela_kW"), arr(fut60,"Dem_Escuela_kW"), arr(fut60,"Precio_Compra"), arr(fut60,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
        t["mpcrl_h"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpcrl_90min = mpcrl_90min.get_action(soc["mpcrl_90min"], arr(fut90,"Gen_Escuela_kW"), arr(fut90,"Dem_Escuela_kW"), arr(fut90,"Precio_Compra"), arr(fut90,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
        t["mpcrl_90min"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpcrl_2h = mpcrl_2h.get_action(soc["mpcrl_2h"], arr(fut120,"Gen_Escuela_kW"), arr(fut120,"Dem_Escuela_kW"), arr(fut120,"Precio_Compra"), arr(fut120,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
        t["mpcrl_2h"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpcrl_3h = mpcrl_3h.get_action(soc["mpcrl_3h"], arr(fut180,"Gen_Escuela_kW"), arr(fut180,"Dem_Escuela_kW"), arr(fut180,"Precio_Compra"), arr(fut180,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
        t["mpcrl_3h"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpcrl_4h = mpcrl_4h.get_action(soc["mpcrl_4h"], arr(fut240,"Gen_Escuela_kW"), arr(fut240,"Dem_Escuela_kW"), arr(fut240,"Precio_Compra"), arr(fut240,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
        t["mpcrl_4h"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpcrl_6h = mpcrl_6h.get_action(soc["mpcrl_6h"], arr(fut360,"Gen_Escuela_kW"), arr(fut360,"Dem_Escuela_kW"), arr(fut360,"Precio_Compra"), arr(fut360,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
        t["mpcrl_6h"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpcrl_12h = mpcrl_12h.get_action(soc["mpcrl_12h"], arr(fut720,"Gen_Escuela_kW"), arr(fut720,"Dem_Escuela_kW"), arr(fut720,"Precio_Compra"), arr(fut720,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
        t["mpcrl_12h"].append((time.perf_counter()-t0)*1000)

        t0 = time.perf_counter()
        a_mpcrl_d = mpcrl_d.get_action(soc["mpcrl_d"], arr(fut1440,"Gen_Escuela_kW"), arr(fut1440,"Dem_Escuela_kW"), arr(fut1440,"Precio_Compra"), arr(fut1440,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
        t["mpcrl_d"].append((time.perf_counter()-t0)*1000)

        # MPC sweep pur (solve/minut)
        a_mpc_sw = {}
        for _key, _ctrl, _h in _mpc_sweep:
            _fut = df_test.iloc[step: step + _h]
            t0 = time.perf_counter()
            a_mpc_sw[_key] = _ctrl.get_action(soc[_key], arr(_fut,"Gen_Escuela_kW"), arr(_fut,"Dem_Escuela_kW"), arr(_fut,"Precio_Compra"), arr(_fut,"Precio_Venta"))
            t[_key].append((time.perf_counter()-t0)*1000)

        # MPC sweep pur (solve/hora)
        a_mpc_sw_h = {}
        for _key, _ctrl, _h in _mpc_sweep_h:
            _fut = df_test.iloc[step: step + _h]
            t0 = time.perf_counter()
            a_mpc_sw_h[_key] = _ctrl.get_action(soc[_key], arr(_fut,"Gen_Escuela_kW"), arr(_fut,"Dem_Escuela_kW"), arr(_fut,"Precio_Compra"), arr(_fut,"Precio_Venta"))
            t[_key].append((time.perf_counter()-t0)*1000)

        # MPC+RL (MPC tracking solve/hora)
        a_mpcrl_hm = {}
        for _key, _ctrl in _mpcrl_hm_ctls:
            _fut_hm = df_test.iloc[step: step + _ctrl.horizon]
            t0 = time.perf_counter()
            a_mpcrl_hm[_key] = _ctrl.get_action(soc[_key], arr(_fut_hm,"Gen_Escuela_kW"), arr(_fut_hm,"Dem_Escuela_kW"), arr(_fut_hm,"Precio_Compra"), arr(_fut_hm,"Precio_Venta"), dem_cas_kw, minuto_del_dia=min_dia)
            t[_key].append((time.perf_counter()-t0)*1000)

        # ── B) FÍSICA + ECONOMIA (incl. baseline sense bateria) ──

        ben_nobt, deg_nobt = calcular_economia(dem_esc_kw, gen_kw, 0.0, dem_cas_kw, precio_c, precio_v, 0.0, 0.0)

        results = {}
        _oracle_pair = [] if skip_oracle else [("oracle", a_oracle)]
        for key, action in [
            ("mpc",a_mpc), ("mpch",a_mpch), *_oracle_pair,
            *a_mpc_sw.items(),
            *a_mpc_sw_h.items(),
            ("rl",a_rl),
            ("mpcrl_30min",a_mpcrl_30min), ("mpcrl_h",a_mpcrl_h), ("mpcrl_90min",a_mpcrl_90min),
            ("mpcrl_2h",a_mpcrl_2h), ("mpcrl_3h",a_mpcrl_3h), ("mpcrl_4h",a_mpcrl_4h),
            ("mpcrl_6h",a_mpcrl_6h), ("mpcrl_12h",a_mpcrl_12h), ("mpcrl_d",a_mpcrl_d),
            *a_mpcrl_hm.items(),
        ]:
            p_safe, _ = filtro_seguridad(action * P_MAX_KW, soc[key])
            soc[key], p_bat, e_real = actualizar_dinamica_bateria(p_safe, soc[key])
            ben, deg = calcular_economia(dem_esc_kw, gen_kw, p_bat, dem_cas_kw, precio_c, precio_v, soc[key], e_real)
            results[key] = (action, ben, deg, e_real)

        registros.append({
            "step": step, "gen_kw": gen_kw, "dem_esc_kw": dem_esc_kw,
            "dem_cas_kw": dem_cas_kw, "precio_compra": precio_c, "precio_venta": precio_v,
            # No-battery
            "beneficio_nobt": ben_nobt, "coste_degradacion_nobt": deg_nobt,
            # Cada controlador
            **{f"{k}_action":            results[k][0] for k in results},
            **{f"soc_{k}":              soc[k]        for k in results},
            **{f"beneficio_{k}":        results[k][1] for k in results},
            **{f"coste_degradacion_{k}": results[k][2] for k in results},
            **{f"e_real_{k}":           results[k][3] for k in results},
            # Temps
            **{f"t_ms_{k}": t[k][-1] for k in t},
        })

    # ── 4. GUARDAR I MOSTRAR ──
    df_r = pd.DataFrame(registros)
    df_r.to_csv(os.path.join(_SRC_DIR, "results", "resultados_comparativa.csv"), index=False)

    STEPS_PER_DAY = 1440

    def metricas(sufijo, t_list):
        if sufijo == "nobt":
            b = df_r["beneficio_nobt"].sum()
            d = df_r["coste_degradacion_nobt"].sum()
            return b, d, b-d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0
        b      = df_r[f"beneficio_{sufijo}"].sum()
        d      = df_r[f"coste_degradacion_{sufijo}"].sum()
        e_real = df_r[f"e_real_{sufijo}"].to_numpy()
        chg    = e_real[e_real > 0].sum()
        dchg   = abs(e_real[e_real < 0].sum())
        cycles = chg / E_MAX_KWH
        avg_soc = df_r[f"soc_{sufijo}"].mean()
        pct_c   = (e_real >  1e-6).mean() * 100
        pct_d   = (e_real < -1e-6).mean() * 100
        # Total computation per day: avg_ms_per_step × steps/day → seconds/day
        total_s = float(np.mean(t_list)) * STEPS_PER_DAY / 1000.0
        return b, d, b-d, chg, dchg, cycles, avg_soc, pct_c, pct_d, 100-pct_c-pct_d, total_s

    _oracle_ctls = [] if skip_oracle else ["oracle"]
    ctls = [
        "nobt","mpc","rl",
        "mpc_30m","mpch","mpc_90m","mpc_2h","mpc_3h",
        "mpc_4h","mpc_6h","mpc_12h", *_oracle_ctls,
        "mph_30m","mph_1h","mph_90m","mph_2h","mph_3h",
        "mph_4h","mph_6h","mph_12h","mph_1d",
        "mpcrl_30min","mpcrl_h","mpcrl_90min","mpcrl_2h","mpcrl_3h",
        "mpcrl_4h","mpcrl_6h","mpcrl_12h","mpcrl_d",
        "mpcrl_hm_h","mpcrl_hm_90min","mpcrl_hm_2h","mpcrl_hm_3h",
        "mpcrl_hm_4h","mpcrl_hm_6h","mpcrl_hm_12h","mpcrl_hm_d",
    ]
    labels = {
        "nobt":          "Sense bateria",
        "mpc":           "MPC(H=1)",
        "mpch":          "MPC(1h)",
        "oracle":        "MPC(H=1440)",
        "mpc_30m":       "MPC(30m)",
        "mpc_90m":       "MPC(90m)",
        "mpc_2h":        "MPC(2h)",
        "mpc_3h":        "MPC(3h)",
        "mpc_4h":        "MPC(4h)",
        "mpc_6h":        "MPC(6h)",
        "mpc_12h":       "MPC(12h)",
        "mph_30m":       "MPC/h(30m)",
        "mph_1h":        "MPC/h(1h)",
        "mph_90m":       "MPC/h(90m)",
        "mph_2h":        "MPC/h(2h)",
        "mph_3h":        "MPC/h(3h)",
        "mph_4h":        "MPC/h(4h)",
        "mph_6h":        "MPC/h(6h)",
        "mph_12h":       "MPC/h(12h)",
        "mph_1d":        "MPC/h(1d)",
        "rl":            "RL",
        "mpcrl_30min":   "MPC+RL(30m)",
        "mpcrl_h":       "MPC+RL(1h)",
        "mpcrl_90min":   "MPC+RL(90m)",
        "mpcrl_2h":      "MPC+RL(2h)",
        "mpcrl_3h":      "MPC+RL(3h)",
        "mpcrl_4h":      "MPC+RL(4h)",
        "mpcrl_6h":      "MPC+RL(6h)",
        "mpcrl_12h":     "MPC+RL(12h)",
        "mpcrl_d":       "MPC+RL(D)",
        "mpcrl_hm_h":    "MPC+RL/h(1h)",
        "mpcrl_hm_90min":"MPC+RL/h(90m)",
        "mpcrl_hm_2h":   "MPC+RL/h(2h)",
        "mpcrl_hm_3h":   "MPC+RL/h(3h)",
        "mpcrl_hm_4h":   "MPC+RL/h(4h)",
        "mpcrl_hm_6h":   "MPC+RL/h(6h)",
        "mpcrl_hm_12h":  "MPC+RL/h(12h)",
        "mpcrl_hm_d":    "MPC+RL/h(D)",
    }
    t_lists = {k: t.get(k, [0.0]) for k in ctls}
    t_lists["nobt"] = [0.0]
    r = {k: metricas(k, t_lists[k]) for k in ctls}

    # Guardar resum de mètriques agregades
    _cols = ["beneficio_transacciones","coste_degradacion","beneficio_neto",
             "energia_carregada_kwh","energia_desc_kwh","cicles_complets",
             "soc_mig","pct_carregant","pct_desc","pct_inactiu","comput_dia_s"]
    df_resum = pd.DataFrame(
        {k: dict(zip(_cols, r[k])) for k in ctls},
    ).T
    df_resum.index.name = "controlador"
    df_resum.insert(0, "label", [labels[k] for k in ctls])
    df_resum.to_csv(os.path.join(_SRC_DIR, "results", "resum_comparativa.csv"))

    W = 13
    M = 24
    grups = [
        ("── BASELINES ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────", ["nobt","mpc","rl"]),
        ("── MPC sweep solve/min ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────", ["mpc_30m","mpch","mpc_90m","mpc_2h","mpc_3h","mpc_4h","mpc_6h","mpc_12h", *_oracle_ctls]),
        ("── MPC sweep solve/hora ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────", ["mph_30m","mph_1h","mph_90m","mph_2h","mph_3h","mph_4h","mph_6h","mph_12h","mph_1d"]),
        ("── MPC+RL MPC/min ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────", ["mpcrl_30min","mpcrl_h","mpcrl_90min","mpcrl_2h","mpcrl_3h","mpcrl_4h","mpcrl_6h","mpcrl_12h","mpcrl_d"]),
        ("── MPC+RL MPC/hora ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────", ["mpcrl_hm_h","mpcrl_hm_90min","mpcrl_hm_2h","mpcrl_hm_3h","mpcrl_hm_4h","mpcrl_hm_6h","mpcrl_hm_12h","mpcrl_hm_d"]),
    ]

    for title, keys in grups:
        SEP = "=" * (M + len(keys) * (W+3) + 1)
        DIV = "-" * len(SEP)
        print(SEP)
        print(f"  {title}")
        print(SEP)
        hdr = f" {'Métrica':<{M}}"
        for k in keys: hdr += f" | {labels[k]:^{W}}"
        print(hdr)
        print(DIV)
        for metric_idx, metric_name in enumerate([
            "Beneficio transacciones",
            "Coste degradación",
            "BENEFICIO NETO",
        ]):
            row = f" {metric_name:<{M}}"
            for k in keys:
                row += f" | {r[k][metric_idx]:>{W}.2f} €"
            print(row)
            if metric_idx == 1:
                print(DIV)
        print(SEP)
        for metric_idx, metric_name in enumerate([
            "Energia carregada (kWh)",
            "Energia desc. (kWh)",
            "Cicles complets",
            "SoC mig",
            "% temps carregant",
            "% temps desc.",
            "% temps inactiu",
        ], start=3):
            row = f" {metric_name:<{M}}"
            for k in keys:
                row += f" | {r[k][metric_idx]:>{W}.1f}  "
            print(row)
        print(DIV)
        row = f" {'Còmput total/dia (s)':<{M}}"
        for k in keys:
            row += f" | {r[k][10]:>{W}.3f}  "
        print(row)
        print(SEP)
        print()

    print("Dades guardades a 'results/resultados_comparativa.csv'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-days", type=int, default=None)
    parser.add_argument("--start-day", type=int, default=0, help="Primer dia del test set (0-indexed)")
    parser.add_argument("--skip-oracle", action="store_true", help="Salta el controlador Oracle(H=1d)")
    args = parser.parse_args()
    simular(n_days=args.n_days, skip_oracle=args.skip_oracle, start_day=args.start_day)
