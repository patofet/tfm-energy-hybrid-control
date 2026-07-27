import json
import os

# ---------------------------------------------------------------------------
# Cargar pipeline_config.json desde la raíz del proyecto TFM
# ---------------------------------------------------------------------------
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_THIS_DIR, "pipeline_config.json")

def _load_config() -> dict:
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

_cfg = _load_config()

# ---------------------------------------------------------------------------
# Parámetros de la batería BESS (Physical parameters)
# ---------------------------------------------------------------------------
# Capacidad nominal de la batería [kWh]
E_MAX_KWH: float   = float(_cfg.get("school_battery_kwh", 50.0))

# Potencia máxima del inversor [kW]
P_MAX_KW: float    = float(_cfg.get("inverter_power_kw", 50.0))

# Granularidad de cada paso temporal [minutos]
GRAN_MIN: float    = float(_cfg.get("granularity_min", 1.0))

# Paso temporal expresado en horas [h]  → usado en la dinámica del SoC
DT_H: float        = GRAN_MIN / 60.0

# Constantes de tiempo absolutas
MINS_IN_DAY: int   = 1440
STEPS_PER_DAY: int = int(MINS_IN_DAY / GRAN_MIN)

# Límites de SoC operativos [0, 1]
SOC_MIN: float     = float(_cfg.get("soc_min", 0))   # Batería mínimo
SOC_MAX: float     = float(_cfg.get("soc_max", 1))   # Batería máximo

SOC_DEG_MIN: float     = float(_cfg.get("soc_deg_min", 0.05))   # Batería mínimo degradacion
SOC_DEG_MAX: float     = float(_cfg.get("soc_deg_max", 0.95))   # Batería máximo degradacion

# Coste unitario de degradación por kWh ciclado [€/kWh]
DEG_COST_EUR_KWH: float = float(_cfg.get("degradation_cost_eur", 0.002))

# Número de casas y potencia de la escuela (comunidad)
N_HOUSES: int = int(_cfg.get("n_houses", 20))
SCHOOL_PEAK_GEN_KW: float = float(_cfg.get("school_peak_gen_kw", 15.0))

# Número de años a simular (repite los datos de generación solar)
N_YEARS: int = int(_cfg.get("n_years", 1))


# Directorio raíz del proyecto TFM
ROOT_DIR: str       = _cfg.get("root_dir", _THIS_DIR)

# Directorio con los resultados del simulador (CSVs diarios)
RESULTS_DIR: str   = os.path.join(ROOT_DIR, "Simulador_comunitat", "results")

# Logs de entrenamiento en local (fuera de Google Drive) para evitar timeouts de CSVOutputFormat
LOCAL_LOGS_DIR: str = os.path.expanduser("~/tfm_logs")

# ---------------------------------------------------------------------------
# Precios fijos de respaldo (si el CSV no tuviera columnas de precio)
# ---------------------------------------------------------------------------
PRICE_BUY_EUR:  float = float(_cfg.get("price_buy_eur",  0.12))   # €/kWh
PRICE_SELL_EUR: float = float(_cfg.get("price_sell_eur", 0.08))   # €/kWh

# ---------------------------------------------------------------------------
# Límites de normalización del espacio de observación (compartidos por
# BessEnv en train_rl.py y RLController en rl.py — deben mantenerse iguales)
# ---------------------------------------------------------------------------
OBS_DEM_ESC_MAX_KW: float = 20.0   # Demanda máxima esperada de la escuela [kW]
OBS_DEM_CAS_MAX_KW: float = 40.0   # Demanda máxima esperada de las casas [kW]
OBS_PRICE_MAX_EUR:  float = 0.50   # Precio máximo esperado de compra [€/kWh]

# ---------------------------------------------------------------------------
# SoC inicial común para todas las simulaciones [0, 1]
# ---------------------------------------------------------------------------
SOC_INIT: float    = 0

# ---------------------------------------------------------------------------
# Resumen para depuración
# ---------------------------------------------------------------------------
def print_params() -> None:
    """Imprime un resumen de los parámetros cargados."""
    print("=========================================")
    print("       Parámetros BESS compartidos       ")
    print("=========================================")
    print(f"  Capacidad (E_MAX)  : {E_MAX_KWH:.1f} kWh")
    print(f"  Potencia máx (P_MAX): {P_MAX_KW:.1f} kW")
    print(f"  SoC mín / máx      : {SOC_MIN:.2f} / {SOC_MAX:.2f}")
    print(f"  Coste degradación  : {DEG_COST_EUR_KWH:.4f} €/kWh")
    print(f"  Comunidad          : {N_HOUSES} casas, {SCHOOL_PEAK_GEN_KW:.1f} kW fotovoltaica")
    print(f"  Granularidad paso  : {GRAN_MIN:.0f} min  ({DT_H*60:.0f} min → {DT_H:.4f} h)")
    print(f"  Precio compra/venta: {PRICE_BUY_EUR:.3f} / {PRICE_SELL_EUR:.3f} €/kWh")
    print(f"  Directorio CSVs    : {RESULTS_DIR}")
    print("=========================================")


if __name__ == "__main__":
    print_params()
