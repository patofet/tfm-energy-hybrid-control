"""MPC+RL amb macro-pas de 90 minuts (16 decisions/dia)."""
import os
import sys
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, Logger

_SRC_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_ROOT_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from params import RESULTS_DIR, MINS_IN_DAY, LOCAL_LOGS_DIR
from models.train_mpc_rl import BessEnv, cargar_dataset, LiveOutputFormat

MACRO_STEP_MIN = 90                            # 90 minuts
EPISODE_LEN    = MINS_IN_DAY // MACRO_STEP_MIN # 16 decisions/dia
N_STEPS        = EPISODE_LEN * 80              # 1280 — divisible per 64
BATCH_SIZE     = 64                            # 1280/64 = 20 minibatches


if __name__ == "__main__":
    print("Pre-procesando datos (MPC+RL 90min)...")
    df_train = cargar_dataset(RESULTS_DIR)

    print(f"Creando Entorno BESS (macro_step={MACRO_STEP_MIN}min, episode_len={EPISODE_LEN})...")
    env = BessEnv(df_train, macro_step_min=MACRO_STEP_MIN)

    print("Instanciando Agente PPO (Macro-Step 90min)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lambda p: 0.001 * p,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
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
        CSVOutputFormat(os.path.join(LOCAL_LOGS_DIR, "logs_mpcrl_90min.csv")),
    ]))

    ruta_guardado = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_mpc_rl_bess_90min")
    timesteps_totales = 2_000_000
    print(f"Comenzando entrenamiento por {timesteps_totales} timesteps...")
    model.learn(total_timesteps=timesteps_totales)

    model.save(ruta_guardado)
    print(f"\nModelo guardado en: {ruta_guardado}.zip")
