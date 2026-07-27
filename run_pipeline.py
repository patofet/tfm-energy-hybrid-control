import os
import subprocess
import sys
import argparse
# source ~/tfm-venv/bin/activate
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))
import params

def parse_args():
    parser = argparse.ArgumentParser(description="Orquestador del Pipeline TFM BESS")
    parser.add_argument('--skip-create',              action='store_true', help="Salta la generació de la comunitat")
    parser.add_argument('--skip-simulation',          action='store_true', help="Salta la simulació de preus/càrrega")
    # MPC+RL MPC/min
    parser.add_argument('--skip-train-rl',            action='store_true', help="Salta l'entrenament RL pur")
    parser.add_argument('--skip-train-mpcrl',         action='store_true', help="Salta MPC+RL 60min")
    parser.add_argument('--skip-train-mpcrl-daily',   action='store_true', help="Salta MPC+RL diari")
    parser.add_argument('--skip-train-mpcrl-30min',   action='store_true', help="Salta MPC+RL 30min")
    parser.add_argument('--skip-train-mpcrl-90min',   action='store_true', help="Salta MPC+RL 90min")
    parser.add_argument('--skip-train-mpcrl-2h',      action='store_true', help="Salta MPC+RL 2h")
    parser.add_argument('--skip-train-mpcrl-3h',      action='store_true', help="Salta MPC+RL 3h")
    parser.add_argument('--skip-train-mpcrl-4h',      action='store_true', help="Salta MPC+RL 4h")
    parser.add_argument('--skip-train-mpcrl-6h',      action='store_true', help="Salta MPC+RL 6h")
    parser.add_argument('--skip-train-mpcrl-12h',     action='store_true', help="Salta MPC+RL 12h")
    # MPC+RL MPC/hora (retrenament amb tracker horari)
    parser.add_argument('--skip-train-mpcrl-hm-1h',   action='store_true', help="Salta MPC+RL hm 1h")
    parser.add_argument('--skip-train-mpcrl-hm-90min',action='store_true', help="Salta MPC+RL hm 90min")
    parser.add_argument('--skip-train-mpcrl-hm-2h',   action='store_true', help="Salta MPC+RL hm 2h")
    parser.add_argument('--skip-train-mpcrl-hm-3h',   action='store_true', help="Salta MPC+RL hm 3h")
    parser.add_argument('--skip-train-mpcrl-hm-4h',   action='store_true', help="Salta MPC+RL hm 4h")
    parser.add_argument('--skip-train-mpcrl-hm-6h',   action='store_true', help="Salta MPC+RL hm 6h")
    parser.add_argument('--skip-train-mpcrl-hm-12h',  action='store_true', help="Salta MPC+RL hm 12h")
    # Avaluació
    parser.add_argument('--skip-oracle',              action='store_true', help="Salta Oracle(H=1440) a l'avaluació")
    parser.add_argument('--skip-eval',                action='store_true', help="Salta l'avaluació final")
    parser.add_argument('--skip-plots',               action='store_true', help="Salta la generació de gràfiques")
    parser.add_argument('--n-days',     type=int, default=None, help="Limita el test set a N dies")
    parser.add_argument('--start-day',  type=int, default=None, help="Primer dia del test set (0-indexed)")
    return parser.parse_args()

def _run(step, total, label, cmd, cwd, flag, flag_name):
    if not flag:
        print(f"🚀 [{step}/{total}] {label}...")
        subprocess.run(cmd, cwd=cwd, check=True)
    else:
        print(f"⏭️  [{step}/{total}] Saltant {flag_name}.")

def main():
    args = parse_args()
    print("=========================================")
    print("    BESS - WORKFLOW PIPELINE INICIADO    ")
    print("=========================================\n")
    params.print_params()

    src = os.path.join(root_dir, "src")
    sim = os.path.join(root_dir, "Simulador_comunitat")
    py  = sys.executable
    TOTAL = 21

    _run(1,  TOTAL, "Generant comunitat",          [py, "create_cornella.py"], sim,  args.skip_create,               "--skip-create")
    _run(2,  TOTAL, "Simulació de base",            [py, "simulation_2.py"],   sim,  args.skip_simulation,           "--skip-simulation")
    _run(3,  TOTAL, "RL pur",                       [py, "models/train_rl.py"],         src, args.skip_train_rl,              "--skip-train-rl")
    _run(4,  TOTAL, "MPC+RL 60min",                 [py, "models/train_mpc_rl.py"],     src, args.skip_train_mpcrl,           "--skip-train-mpcrl")
    _run(5,  TOTAL, "MPC+RL diari",                 [py, "models/train_mpc_rl_daily.py"],src,args.skip_train_mpcrl_daily,     "--skip-train-mpcrl-daily")
    _run(6,  TOTAL, "MPC+RL 30min",                 [py, "models/train_mpc_rl_30min.py"],src,args.skip_train_mpcrl_30min,     "--skip-train-mpcrl-30min")
    _run(7,  TOTAL, "MPC+RL 90min",                 [py, "models/train_mpc_rl_90min.py"],src,args.skip_train_mpcrl_90min,     "--skip-train-mpcrl-90min")
    _run(8,  TOTAL, "MPC+RL 2h",                    [py, "models/train_mpc_rl_2h.py"],  src, args.skip_train_mpcrl_2h,        "--skip-train-mpcrl-2h")
    _run(9,  TOTAL, "MPC+RL 3h",                    [py, "models/train_mpc_rl_3h.py"],  src, args.skip_train_mpcrl_3h,        "--skip-train-mpcrl-3h")
    _run(10, TOTAL, "MPC+RL 4h",                    [py, "models/train_mpc_rl_4h.py"],  src, args.skip_train_mpcrl_4h,        "--skip-train-mpcrl-4h")
    _run(11, TOTAL, "MPC+RL 6h",                    [py, "models/train_mpc_rl_6h.py"],  src, args.skip_train_mpcrl_6h,        "--skip-train-mpcrl-6h")
    _run(12, TOTAL, "MPC+RL 12h",                   [py, "models/train_mpc_rl_12h.py"], src, args.skip_train_mpcrl_12h,       "--skip-train-mpcrl-12h")
    _run(13, TOTAL, "MPC+RL hm 1h",                 [py, "models/train_mpc_rl_hm_1h.py"],  src, args.skip_train_mpcrl_hm_1h,   "--skip-train-mpcrl-hm-1h")
    _run(14, TOTAL, "MPC+RL hm 90min",              [py, "models/train_mpc_rl_hm_90min.py"],src,args.skip_train_mpcrl_hm_90min,"--skip-train-mpcrl-hm-90min")
    _run(15, TOTAL, "MPC+RL hm 2h",                 [py, "models/train_mpc_rl_hm_2h.py"],  src, args.skip_train_mpcrl_hm_2h,   "--skip-train-mpcrl-hm-2h")
    _run(16, TOTAL, "MPC+RL hm 3h",                 [py, "models/train_mpc_rl_hm_3h.py"],  src, args.skip_train_mpcrl_hm_3h,   "--skip-train-mpcrl-hm-3h")
    _run(17, TOTAL, "MPC+RL hm 4h",                 [py, "models/train_mpc_rl_hm_4h.py"],  src, args.skip_train_mpcrl_hm_4h,   "--skip-train-mpcrl-hm-4h")
    _run(18, TOTAL, "MPC+RL hm 6h",                 [py, "models/train_mpc_rl_hm_6h.py"],  src, args.skip_train_mpcrl_hm_6h,   "--skip-train-mpcrl-hm-6h")
    _run(19, TOTAL, "MPC+RL hm 12h",                [py, "models/train_mpc_rl_hm_12h.py"], src, args.skip_train_mpcrl_hm_12h,  "--skip-train-mpcrl-hm-12h")

    # Avaluació
    if not args.skip_eval:
        print(f"🚀 [20/{TOTAL}] Avaluant simulador (main_evaluation.py)...")
        cmd = [py, "main_evaluation.py"]
        if args.n_days:
            cmd += ["--n-days", str(args.n_days)]
        if args.start_day:
            cmd += ["--start-day", str(args.start_day)]
        if args.skip_oracle:
            cmd += ["--skip-oracle"]
        subprocess.run(cmd, cwd=src, check=True)
    else:
        print(f"⏭️  [20/{TOTAL}] Saltant avaluació final (--skip-eval).")

    # Gràfiques
    if not args.skip_plots:
        print(f"🚀 [21/{TOTAL}] Generant gràfiques d'entrenament...")
        subprocess.run([py, "plot_training.py"], cwd=src, check=True)
        print(f"🚀 [21/{TOTAL}] Generant gràfiques de comparativa (plot_comparativa.py)...")
        subprocess.run([py, "plot_comparativa.py", "--skip-individual"], cwd=src, check=True)
    else:
        print(f"⏭️  [21/{TOTAL}] Saltant generació de gràfiques (--skip-plots).")

    print("\n🎉 ¡Tot el pipeline ha finalitzat amb èxit!")

if __name__ == "__main__":
    main()
