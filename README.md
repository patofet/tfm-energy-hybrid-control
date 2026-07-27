# ⚡ Hybrid MPC + Reinforcement Learning Control for Battery Energy Storage Systems

> **Master's Thesis (TFM)** — Hierarchical control framework combining Model Predictive Control (MPC) and Deep Reinforcement Learning (PPO) for optimal Battery Energy Storage System (BESS) management in residential energy communities.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Full Pipeline](#full-pipeline)
  - [Selective Execution](#selective-execution)
- [Controllers](#controllers)
- [Community Simulator](#community-simulator)
- [Evaluation & Visualization](#evaluation--visualization)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This project investigates **hierarchical hybrid control strategies** for managing a shared Battery Energy Storage System (BESS) within a simulated residential energy community in Cornellà de Llobregat (Barcelona). The community consists of **1 school prosumer** (with rooftop PV panels and a 50 kWh battery) and **20 residential consumers**.

The core research question is: *Can a hierarchical controller — where a Reinforcement Learning agent sets high-level SoC targets and a Model Predictive Controller tracks them at the minutely level — outperform standalone MPC or standalone RL approaches?*

The framework benchmarks **38+ controller variants** across multiple dimensions:
- **Baseline controllers**: No-battery, single-step MPC
- **Receding-horizon MPC**: With configurable lookahead horizons (30 min – 24 h)
- **Pure RL**: End-to-end PPO agent directly controlling charge/discharge power
- **Hybrid MPC+RL**: Hierarchical controller with various macro-step intervals (30 min – 12 h) and tracker granularities (minutely and hourly)
- **Oracle MPC**: Perfect-foresight upper bound (full 24 h horizon with ground-truth data)

---

## Key Features

- 🔋 **Realistic battery physics**: SoC dynamics, inverter power limits, degradation cost modeling
- 🏘️ **Community energy sharing model**: Internal self-consumption prioritization (school → houses) before grid export
- 🧠 **Hierarchical control**: RL macro-planning + MPC micro-tracking with configurable time scales
- 📊 **Comprehensive benchmarking**: 38+ controller variants evaluated on financial profit, degradation, cycle count, and computational cost
- 🔄 **End-to-end pipeline**: From community generation to training, evaluation, and publication-ready plots
- 📈 **Publication-ready visualizations**: Matplotlib static charts + Plotly interactive HTML dashboards

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Hybrid MPC+RL Controller              │
│                                                         │
│  ┌───────────────┐    SoC target    ┌────────────────┐  │
│  │   RL Agent    │ ───────────────► │  MPC Tracker   │  │
│  │   (PPO)       │  every N min     │  (cvxpy LP/QP) │  │
│  │               │                  │                │  │
│  │  Observation:  │                  │  Horizon: H    │  │
│  │  • SoC        │                  │  Steps: 1 min  │  │
│  │  • PV gen     │                  │  Output: P_bat │  │
│  │  • Demand     │                  │                │  │
│  │  • Prices     │                  │                │  │
│  │  • Time (sin/ │                  │                │  │
│  │    cos enc.)  │                  │                │  │
│  └───────────────┘                  └────────────────┘  │
│                                            │            │
│                                     Battery action      │
│                                      (kW charge/        │
│                                       discharge)        │
└────────────────────────────────────────────┬────────────┘
                                             │
                                             ▼
                          ┌──────────────────────────────┐
                          │    Community Microgrid        │
                          │                              │
                          │  🏫 School (PV + BESS)       │
                          │  🏠 20 Residential houses    │
                          │  ⚡ Grid connection           │
                          └──────────────────────────────┘
```

---

## Project Structure

```
tfm-energy-hybrid-control/
│
├── pipeline_config.json          # Global simulation parameters (JSON)
├── params.py                     # Shared parameter loader for all modules
├── run_pipeline.py               # 🚀 Main orchestrator (21-step pipeline)
│
├── Simulador_comunitat/          # Community energy simulator
│   ├── create_cornella.py        # Generates community topology & profiles
│   ├── cornella_community.txt    # Community configuration (JSON)
│   ├── simulation_2.py           # Stochastic multi-year energy simulation
│   ├── generation.csv            # Solar irradiance data (minutely)
│   ├── prices.csv                # Electricity tariff data (buy/sell)
│   ├── functions_0.py            # Stochastic activity-based load generator
│   └── results/                  # Daily simulation CSVs (output)
│
├── src/                          # Core source code
│   ├── controllers/              # Control strategy implementations
│   │   ├── mpc.py                # Single-step MPC (no lookahead)
│   │   ├── mpc_h.py              # Receding-horizon MPC (configurable H)
│   │   ├── mpc_oracle.py         # Oracle MPC (H=1440, perfect foresight)
│   │   ├── rl.py                 # Pure RL controller (PPO inference)
│   │   ├── mpc_rl.py             # Hybrid MPC+RL (configurable macro-step)
│   │   └── mpc_rl_daily.py       # Hybrid MPC+RL (daily macro-step)
│   │
│   ├── models/                   # Training scripts & saved models
│   │   ├── train_rl.py           # PPO training for pure RL controller
│   │   ├── train_mpc_rl.py       # PPO training for hybrid MPC+RL
│   │   ├── train_mpc_rl_daily.py # PPO training for daily hybrid
│   │   ├── train_mpc_rl_*.py     # Horizon-specific training variants
│   │   ├── train_ablation*.py    # Ablation study training scripts
│   │   ├── train_multiseed_rl.py # Multi-seed training for CI analysis
│   │   └── *.zip                 # Pre-trained PPO model checkpoints
│   │
│   ├── main_evaluation.py        # 📊 Backtest evaluation framework
│   ├── utils.py                  # Battery physics & economic utilities
│   ├── plot_comparativa.py       # Comparison visualization (Matplotlib/Plotly)
│   ├── plot_training.py          # Training curve plots
│   ├── plot_learning_evolution.py# Multi-seed learning evolution (95% CI)
│   └── results/                  # Evaluation output CSVs & plots
│
└── Memoria/                      # LaTeX thesis document
    ├── main.tex                  # Full thesis source
    ├── main.pdf                  # Compiled thesis PDF
    ├── biblio.bib                # Bibliography references
    └── imatges/                  # Thesis figures
```

---

## Installation

### Prerequisites

- **Python** ≥ 3.9
- A virtual environment is recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/patofet/tfm-energy-hybrid-control.git
cd tfm-energy-hybrid-control

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install numpy pandas matplotlib cvxpy stable-baselines3 gymnasium torch plotly
```

### Required Python Packages

| Package              | Purpose                                      |
|----------------------|----------------------------------------------|
| `numpy`              | Numerical computation                        |
| `pandas`             | Data manipulation and CSV I/O                |
| `cvxpy`              | Convex optimization (MPC solvers)            |
| `stable-baselines3`  | Proximal Policy Optimization (PPO) agent     |
| `gymnasium`          | RL environment interface                     |
| `torch` (PyTorch)    | Neural network backend for PPO               |
| `matplotlib`         | Static publication-quality plots             |
| `plotly` *(optional)*| Interactive HTML dashboards                  |

---

## Configuration

All simulation parameters are centralized in [`pipeline_config.json`](pipeline_config.json):

```json
{
    "n_houses": 20,
    "n_years": 20,
    "school_peak_gen_kw": 15.0,
    "school_battery_kwh": 50.0,
    "inverter_power_kw": 50.0,
    "soc_min": 0,
    "soc_max": 1,
    "soc_deg_min": 0.1,
    "soc_deg_max": 0.90,
    "degradation_cost_eur": 0.01,
    "price_sell_eur": 0.08,
    "price_buy_eur": 0.12,
    "granularity_min": 1,
    "root_dir": "/path/to/your/data/directory"
}
```

| Parameter              | Description                                    | Default  |
|------------------------|------------------------------------------------|----------|
| `n_houses`             | Number of residential consumers                | 20       |
| `n_years`              | Years of simulation data to generate           | 20       |
| `school_peak_gen_kw`   | School PV peak generation capacity (kW)        | 15.0     |
| `school_battery_kwh`   | Battery capacity (kWh)                         | 50.0     |
| `inverter_power_kw`    | Maximum inverter power (kW)                    | 50.0     |
| `soc_min` / `soc_max`  | Operational SoC limits [0, 1]                  | 0 / 1    |
| `soc_deg_min/max`      | SoC degradation protection bounds              | 0.1/0.9  |
| `degradation_cost_eur` | Battery degradation cost (€/kWh cycled)        | 0.01     |
| `price_buy_eur`        | Grid purchase price (€/kWh)                    | 0.12     |
| `price_sell_eur`       | Grid sale price (€/kWh)                        | 0.08     |
| `granularity_min`      | Simulation time step (minutes)                 | 1        |
| `root_dir`             | Root directory for data files                  | Project dir |

> **Note:** Update `root_dir` to point to your local data directory before running.

---

## Usage

### Full Pipeline

The complete workflow (21 steps) is orchestrated by [`run_pipeline.py`](run_pipeline.py):

```bash
python run_pipeline.py
```

This sequentially executes:

| Step | Description                                      |
|------|--------------------------------------------------|
| 1    | Generate community topology (`create_cornella`)  |
| 2    | Run stochastic energy simulation                 |
| 3    | Train pure RL agent (PPO)                        |
| 4–12 | Train MPC+RL agents (minutely tracker, various macro-step intervals) |
| 13–19| Train MPC+RL agents (hourly tracker variants)    |
| 20   | Run evaluation backtest across all controllers   |
| 21   | Generate comparison plots                        |

### Selective Execution

Skip specific pipeline stages with `--skip-*` flags:

```bash
# Skip community generation and simulation (use existing data)
python run_pipeline.py --skip-create --skip-simulation

# Only run evaluation and plots (skip all training)
python run_pipeline.py --skip-create --skip-simulation \
    --skip-train-rl --skip-train-mpcrl --skip-train-mpcrl-daily \
    --skip-train-mpcrl-30min --skip-train-mpcrl-90min \
    --skip-train-mpcrl-2h --skip-train-mpcrl-3h \
    --skip-train-mpcrl-4h --skip-train-mpcrl-6h \
    --skip-train-mpcrl-12h \
    --skip-train-mpcrl-hm-1h --skip-train-mpcrl-hm-90min \
    --skip-train-mpcrl-hm-2h --skip-train-mpcrl-hm-3h \
    --skip-train-mpcrl-hm-4h --skip-train-mpcrl-hm-6h \
    --skip-train-mpcrl-hm-12h

# Limit evaluation to specific days
python run_pipeline.py --skip-create --skip-simulation \
    --n-days 30 --start-day 0

# Skip the Oracle controller (very slow, H=1440)
python run_pipeline.py --skip-oracle
```

### Running Individual Components

```bash
# Generate community profiles
python Simulador_comunitat/create_cornella.py

# Run energy simulation
python Simulador_comunitat/simulation_2.py

# Train a specific model
python src/models/train_rl.py           # Pure RL
python src/models/train_mpc_rl.py       # Hybrid MPC+RL (60 min macro-step)

# Run evaluation only
python src/main_evaluation.py --n-days 30

# Generate plots
python src/plot_comparativa.py
python src/plot_training.py
python src/plot_learning_evolution.py
```

---

## Controllers

### Overview

| Controller         | Type        | Decision Frequency | Lookahead | Description |
|--------------------|-------------|-------------------|-----------|-------------|
| **No Battery**     | Baseline    | —                 | —         | Financial baseline without storage |
| **MPC (H=1)**      | Optimization| Every minute      | 1 step    | Greedy single-step convex optimizer |
| **MPC (H=N)**      | Optimization| Every minute      | N steps   | Receding-horizon LP with configurable horizon |
| **Oracle MPC**     | Optimization| Every minute      | 1440 steps| Perfect-foresight full-day LP upper bound |
| **Pure RL**        | Learning    | Every minute      | —         | End-to-end PPO agent (direct power output) |
| **MPC+RL (N min)** | Hybrid      | Every N minutes   | H steps   | RL sets SoC target → MPC tracks it |
| **MPC+RL (daily)** | Hybrid      | Once per day      | H steps   | RL sets daily SoC target → MPC tracks |

### Hybrid MPC+RL Controller Detail

The core contribution of this work is the **hierarchical hybrid controller**:

1. **Macro-step (RL)**: Every *N* minutes, the PPO agent observes the system state (SoC, PV generation, demand, prices, cyclic time encoding) and outputs a **target SoC** value in $[SoC_{min}, SoC_{max}]$.

2. **Micro-step (MPC Tracker)**: At every minute, a convex QP solver computes the optimal battery power $P_{bat}$ that minimizes grid energy costs while tracking the RL-prescribed SoC target through a weighted quadratic penalty:

$$\min_{P_{bat}} \sum_{t=0}^{H-1} \left[ c_{buy} \cdot P_{buy}(t) - c_{sell} \cdot P_{sell}(t) + c_{deg} \cdot |P_{bat}(t)| \right] + \lambda \sum_{t=0}^{H-1} \left( SoC(t) - SoC_{target} \right)^2$$

Subject to battery capacity, power, and SoC constraints.

### Solver Details

- **Primary solver**: CLARABEL (with warm-start)
- **Fallback solver**: SCS (if CLARABEL fails or returns non-optimal status)
- **Framework**: CVXPY with DPP-compliant formulations for efficient re-solves

---

## Community Simulator

The simulator generates realistic minutely energy data for a residential community in Cornellà de Llobregat:

- **Solar generation**: Based on real irradiance profiles (`generation.csv`), linearly interpolated to 1-minute resolution
- **Load profiles**: Stochastic activity-based model generating diverse household consumption patterns
- **Pricing**: Time-of-use electricity tariffs from `prices.csv`
- **Output**: Daily CSV files (`YYYY-MM-DD_datos_cornella.csv`) with columns for PV generation, school demand, residential demand, and buy/sell prices

---

## Evaluation & Visualization

### Metrics

The evaluation framework (`main_evaluation.py`) computes the following metrics for each controller:

| Metric                    | Description                                            |
|---------------------------|--------------------------------------------------------|
| **Net Profit (€/day)**    | Grid revenue − grid cost − degradation cost            |
| **Grid Purchases (€)**    | Total cost of electricity bought from the grid         |
| **Grid Sales (€)**        | Total revenue from electricity sold to the grid        |
| **Degradation Cost (€)**  | Battery wear cost based on energy cycled               |
| **Full Cycles**           | Number of equivalent full charge/discharge cycles      |
| **Average SoC**           | Mean State of Charge over the evaluation period        |
| **Computation Time (s)**  | Wall-clock time per simulated day                      |

### Visualization Tools

- **`plot_comparativa.py`**: Side-by-side comparison of SoC trajectories, battery power profiles, and PV generation vs. demand across controllers
- **`plot_training.py`**: PPO training reward curves
- **`plot_learning_evolution.py`**: Multi-seed learning curves with 95% confidence intervals

---

## Dependencies

```
numpy
pandas
cvxpy
stable-baselines3
gymnasium
torch
matplotlib
plotly  # optional, for interactive HTML plots
```

---

## License

This project was developed as part of a Master's Thesis (TFM). Please contact the author for licensing and usage information.
