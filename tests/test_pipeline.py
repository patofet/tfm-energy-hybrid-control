import os
import sys
import pytest

# Add root directory to python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import params

def test_params_loading():
    """Verify that core BESS simulation parameters load correctly."""
    assert params.E_MAX_KWH > 0, "Nominal battery capacity must be strictly positive"
    assert params.P_MAX_KW > 0, "Max inverter power must be strictly positive"
    assert 0 <= params.SOC_MIN < params.SOC_MAX <= 1, "SoC limits must be normalized within [0, 1]"
    assert params.GRAN_MIN > 0, "Time step granularity must be positive"

def test_default_simulation_fallbacks():
    """Ensure directory paths and fallback economic parameters operate safely."""
    assert isinstance(params.ROOT_DIR, str) and len(params.ROOT_DIR) > 0, "ROOT_DIR must be a valid string"
    assert isinstance(params.RESULTS_DIR, str) and len(params.RESULTS_DIR) > 0, "RESULTS_DIR must be valid"
    assert params.PRICE_BUY_EUR >= params.PRICE_SELL_EUR >= 0, "Buy price should be >= sell price and non-negative"
    assert params.DEG_COST_EUR_KWH >= 0, "Degradation cost must be non-negative"

def test_step_granularity_consistency():
    """Test minutely granularity translation to daily simulation steps."""
    expected_steps = int(1440 / params.GRAN_MIN)
    assert params.STEPS_PER_DAY == expected_steps, f"Expected {expected_steps} steps per day"
