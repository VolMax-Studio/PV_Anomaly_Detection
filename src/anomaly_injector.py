"""
anomaly_injector.py — PV Fault Injection
==========================================
Injects four types of known PV system faults into a clean baseline simulation.
All faults are synthetic (controlled, labeled ground truth) applied to the
physics-based pvlib simulation output.

Fault types and physical basis
-------------------------------

Soiling
    Dust, pollen, bird droppings, and pollution accumulate on panel glass,
    blocking incident irradiance. Effect: proportional reduction in Isc
    (short-circuit current), which reduces P_mpp linearly.
    Modeled as: P_dc_soiled = P_dc_clean × (1 − soiling_ratio(t))
    Soiling rate: 0.1–0.4% per day (arid climates), reset by rain events.
    References: Kimber et al. (2006), HSU soiling model (pvlib).

Partial Shading
    Tree growth, nearby structures, or inverter-level module mismatch causes
    one or more strings to be partially or fully shaded.
    Effect: bypass diode activation reduces P_dc by approximately
    1/N_strings per shaded string. Step-change in power.
    Modeled as: P_dc_shaded = P_dc_clean × (1 − shaded_fraction)
    Shaded fraction: 0.25–0.50 (typically one string or half a string).

PID (Potential Induced Degradation)
    High DC system voltage causes leakage current through module frame,
    degrading cell efficiency. Slow, irreversible voltage-dependent effect.
    Onset: typically after 3–6 months of operation in humid conditions.
    Effect: gradual reduction in P_mp (voltage, Isc, and fill factor).
    Modeled as: linear degradation 0–15% over the second half of the year.
    Reference: Pingel et al. (2010), IEC 62804.

Inverter Fault
    AC-side inverter disconnection (overtemperature, MPPT firmware error,
    grid fault). Hard P_ac → 0 while P_dc (solar side) remains available.
    Duration: hours to days before maintenance.
    Modeled as: P_ac multiplied by 0 during fault hours.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
import warnings


@dataclass
class FaultConfig:
    """Configuration for one or more injected fault events."""
    # Soiling
    soiling_start_hour: int = 500     # Hour of year when soiling begins
    soiling_end_hour:   int = 1200    # Hour of year when rain event clears it
    soiling_rate_per_day: float = 0.003  # 0.3%/day loss rate
    soiling_second_event: bool = True

    # Partial shading
    shading_start_hour: int = 2000
    shading_end_hour:   int = 2168    # 1 week of shading
    shading_fraction:   float = 0.33  # 33% of array shaded (1 of 3 strings)

    # PID
    pid_start_hour: int = 4380       # Starts mid-year
    pid_end_hour:   int = 8760       # Through end of year
    pid_max_loss_fraction: float = 0.10  # Up to 10% degradation by year end

    # Inverter fault
    inverter_fault_hours: List[int] = None  # Hours where inverter trips

    def __post_init__(self):
        if self.inverter_fault_hours is None:
            # Single 48-hour inverter fault event at peak summer
            self.inverter_fault_hours = list(range(3624, 3672))  # 48h in July


DEFAULT_FAULTS = FaultConfig()


@dataclass
class AnomalyResult:
    """PV output with injected anomalies and ground-truth labels."""
    timestamps: pd.DatetimeIndex
    p_ac_clean: pd.Series          # Clean baseline P_ac [W]
    p_ac_faulted: pd.Series        # P_ac with all faults applied [W]
    p_dc_faulted: pd.Series        # P_dc with all faults applied [W]
    performance_ratio_faulted: pd.Series
    fault_labels: pd.DataFrame     # Boolean columns per fault type
    power_multiplier: pd.Series    # Combined power multiplier [0–1]

    @property
    def n_fault_hours(self) -> dict:
        return {col: int(self.fault_labels[col].sum())
                for col in self.fault_labels.columns}


def inject_faults(
    sim_result,   # SimulationResult from pv_simulator
    config: Optional[FaultConfig] = None,
    seed: int = 42,
) -> AnomalyResult:
    """
    Apply synthetic faults to a clean PV simulation baseline.

    Parameters
    ----------
    sim_result : SimulationResult from pv_simulator.simulate_pv_system()
    config : FaultConfig. Uses DEFAULT_FAULTS if None.
    seed : int  Random seed for any stochastic elements.

    Returns
    -------
    AnomalyResult with per-hour fault labels and degraded P_ac.
    """
    if config is None:
        config = DEFAULT_FAULTS

    rng = np.random.default_rng(seed)
    n = len(sim_result.timestamps)
    idx = sim_result.timestamps

    # ── Initialize multipliers (1.0 = no effect) ──────────────────────────
    soiling_mult    = np.ones(n, dtype=np.float64)
    shading_mult    = np.ones(n, dtype=np.float64)
    pid_mult        = np.ones(n, dtype=np.float64)
    inverter_mult   = np.ones(n, dtype=np.float64)

    soiling_label   = np.zeros(n, dtype=bool)
    shading_label   = np.zeros(n, dtype=bool)
    pid_label       = np.zeros(n, dtype=bool)
    inverter_label  = np.zeros(n, dtype=bool)

    # ── Soiling ───────────────────────────────────────────────────────────
    # First soiling event
    _apply_soiling(
        soiling_mult, soiling_label,
        start=config.soiling_start_hour,
        end=config.soiling_end_hour,
        rate_per_hour=config.soiling_rate_per_day / 24.0,
        n=n,
    )
    if config.soiling_second_event:
        # Second event: shorter, less severe (e.g., pollen season)
        _apply_soiling(
            soiling_mult, soiling_label,
            start=min(config.soiling_end_hour + 500, n - 500),
            end=min(config.soiling_end_hour + 900, n - 1),
            rate_per_hour=config.soiling_rate_per_day * 0.5 / 24.0,
            n=n,
        )

    # ── Partial shading ───────────────────────────────────────────────────
    s_start = max(0, min(config.shading_start_hour, n - 1))
    s_end   = max(0, min(config.shading_end_hour, n))
    shading_mult[s_start:s_end] = 1.0 - config.shading_fraction
    shading_label[s_start:s_end] = True

    # ── PID ───────────────────────────────────────────────────────────────
    p_start = max(0, min(config.pid_start_hour, n - 1))
    p_end   = max(0, min(config.pid_end_hour, n))
    pid_duration = p_end - p_start
    if pid_duration > 0:
        ramp = np.linspace(0, config.pid_max_loss_fraction, pid_duration)
        pid_mult[p_start:p_end] = 1.0 - ramp
        pid_label[p_start:p_end] = True

    # ── Inverter fault ────────────────────────────────────────────────────
    for h in config.inverter_fault_hours:
        if 0 <= h < n:
            inverter_mult[h]  = 0.0
            inverter_label[h] = True

    # ── Combine: DC multiplier (soiling + shading + PID affect DC side) ──
    dc_mult = soiling_mult * shading_mult * pid_mult
    dc_mult_series = pd.Series(dc_mult, index=idx)

    # Apply DC degradation to original P_dc
    p_dc_faulted = sim_result.p_mp * dc_mult_series

    # Apply inverter fault to AC side (DC unaffected by inverter trip)
    inv_mult_series = pd.Series(inverter_mult, index=idx)
    p_ac_faulted = sim_result.p_ac * dc_mult_series * inv_mult_series

    # Combined power multiplier for anomaly detection feature
    combined_mult = pd.Series(dc_mult * inverter_mult, index=idx)

    # PR with faults
    g_poa = sim_result.g_poa
    p_rated = sim_result.system.p_rated_w
    pr_denom = g_poa * (p_rated / 1000.0)
    pr_faulted = np.where(
        pr_denom > 5,
        p_ac_faulted / pr_denom,
        np.nan,
    )
    pr_faulted_series = pd.Series(pr_faulted, index=idx)

    # Label DataFrame
    labels = pd.DataFrame({
        'soiling':        soiling_label,
        'partial_shading': shading_label,
        'pid':            pid_label,
        'inverter_fault': inverter_label,
    }, index=idx)
    labels['any_fault'] = labels.any(axis=1)

    return AnomalyResult(
        timestamps=idx,
        p_ac_clean=sim_result.p_ac,
        p_ac_faulted=p_ac_faulted,
        p_dc_faulted=p_dc_faulted,
        performance_ratio_faulted=pr_faulted_series,
        fault_labels=labels,
        power_multiplier=combined_mult,
    )


def _apply_soiling(
    mult_array: np.ndarray,
    label_array: np.ndarray,
    start: int,
    end: int,
    rate_per_hour: float,
    n: int,
) -> None:
    """
    Apply linear soiling accumulation to the multiplier array in-place.
    Soiling accumulates at rate_per_hour until the event ends (rain reset).
    """
    start = max(0, min(start, n - 1))
    end   = max(0, min(end, n))
    if end <= start:
        return

    duration = end - start
    # Soiling ratio at end of event: min(1.0, rate × duration)
    final_soiling = min(0.70, rate_per_hour * duration)  # cap at 70% loss
    soiling_ramp = np.linspace(0, final_soiling, duration)

    mult_array[start:end] *= (1.0 - soiling_ramp)
    label_array[start:end] = True
