"""
mppt_transient.py — Cloud Transient Impact on MPPT Efficiency
=============================================================
Simulates sub-minute irradiance transients caused by cloud shadows and
evaluates MPPT tracking efficiency degradation during rapid irradiance changes.

This module operates at 10 Hz (100 ms) temporal resolution — four orders of
magnitude finer than the hourly TMY3 baseline in pv_simulator.py.

Physical context — why cloud transients matter
----------------------------------------------
Standard P&O (Perturb & Observe) MPPT algorithms track the Maximum Power
Point by periodically perturbing the operating voltage and observing the
power response:
    - Scan frequency: typically 1–10 Hz (100 ms – 1 s per step)
    - Perturbation step ΔV: 0.1–1.0% of Voc

When irradiance changes faster than the scan frequency, the algorithm
"chases" a moving target:
    - G rises rapidly → MPP voltage shifts right (higher Vmpp)
    - P&O is still at old Vmpp → misses 5–15% of available power during ramp
    - G drops rapidly → algorithm over-shoots into current-limited region

For a 10 kWp system with 5% average transient loss:
    10 kW × 0.05 loss × 200 transient hours/year ≈ 100 kWh/year missed

BPM FiberNetworks relevance
----------------------------
PV systems connected to MV grid at EPS substations are monitored by current
transformers and voltage sensors at the connection point. Cloud transient
signatures appear as:
    1. Rapid dP/dt excursions (|dP/dt| > 5 kW/s)
    2. Sudden reactive power demand from inverter tracking error
    3. Momentary voltage sags at weak grid connection points

P&O MPPT algorithm parameters
------------------------------
Standard implementation (IEEE 1562, IEC 62894):
    - Scan period T_scan: 100 ms (10 Hz)
    - Perturbation ΔV:    0.5% of Vmpp_STC
    - Controller type:    Incremental conductance with hysteresis deadband

Cloud transient characterization
----------------------------------
Measured cloud shadow profiles (Lappalainen & Valkealahti 2017,
Progress in Photovoltaics):
    - Fast edge (high wind): dG/dt = 200–800 W/m²/s, transition time 0.5–2 s
    - Slow edge (low wind):  dG/dt = 20–80  W/m²/s, transition time 5–20 s
    - Shadow depth:          10–90% of clear-sky irradiance
    - Duration of shading:   10 s – 10 min

References
----------
Lappalainen, K. & Valkealahti, S. (2017). Analysis of shading periods caused
by moving clouds. Solar Energy, 154, 283–294.

De Brito, M.A.G. et al. (2013). Evaluation of the main MPPT techniques for
photovoltaic applications. IEEE Trans. Ind. Electron., 60(3), 1156–1167.

Urtasun, A. et al. (2015). Influence of the DC/DC converter topologies on the
MPPT performance. Solar Energy, 111, 173–184.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.signal import butter, sosfiltfilt


# ── Module / system constants (CS5P-250M at STC) ────────────────────────────

V_OC_REF  = 59.60   # V — open circuit voltage at STC
V_MP_REF  = 48.70   # V — MPP voltage at STC
I_SC_REF  = 5.49    # A — short circuit current at STC
I_MP_REF  = 5.14    # A — MPP current at STC
P_MP_REF  = I_MP_REF * V_MP_REF    # W ≈ 250 W
GAMMA_PMP = -0.0041  # /°C temperature coefficient
G_STC     = 1000.0   # W/m²
T_STC     = 25.0     # °C

# System: 4 strings × 10 series
N_STRINGS = 4
N_SERIES  = 10


# ── Cloud transient profiles ─────────────────────────────────────────────────

@dataclass
class CloudEvent:
    """Single cloud shadow event."""
    t_start_s: float          # Start time [s]
    shadow_depth: float       # Fractional irradiance reduction [0–1]
    duration_s: float         # Shadow duration [s]
    ramp_in_s: float          # Shadow edge rise time [s] (fast = 0.5s, slow = 10s)
    ramp_out_s: float         # Recovery time [s]
    label: str = "cloud"


def generate_cloud_events(
    total_duration_s: float = 600.0,
    n_events: int = 5,
    clear_sky_g: float = 800.0,
    seed: int = 42,
) -> List[CloudEvent]:
    """
    Generate a sequence of realistic cloud shadow events.

    Parameters
    ----------
    total_duration_s : float  Simulation window [seconds]. Default: 10 minutes.
    n_events : int            Number of shadow events. Default: 5.
    clear_sky_g : float       Clear-sky irradiance [W/m²]. Default: 800 W/m².
    seed : int                Random seed for event parameters.

    Returns
    -------
    list of CloudEvent, sorted by t_start_s.
    """
    rng = np.random.default_rng(seed)

    # Space events roughly uniformly with gaps
    spacing = total_duration_s / (n_events + 1)
    events = []

    for i in range(n_events):
        t_start = spacing * (i + 1) + rng.uniform(-spacing * 0.3, spacing * 0.3)
        t_start = max(5.0, min(t_start, total_duration_s - 30.0))

        shadow_depth = rng.uniform(0.20, 0.85)    # 20-85% irradiance reduction
        duration     = rng.uniform(5.0, 60.0)     # 5s - 1min shadow
        # Wind-dependent edge sharpness: high-depth events tend to have sharper edges
        ramp_in  = rng.uniform(0.3, 3.0) * (1 - 0.5 * shadow_depth)
        ramp_out = ramp_in * rng.uniform(0.8, 1.5)  # similar rise/fall

        events.append(CloudEvent(
            t_start_s=t_start,
            shadow_depth=shadow_depth,
            duration_s=duration,
            ramp_in_s=ramp_in,
            ramp_out_s=ramp_out,
            label=f"cloud_{i+1}",
        ))

    events.sort(key=lambda e: e.t_start_s)
    return events


def build_irradiance_profile(
    events: List[CloudEvent],
    total_duration_s: float = 600.0,
    fs: float = 10.0,              # Hz — 10 Hz = 100ms resolution
    clear_sky_g: float = 800.0,
    noise_std_w: float = 5.0,      # W/m² measurement noise
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a sub-second irradiance time series with cloud shadow profiles.

    Each cloud shadow uses a smooth trapezoidal profile (ramp-in, flat shadow,
    ramp-out) rather than a rectangular step, matching real measured profiles.

    Parameters
    ----------
    events : list of CloudEvent
    total_duration_s : float  [s]
    fs : float               Sampling frequency [Hz]. Default: 10 Hz.
    clear_sky_g : float      Clear-sky irradiance [W/m²].
    noise_std_w : float      Pyranometer measurement noise [W/m²].

    Returns
    -------
    t : np.ndarray   Time axis [s], shape (N,)
    g : np.ndarray   Irradiance [W/m²], shape (N,)
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    t = np.arange(0, total_duration_s, dt)
    N = len(t)

    # Start with clear-sky irradiance
    g = clear_sky_g * np.ones(N)

    dt = 1.0 / fs  # timestep [s]
    for ev in events:
        # Integer-index-based shadow application: robust against np.arange
        # floating-point accumulation errors (e.g. t[203] = 20.2999... ≠ 20.3)
        i_in_start   = int(round(ev.t_start_s * fs))
        i_in_end     = int(round((ev.t_start_s + ev.ramp_in_s) * fs))
        i_shad_end   = int(round((ev.t_start_s + ev.ramp_in_s + ev.duration_s) * fs))
        i_out_end    = int(round((ev.t_start_s + ev.ramp_in_s + ev.duration_s + ev.ramp_out_s) * fs))

        # Clamp to valid array range
        i_in_start = max(0, min(i_in_start, N))
        i_in_end   = max(i_in_start, min(i_in_end, N))
        i_shad_end = max(i_in_end,   min(i_shad_end, N))
        i_out_end  = max(i_shad_end, min(i_out_end, N))

        # Ramp-in: linear attenuation 0 → shadow_depth
        n_ramp_in = i_in_end - i_in_start
        if n_ramp_in > 0:
            frac = np.linspace(0, 1, n_ramp_in + 1)[:-1]
            g[i_in_start:i_in_end] *= 1.0 - ev.shadow_depth * frac

        # Full shadow
        g[i_in_end:i_shad_end] *= 1.0 - ev.shadow_depth

        # Ramp-out: linear recovery shadow_depth → 0
        n_ramp_out = i_out_end - i_shad_end
        if n_ramp_out > 0:
            frac = np.linspace(0, 1, n_ramp_out + 1)[:-1]
            g[i_shad_end:i_out_end] *= 1.0 - ev.shadow_depth * (1.0 - frac)

    # Add pyranometer measurement noise
    g += rng.normal(0, noise_std_w, N)
    g = np.clip(g, 0, 1200)

    return t, g


# ── Single-diode model (simplified for transient simulation) ─────────────────

def mpp_from_irradiance(
    g: np.ndarray,
    t_cell: float = 35.0,
    n_strings: int = N_STRINGS,
    n_series: int = N_SERIES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute theoretical MPP (Vmpp, Impp, Pmpp) per module at each irradiance sample.

    Uses linear irradiance scaling + temperature correction from STC:
        Impp = Impp_ref × (G / G_stc)
        Vmpp = Vmpp_ref × (1 + γ_pmp × (T_cell − T_stc))
        Pmpp = Impp × Vmpp × N_strings × N_series

    This is the first-order approximation valid for G > 100 W/m² and
    T_cell within ±30°C of STC. More accurate simulation would use the
    full five-parameter CEC model.

    Returns
    -------
    vmpp : array   Theoretical MPP voltage (array) [V]
    impp : array   Theoretical MPP current (array) [A]
    pmpp : array   Theoretical MPP power (array) [W]
    """
    g_ratio = np.clip(g / G_STC, 0, 1.5)

    impp = I_MP_REF * g_ratio * n_strings
    temp_factor = 1.0 + GAMMA_PMP * (t_cell - T_STC)
    # Vmpp irradiance correction: logarithmic from single-diode model
    # Vmpp = Vmpp_ref + nVt × ln(G/G_stc) per module, n_series modules in series
    # nVt ≈ 26mV × ideality × n_cells ≈ 0.06V per module at 25°C
    # Effectively: Vmpp rises ~3% from G=200 to G=1000 W/m²
    # This is what causes P&O tracking error under rapidly changing irradiance.
    nVt_series = 0.06 * n_series   # thermal voltage × n_cells in series [V]
    g_ratio_safe = np.where(g_ratio > 0.01, g_ratio, 0.01)  # avoid log(0)
    vmpp = (V_MP_REF * temp_factor + nVt_series * np.log(g_ratio_safe)) * n_series
    vmpp = np.maximum(vmpp, 0.3 * V_MP_REF * n_series)  # physical lower bound
    vmpp = np.asarray(vmpp, dtype=np.float64)
    pmpp = impp * vmpp

    return vmpp, impp, pmpp


# ── P&O MPPT algorithm ────────────────────────────────────────────────────────

@dataclass
class MPPTState:
    """Internal state of the P&O MPPT controller."""
    v_ref: float       # Current voltage reference [V]
    v_prev: float      # Previous voltage [V]
    p_prev: float      # Previous power measurement [W]
    dv: float          # Perturbation step [V]
    scan_period: float # Time between perturbations [s]
    t_last_scan: float # Time of last perturbation [s]


def simulate_mppt_po(
    t: np.ndarray,
    g: np.ndarray,
    t_cell: float = 35.0,
    scan_period_s: float = 0.1,    # 10 Hz scan rate
    dv_fraction: float = 0.005,    # 0.5% of Vmpp_ref per step
    controller_delay_s: float = 0.02,  # 20ms ADC + control loop delay
    n_strings: int = N_STRINGS,
    n_series: int = N_SERIES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate P&O MPPT algorithm tracking through irradiance transients.

    The algorithm perturbs the voltage reference every scan_period_s and
    measures the resulting power change. It moves toward higher power.

    Parameters
    ----------
    t : np.ndarray   Time axis [s]
    g : np.ndarray   Irradiance [W/m²]
    t_cell : float   Cell temperature [°C] (assumed constant for transient)
    scan_period_s : float  MPPT update period [s]. Default: 0.1 s (10 Hz).
    dv_fraction : float    Perturbation as fraction of Vmpp_ref. Default: 0.5%.
    controller_delay_s : float  ADC measurement + control computation delay.
    n_strings, n_series : int   Array configuration.

    Returns
    -------
    v_actual : np.ndarray  Actual operating voltage [V]
    p_actual : np.ndarray  Actual operating power [W]
    p_theoretical : np.ndarray  Theoretical MPP power at same irradiance [W]
    """
    vmpp_t, impp_t, pmpp_t = mpp_from_irradiance(g, t_cell, n_strings, n_series)

    dv = dv_fraction * V_MP_REF * n_series

    v_actual = np.zeros_like(t)
    p_actual = np.zeros_like(t)

    # Initial state: start at STC Vmpp
    state = MPPTState(
        v_ref=V_MP_REF * n_series,
        v_prev=V_MP_REF * n_series,
        p_prev=float(pmpp_t[0]) if len(pmpp_t) > 0 else 0.0,
        dv=dv,
        scan_period=scan_period_s,
        t_last_scan=-scan_period_s,
    )

    for i, ts in enumerate(t):
        # Controller delay: use irradiance from (t - delay) for power measurement
        delay_idx = max(0, i - int(controller_delay_s * (1 / (t[1] - t[0]))))
        g_measured = g[delay_idx]

        # Current power at v_ref (simplified: linear I-V around MPP)
        # P = Vmpp_theoretical × Impp_theoretical × (v_ref / Vmpp_theoretical)
        # × (1 - 0.5*(v_ref/Vmpp_theoretical - 1)^2) — parabolic I-V approximation
        v_mp = float(vmpp_t[delay_idx])
        p_mp = float(pmpp_t[delay_idx])

        if v_mp > 0.1 and p_mp > 0.1:
            v_norm = state.v_ref / v_mp
            # Parabolic approximation: P(v) = Pmpp × (2*v_norm - v_norm²)
            # Valid near MPP; breaks down far from it
            v_norm_clamped = np.clip(v_norm, 0.5, 1.5)
            p_at_vref = p_mp * (2 * v_norm_clamped - v_norm_clamped ** 2)
        else:
            p_at_vref = 0.0

        v_actual[i] = state.v_ref
        p_actual[i] = max(0.0, p_at_vref)

        # P&O update: perturb voltage every scan_period
        if ts - state.t_last_scan >= scan_period_s:
            p_curr = p_at_vref
            p_prev = state.p_prev

            # Core P&O logic: move toward higher power
            if p_curr >= p_prev:
                # Power increased: continue in same direction
                if state.v_ref >= state.v_prev:
                    v_new = state.v_ref + dv
                else:
                    v_new = state.v_ref - dv
            else:
                # Power decreased: reverse direction
                if state.v_ref >= state.v_prev:
                    v_new = state.v_ref - dv
                else:
                    v_new = state.v_ref + dv

            # Voltage limits: [0.5*Vmpp_ref, 1.1*Voc_ref]
            v_min = 0.5 * V_MP_REF * n_series
            v_max = 1.1 * V_OC_REF * n_series
            v_new = np.clip(v_new, v_min, v_max)

            state.v_prev = state.v_ref
            state.p_prev = p_curr
            state.v_ref  = v_new
            state.t_last_scan = ts

    return v_actual, p_actual, pmpp_t


# ── MPPT efficiency analysis ──────────────────────────────────────────────────

@dataclass
class MPPTAnalysis:
    """MPPT tracking efficiency analysis results."""
    t: np.ndarray
    g: np.ndarray
    p_actual: np.ndarray
    p_theoretical: np.ndarray
    eta_mppt: np.ndarray           # Per-sample MPPT efficiency [0–1]
    eta_mean: float                # Mean efficiency over full window
    eta_during_transient: float    # Mean efficiency during cloud events
    eta_clear_sky: float           # Mean efficiency during clear periods
    energy_loss_pct: float         # Percentage energy loss vs perfect MPPT
    rocof_irr: np.ndarray          # Rate of change of irradiance [W/m²/s]
    transient_mask: np.ndarray     # Boolean: True during cloud shadow events
    events: List[CloudEvent]


def analyze_mppt_efficiency(
    t: np.ndarray,
    g: np.ndarray,
    p_actual: np.ndarray,
    p_theoretical: np.ndarray,
    events: List[CloudEvent],
    rocof_threshold_w_m2_s: float = 30.0,
) -> MPPTAnalysis:
    """
    Compute MPPT efficiency and energy loss statistics.

    Parameters
    ----------
    rocof_threshold_w_m2_s : float
        Irradiance rate-of-change threshold for "transient" classification [W/m²/s].
        Default: 30 W/m²/s (moderate cloud edge).

    Returns
    -------
    MPPTAnalysis
    """
    dt = float(np.mean(np.diff(t)))

    # MPPT efficiency per sample
    eta = np.where(
        p_theoretical > 10.0,
        np.clip(p_actual / p_theoretical, 0.0, 1.0),
        np.nan,
    )

    # Rate of change of irradiance (ROCOF_irr)
    rocof_irr = np.gradient(g, t)

    # Transient mask: |dG/dt| > threshold
    transient_from_rocof = np.abs(rocof_irr) > rocof_threshold_w_m2_s

    # Also mark shadow periods from event definitions
    transient_from_events = np.zeros(len(t), dtype=bool)
    for ev in events:
        mask = (t >= ev.t_start_s - ev.ramp_in_s) & \
               (t <= ev.t_start_s + ev.ramp_in_s + ev.duration_s + ev.ramp_out_s)
        transient_from_events |= mask

    transient_mask = transient_from_rocof | transient_from_events

    # Efficiency statistics (ignore NaN)
    valid = ~np.isnan(eta)
    eta_mean    = float(np.nanmean(eta[valid]))
    eta_trans   = float(np.nanmean(eta[valid & transient_mask])) if (valid & transient_mask).any() else np.nan
    eta_clear   = float(np.nanmean(eta[valid & ~transient_mask])) if (valid & ~transient_mask).any() else np.nan

    # Energy loss: (P_theoretical - P_actual) integrated
    e_theoretical = float(np.sum(p_theoretical[valid]) * dt)
    e_actual      = float(np.sum(p_actual[valid]) * dt)
    energy_loss_pct = 100.0 * (e_theoretical - e_actual) / (e_theoretical + 1e-9)

    return MPPTAnalysis(
        t=t,
        g=g,
        p_actual=p_actual,
        p_theoretical=p_theoretical,
        eta_mppt=eta,
        eta_mean=eta_mean,
        eta_during_transient=eta_trans,
        eta_clear_sky=eta_clear,
        energy_loss_pct=energy_loss_pct,
        rocof_irr=rocof_irr,
        transient_mask=transient_mask,
        events=events,
    )


def run_cloud_transient_analysis(
    clear_sky_g: float = 800.0,
    n_events: int = 5,
    duration_s: float = 600.0,
    scan_period_s: float = 0.1,
    dv_fraction: float = 0.005,
    seed: int = 42,
) -> MPPTAnalysis:
    """
    Full pipeline: generate cloud events → irradiance profile → P&O MPPT → analysis.

    Parameters
    ----------
    clear_sky_g : float  Clear-sky irradiance [W/m²].
    n_events : int       Number of cloud shadow events.
    duration_s : float   Total simulation duration [s].
    scan_period_s : float  MPPT scan period [s]. Default: 0.1 s (10 Hz).
    dv_fraction : float    P&O voltage step [fraction of Vmpp].

    Returns
    -------
    MPPTAnalysis
    """
    events = generate_cloud_events(duration_s, n_events, clear_sky_g, seed=seed)
    t, g = build_irradiance_profile(events, duration_s, fs=10.0,
                                     clear_sky_g=clear_sky_g, seed=seed)
    _, p_actual, p_theoretical = simulate_mppt_po(
        t, g, scan_period_s=scan_period_s, dv_fraction=dv_fraction
    )
    return analyze_mppt_efficiency(t, g, p_actual, p_theoretical, events)
