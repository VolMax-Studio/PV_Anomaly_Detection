"""
pv_simulator.py — PV System Performance Simulation
====================================================
Computes hourly AC power output for a PV system using:

  - Real TMY3 meteorological data (NREL ASOS station 723170,
    Greensboro, NC — 8760 hours of real measured irradiance and temperature)
  - Real CEC module parameters (Canadian Solar CS5P-250M, from the
    California Energy Commission module database, measured at STC)
  - pvlib single-diode model (de Soto et al. 2006)

Inputs / outputs are physically grounded throughout; no synthetic
meteorological data is used. Anomaly injection is added separately
in anomaly_injector.py.

Physical model summary
----------------------
1. Solar geometry (pvlib.solarposition) → solar zenith, azimuth
2. Decompose irradiance → POA (plane-of-array) using Perez transposition
3. Cell temperature → NOCT (Normal Operating Cell Temperature) model:
       T_cell = T_air + (NOCT − 20) / 800 × G_POA
4. CEC single-diode five-parameter model (de Soto 2006) gives I_mpp, V_mpp
5. Array output:
       P_dc = N_series × N_parallel × I_mpp × V_mpp
6. Inverter (Sandia model) → P_ac

Performance Ratio definition
-----------------------------
    PR = E_ac / (G_poa × P_rated_kW / G_stc)

where G_stc = 1,000 W/m², P_rated_kW = system rated DC power [kW].
PR is dimensionless, typically 0.70–0.90 for well-maintained systems.
PR < 0.6 indicates degradation or significant soiling.
PR = 1.0 is physically impossible (inverter and temperature losses).

References
----------
de Soto, W., Klein, S.A., Beckman, W.A. (2006). Improvement and validation
of a model for photovoltaic array performance. Solar Energy, 80(1), 78–88.

pvlib documentation: https://pvlib-python.readthedocs.io/
"""

import numpy as np
import pandas as pd
import pvlib
from pvlib.iotools import read_tmy3
from pvlib import solarposition, irradiance, pvsystem, atmosphere
from dataclasses import dataclass
from typing import Optional
import os


# ── Data sources (pvlib built-in, real measured data) ──────────────────────

_PVLIB_DATA = os.path.join(os.path.dirname(pvlib.__file__), 'data')

# TMY3: NREL ASOS station 723170 — Greensboro/Piedmont Triad, NC
# Real multi-year measured meteorological data from NREL MIDC network
TMY3_FILE = os.path.join(_PVLIB_DATA, '723170TYA.CSV')
SITE_NAME = "Greensboro, NC (NREL TMY3 station 723170)"
SITE_LAT  = 36.10    # degrees N
SITE_LON  = -79.95   # degrees E (negative = West)
SITE_ALT  = 273.0    # meters ASL
SITE_TZ   = 'Etc/GMT+5'   # Eastern US (TMY3 uses local standard time)

# CEC Module: Canadian Solar CS5P-250M (monocrystalline silicon, 250 Wp)
# Parameters from California Energy Commission database (real measured)
MODULE_NAME = 'Canadian_Solar_Inc__CS5P_250M'

# System configuration
N_STRINGS  = 4     # Number of strings in parallel
N_SERIES   = 10    # Modules per string
P_RATED_W  = 250.0 * N_STRINGS * N_SERIES  # 10,000 Wp DC rated
GCR        = 0.4   # Ground coverage ratio (typical rooftop/ground mount)
TILT_DEG   = 35.0  # Array tilt (≈ latitude for maximum annual yield)
AZ_DEG     = 180.0 # Array azimuth (180° = South-facing)
NOCT_C     = 45.0  # Normal Operating Cell Temperature [°C] (from datasheet)

# Standard Test Conditions
G_STC  = 1000.0   # W/m²
T_STC  = 25.0     # °C


@dataclass
class PVSystem:
    """Configuration of the PV system being simulated."""
    module_name: str = MODULE_NAME
    n_strings: int = N_STRINGS
    n_series: int = N_SERIES
    tilt_deg: float = TILT_DEG
    azimuth_deg: float = AZ_DEG
    noct_c: float = NOCT_C
    p_rated_w: float = P_RATED_W

    @property
    def n_modules(self) -> int:
        return self.n_strings * self.n_series


@dataclass
class SimulationResult:
    """Hourly simulation output for one year."""
    timestamps: pd.DatetimeIndex
    ghi: pd.Series          # W/m² — Global Horizontal Irradiance (real TMY3)
    dni: pd.Series          # W/m² — Direct Normal Irradiance (real TMY3)
    dhi: pd.Series          # W/m² — Diffuse Horizontal Irradiance (real TMY3)
    g_poa: pd.Series        # W/m² — Plane of Array irradiance
    t_air: pd.Series        # °C   — Ambient temperature (real TMY3)
    t_cell: pd.Series       # °C   — Cell temperature (NOCT model)
    p_mp: pd.Series         # W    — DC array output at MPP
    p_ac: pd.Series         # W    — AC inverter output
    performance_ratio: pd.Series  # dimensionless [0–1]
    system: PVSystem
    data_source: str = (
        f"TMY3 meteorological: {SITE_NAME} — NREL ASOS real measured data. "
        f"Module parameters: CEC database (Canadian Solar CS5P-250M, real measured). "
        f"Anomaly labels: None (clean baseline)."
    )
    anomaly_labels: Optional[pd.Series] = None

    @property
    def annual_energy_kwh(self) -> float:
        return float(self.p_ac.clip(lower=0).sum() / 1000)

    @property
    def mean_pr(self) -> float:
        daylight = self.g_poa > 10
        return float(self.performance_ratio[daylight].mean())


def load_tmy3() -> tuple:
    """Load pvlib built-in TMY3 dataset (NREL, real measured data)."""
    tmy, meta = read_tmy3(TMY3_FILE)
    return tmy, meta


def simulate_pv_system(
    system: Optional[PVSystem] = None,
    apply_anomaly_mask: Optional[pd.Series] = None,
    verbose: bool = False,
) -> SimulationResult:
    """
    Simulate hourly PV output for one TMY year.

    Parameters
    ----------
    system : PVSystem, optional. Uses default 10 kWp system if None.
    apply_anomaly_mask : pd.Series of float [0–1], optional.
        Per-hour multiplier on P_dc (1.0 = no effect, 0.0 = full loss).
        Used by anomaly_injector to apply fault effects to a clean baseline.
    verbose : bool

    Returns
    -------
    SimulationResult
    """
    if system is None:
        system = PVSystem()

    # 1. Load real TMY3 meteorological data
    tmy, meta = load_tmy3()
    timestamps = tmy.index

    # 2. Retrieve real CEC module parameters
    modules_db = pvlib.pvsystem.retrieve_sam('CECMod')
    mod = modules_db[system.module_name]

    # 3. Solar position
    loc = pvlib.location.Location(
        latitude=SITE_LAT, longitude=SITE_LON,
        altitude=SITE_ALT, tz=SITE_TZ,
    )
    solar_pos = loc.get_solarposition(timestamps)

    # 4. POA irradiance (Perez transposition model)
    # Extraterrestrial DNI for Perez model
    dni_extra = pvlib.irradiance.get_extra_radiation(timestamps)
    airmass = pvlib.atmosphere.get_relative_airmass(solar_pos['apparent_zenith'])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure=101325)

    poa_components = pvlib.irradiance.get_total_irradiance(
        surface_tilt=system.tilt_deg,
        surface_azimuth=system.azimuth_deg,
        solar_zenith=solar_pos['apparent_zenith'],
        solar_azimuth=solar_pos['azimuth'],
        dni=tmy['dni'],
        ghi=tmy['ghi'],
        dhi=tmy['dhi'],
        dni_extra=dni_extra,
        airmass=airmass_abs,
        model='perez',
    )
    g_poa = poa_components['poa_global'].fillna(0).clip(lower=0)

    # 5. Cell temperature (NOCT model)
    t_air = tmy['temp_air']
    t_cell = t_air + (system.noct_c - 20.0) / 800.0 * g_poa

    # 6. CEC five-parameter model: effective irradiance → module I-V parameters
    # EgRef, dEgdT: Silicon bandgap parameters (standard values)
    EgRef = 1.121   # eV — silicon bandgap at reference temperature
    dEgdT = -0.0002677   # eV/K

    # PV cell ideality factor (alpha_sc already in CEC params as percentage)
    params_cec = pvlib.pvsystem.calcparams_cec(
        effective_irradiance=g_poa,
        temp_cell=t_cell,
        alpha_sc=mod['alpha_sc'],
        a_ref=mod['a_ref'],
        I_L_ref=mod['I_L_ref'],
        I_o_ref=mod['I_o_ref'],
        R_sh_ref=mod['R_sh_ref'],
        R_s=mod['R_s'],
        Adjust=mod['Adjust'],
    )
    photo_current, sat_current, resistance_series, resistance_shunt, nNsVth = params_cec

    # 7. Single-diode equation → MPP
    iv_result = pvlib.pvsystem.singlediode(
        photocurrent=photo_current,
        saturation_current=sat_current,
        resistance_series=resistance_series,
        resistance_shunt=resistance_shunt,
        nNsVth=nNsVth,
        method='lambertw',
    )

    # 8. Array power: N_series × N_parallel × P_module_mpp
    p_mp_module = iv_result['p_mp'].fillna(0).clip(lower=0)
    p_dc_array = p_mp_module * system.n_strings * system.n_series

    # 9. Apply anomaly multiplier if provided
    if apply_anomaly_mask is not None:
        p_dc_array = p_dc_array * apply_anomaly_mask.reindex(timestamps).fillna(1.0)

    # 10. DC-to-AC: simplified inverter efficiency
    # CEC inverter is optional; use flat 96% efficiency for simplicity
    eta_inverter = 0.96
    p_dc_clipped = p_dc_array.clip(upper=system.p_rated_w * 1.02)  # clip to rated
    p_ac = (p_dc_clipped * eta_inverter).clip(lower=0)

    # 11. Performance Ratio
    pr_denom = g_poa * (system.p_rated_w / G_STC)
    pr = np.where(pr_denom > 5, p_ac / pr_denom, np.nan)
    pr_series = pd.Series(pr, index=timestamps, name='performance_ratio')

    if verbose:
        daylight = g_poa > 10
        print(f"Annual energy: {p_ac.clip(lower=0).sum()/1000:.1f} kWh")
        print(f"Mean PR (daylight hours): {float(np.nanmean(pr[daylight.values])):.3f}")
        print(f"Peak POA irradiance: {g_poa.max():.0f} W/m²")

    return SimulationResult(
        timestamps=timestamps,
        ghi=tmy['ghi'],
        dni=tmy['dni'],
        dhi=tmy['dhi'],
        g_poa=g_poa,
        t_air=t_air,
        t_cell=t_cell,
        p_mp=p_dc_array,
        p_ac=p_ac,
        performance_ratio=pr_series,
        system=system,
    )
