"""
pv_analyzer.py — PV Anomaly Detection and Performance Analysis
==============================================================
Three analysis modules:

1. Performance Ratio (PR) monitoring
   PR = E_ac / (G_poa × P_rated / G_stc)
   Rolling daily PR highlights gradual degradation (soiling, PID).

2. MPPT Efficiency tracking
   η_MPPT = P_actual / P_theoretical
   P_theoretical = P_rated × (G_poa/G_stc) × (1 + γ_pmp × (T_cell − T_stc))
   Deviation from expected indicates inverter MPPT algorithm degradation.

3. Isolation Forest anomaly detection
   Features: [PR, normalized_power, temperature_corrected_power, G_poa_norm]
   Trained on clean hours (first 20% of year = winter/spring baseline).
   Flags anomalous hours via contamination parameter.
   Precision/recall reported against injected ground-truth labels.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from dataclasses import dataclass
from typing import Optional, Tuple


# CEC module temperature coefficient (CS5P-250M from datasheet)
GAMMA_PMP = -0.0041   # fraction/°C (−0.41%/°C — typical monocrystalline)
G_STC     = 1000.0    # W/m²
T_STC     = 25.0      # °C
MIN_IRRADIANCE_FOR_ANALYSIS = 50.0  # W/m² — below this, don't compute PR/efficiency


@dataclass
class PRAnalysis:
    """Performance Ratio analysis results."""
    hourly_pr: pd.Series
    daily_pr: pd.Series           # Daily average PR (daylight hours only)
    rolling_pr_7d: pd.Series      # 7-day rolling mean
    pr_baseline: float            # Expected PR (first 30 clean days)
    pr_mean: float                # Mean PR over analysis period
    pr_trend_per_day: float       # Linear trend [PR_units/day] — negative = degradation
    n_low_pr_days: int            # Days with PR < 0.6 × baseline


@dataclass
class DetectionResult:
    """Isolation Forest detection results."""
    anomaly_scores: pd.Series     # [-1, 1] → anomaly = -1 in sklearn convention
    anomaly_flags: pd.Series      # Boolean: True = anomaly detected
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    n_detected: int
    n_false_positives: int
    n_false_negatives: int
    feature_names: list


def compute_pr(
    p_ac: pd.Series,
    g_poa: pd.Series,
    p_rated_w: float,
    min_g_poa: float = MIN_IRRADIANCE_FOR_ANALYSIS,
) -> pd.Series:
    """
    Compute hourly Performance Ratio.

    PR is only meaningful when there is sufficient irradiance. Returns NaN
    for low-irradiance hours (night, heavy overcast).
    """
    pr_denom = g_poa * (p_rated_w / G_STC)
    pr = np.where(
        g_poa > min_g_poa,
        p_ac / pr_denom,
        np.nan,
    )
    return pd.Series(pr, index=p_ac.index, name='pr')


def compute_expected_power(
    g_poa: pd.Series,
    t_cell: pd.Series,
    p_rated_w: float,
    gamma_pmp: float = GAMMA_PMP,
) -> pd.Series:
    """
    Temperature-corrected expected DC power from irradiance.

    P_expected = P_rated × (G_poa/G_stc) × (1 + γ_pmp × (T_cell − T_stc))

    This is the single-diode model linearized around STC. Valid when
    G_poa > 100 W/m² and T_cell ∈ [−10, 75°C].
    """
    temp_factor = 1.0 + gamma_pmp * (t_cell - T_STC)
    p_exp = p_rated_w * (g_poa / G_STC) * temp_factor
    return p_exp.clip(lower=0).rename('p_expected_w')


def pr_analysis(
    p_ac: pd.Series,
    g_poa: pd.Series,
    p_rated_w: float,
    baseline_days: int = 30,
) -> PRAnalysis:
    """
    Full PR analysis: daily trends, baseline establishment, degradation detection.

    Parameters
    ----------
    baseline_days : int
        Number of days at the start of the series used as clean reference.
    """
    pr_hourly = compute_pr(p_ac, g_poa, p_rated_w)

    # Daily PR: mean of daylight hours (ignores NaN)
    pr_daily = pr_hourly.resample('D').mean()
    pr_daily = pr_daily.dropna()

    # Baseline: mean PR over first baseline_days days
    pr_baseline = float(pr_daily.iloc[:baseline_days].mean()) if len(pr_daily) >= baseline_days else float(pr_daily.mean())

    # Rolling 7-day mean for slow trend visualization
    pr_7d = pr_daily.rolling(7, min_periods=1).mean()

    # Linear trend via least squares
    if len(pr_daily) > 1:
        days_num = np.arange(len(pr_daily))
        valid = ~pr_daily.isna()
        if valid.sum() > 2:
            coeffs = np.polyfit(days_num[valid], pr_daily.values[valid], 1)
            trend = float(coeffs[0])
        else:
            trend = 0.0
    else:
        trend = 0.0

    n_low = int((pr_daily < 0.6 * pr_baseline).sum())

    return PRAnalysis(
        hourly_pr=pr_hourly,
        daily_pr=pr_daily,
        rolling_pr_7d=pr_7d,
        pr_baseline=pr_baseline,
        pr_mean=float(pr_daily.mean()),
        pr_trend_per_day=trend,
        n_low_pr_days=n_low,
    )


def build_features(
    p_ac: pd.Series,
    g_poa: pd.Series,
    t_cell: pd.Series,
    p_rated_w: float,
) -> pd.DataFrame:
    """
    Build feature matrix for Isolation Forest anomaly detection.

    Features chosen to capture multiple fault modes:
    - pr:              catches soiling, PID, partial shading (long-duration)
    - power_residual:  |P_actual - P_expected| / P_expected — catches all faults
    - t_corrected_pr:  PR after temperature correction — isolates thermal effects
    - g_norm:          G_poa normalized — filters irradiance variation

    Only includes daylight hours with irradiance > threshold.
    """
    daylight = g_poa > MIN_IRRADIANCE_FOR_ANALYSIS

    pr = compute_pr(p_ac, g_poa, p_rated_w)
    p_exp = compute_expected_power(g_poa, t_cell, p_rated_w)

    # Power residual (signed: negative = underperforming)
    power_residual = (p_ac - p_exp) / (p_exp + 1e-3)

    # Temperature-corrected power ratio
    temp_factor = (1.0 + GAMMA_PMP * (t_cell - T_STC)).clip(lower=0.5)
    p_t_corrected = p_ac / (p_rated_w * temp_factor * g_poa / G_STC + 1e-3)

    # Normalized irradiance (z-score over daylight hours only)
    g_norm = g_poa.copy()
    g_mean = float(g_poa[daylight].mean())
    g_std  = float(g_poa[daylight].std())
    g_norm = (g_poa - g_mean) / (g_std + 1e-3)

    features = pd.DataFrame({
        'pr':               pr,
        'power_residual':   power_residual,
        't_corrected_pr':   p_t_corrected,
        'g_norm':           g_norm,
    })

    # Return only daylight rows with no NaN
    features = features[daylight].dropna()
    return features


def detect_anomalies(
    p_ac: pd.Series,
    g_poa: pd.Series,
    t_cell: pd.Series,
    p_rated_w: float,
    ground_truth: Optional[pd.Series] = None,
    training_fraction: float = 0.20,
    contamination: float = 0.05,
    random_state: int = 42,
) -> DetectionResult:
    """
    Isolation Forest anomaly detection on PV performance features.

    Parameters
    ----------
    training_fraction : float
        Fraction of data (first N hours) used for training. Should be from
        a period known to be clean (first winter/spring months).
    contamination : float
        Expected fraction of anomalous samples. Set to approximate fault rate.
        Default: 0.05 (5%) — conservative for real PV datasets.
    ground_truth : pd.Series of bool, optional
        Per-hour fault labels for precision/recall calculation (evaluation only).
    """
    features = build_features(p_ac, g_poa, t_cell, p_rated_w)

    if len(features) < 100:
        raise ValueError(f"Insufficient daylight samples: {len(features)} < 100")

    # Train on first training_fraction of daylight hours (assumed clean)
    n_train = int(len(features) * training_fraction)
    X_train = features.iloc[:n_train]
    X_all   = features

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled   = scaler.transform(X_all)

    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_scaled)

    # Predictions: -1 = anomaly, 1 = normal
    predictions = iso_forest.predict(X_all_scaled)
    scores      = iso_forest.score_samples(X_all_scaled)

    anomaly_flags = pd.Series(predictions == -1, index=features.index, name='anomaly')
    anomaly_scores = pd.Series(scores, index=features.index, name='anomaly_score')

    # Precision / recall against ground truth (if provided)
    precision_val = recall_val = f1_val = None
    n_fp = n_fn = 0

    if ground_truth is not None:
        gt_aligned = ground_truth.reindex(features.index).fillna(False).astype(int)
        pred_int   = anomaly_flags.astype(int)

        if gt_aligned.sum() > 0:
            precision_val = float(precision_score(gt_aligned, pred_int, zero_division=0))
            recall_val    = float(recall_score(gt_aligned, pred_int, zero_division=0))
            f1_val        = float(f1_score(gt_aligned, pred_int, zero_division=0))
            n_fp = int(((pred_int == 1) & (gt_aligned == 0)).sum())
            n_fn = int(((pred_int == 0) & (gt_aligned == 1)).sum())

    return DetectionResult(
        anomaly_scores=anomaly_scores,
        anomaly_flags=anomaly_flags,
        precision=precision_val,
        recall=recall_val,
        f1=f1_val,
        n_detected=int(anomaly_flags.sum()),
        n_false_positives=n_fp,
        n_false_negatives=n_fn,
        feature_names=list(features.columns),
    )
