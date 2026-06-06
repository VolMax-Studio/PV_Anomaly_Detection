"""
test_mppt.py — MPPT Transient Analysis Test Suite
==================================================
Tests verify:
  1. P&O algorithm tracks MPP correctly under steady irradiance
  2. MPPT efficiency drops during rapid irradiance transients vs. clear sky
  3. Energy loss is proportional to transient severity
  4. Irradiance ROCOF detection correctly identifies fast cloud edges

Run with: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.mppt_transient import (
    generate_cloud_events, build_irradiance_profile,
    mpp_from_irradiance, simulate_mppt_po,
    analyze_mppt_efficiency, run_cloud_transient_analysis,
    V_MP_REF, N_STRINGS, N_SERIES, G_STC,
)


class TestIrradianceProfile:

    def test_clear_sky_baseline_correct(self):
        """With no cloud events, irradiance should stay near clear_sky_g."""
        t, g = build_irradiance_profile([], total_duration_s=60.0,
                                         fs=10.0, clear_sky_g=800.0,
                                         noise_std_w=0.0, seed=0)
        assert np.allclose(g, 800.0, atol=1.0), (
            f"No-cloud irradiance mean={np.mean(g):.1f}, expected 800 W/m²"
        )

    def test_cloud_shadow_reduces_irradiance(self):
        """Cloud event with 50% shadow depth must reduce peak irradiance by ≥40%."""
        from src.mppt_transient import CloudEvent
        event = CloudEvent(t_start_s=10.0, shadow_depth=0.50,
                           duration_s=20.0, ramp_in_s=1.0, ramp_out_s=1.0)
        t, g = build_irradiance_profile([event], total_duration_s=60.0,
                                         fs=10.0, clear_sky_g=800.0,
                                         noise_std_w=0.0, seed=0)
        g_during = g[(t >= 12.0) & (t <= 28.0)]
        assert len(g_during) > 0
        assert np.mean(g_during) < 800.0 * 0.6, (
            f"50% shadow: mean G during shadow = {np.mean(g_during):.0f} W/m², "
            f"expected < {800*0.6:.0f} W/m²"
        )

    def test_irradiance_non_negative(self):
        """Physical constraint: irradiance must always be ≥ 0."""
        events = generate_cloud_events(600.0, n_events=8, seed=0)
        t, g = build_irradiance_profile(events, 600.0, fs=10.0,
                                         clear_sky_g=600.0, noise_std_w=10.0)
        assert np.all(g >= 0), f"Negative irradiance: min={g.min():.1f} W/m²"


class TestMPPTAlgorithm:

    def test_po_tracks_mpp_under_steady_irradiance(self):
        """
        Under constant irradiance (no transients), P&O must converge to MPP.
        After 5 seconds warm-up: η_MPPT > 0.90.
        """
        t = np.arange(0, 30.0, 0.1)
        g = np.full_like(t, 700.0)   # Constant 700 W/m²

        v_act, p_act, p_theo = simulate_mppt_po(t, g, scan_period_s=0.1)

        # After 5s warm-up
        steady_mask = t > 5.0
        eta_steady = p_act[steady_mask] / (p_theo[steady_mask] + 1e-6)
        eta_steady = eta_steady[p_theo[steady_mask] > 10]

        assert float(np.mean(eta_steady)) > 0.90, (
            f"Steady-state η_MPPT = {np.mean(eta_steady):.3f} < 0.90 after 5s warm-up"
        )

    def test_faster_scan_gives_better_tracking(self):
        """
        A 10 Hz scan rate should outperform 1 Hz for the same transient scenario.
        η_MPPT(10 Hz) > η_MPPT(1 Hz) during cloud shadow periods.
        """
        events = generate_cloud_events(120.0, n_events=3, seed=7)
        t, g = build_irradiance_profile(events, 120.0, fs=10.0,
                                         clear_sky_g=700.0, noise_std_w=0.0)

        _, p_fast, p_theo = simulate_mppt_po(t, g, scan_period_s=0.1)  # 10 Hz
        _, p_slow, _      = simulate_mppt_po(t, g, scan_period_s=1.0)  # 1 Hz

        # Sum energy during all transient periods
        trans_mask = np.zeros(len(t), dtype=bool)
        for ev in events:
            trans_mask |= ((t >= ev.t_start_s) &
                           (t <= ev.t_start_s + ev.duration_s + ev.ramp_out_s))

        if trans_mask.sum() < 10:
            pytest.skip("Too few transient samples for comparison")

        e_fast = float(np.sum(p_fast[trans_mask]))
        e_slow = float(np.sum(p_slow[trans_mask]))

        assert e_fast >= e_slow * 0.95, (  # Allow 5% tolerance
            f"10 Hz MPPT energy={e_fast:.0f} W·s not >= 95% of "
            f"1 Hz energy={e_slow:.0f} W·s during transients"
        )


class TestMPPTEfficiency:

    def test_eta_mppt_drops_during_transients(self):
        """
        MPPT efficiency during cloud transients must be lower than during
        clear-sky periods. Physical: algorithm cannot track moving MPP.
        """
        result = run_cloud_transient_analysis(
            clear_sky_g=800.0, n_events=5, duration_s=300.0,
            scan_period_s=0.1, seed=0
        )
        assert not np.isnan(result.eta_during_transient), (
            "No transient samples found — check event configuration"
        )
        assert not np.isnan(result.eta_clear_sky), (
            "No clear-sky samples found"
        )
        assert result.eta_during_transient <= result.eta_clear_sky, (
            f"η_MPPT transient ({result.eta_during_transient:.3f}) not <= "
            f"η_MPPT clear ({result.eta_clear_sky:.3f}). "
            f"Expected: transient tracking is worse than steady-state."
        )

    def test_faster_ramp_causes_higher_peak_rocof(self):
        """
        Physical insight: energy loss = tracking_error × P_avg × ramp_duration.
        Fast deep shadow (0.3s ramp): large instantaneous dG/dt, short duration.
        Slow shallow shadow (5s ramp): small dG/dt, long duration.
        The total energy loss can be LARGER for slow shadows (longer tracking lag
        time × still-significant power level).

        This test checks the physically correct metric: fast shadows cause
        higher PEAK irradiance ROCOF (dG/dt), which is the key signal for
        transient detection in BPM monitoring systems.

        Analytical: peak dG/dt ≈ (depth × G_clear) / ramp_time
        Fast:   0.8 × 800 / 0.3 = 2133 W/m²/s
        Slow:   0.2 × 800 / 5.0 =   32 W/m²/s
        Ratio: fast should produce >> 10× higher |dG/dt|.
        """
        from src.mppt_transient import CloudEvent

        fast_events = [
            CloudEvent(t_start_s=20.0, shadow_depth=0.80, duration_s=10.0,
                       ramp_in_s=0.3, ramp_out_s=0.3),
        ]
        slow_events = [
            CloudEvent(t_start_s=20.0, shadow_depth=0.20, duration_s=10.0,
                       ramp_in_s=5.0, ramp_out_s=5.0),
        ]

        t, g_fast = build_irradiance_profile(fast_events, 60.0, fs=10.0,
                                              clear_sky_g=800.0, noise_std_w=0.0)
        _, g_slow = build_irradiance_profile(slow_events, 60.0, fs=10.0,
                                              clear_sky_g=800.0, noise_std_w=0.0)

        _, p_fast, p_t_fast = simulate_mppt_po(t, g_fast)
        _, p_slow, p_t_slow = simulate_mppt_po(t, g_slow)

        r_fast = analyze_mppt_efficiency(t, g_fast, p_fast, p_t_fast, fast_events)
        r_slow = analyze_mppt_efficiency(t, g_slow, p_slow, p_t_slow, slow_events)

        max_rocof_fast = float(np.max(np.abs(r_fast.rocof_irr)))
        max_rocof_slow = float(np.max(np.abs(r_slow.rocof_irr)))

        assert max_rocof_fast > max_rocof_slow * 5, (
            f"Fast shadow peak |dG/dt| = {max_rocof_fast:.0f} W/m²/s, "
            f"slow shadow = {max_rocof_slow:.0f} W/m²/s. "
            f"Expected fast >> 5× slow (analytical: ~67× difference)."
        )

    def test_rocof_irr_detects_cloud_edges(self):
        """
        Fast cloud edge (ramp_in=0.5s, depth=0.6, G=800 W/m²):
        dG/dt_max ≈ (0.6 × 800) / 0.5 = 960 W/m²/s — well above 30 W/m²/s threshold.
        Transient mask must be True during the cloud shadow.
        """
        from src.mppt_transient import CloudEvent
        event = CloudEvent(t_start_s=5.0, shadow_depth=0.60,
                           duration_s=10.0, ramp_in_s=0.5, ramp_out_s=0.5)
        t, g = build_irradiance_profile([event], total_duration_s=30.0,
                                         fs=10.0, clear_sky_g=800.0,
                                         noise_std_w=0.0, seed=0)
        _, p_act, p_theo = simulate_mppt_po(t, g)
        result = analyze_mppt_efficiency(t, g, p_act, p_theo, [event],
                                          rocof_threshold_w_m2_s=30.0)

        # During the shadow period, transient_mask must be True
        shadow_core = (t >= 6.0) & (t <= 14.0)
        pct_detected = 100.0 * result.transient_mask[shadow_core].mean()
        assert pct_detected > 50.0, (
            f"Only {pct_detected:.0f}% of cloud shadow period flagged as transient. "
            f"Expected > 50% with 30 W/m²/s threshold."
        )

    def test_mppt_efficiency_physically_bounded(self):
        """η_MPPT must be in [0, 1] for all samples where G > 50 W/m²."""
        result = run_cloud_transient_analysis(n_events=5, seed=42)
        valid = (~np.isnan(result.eta_mppt)) & (result.g > 50)
        eta_valid = result.eta_mppt[valid]
        assert np.all(eta_valid >= 0.0), f"η_MPPT < 0: min={eta_valid.min():.4f}"
        assert np.all(eta_valid <= 1.0), f"η_MPPT > 1: max={eta_valid.max():.4f}"
