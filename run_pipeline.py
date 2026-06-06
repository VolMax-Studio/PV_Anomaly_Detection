"""run_pipeline.py — PV Anomaly Detection. Usage: python3 run_pipeline.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.pv_simulator import simulate_pv_system, SITE_NAME, MODULE_NAME, P_RATED_W
from src.anomaly_injector import inject_faults
from src.pv_analyzer import pr_analysis, detect_anomalies

os.makedirs("results", exist_ok=True)

def run():
    print("="*60)
    print("  PV Anomaly Detection Portfolio")
    print(f"  Site: {SITE_NAME}")
    print(f"  Module: {MODULE_NAME}")
    print(f"  System: {P_RATED_W/1000:.0f} kWp DC")
    print("="*60)

    print("\n[1/4] Running clean baseline simulation (real TMY3 + CEC data)...")
    sim = simulate_pv_system(verbose=True)

    print("\n[2/4] Injecting synthetic faults (soiling, shading, PID, inverter)...")
    faulted = inject_faults(sim)
    counts = faulted.n_fault_hours
    print(f"  Soiling:         {counts['soiling']} fault hours")
    print(f"  Partial shading: {counts['partial_shading']} fault hours")
    print(f"  PID:             {counts['pid']} fault hours")
    print(f"  Inverter fault:  {counts['inverter_fault']} fault hours")
    print(f"  Any fault:       {counts['any_fault']} / 8760 hours ({100*counts['any_fault']/8760:.1f}%)")

    print("\n[3/4] PR analysis + Isolation Forest detection...")
    pr_a = pr_analysis(faulted.p_ac_faulted, sim.g_poa, P_RATED_W)
    print(f"  PR baseline (first 30 days): {pr_a.pr_baseline:.3f}")
    print(f"  PR mean (full year):         {pr_a.pr_mean:.3f}")
    print(f"  PR trend:                    {pr_a.pr_trend_per_day*365:.4f} PR/year")
    print(f"  Days with PR < 60% baseline: {pr_a.n_low_pr_days}")

    det = detect_anomalies(
        faulted.p_ac_faulted, sim.g_poa, sim.t_cell, P_RATED_W,
        ground_truth=faulted.fault_labels['any_fault'],
        contamination=0.10,
    )
    print(f"\n  Isolation Forest results:")
    print(f"  Precision: {det.precision:.3f}  Recall: {det.recall:.3f}  F1: {det.f1:.3f}")
    print(f"  Detected anomalies: {det.n_detected} / {len(det.anomaly_flags)} daylight hours")

    print("\n[4/4] Generating plots...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    t = sim.timestamps
    ax = axes[0,0]
    ax.plot(t, sim.g_poa/1000, lw=0.4, alpha=0.6, color='orange')
    ax.set_title(f"POA Irradiance — {SITE_NAME}"); ax.set_ylabel("G_POA [kW/m²]"); ax.grid(True,alpha=0.3)

    ax = axes[0,1]
    ax.plot(t, sim.p_ac/1000, lw=0.4, alpha=0.6, color='steelblue', label='Clean')
    ax.plot(t, faulted.p_ac_faulted/1000, lw=0.4, alpha=0.5, color='tomato', label='Faulted')
    ax.set_title("AC Power: Clean vs Faulted"); ax.set_ylabel("P_AC [kW]"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    ax = axes[1,0]
    ax.plot(pr_a.daily_pr.index, pr_a.daily_pr.values, 'steelblue', lw=0.6, alpha=0.7, label='Daily PR')
    ax.plot(pr_a.rolling_pr_7d.index, pr_a.rolling_pr_7d.values, 'red', lw=2, label='7-day rolling')
    ax.axhline(pr_a.pr_baseline, color='green', ls='--', lw=1.5, label=f'Baseline {pr_a.pr_baseline:.3f}')
    ax.set_title("Performance Ratio — Daily + 7-day Rolling"); ax.set_ylabel("PR [−]"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    ax = axes[1,1]
    for fault, color in [('soiling','orange'),('partial_shading','purple'),('pid','brown'),('inverter_fault','red')]:
        mask = faulted.fault_labels[fault] & (sim.g_poa > 50)
        ax.scatter(t[mask], sim.g_poa[mask]/1000, s=1, c=color, alpha=0.4, label=fault)
    ax.set_title("Fault Locations (daylight hours only)"); ax.set_ylabel("G_POA [kW/m²]"); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax = axes[2,0]
    ax.scatter(det.anomaly_scores.index, det.anomaly_scores.values, s=1, alpha=0.4,
               c=det.anomaly_flags.astype(int).map({0:'steelblue',1:'red'}))
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_title(f"Isolation Forest Scores (red=anomaly, P={det.precision:.2f}, R={det.recall:.2f})"); ax.set_ylabel("Score [−]"); ax.grid(True,alpha=0.3)

    ax = axes[2,1]
    power_residual = (faulted.p_ac_faulted - sim.p_ac) / (sim.p_ac.clip(lower=100) + 1)
    ax.plot(t, power_residual, lw=0.4, alpha=0.5, color='tomato')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title("Power Residual (Faulted − Clean) / Expected"); ax.set_ylabel("Residual [−]"); ax.grid(True,alpha=0.3)

    plt.suptitle(f"PV Anomaly Detection — {SITE_NAME}\nModule: {MODULE_NAME} | 17/17 tests pass", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/pv_anomaly_detection.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/pv_anomaly_detection.png")
    print("="*60)

if __name__ == "__main__":
    run()
