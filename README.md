# PV Anomaly Detection Portfolio

Physics-based PV system performance simulation with fault injection and Isolation Forest anomaly detection.

**Data sources: real NREL TMY3 meteorological measurements + real CEC module parameters from California Energy Commission database. Fault injection: synthetic with ground-truth labels.**

---

## Data

| Source | Type | Reference |
|--------|------|-----------|
| TMY3 meteorological (GHI, DNI, DHI, T_air) | **Real measured** — NREL ASOS station 723170, Greensboro, NC | NREL MIDC |
| Module parameters | **Real measured** — Canadian Solar CS5P-250M from CEC Module Database | pvlib built-in |
| Fault events (soiling, shading, PID, inverter) | **Synthetic** — injected with known labels | This repo |

TMY3 is a Typical Meteorological Year derived from multi-year real measurements, used as the global standard for PV yield analysis (IEC 61853-3).

---

## System configuration

- **Site:** Greensboro, NC — 36.1°N, 79.95°W, 273 m ASL
- **Module:** Canadian Solar CS5P-250M (250 Wp monocrystalline silicon)
- **System:** 10 strings × 4 parallel = 40 modules, 10 kWp DC rated
- **Tilt / Azimuth:** 35° / 180° (south-facing, fixed)
- **Inverter efficiency:** 96% (flat, simplified)

---

## Physics model

**POA irradiance** — Perez transposition model (most accurate diffuse model):  
`G_POA = G_beam + G_sky_diffuse + G_ground_reflected`

**Cell temperature** — NOCT model:  
`T_cell = T_air + (NOCT − 20) / 800 × G_POA`  (NOCT = 45°C for this module)

**Module output** — CEC five-parameter single-diode model (de Soto et al. 2006)  
→ pvlib `calcparams_cec` + `singlediode` → I_mpp, V_mpp per module

**Performance Ratio:**  
`PR = P_AC / (G_POA × P_rated / 1000)` — dimensionless, 0.70–0.95 typical

**Temperature-corrected expected power:**  
`P_exp = P_rated × (G_POA/1000) × (1 + γ_pmp × (T_cell − 25))`  
γ_pmp = −0.41%/°C for CS5P-250M

---

## Fault types injected (synthetic)

| Fault | Physical mechanism | Detection signature |
|-------|-------------------|---------------------|
| **Soiling** | Dust/pollen on glass → reduced Isc | Gradual PR drop; restores after rain |
| **Partial shading** | 1 string shaded → bypass diodes activate | Step-change in P_dc; Vmpp shift |
| **PID** | Voltage-driven leakage → cell degradation | Slow linear PR decline over months |
| **Inverter fault** | Trip/overtemperature → P_ac = 0 | Sudden zero output despite solar availability |

---

## Results

| Metric | Value |
|--------|-------|
| Annual energy (clean) | 15,889 kWh |
| Mean PR (clean, daylight) | 0.918 |
| Isolation Forest Precision | 0.985 |
| Isolation Forest Recall | 0.565 |
| F1-score | 0.718 |

Recall of 0.565 reflects that slow PID degradation (gradual power reduction) is harder to isolate with Isolation Forest at 1-hour granularity. High precision (0.985) means very few false alarms — important for field deployment.

**Note:** Precision and recall are scored against **injected** ground-truth labels, not field-verified faults. These results characterize detector behaviour on a controlled benchmark, not field fault-detection accuracy.

PR values briefly exceeding 1.0 in winter are **physically correct**: at low temperatures (T_cell < 25°C), the negative temperature coefficient boosts P_mp above STC rating.

---

## Quick start

```bash
pip install -r requirements.txt
python3 run_pipeline.py          # full pipeline
pytest tests/ -v                 # 26 tests, all pass
```

---

## Domain relevance

Direct application to Serbia's solar capacity (~1.5 GW installed as of 2024) and BPM FiberNetworks' monitoring infrastructure. Performance Ratio monitoring is the IEC 61724-1 standardized KPI for PV plant operators. The MPPT efficiency metric connects directly to P4 hardware arbiter work — fast hardware interrupts enable sub-cycle MPPT tracking corrections.

## References

- de Soto, W. et al. (2006). Solar Energy, 80(1), 78–88.
- Marion, B. (2017). Progress in Photovoltaics, 25(3), 303–312.
- pvlib-python: https://pvlib-python.readthedocs.io/

## License

MIT
