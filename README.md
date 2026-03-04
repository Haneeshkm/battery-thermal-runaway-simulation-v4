# Thermal Runaway Propagation Simulation — v4.0

**Physics-corrected Monte Carlo model for 21700 cylindrical Li-ion battery modules**

### Root-cause fixes from v3.4
| Parameter              | Old   | New    | Reason |
|------------------------|-------|--------|--------|
| `k_contact`            | 1.0   | **0.20** | Air-gap dominated (lit. 0.05–0.5) |
| `cell_spacing`         | 1.5 mm| **2 mm** | Typical 21700 pack |
| `vent_energy_per_nbr`  | 500 J | **10 kJ**| ~10–20 % of cell energy |
| `vent_duration`        | 20 s  | **35 s** | Realistic venting phase |
| `radiation_scale`      | 0.3   | **0.20** | Conservative view factor |
| `T_trigger_std`        | 8 °C  | **15 °C**| Measured manufacturing scatter |

### Expected baseline results (3×3 air-cooled, T_seed = 250 °C)
- Full-propagation probability ≈ **45–70 %**
- Liquid cooling → **< 5 %** propagation probability

## Features
- Lumped single-cell TR model (SEI + anode + cathode + electrolyte)
- Module heat transfer: conduction + radiation + vent-gas jets
- Full parallel Monte Carlo ensemble
- Publication-quality figures (Nature/JPS style)
- Air vs liquid cooling comparison

## Quick start
```bash
pip install numpy pandas matplotlib tqdm
python thermal_runaway_v4.py
