"""
THERMAL RUNAWAY PROPAGATION SIMULATION — v4.4  (Publication-ready figures)
======================================================================
Changes from v4.3:
    • Fig 6 (NEW): cooling design guideline curve — sigmoid fit to h_c sweep,
    bootstrap 95 % CI band, three engineering safety threshold annotations,
    plus a companion panel showing mean cells triggered vs h_c
  • Regime analysis figure: suptitle removed; panel titles made more concise
  • All save DPI kept at 300 (journal minimum)

FIX (v4.4 → v4.4-fixed):
  

from __future__ import annotations

import os
import json
import warnings
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

warnings.filterwarnings("ignore")

# ── numpy trapz compat ────────────────────────────────────────────────────────
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ── tqdm fallback ─────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

if not _HAS_TQDM:
    class tqdm:
        def __init__(self, iterable=None, desc="", total=None, disable=False, **kw):
            self._disable = disable
            self.desc = desc
            self._items = list(iterable) if iterable is not None else None
            self.total = (len(self._items) if (self._items is not None and total is None)
                          else (total or 0))
            self.n = 0
        def update(self, n=1):   self.n += n
        def set_postfix_str(self, s): pass
        def close(self):
            if not self._disable:
                print(f"  {self.desc}: done ({self.n})", flush=True)
        def __enter__(self):  return self
        def __exit__(self, *a): self.close()
        def __iter__(self):   return self
        def __next__(self):
            if self._items is None or self.n >= len(self._items):
                if not self._disable and self._items is not None:
                    print(f"  {self.desc}: done ({self.n})", flush=True)
                raise StopIteration
            item = self._items[self.n];  self.n += 1;  return item


# ── Output directory ──────────────────────────────────────────────────────────
try:
    FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
except NameError:
    FIG_DIR = os.path.join(os.getcwd(), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# ── PUBLICATION STYLE  (v4.1 — enlarged fonts, clean palette) ────────────────
# =============================================================================
PUB_STYLE = {
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "font.size":           14,
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.titlesize":      13,
    "axes.labelsize":      14,
    "xtick.labelsize":     13,
    "ytick.labelsize":     13,
    "axes.linewidth":      1.3,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.major.size":    6,
    "ytick.major.size":    6,
    "xtick.minor.size":    3,
    "ytick.minor.size":    3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.framealpha":   0.93,
    "legend.edgecolor":    "0.7",
    "legend.fontsize":     12,
    "legend.title_fontsize": 12,
    "lines.linewidth":     2.0,
    "patch.edgecolor":     "0.2",
}
matplotlib.rcParams.update(PUB_STYLE)

# =============================================================================
# ── COLOUR PALETTE  (Nature/Science 7-colour, print-safe) ────────────────────
# =============================================================================
C = {
    "blue":     "#0072B2",
    "orange":   "#E69F00",
    "green":    "#009E73",
    "red":      "#D55E00",
    "purple":   "#CC79A7",
    "sky":      "#56B4E9",
    "yellow":   "#F0E442",
    "black":    "#000000",
    "grey":     "#777777",
    "darkgrey": "#444444",
}

REGIME_COLORS  = {"none": "#4393C3", "partial": "#FDAE61", "full": "#D73027"}
REGIME_HATCHES = {"none": "///",     "partial": "...",     "full": "xxx"}
REGIME_LABELS  = {
    "none":    "No propagation (≤ 1 cell)",
    "partial": "Partial propagation (2 – N−1 cells)",
    "full":    "Full propagation (all cells)",
}


def save_figure(fig: plt.Figure, name: str) -> None:
    base = os.path.join(FIG_DIR, name)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    print(f"  Saved → {base}.png  |  {base}.pdf")


def _in_jupyter() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except Exception:
        return False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def export_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved → {path}")


def _panel_label(ax: plt.Axes, label: str,
                 x: float = -0.13, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="left")


# =============================================================================
# PART 1 — SINGLE CELL MODEL
# =============================================================================
class CellThermalRunaway:
    def __init__(self, cell_id: int = 0, params: dict | None = None,
                 rng: np.random.Generator | None = None):
        self.cell_id = int(cell_id)
        self.rng     = rng if rng is not None else np.random.default_rng()
        self.params: Dict[str, Any] = {
            "mass": 0.070, "Cp": 1100.0, "area": 0.008,
            "h_conv": 2.0, "T_amb": 25.0,
            "A_sei": 4.583e14, "E_sei": 1.35e5,  "H_sei": 962_500.0,  "x_sei": 0.15,
            "A_an":  3.404e7,  "E_an":  7.5e4,   "H_an":  1_232_000.0,"x_an":  0.25,
            "A_ca":  2.398e10, "E_ca":  1.0e5,   "H_ca":  1_078_000.0,"x_ca":  0.35,
            "A_el":  1.991e11, "E_el":  1.1e5,   "H_el":  1_540_000.0,"x_el":  0.25,
            "k_max_sei": 2.0, "k_max_sec": 0.5,
            "T_trigger_mean": 170.0, "T_trigger_std": 15.0,
            "H_variation_std": 0.08, "R": 8.314,
        }
        if params:
            self.params.update(params)
        H_std = float(self.params["H_variation_std"])
        if H_std > 0.0:
            for key in ("H_sei", "H_an", "H_ca", "H_el"):
                self.params[key] *= 1.0 + H_std * self.rng.standard_normal()
        self.reset()

    def reset(self) -> None:
        self.T = float(self.params["T_amb"])
        self.c_sei = self.c_an = self.c_ca = self.c_el = 1.0
        self.time_history:  List[float] = []
        self.T_history:     List[float] = []
        self.Q_gen_history: List[float] = []
        self.c_sei_history: List[float] = []
        self.c_an_history:  List[float] = []
        self.c_ca_history:  List[float] = []
        self.c_el_history:  List[float] = []
        self.T_trigger_sample = self._sample_trigger_temperature()
        self.triggered    = False
        self.trigger_time: Optional[float] = None

    def _sample_trigger_temperature(self) -> float:
        T = self.rng.normal(self.params["T_trigger_mean"],
                            self.params["T_trigger_std"])
        return float(max(130.0, T))

    def _arrhenius_k(self, T_K: float) -> dict:
        R = float(self.params["R"])
        k_sei = k_an = k_ca = k_el = 0.0
        if T_K > 363.0 and self.c_sei > 0.0:
            k_sei = min(self.params["A_sei"] * np.exp(-self.params["E_sei"] / (R * T_K)),
                        float(self.params["k_max_sei"]))
        if T_K > 393.0 and self.c_sei < 0.1 and self.c_an > 0.0:
            k_an  = min(self.params["A_an"]  * np.exp(-self.params["E_an"]  / (R * T_K)),
                        float(self.params["k_max_sec"]))
        if T_K > 453.0 and self.c_ca > 0.0:
            k_ca  = min(self.params["A_ca"]  * np.exp(-self.params["E_ca"]  / (R * T_K)),
                        float(self.params["k_max_sec"]))
        if T_K > 423.0 and self.c_el > 0.0:
            k_el  = min(self.params["A_el"]  * np.exp(-self.params["E_el"]  / (R * T_K)),
                        float(self.params["k_max_sec"]))
        return {"sei": k_sei, "an": k_an, "ca": k_ca, "el": k_el}

    def _integrate_reactions(self, k: dict, h: float) -> tuple[dict, float]:
        m  = float(self.params["mass"])
        Q  = 0.0
        c_new = {}
        for rxn, c_old, H_key, x_key in (
            ("sei", self.c_sei, "H_sei", "x_sei"),
            ("an",  self.c_an,  "H_an",  "x_an"),
            ("ca",  self.c_ca,  "H_ca",  "x_ca"),
            ("el",  self.c_el,  "H_el",  "x_el"),
        ):
            ki = float(k[rxn])
            if ki > 0.0 and c_old > 0.0:
                c_n = c_old * np.exp(-ki * h)
                dc  = c_old - c_n
                Q  += dc * float(self.params[H_key]) * float(self.params[x_key]) * m / h
                c_new[rxn] = max(0.0, float(c_n))
            else:
                c_new[rxn] = max(0.0, float(c_old))
        return c_new, float(Q)

    def step(self, dt: float, t_now: float,
             T_ext: float | None = None, Q_ext: float = 0.0) -> dict:
        T_amb_eff = float(T_ext if T_ext is not None else self.params["T_amb"])
        active    = (self.T > 150.0 or any(
            c < 0.95 for c in (self.c_sei, self.c_an, self.c_ca, self.c_el)))
        sub_dt = 0.001 if active else 0.01
        n_sub  = max(1, int(np.ceil(dt / sub_dt)))
        h      = dt / n_sub
        Q_gen_accum = 0.0
        t_start     = t_now - dt
        for sub_i in range(n_sub):
            T_K      = self.T + 273.15
            k        = self._arrhenius_k(T_K)
            c_new, Q_gen = self._integrate_reactions(k, h)
            Q_gen_accum += Q_gen
            Q_loss = (float(self.params["h_conv"]) *
                      float(self.params["area"]) * (self.T - T_amb_eff))
            dT = ((Q_gen + Q_ext) - Q_loss) * h / (
                float(self.params["mass"]) * float(self.params["Cp"]))
            self.T = max(self.T + dT, float(self.params["T_amb"]))
            self.c_sei, self.c_an, self.c_ca, self.c_el = (
                c_new["sei"], c_new["an"], c_new["ca"], c_new["el"])
            if self.T > 2000.0:
                self.T = 2000.0
                self.c_sei = self.c_an = self.c_ca = self.c_el = 0.0
                break
            if (not self.triggered) and (self.T >= self.T_trigger_sample):
                self.triggered    = True
                self.trigger_time = t_start + (sub_i + 1) * h
        Q_gen_avg = Q_gen_accum / n_sub
        self.time_history.append(float(t_now))
        self.T_history.append(float(self.T))
        self.Q_gen_history.append(float(Q_gen_avg))
        self.c_sei_history.append(float(self.c_sei))
        self.c_an_history.append(float(self.c_an))
        self.c_ca_history.append(float(self.c_ca))
        self.c_el_history.append(float(self.c_el))
        return {"T": float(self.T), "triggered": bool(self.triggered),
                "trigger_time": None if self.trigger_time is None
                                else float(self.trigger_time),
                "Q_gen": float(Q_gen_avg)}


# =============================================================================
# PART 2 — MODULE-LEVEL PROPAGATION
# =============================================================================
@dataclass
class ModuleSeedConfig:
    mode:             str   = "temperature"
    T_seed_C:         float = 250.0
    Q_seed_W:         float = 2000.0
    pulse_duration_s: float = 3.0


class BatteryModule:
    _K_CONTACT_DEFAULT          = 0.20
    _CONTACT_AREA_DEFAULT       = 0.001
    _CELL_SPACING_DEFAULT       = 0.002
    _VENT_ENERGY_DEFAULT        = 10_000.0
    _VENT_DURATION_DEFAULT      = 35.0
    _RADIATION_SCALE_DEFAULT    = 0.20

    def __init__(self, n_rows=3, n_cols=3,
                 cell_spacing=_CELL_SPACING_DEFAULT, cooling="air",
                 rng=None, disable_cell_internal_conv=True,
                 radiation_scale=_RADIATION_SCALE_DEFAULT,
                 seed_cfg=None, gap_eps=1e-6):
        self.n_rows    = int(n_rows);  self.n_cols = int(n_cols)
        self.n_cells   = self.n_rows * self.n_cols
        self.cell_spacing = float(cell_spacing)
        self.cooling   = str(cooling).lower()
        self.rng       = rng if rng is not None else np.random.default_rng()
        self.disable_cell_internal_conv = bool(disable_cell_internal_conv)
        self.radiation_scale = float(radiation_scale)
        self.seed_cfg  = seed_cfg if seed_cfg is not None else ModuleSeedConfig()
        self.gap_eps   = float(gap_eps)
        self.k_contact    = self._K_CONTACT_DEFAULT
        self.contact_area = self._CONTACT_AREA_DEFAULT
        self.sigma        = 5.670374419e-8
        self.emissivity   = 0.80
        self.view_factor  = 0.30
        self.lateral_area = np.pi * 0.021 * 0.070
        _h_map = {"air": 2.0, "liquid": 200.0, "none": 0.0}
        self.h_cooling = float(_h_map.get(self.cooling, 0.0))
        self.vent_energy_per_neighbour = self._VENT_ENERGY_DEFAULT
        self.vent_duration             = self._VENT_DURATION_DEFAULT
        self.vent_start_time           = np.full(self.n_cells, np.nan, dtype=float)
        self.cells: List[CellThermalRunaway] = []
        for i in range(self.n_cells):
            extra: dict = {
                "T_trigger_mean": 170.0, "T_trigger_std": 15.0,
                "mass": 0.070 * (1.0 + 0.03 * self.rng.standard_normal()),
                "Cp":   1100.0 * (1.0 + 0.02 * self.rng.standard_normal()),
                "H_variation_std": 0.08,
            }
            if self.disable_cell_internal_conv:
                extra["h_conv"] = 0.0
            self.cells.append(
                CellThermalRunaway(cell_id=i, params=extra, rng=self.rng))
        self.reset()

    def reset(self) -> None:
        for c in self.cells:  c.reset()
        self.time = 0.0
        self.cell_states:      List[dict] = []
        self.trigger_order:    List[int]  = []
        self.propagation_complete = False
        self.vent_start_time[:] = np.nan

    def idx(self, r, c): return r * self.n_cols + c

    def neighbours(self, idx):
        r, c = divmod(idx, self.n_cols)
        out  = []
        if r > 0:               out.append(self.idx(r-1, c))
        if r < self.n_rows - 1: out.append(self.idx(r+1, c))
        if c > 0:               out.append(self.idx(r,   c-1))
        if c < self.n_cols - 1: out.append(self.idx(r,   c+1))
        return out

    def _pair_heat(self, T_i, T_j):
        gap    = max(self.cell_spacing, self.gap_eps)
        Q_cond = self.k_contact * self.contact_area * (T_i - T_j) / gap
        Q_rad  = 0.0
        if T_i > 300.0 or T_j > 300.0:
            T_iK = T_i + 273.15;  T_jK = T_j + 273.15
            Q_rad = (self.sigma * self.emissivity * self.view_factor *
                     self.radiation_scale * self.lateral_area *
                     (T_iK**4 - T_jK**4))
        return float(Q_cond + Q_rad)

    def _seed_centre(self):
        ci     = self.idx(self.n_rows // 2, self.n_cols // 2)
        centre = self.cells[ci]
        centre.triggered    = True
        centre.trigger_time = 0.0
        if ci not in self.trigger_order:
            self.trigger_order.append(ci)
        self.vent_start_time[ci] = 0.0
        if self.seed_cfg.mode.lower() == "temperature":
            centre.T = max(float(self.seed_cfg.T_seed_C),
                           centre.T_trigger_sample + 1.0)

    def _sync_vent_start_times(self):
        for i, cell in enumerate(self.cells):
            if cell.triggered and np.isnan(self.vent_start_time[i]):
                tt = cell.trigger_time
                self.vent_start_time[i] = float(tt) if tt is not None else float(self.time)

    def step(self, dt, trigger_center=True):
        self.time += dt
        if trigger_center and len(self.cell_states) == 0:
            self._seed_centre()
        HT_net = np.zeros(self.n_cells, dtype=float)
        for i in range(self.n_cells):
            for j in self.neighbours(i):
                if i < j:
                    Q_ij = self._pair_heat(self.cells[i].T, self.cells[j].T)
                    HT_net[i] -= Q_ij;  HT_net[j] += Q_ij
        self._sync_vent_start_times()
        Q_vent    = np.zeros(self.n_cells, dtype=float)
        vent_rate = self.vent_energy_per_neighbour / max(1e-12, self.vent_duration)
        for i in range(self.n_cells):
            if not self.cells[i].triggered:  continue
            t0 = self.vent_start_time[i]
            if np.isnan(t0):  continue
            elapsed = self.time - t0
            if 0.0 <= elapsed < self.vent_duration:
                for j in self.neighbours(i):
                    Q_vent[j] += vent_rate
        T_amb      = float(self.cells[0].params["T_amb"])
        centre_idx = self.idx(self.n_rows // 2, self.n_cols // 2)
        cur_states = []
        for i, cell in enumerate(self.cells):
            Q_cool = (self.h_cooling * float(cell.params["area"]) *
                      (cell.T - T_amb)) if self.h_cooling > 0.0 else 0.0
            Q_seed = 0.0
            if (self.seed_cfg.mode.lower() == "pulse" and i == centre_idx
                    and self.time <= self.seed_cfg.pulse_duration_s):
                Q_seed = float(self.seed_cfg.Q_seed_W)
            Q_ext  = float(HT_net[i] + Q_vent[i] + Q_seed - Q_cool)
            state  = cell.step(dt=dt, t_now=self.time, T_ext=T_amb, Q_ext=Q_ext)
            cur_states.append(state)
            if state["triggered"] and i not in self.trigger_order:
                self.trigger_order.append(i)
        self._sync_vent_start_times()
        self.cell_states.append({"time": float(self.time), "states": cur_states})
        if len(self.trigger_order) == self.n_cells:
            self.propagation_complete = True

    def run_simulation(self, t_max=300.0, dt=0.1,
                       trigger_center=True, show_progress=True):
        self.reset()
        n_steps = int(np.ceil(t_max / dt))
        pbar    = tqdm(total=n_steps, desc="Simulating", disable=not show_progress)
        while self.time < t_max and not self.propagation_complete:
            self.step(dt, trigger_center=trigger_center);  pbar.update(1)
        if self.propagation_complete and self.time < t_max:
            pbar.set_postfix_str(f"complete at {self.time:.1f} s")
        pbar.close()
        return self.compile_results()

    def compile_results(self):
        times     = np.array([s["time"]  for s in self.cell_states], dtype=float)
        T_history = np.zeros((len(times), self.n_cells), dtype=float)
        for t_idx, st in enumerate(self.cell_states):
            for c_idx, cs in enumerate(st["states"]):
                T_history[t_idx, c_idx] = float(cs["T"])
        trigger_times = np.full(self.n_cells, np.nan, dtype=float)
        for idx in self.trigger_order:
            tt = self.cells[idx].trigger_time
            if tt is not None:
                trigger_times[idx] = float(tt)
        return {"times": times, "T_history": T_history,
                "trigger_order": list(self.trigger_order),
                "trigger_times": trigger_times,
                "n_cells": self.n_cells, "n_rows": self.n_rows,
                "n_cols": self.n_cols,
                "propagation_complete": bool(self.propagation_complete),
                "final_time": float(self.time)}


# =============================================================================
# MULTIPROCESSING WORKER
# =============================================================================
def _run_single_simulation(args):
    param_dict, t_max, dt, seed = args
    rng      = np.random.default_rng(int(seed))
    seed_cfg = param_dict.get("seed_cfg")
    if isinstance(seed_cfg, dict):   seed_cfg = ModuleSeedConfig(**seed_cfg)
    elif seed_cfg is None:           seed_cfg = ModuleSeedConfig()
    module = BatteryModule(
        n_rows=int(param_dict.get("n_rows", 3)),
        n_cols=int(param_dict.get("n_cols", 3)),
        cell_spacing=float(param_dict.get("cell_spacing",
                                          BatteryModule._CELL_SPACING_DEFAULT)),
        cooling=str(param_dict.get("cooling", "air")), rng=rng,
        disable_cell_internal_conv=bool(
            param_dict.get("disable_cell_internal_conv", True)),
        radiation_scale=float(param_dict.get("radiation_scale",
                                             BatteryModule._RADIATION_SCALE_DEFAULT)),
        seed_cfg=seed_cfg,
        gap_eps=float(param_dict.get("gap_eps", 1e-6)),
    )
    module.k_contact               = float(param_dict.get("k_contact",  module.k_contact))
    module.contact_area            = float(param_dict.get("contact_area", module.contact_area))
    module.h_cooling               = float(param_dict.get("h_cooling",   module.h_cooling))
    module.vent_energy_per_neighbour = float(
        param_dict.get("vent_energy_per_neighbour", module.vent_energy_per_neighbour))
    module.vent_duration           = float(param_dict.get("vent_duration", module.vent_duration))
    results         = module.run_simulation(t_max=float(t_max), dt=float(dt),
                                            trigger_center=True, show_progress=False)
    results["seed"] = int(seed)
    return results


# =============================================================================
# PART 3 — SINGLE-CELL MONTE CARLO
# =============================================================================
def run_single_cell_mc(n_simulations=100, heating_power=50.0,
                       t_max=1200.0, dt=0.1, seed=0):
    trigger_temps  = []
    total_energies = []
    for k in tqdm(range(n_simulations), desc="Single-cell MC"):
        rng    = np.random.default_rng(seed + k)
        params = {"mass": 0.070 * (1.0 + 0.03 * rng.standard_normal()),
                  "Cp":   1100.0 * (1.0 + 0.02 * rng.standard_normal()),
                  "T_trigger_std": 15.0, "H_variation_std": 0.08}
        cell   = CellThermalRunaway(cell_id=0, params=params, rng=rng)
        t = 0.0
        while t < t_max and not cell.triggered:
            t += dt;  cell.step(dt, t_now=t, Q_ext=heating_power)
        while t < t_max:
            t += dt;  cell.step(dt, t_now=t, Q_ext=0.0)
        energy_J = _trapz(np.array(cell.Q_gen_history), np.array(cell.time_history))
        total_energies.append(float(energy_J / 1000.0))
        trigger_temps.append(float(cell.T_trigger_sample))
    return {"trigger_temps":  np.array(trigger_temps,  dtype=float),
            "total_energies": np.array(total_energies, dtype=float)}


# =============================================================================
# PART 4 — PUBLICATION-QUALITY FIGURES
# =============================================================================

def plot_single_cell_response(cell, t_max=1200.0, heating_power=50.0,
                              save_name="fig1_single_cell_response"):
    print("Simulating single cell …")
    dt = 0.1;  cell.reset();  t = 0.0
    while t < t_max and not cell.triggered:
        t += dt;  cell.step(dt, t_now=t, Q_ext=heating_power)
    while t < t_max:
        t += dt;  cell.step(dt, t_now=t, Q_ext=0.0)

    times = np.array(cell.time_history,   dtype=float)
    T_arr = np.array(cell.T_history,      dtype=float)
    Q_arr = np.array(cell.Q_gen_history,  dtype=float)
    t_tr  = cell.trigger_time

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(times, T_arr, color=C["red"], lw=2.4, label="Cell temperature", zorder=4)
    ax.axhline(cell.params["T_trigger_mean"], ls="--", color=C["grey"], lw=1.8,
               label=f"Mean $T_{{\\mathrm{{tr}}}}$ = {cell.params['T_trigger_mean']:.0f} °C")
    ax.axhline(cell.T_trigger_sample, ls=":", color=C["orange"], lw=2.0,
               label=f"Sampled $T_{{\\mathrm{{tr}}}}$ = {cell.T_trigger_sample:.1f} °C")
    if t_tr is not None:
        ax.axvline(t_tr, ls="-.", color=C["purple"], lw=1.8,
                   label=f"TR onset  {t_tr:.1f} s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Cell temperature", fontsize=13, pad=6)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.93,
              borderpad=0.8, handlelength=2.0)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_xlim(left=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(times, np.maximum(Q_arr, 1e-12), color=C["orange"], lw=2.4)
    if t_tr is not None:
        ax.axvline(t_tr, ls="-.", color=C["purple"], lw=1.8,
                   label=f"TR onset  {t_tr:.1f} s")
        ax.legend(loc="lower right", fontsize=11, framealpha=0.93)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heat generation rate (W)")
    ax.set_title("Heat generation rate (log scale)", fontsize=13, pad=6)
    ax.grid(True, alpha=0.25, lw=0.6, which="both")
    ax.set_xlim(left=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 0])
    rxn_colors = [C["blue"], C["red"], C["green"], C["orange"]]
    rxn_labels = ["SEI decomposition", "Anode–electrolyte",
                  "Cathode decomposition", "Electrolyte combustion"]
    rxn_data   = [cell.c_sei_history, cell.c_an_history,
                  cell.c_ca_history,  cell.c_el_history]
    for data, col, lbl in zip(rxn_data, rxn_colors, rxn_labels):
        ax.plot(times, data, color=col, lw=2.2, label=lbl)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised reactant fraction (−)")
    ax.set_title("Reactant depletion", fontsize=13, pad=6)
    ax.legend(loc="upper right", fontsize=11, ncol=1, framealpha=0.93)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_xlim(left=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(T_arr, Q_arr, c=times, s=8, alpha=0.75,
                    cmap="viridis", rasterized=True)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Heat generation rate (W)")
    ax.set_title("Phase plane (coloured by time)", fontsize=13, pad=6)
    cbar = fig.colorbar(sc, ax=ax, pad=0.04)
    cbar.set_label("Time (s)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25, lw=0.6)
    _panel_label(ax, "(d)")

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    return fig


def plot_module_propagation(results, save_name="fig2_module_propagation"):
    times      = results["times"]
    T_hist     = results["T_history"]
    n_cells    = results["n_cells"]
    n_rows     = results["n_rows"]
    n_cols     = results["n_cols"]
    trig_order = results["trigger_order"]
    trig_times = results["trigger_times"]

    cell_palette = [C["blue"], C["red"], C["green"], C["orange"],
                    C["purple"], C["sky"], C["yellow"], C["darkgrey"], C["black"]]

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    for i in range(n_cells):
        bold = trig_order and i == trig_order[0]
        col  = cell_palette[i % len(cell_palette)]
        ax.plot(times, T_hist[:, i],
                lw=2.6 if bold else 1.2,
                alpha=1.0 if bold else 0.60,
                color=col,
                label=f"Cell {i} (seed)" if bold else f"Cell {i}",
                zorder=5 if bold else 2)
    for tt in trig_times:
        if not np.isnan(tt):
            ax.axvline(tt, ls=":", alpha=0.20, color="black", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Cell temperature evolution", fontsize=13, pad=6)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc="lower left", fontsize=11, ncol=2,
              framealpha=0.93, borderpad=0.7, handlelength=1.6,
              columnspacing=0.8)
    _panel_label(ax, "(a)")

    ax  = fig.add_subplot(gs[0, 1])
    fT  = T_hist[-1, :].reshape(n_rows, n_cols)
    vmax = max(500.0, float(np.nanmax(fT)))
    hot_cmap = LinearSegmentedColormap.from_list(
        "hot_pub", ["#f7f7f7", "#fee08b", "#f46d43", "#a50026"])
    im  = ax.imshow(fT, cmap=hot_cmap, vmin=25.0, vmax=vmax,
                    interpolation="nearest", aspect="auto")
    # FIX 4 (minor): display precise time value (36.9 s) in the title
    # rather than the rounded integer (37 s) used in earlier versions.
    ax.set_title(f"Final temperature map  (t = {times[-1]:.1f} s)",
                 fontsize=13, pad=6)
    ax.set_xlabel("Column index");  ax.set_ylabel("Row index")
    cbar = fig.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label("Temperature (°C)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    for r in range(n_rows):
        for c in range(n_cols):
            idx  = r * n_cols + c
            lbl  = str(trig_order.index(idx)) if idx in trig_order else "–"
            tcol = "white" if fT[r, c] > vmax * 0.55 else "black"
            ax.text(c, r, lbl, ha="center", va="center",
                    color=tcol, fontweight="bold", fontsize=13)
    _panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 0])
    if trig_order:
        y    = np.arange(len(trig_order))
        tts  = [trig_times[idx] for idx in trig_order]
        bars = ax.barh(y, tts, color=C["blue"], edgecolor="white",
                       height=0.65, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([f"Cell {idx}" for idx in trig_order], fontsize=12)
        ax.set_xlabel("Trigger time (s)")
        ax.set_title("Trigger time sequence", fontsize=13, pad=6)
        ax.grid(True, alpha=0.25, lw=0.6, axis="x")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        for bar, val in zip(bars, tts):
            ax.text(val + max(tts) * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f} s", va="center", fontsize=11)
    else:
        ax.text(0.5, 0.5, "No cells triggered",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
    _panel_label(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    if len(trig_order) > 1:
        t0r, t0c = divmod(trig_order[0], n_cols)
        dists, speeds = [], []
        for idx in trig_order[1:]:
            r, c  = divmod(idx, n_cols)
            d_m   = np.sqrt((r - t0r)**2 + (c - t0c)**2) * 0.021
            dt_   = trig_times[idx] - trig_times[trig_order[0]]
            if dt_ > 0:
                dists.append(d_m * 1000);  speeds.append((d_m / dt_) * 1000)
        if speeds:
            ax.plot(dists, speeds, "o", color=C["green"], ms=10,
                    markerfacecolor="white", markeredgewidth=2.2,
                    linestyle="-", label="Cell propagation speed")
            mu_s = np.mean(speeds)
            ax.axhline(mu_s, ls="--", color=C["grey"], lw=1.8,
                       label=f"Mean = {mu_s:.2f} mm s⁻¹")
            ax.set_xlabel("Distance from trigger cell (mm)")
            ax.set_ylabel("Propagation speed (mm s⁻¹)")
            ax.set_title("Propagation speed", fontsize=13, pad=6)
            ax.legend(loc="best", fontsize=11, framealpha=0.93)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        else:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=13)
    else:
        ax.text(0.5, 0.5, "Only one cell triggered",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
    _panel_label(ax, "(d)")

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    return fig


def plot_comparison_with_experiment(sim_trigger, sim_energy, exp_csv_path,
                                    save_name="fig3_validation_vs_experiment"):
    exp_df      = pd.read_csv(exp_csv_path)
    exp_trigger = pd.to_numeric(
        exp_df["Avg-Cell-Temp-At-Trigger-degC"], errors="coerce").dropna()
    exp_energy  = pd.to_numeric(
        exp_df["Corrected-Total-Energy-Yield-kJ"], errors="coerce").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             gridspec_kw={"wspace": 0.42})

    meta = [
        ("(a)  Trigger temperature distribution",
         "Trigger temperature (°C)", exp_trigger, sim_trigger, "(a)"),
        ("(b)  Total energy release distribution",
         "Total energy released (kJ)",  exp_energy,  sim_energy,  "(b)"),
    ]
    for ax, (title, xlabel, exp, sim, lbl) in zip(axes, meta):
        bins = np.linspace(min(exp.min(), sim.min()),
                           max(exp.max(), sim.max()), 12)
        ax.hist(exp, bins=bins, alpha=0.62, label="Experimental",
                color=C["blue"],   edgecolor="white", density=True, lw=1.2)
        ax.hist(sim, bins=bins, alpha=0.62, label="Simulation",
                color=C["orange"], edgecolor="white", density=True, lw=1.2)
        ax.axvline(exp.mean(), color=C["blue"],   ls="--", lw=2.0,
                   label=f"Exp. mean = {exp.mean():.1f}")
        ax.axvline(float(np.mean(sim)), color=C["orange"], ls="--", lw=2.0,
                   label=f"Sim. mean = {np.mean(sim):.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.set_title(title, fontsize=13, pad=6)
        ax.legend(fontsize=11, framealpha=0.93)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        _panel_label(ax, lbl)

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    return fig


def monte_carlo_parallel(param_dict, n_simulations=100, t_max=300.0, dt=0.1,
                         max_workers=None, force_serial=None,
                         save_name="fig4_monte_carlo_baseline", seed=0):
    seeds  = list(range(seed, seed + n_simulations))
    tasks  = [(param_dict, t_max, dt, s) for s in seeds]
    if force_serial is None:
        force_serial = _in_jupyter()

    def _serial():
        out = []
        for s in tqdm(seeds, desc="Monte Carlo (serial)"):
            out.append(_run_single_simulation((param_dict, t_max, dt, s)))
        return out

    results_list = []
    if force_serial:
        print(f"\nMonte Carlo ({n_simulations} runs) — SERIAL")
        results_list = _serial()
    else:
        try:
            workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
            print(f"\nMonte Carlo ({n_simulations} runs) — PARALLEL ({workers} workers)")
            with mp.Pool(processes=workers) as pool:
                for res in tqdm(pool.imap_unordered(_run_single_simulation, tasks),
                                total=n_simulations, desc="Monte Carlo (parallel)"):
                    results_list.append(res)
        except Exception as exc:
            print(f"\n[Parallel failed → serial fallback] {exc}")
            results_list = _serial()

    r0          = results_list[0]
    total_cells = r0["n_cells"]
    n_rows_mc   = r0["n_rows"];  n_cols_mc = r0["n_cols"]
    n_trig_arr  = np.array([len(r["trigger_order"]) for r in results_list], dtype=float)
    trig_counts = np.zeros(total_cells, dtype=float)
    prop_times  = [];  final_temps = []
    for r in results_list:
        for idx in r["trigger_order"]:
            trig_counts[idx] += 1.0
        if len(r["trigger_order"]) == total_cells:
            last_idx = r["trigger_order"][-1]
            t_last   = r["trigger_times"][last_idx]
            prop_times.append(float(t_last) if not np.isnan(t_last) else np.nan)
        else:
            prop_times.append(np.nan)
        final_temps.append(float(np.mean(r["T_history"][-1, :])))

    prop_times_arr  = np.array(prop_times,  dtype=float)
    final_temps_arr = np.array(final_temps, dtype=float)
    valid_prop      = prop_times_arr[~np.isnan(prop_times_arr)]
    prop_prob       = float(np.mean(n_trig_arr == total_cells))
    prob_map        = (trig_counts / n_simulations).reshape(n_rows_mc, n_cols_mc)

    bs_props = []
    rng_bs   = np.random.default_rng(seed + 999)
    for _ in range(2000):
        idx_bs = rng_bs.integers(0, n_simulations, n_simulations)
        bs_props.append(float(np.mean(n_trig_arr[idx_bs] == total_cells)))
    ci_low  = float(np.percentile(bs_props, 2.5))
    ci_high = float(np.percentile(bs_props, 97.5))

    plt.close("all")
    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    bins_trig = np.arange(0.5, total_cells + 1.5, 1.0)
    ax.hist(n_trig_arr, bins=bins_trig, color=C["blue"], edgecolor="white",
            alpha=0.82, density=False, lw=1.2)
    ax.axvline(n_trig_arr.mean(), ls="--", color=C["red"], lw=2.0,
               label=f"Mean = {n_trig_arr.mean():.2f} ± {n_trig_arr.std():.2f}")
    ax.set_xlabel("Number of cells triggered")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"(a)  Full propagation: {prop_prob:.1%}  "
        f"(95% CI: {ci_low:.1%}–{ci_high:.1%})",
        fontsize=12, pad=5)
    ax.set_xticks(range(1, total_cells + 1))
    ax.legend(fontsize=11, framealpha=0.93)
    ax.grid(True, alpha=0.25, lw=0.6, axis="y")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    if len(valid_prop) > 0:
        ax.hist(valid_prop, bins=min(15, max(5, len(valid_prop) // 5)),
                color=C["red"], edgecolor="white", alpha=0.82, lw=1.2)
        ax.axvline(np.mean(valid_prop), ls="--", color="black", lw=2.0,
                   label=f"Mean = {np.mean(valid_prop):.1f} s")
        ax.axvline(np.percentile(valid_prop, 25), ls=":", color=C["grey"], lw=1.4)
        ax.axvline(np.percentile(valid_prop, 75), ls=":", color=C["grey"], lw=1.4,
                   label="IQR")
        ax.set_xlabel("Full propagation time (s)")
        ax.set_ylabel("Frequency")
        ax.set_title("Propagation time — full-cascade runs", fontsize=13, pad=5)
        ax.legend(fontsize=11, framealpha=0.93)
        ax.grid(True, alpha=0.25, lw=0.6, axis="y")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    else:
        ax.text(0.5, 0.5, "No full-propagation runs\nin this ensemble",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
    _panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(final_temps_arr, bins=20, color=C["orange"], edgecolor="white",
            alpha=0.82, lw=1.2)
    ax.axvline(final_temps_arr.mean(), ls="--", color="black", lw=2.0,
               label=f"Mean = {final_temps_arr.mean():.0f} °C")
    ax.set_xlabel("Mean module temperature at end of simulation (°C)")
    ax.set_ylabel("Frequency")
    ax.set_title("Final mean module temperature", fontsize=13, pad=5)
    ax.legend(fontsize=11, framealpha=0.93)
    ax.grid(True, alpha=0.25, lw=0.6, axis="y")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    cmap_prob = LinearSegmentedColormap.from_list(
        "prob", ["#ffffff", "#fee08b", "#f46d43", "#a50026"])
    im = ax.imshow(prob_map, cmap=cmap_prob, vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    ax.set_title("Cell trigger probability map", fontsize=13, pad=5)
    ax.set_xlabel("Column index");  ax.set_ylabel("Row index")
    cbar = fig.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label("Trigger probability", fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    for i in range(n_rows_mc):
        for j in range(n_cols_mc):
            ax.text(j, i, f"{prob_map[i, j]:.2f}",
                    ha="center", va="center", fontweight="bold", fontsize=13,
                    color="white" if prob_map[i, j] > 0.60 else "black")
    _panel_label(ax, "(d)")

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()

    return {
        "propagation_probability_full":  prop_prob,
        "ci_low_95":                     ci_low,
        "ci_high_95":                    ci_high,
        "mean_triggered_cells":          float(np.mean(n_trig_arr)),
        "std_triggered_cells":           float(np.std(n_trig_arr)),
        "mean_propagation_time_full":    float(np.nanmean(prop_times_arr)),
        "std_propagation_time_full":     float(np.nanstd(prop_times_arr)),
        "mean_final_temp":               float(np.mean(final_temps_arr)),
        "probability_map":               prob_map,
        "raw_n_triggered":               n_trig_arr,
        "raw_prop_times":                prop_times_arr,
        "raw_final_temps":               final_temps_arr,
    }


def plot_air_vs_liquid(mc_air, mc_liq, n_simulations=100,
                       save_name="fig5_air_vs_liquid"):
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.42)
    total_cells = int(round(mc_air["raw_n_triggered"].max()))

    ax   = fig.add_subplot(gs[0, 0])
    bins = np.arange(0.5, total_cells + 1.5, 1.0)
    ax.hist(mc_air["raw_n_triggered"], bins=bins, alpha=0.68,
            color=C["orange"], edgecolor="white", lw=1.2,
            label=f"Air cooling  (mean {mc_air['mean_triggered_cells']:.1f})")
    ax.hist(mc_liq["raw_n_triggered"], bins=bins, alpha=0.68,
            color=C["blue"], edgecolor="white", lw=1.2,
            label=f"Liquid cooling  (mean {mc_liq['mean_triggered_cells']:.1f})")
    ax.set_xlabel("Cells triggered")
    ax.set_ylabel("Frequency")
    ax.set_title("Cells triggered per run", fontsize=13, pad=6)
    ax.set_xticks(range(1, total_cells + 1))
    ax.legend(fontsize=11, framealpha=0.93, loc="upper left")
    ax.grid(True, alpha=0.25, lw=0.6, axis="y")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(a)")

    ax     = fig.add_subplot(gs[0, 1])
    labels = ["Air\ncooling", "Liquid\ncooling"]
    probs  = [mc_air["propagation_probability_full"],
              mc_liq["propagation_probability_full"]]
    errs_lo = [probs[0] - mc_air["ci_low_95"],  probs[1] - mc_liq["ci_low_95"]]
    errs_hi = [mc_air["ci_high_95"] - probs[0], mc_liq["ci_high_95"] - probs[1]]
    colors  = [C["orange"], C["blue"]]
    bars    = ax.bar(labels, probs, color=colors, alpha=0.85,
                     edgecolor="white", width=0.48,
                     yerr=[errs_lo, errs_hi], capsize=9,
                     error_kw={"elinewidth": 2.0, "ecolor": "black"})
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.05,
                f"{p:.0%}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Full-propagation probability")
    ax.set_title("Cascade probability comparison", fontsize=13, pad=6)
    ax.grid(True, alpha=0.25, lw=0.6, axis="y")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(b)")

    ax    = fig.add_subplot(gs[1, 0])
    t_air = mc_air["raw_final_temps"]
    t_liq = mc_liq["raw_final_temps"]
    all_T = np.concatenate([t_air, t_liq])
    bins2 = np.linspace(all_T.min(), all_T.max(), 20)
    ax.hist(t_air, bins=bins2, alpha=0.68, color=C["orange"], edgecolor="white",
            lw=1.2, label=f"Air  (mean {t_air.mean():.0f} °C)")
    ax.hist(t_liq, bins=bins2, alpha=0.68, color=C["blue"],   edgecolor="white",
            lw=1.2, label=f"Liquid  (mean {t_liq.mean():.0f} °C)")
    ax.set_xlabel("Mean final module temperature (°C)")
    ax.set_ylabel("Frequency")
    ax.set_title("Final mean module temperature", fontsize=13, pad=6)
    ax.legend(fontsize=11, framealpha=0.93)
    ax.grid(True, alpha=0.25, lw=0.6, axis="y")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    prob_map = mc_liq["probability_map"]
    n_rows_m, n_cols_m = prob_map.shape
    cmap_liq = LinearSegmentedColormap.from_list(
        "liq_prob", ["#ffffff", "#c6dbef", "#2171b5", "#08306b"])
    im = ax.imshow(prob_map, cmap=cmap_liq, vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    ax.set_title("Liquid cooling — trigger probability map", fontsize=13, pad=6)
    ax.set_xlabel("Column index");  ax.set_ylabel("Row index")
    cbar = fig.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label("Trigger probability", fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    for i in range(n_rows_m):
        for j in range(n_cols_m):
            val  = prob_map[i, j]
            tcol = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center", fontweight="bold",
                    fontsize=13, color=tcol)
    _panel_label(ax, "(d)")

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    return fig


# =============================================================================
# PART 5 — PROPAGATION REGIME ANALYSIS
# =============================================================================
def classify_regime(n_triggered, n_cells):
    if n_triggered <= 1:       return 0
    if n_triggered < n_cells:  return 1
    return 2


def _single_run_regime(args):
    params, seed = args
    r      = _run_single_simulation(
        (params, params.get("t_max", 300.0), params.get("dt", 0.2), seed))
    n_trig = len(r["trigger_order"])
    return n_trig, classify_regime(n_trig, r["n_cells"])


def run_1d_regime_sweep(param_name, param_values, fixed, n_mc=20):
    mean_frac = [];  p_none, p_part, p_full = [], [], [];  sem_frac = []
    for val in param_values:
        p = dict(fixed);  p[param_name] = float(val)
        results = [_single_run_regime((p, s)) for s in range(n_mc)]
        n_cells = int(fixed.get("n_rows", 3)) * int(fixed.get("n_cols", 3))
        fracs   = np.array([r[0] / n_cells for r in results])
        regs    = [r[1] for r in results]
        mean_frac.append(float(fracs.mean()))
        sem_frac.append(float(fracs.std() / np.sqrt(n_mc)))
        p_none.append(regs.count(0) / n_mc)
        p_part.append(regs.count(1) / n_mc)
        p_full.append(regs.count(2) / n_mc)
    return (np.array(mean_frac), np.array(sem_frac),
            np.array(p_none), np.array(p_part), np.array(p_full))


def propagation_regime_analysis(output_prefix="fig_regime",
                                fixed=None, n_mc=20):
    print("\n" + "=" * 60)
    print("SECTION — PROPAGATION REGIME ANALYSIS")
    print("=" * 60)

    if fixed is None:
        fixed = dict(
            n_rows=3, n_cols=3, cooling="air",
            disable_cell_internal_conv=True,
            radiation_scale=0.20, cell_spacing=0.002,
            k_contact=0.20, contact_area=0.001, h_cooling=2.0,
            vent_energy_per_neighbour=10_000.0, vent_duration=35.0,
            gap_eps=1e-6,
            seed_cfg={"mode": "temperature", "T_seed_C": 250.0},
            t_max=300.0, dt=0.2,
        )

    sweeps  = {
        "vent_energy_per_neighbour": np.linspace(1_000,  18_000, 12),
        "cell_spacing":              np.linspace(0.001,  0.010,  10),
        "k_contact":                 np.linspace(0.05,   0.80,   10),
        "h_cooling":                 np.linspace(1.0,    500.0,  10),
    }
    xlabels = {
        "vent_energy_per_neighbour": "Vent energy per neighbour  (J)",
        "cell_spacing":              "Cell-to-cell gap  (m)",
        "k_contact":                 "Contact thermal conductance  (W m⁻¹ K⁻¹)",
        "h_cooling":                 "Cooling coefficient  (W m⁻² K⁻¹)",
    }
    panel_titles = {
        "vent_energy_per_neighbour": "(a)  Vent energy",
        "cell_spacing":              "(b)  Cell spacing",
        "k_contact":                 "(c)  Thermal contact conductance",
        "h_cooling":                 "(d)  Cooling coefficient",
    }

    sweep_results = {}
    for pname, pvals in sweeps.items():
        print(f"  Sweeping {pname}  ({len(pvals)} values × {n_mc} MC) …")
        mf, sem, pn, pp, pf = run_1d_regime_sweep(pname, pvals, fixed, n_mc=n_mc)
        sweep_results[pname] = (pvals, mf, sem, pn, pp, pf)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                             gridspec_kw={"hspace": 0.50, "wspace": 0.42})

    legend_patches = [
        mpatches.Patch(facecolor=REGIME_COLORS["none"],
                       hatch=REGIME_HATCHES["none"],
                       edgecolor="white", label=REGIME_LABELS["none"]),
        mpatches.Patch(facecolor=REGIME_COLORS["partial"],
                       hatch=REGIME_HATCHES["partial"],
                       edgecolor="white", label=REGIME_LABELS["partial"]),
        mpatches.Patch(facecolor=REGIME_COLORS["full"],
                       hatch=REGIME_HATCHES["full"],
                       edgecolor="white", label=REGIME_LABELS["full"]),
    ]

    for ax, (pname, (pvals, mf, sem, pn, pp, pf)) in zip(
        axes.flat, sweep_results.items()
    ):
        polys = ax.stackplot(
            pvals, pn, pp, pf,
            colors=[REGIME_COLORS["none"],
                    REGIME_COLORS["partial"],
                    REGIME_COLORS["full"]],
            alpha=0.82,
        )
        for poly, hatch in zip(polys, [REGIME_HATCHES["none"],
                                        REGIME_HATCHES["partial"],
                                        REGIME_HATCHES["full"]]):
            poly.set_hatch(hatch)

        ax.plot(pvals, mf, color="black", lw=2.4, ls="-",
                label="Mean triggered fraction", zorder=5)
        ax.fill_between(pvals, mf - sem, mf + sem,
                        color="black", alpha=0.18, label="±1 SE")
        ax.set_xlabel(xlabels[pname], fontsize=14)
        ax.set_ylabel("Probability  /  Fraction", fontsize=14)
        ax.set_title(panel_titles[pname], fontsize=13, fontweight="bold", pad=6)
        ax.set_ylim(0, 1);  ax.set_xlim(pvals[0], pvals[-1])
        ax.grid(True, alpha=0.22, lw=0.6)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        _panel_label(ax, panel_titles[pname].split()[0], x=-0.15)

    mean_line = plt.Line2D([0], [0], color="black", lw=2.4,
                           label="Mean triggered fraction")
    se_patch  = mpatches.Patch(facecolor="black", alpha=0.18, label="±1 SE")
    fig.legend(handles=legend_patches + [mean_line, se_patch],
               loc="lower center", ncol=5, fontsize=12,
               framealpha=0.93, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, f"{output_prefix}_1d_sweeps")
    plt.show()

    return sweep_results


# =============================================================================
# FIG 6 — COOLING DESIGN GUIDELINE CURVE
# =============================================================================

def run_hc_design_sweep(hc_values, base_params, n_mc=50, t_max=300.0, dt=0.2,
                        seed_offset=5000):
    n_cells = int(base_params.get("n_rows", 3)) * int(base_params.get("n_cols", 3))
    all_outcomes = []
    all_n_trig   = []
    for i, hc in enumerate(hc_values):
        p = dict(base_params)
        p["h_cooling"] = float(hc)
        p["cooling"]   = "liquid" if hc > 5.0 else "air"
        outcomes = []
        n_trigs  = []
        for s in range(n_mc):
            r       = _run_single_simulation((p, t_max, dt, seed_offset + i * 1000 + s))
            n_trig  = len(r["trigger_order"])
            outcomes.append(n_trig == n_cells)
            n_trigs.append(n_trig)
        all_outcomes.append(np.array(outcomes, dtype=float))
        all_n_trig.append(np.array(n_trigs, dtype=float))
        pf = float(np.mean(outcomes))
        print(f"  h_c = {hc:6.1f}  →  P_full = {pf:.2f}", flush=True)
    return all_outcomes, all_n_trig


def _sigmoid(hc, k, h50):
    """Logistic sigmoid: P_full = 1 / (1 + exp(k * (hc - h50)))."""
    return 1.0 / (1.0 + np.exp(k * (hc - h50)))


def _hc_at_p(p_target: float, k: float, h50: float) -> float:
    """
    Invert the logistic sigmoid to find the h_c value at which P_full = p_target.

    From  P = 1 / (1 + exp(k*(h - h50)))
    =>    h = h50 + log(1/P - 1) / k

    Parameters
    ----------
    p_target : target probability (e.g. 0.20, 0.10, 0.05)
    k        : fitted sigmoid steepness  (m² K W⁻¹)
    h50      : fitted sigmoid inflection point  (W m⁻² K⁻¹)

    Returns
    -------
    h_c at which P_full equals p_target  (W m⁻² K⁻¹)
    """
    return h50 + np.log(1.0 / p_target - 1.0) / k


def plot_design_guideline_curve(hc_values, all_outcomes, all_n_trig,
                                n_cells=9,
                                save_name="fig6_design_curve"):
    """
    Two-panel cooling design guideline figure (1 × 2).

    Panel (a): Full-propagation probability P_full vs h_c.
        — Bootstrap 95 % CI band (shaded)
        — Raw MC data points
        — Sigmoid fit (if scipy available)
        — Three safety threshold annotations derived analytically from the
          fitted sigmoid (NOT hardcoded), consistent with manuscript text
        — Operating point markers for air and liquid configurations

    Panel (b): Mean number of cells triggered vs h_c.
        — Error bars = ±1 standard error
    """
    hc_arr    = np.asarray(hc_values, dtype=float)
    n_pts     = len(hc_arr)
    n_mc      = len(all_outcomes[0])

    # ── Per-point statistics ──────────────────────────────────────────────────
    p_full     = np.array([float(np.mean(o)) for o in all_outcomes])
    mean_ntrig = np.array([float(np.mean(n)) for n in all_n_trig])
    se_ntrig   = np.array([float(np.std(n) / np.sqrt(n_mc)) for n in all_n_trig])

    # Bootstrap 95 % CI at each h_c point
    rng_bs  = np.random.default_rng(42)
    ci_lo   = np.zeros(n_pts)
    ci_hi   = np.zeros(n_pts)
    for i, outcomes in enumerate(all_outcomes):
        bs = []
        for _ in range(2000):
            idx = rng_bs.integers(0, n_mc, n_mc)
            bs.append(float(np.mean(outcomes[idx])))
        ci_lo[i] = np.percentile(bs, 2.5)
        ci_hi[i] = np.percentile(bs, 97.5)

    # ── Sigmoid fit ───────────────────────────────────────────────────────────
    fit_ok = False
    k_fit  = 0.018;  h50_fit = 185.0          # fallback values
    if _HAS_SCIPY:
        try:
            p0      = [0.018, 185.0]
            popt, _ = curve_fit(_sigmoid, hc_arr, p_full, p0=p0,
                                 bounds=([0.001, 50.0], [1.0, 500.0]),
                                 maxfev=5000)
            k_fit, h50_fit = popt
            fit_ok = True
            print(f"  Sigmoid fit: k = {k_fit:.4f}  h50 = {h50_fit:.1f} W m⁻²K⁻¹")
        except Exception as e:
            print(f"  [Sigmoid fit failed: {e}]")

    hc_fine = np.linspace(hc_arr[0], hc_arr[-1], 500)
    p_fit   = (_sigmoid(hc_fine, k_fit, h50_fit) if fit_ok
               else np.interp(hc_fine, hc_arr, p_full))

    # ── Engineering thresholds — computed from fitted sigmoid (FIX 1) ─────────
    #
    # BEFORE (bug): hardcoded dict values {0.20: 310.0, 0.10: 380.0, 0.05: 440.0}
    # AFTER  (fix): computed via _hc_at_p() using the fitted k and h50.
    #
    # With the actual sweep data (k ≈ 0.21, h50 ≈ 199) this gives:
    #   P_full < 20%  →  h_c ≈ 206  W m⁻²K⁻¹
    #   P_full < 10%  →  h_c ≈ 209  W m⁻²K⁻¹
    #   P_full < 5%   →  h_c ≈ 213  W m⁻²K⁻¹
    # consistent with Section 3.6, the Fig 6 caption, and the Conclusion.
    # ─────────────────────────────────────────────────────────────────────────
    threshold_targets = [
        (0.20, C["orange"], "--"),
        (0.10, C["red"],    "-."),
        (0.05, C["purple"], ":"),
    ]
    thresholds = []
    for p_thresh, col, ls in threshold_targets:
        hc_thresh = _hc_at_p(p_thresh, k_fit, h50_fit)
        thresholds.append((p_thresh, hc_thresh, col, ls))
        print(f"  Threshold: P_full < {p_thresh:.0%}  →  h_c > {hc_thresh:.1f} W m⁻²K⁻¹")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"wspace": 0.44})

    # ── (a) P_full vs h_c ────────────────────────────────────────────────────
    ax = axes[0]

    ax.fill_between(hc_arr, ci_lo, ci_hi,
                    color=C["blue"], alpha=0.18, label="95% bootstrap CI")
    ax.plot(hc_arr, p_full, "o", color=C["blue"], ms=8,
            markerfacecolor="white", markeredgewidth=2.2, zorder=5,
            label="MC data")
    lbl_fit = (f"Sigmoid fit  ($h_{{50}}$ = {h50_fit:.0f} W m⁻²K⁻¹)"
               if fit_ok else "Interpolated curve")
    ax.plot(hc_fine, p_fit, "-", color=C["blue"], lw=2.4, zorder=4,
            label=lbl_fit)

    # Annotate each threshold with a vertical dashed line and arrowhead label.
    # All h_c values are now derived from the sigmoid fit — no hardcoding.
    for p_thresh, hc_thresh, col, ls in thresholds:
        ax.axhline(p_thresh,  ls=ls, color=col, lw=1.6, alpha=0.85)
        ax.axvline(hc_thresh, ls=ls, color=col, lw=1.6, alpha=0.85,
                   label=(f"$P_{{\\mathrm{{full}}}}$ < {p_thresh:.0%}  "
                          f"→  $h_c$ > {hc_thresh:.0f}"))
        ax.annotate(
            f"{hc_thresh:.0f}",
            xy=(hc_thresh, p_thresh),
            xytext=(hc_thresh + 15, p_thresh + 0.04),
            fontsize=11, color=col,
            arrowprops=dict(arrowstyle="-|>", color=col,
                            lw=1.2, mutation_scale=10),
        )

    # Operating-point markers
    ax.plot(2.0,   1.0, "^", color=C["orange"], ms=12, zorder=6,
            markeredgecolor="black", markeredgewidth=1.0,
            label="Air cooling  ($h_c$ = 2)")
    ax.plot(200.0, float(np.interp(200.0, hc_arr, p_full)),
            "s", color=C["sky"], ms=12, zorder=6,
            markeredgecolor="black", markeredgewidth=1.0,
            label="Liquid cooling  ($h_c$ = 200)")

    ax.set_xlabel("Heat transfer coefficient, $h_c$  (W m⁻² K⁻¹)")
    ax.set_ylabel("Full-propagation probability, $P_{\\mathrm{full}}$")
    ax.set_title("Cooling design guideline curve", fontsize=13, pad=6)
    ax.set_xlim(hc_arr[0] - 5, hc_arr[-1] + 10)
    ax.set_ylim(-0.04, 1.12)
    ax.legend(fontsize=10, framealpha=0.93, loc="upper right",
              ncol=1, handlelength=2.0)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(a)")

    # ── (b) Mean cells triggered vs h_c ──────────────────────────────────────
    ax = axes[1]

    ax.errorbar(hc_arr, mean_ntrig, yerr=se_ntrig,
                fmt="o-", color=C["red"], lw=2.2, ms=7,
                markerfacecolor="white", markeredgewidth=2.0,
                capsize=5, elinewidth=1.4, zorder=5,
                label="Mean cells triggered  (±1 SE)")
    ax.axhline(n_cells, ls="--", color=C["grey"], lw=1.6,
               label=f"Full cascade  (n = {n_cells})")
    ax.axhline(1.0,     ls=":",  color=C["grey"], lw=1.6,
               label="Containment  (n = 1)")
    ax.plot(2.0,   float(np.interp(2.0, hc_arr, mean_ntrig)),
            "^", color=C["orange"], ms=12, zorder=6,
            markeredgecolor="black", markeredgewidth=1.0,
            label="Air cooling  ($h_c$ = 2)")
    ax.plot(200.0, float(np.interp(200.0, hc_arr, mean_ntrig)),
            "s", color=C["sky"],    ms=12, zorder=6,
            markeredgecolor="black", markeredgewidth=1.0,
            label="Liquid cooling  ($h_c$ = 200)")

    ax.set_xlabel("Heat transfer coefficient, $h_c$  (W m⁻² K⁻¹)")
    ax.set_ylabel("Mean number of cells triggered")
    ax.set_title("Mean cascade extent", fontsize=13, pad=6)
    ax.set_xlim(hc_arr[0] - 5, hc_arr[-1] + 10)
    ax.set_ylim(0, n_cells + 0.8)
    ax.set_yticks(range(0, n_cells + 1))
    ax.legend(fontsize=11, framealpha=0.93, loc="upper right")
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _panel_label(ax, "(b)")

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    return fig


# =============================================================================
# BASELINE PARAMETERS
# =============================================================================
BASELINE_PARAMS = dict(
    n_rows=3, n_cols=3, cooling="air",
    disable_cell_internal_conv=True,
    cell_spacing=0.002, k_contact=0.20, contact_area=0.001,
    h_cooling=2.0, radiation_scale=0.20,
    vent_energy_per_neighbour=10_000.0, vent_duration=35.0,
    gap_eps=1e-6,
    seed_cfg={"mode": "temperature", "T_seed_C": 250.0},
)
LIQUID_PARAMS = {**BASELINE_PARAMS, "cooling": "liquid", "h_cooling": 200.0}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    SEED = 0
    SEP  = "=" * 70
    sep  = "-" * 70

    print(SEP)
    print("  THERMAL RUNAWAY PROPAGATION SIMULATION — v4.4-fixed")
    print(f"  Output figures → {FIG_DIR}")
    print(SEP)

    # ── 1) Single-cell Monte Carlo ────────────────────────────────────────────
    N_SC    = 100
    sc_data = run_single_cell_mc(
        n_simulations=N_SC, heating_power=50, t_max=1200, dt=0.1, seed=SEED)
    print(sep)
    print("  SINGLE-CELL MONTE CARLO")
    print(sep)
    print(f"  Runs                     : {N_SC}")
    print(f"  Mean trigger temperature : {sc_data['trigger_temps'].mean():.1f} °C")
    print(f"  Std  trigger temperature : {sc_data['trigger_temps'].std():.1f}  °C")
    print(f"  Mean total energy        : {sc_data['total_energies'].mean():.1f}  kJ")
    print(f"  Std  total energy        : {sc_data['total_energies'].std():.2f}  kJ")

    demo_cell = CellThermalRunaway(cell_id=0, rng=np.random.default_rng(SEED))
    plot_single_cell_response(demo_cell, t_max=1200, heating_power=50)

    exp_csv = r"E:\Thermal Runaway\lg_m50_failure_data_with_trigger.csv"
    if os.path.exists(exp_csv):
        plot_comparison_with_experiment(
            sc_data["trigger_temps"], sc_data["total_energies"], exp_csv)
    else:
        print(f"\n  [INFO] Experimental CSV not found at {exp_csv!r} — "
              "skipping validation figure.")

    # ── 2) Single deterministic module run ───────────────────────────────────
    print(sep);  print("  SINGLE MODULE RUN (deterministic seed)");  print(sep)
    rng    = np.random.default_rng(SEED)
    module = BatteryModule(
        n_rows=3, n_cols=3,
        cell_spacing=BASELINE_PARAMS["cell_spacing"],
        cooling=BASELINE_PARAMS["cooling"], rng=rng,
        disable_cell_internal_conv=True,
        radiation_scale=BASELINE_PARAMS["radiation_scale"],
        seed_cfg=ModuleSeedConfig(mode="temperature", T_seed_C=250.0),
    )
    module.k_contact               = BASELINE_PARAMS["k_contact"]
    module.vent_energy_per_neighbour = BASELINE_PARAMS["vent_energy_per_neighbour"]
    module.vent_duration           = BASELINE_PARAMS["vent_duration"]
    results = module.run_simulation(t_max=300, dt=0.1, show_progress=True)
    plot_module_propagation(results)
    print(f"  Cells triggered     : {len(results['trigger_order'])}/{results['n_cells']}")
    print(f"  Trigger order       : {results['trigger_order']}")
    print(f"  Propagation complete: {results['propagation_complete']}")
    print(f"  Final sim time      : {results['final_time']:.1f} s")
    if results["propagation_complete"] and len(results["trigger_order"]) > 1:
        tt = results["trigger_times"];  o = results["trigger_order"]
        print(f"  Propagation span    : {tt[o[-1]] - tt[o[0]]:.1f} s")
    export_json(results, os.path.join(FIG_DIR, "module_single_run_results.json"))

    # ── 3) Monte Carlo — air cooling ─────────────────────────────────────────
    print(sep);  print("  MONTE CARLO — AIR COOLING (100 runs)");  print(sep)
    mc_air = monte_carlo_parallel(
        BASELINE_PARAMS, n_simulations=100, t_max=300, dt=0.1,
        save_name="fig4_mc_air", force_serial=True, seed=SEED)
    print(f"  Full propagation probability : {mc_air['propagation_probability_full']:.1%}"
          f"  (95 % CI: {mc_air['ci_low_95']:.1%}–{mc_air['ci_high_95']:.1%})")
    print(f"  Mean cells triggered         : {mc_air['mean_triggered_cells']:.2f}"
          f"  ±  {mc_air['std_triggered_cells']:.2f}")
    if not np.isnan(mc_air["mean_propagation_time_full"]):
        print(f"  Mean full-propagation time   : {mc_air['mean_propagation_time_full']:.1f}"
              f"  ±  {mc_air['std_propagation_time_full']:.1f}  s")
    print(f"  Mean final temperature       : {mc_air['mean_final_temp']:.1f}  °C")

    # ── 4) Monte Carlo — liquid cooling ──────────────────────────────────────
    print(sep);  print("  MONTE CARLO — LIQUID COOLING (100 runs)");  print(sep)
    mc_liq = monte_carlo_parallel(
        LIQUID_PARAMS, n_simulations=100, t_max=300, dt=0.1,
        save_name="fig4_mc_liquid", force_serial=True, seed=SEED)
    print(f"  Full propagation probability : {mc_liq['propagation_probability_full']:.1%}"
          f"  (95 % CI: {mc_liq['ci_low_95']:.1%}–{mc_liq['ci_high_95']:.1%})")
    print(f"  Mean cells triggered         : {mc_liq['mean_triggered_cells']:.2f}"
          f"  ±  {mc_liq['std_triggered_cells']:.2f}")
    print(f"  Mean final temperature       : {mc_liq['mean_final_temp']:.1f}  °C")

    plot_air_vs_liquid(mc_air, mc_liq, n_simulations=100)
    export_json(mc_air, os.path.join(FIG_DIR, "mc_air_summary.json"))
    export_json(mc_liq, os.path.join(FIG_DIR, "mc_liquid_summary.json"))

    # ── 5) Propagation regime analysis ───────────────────────────────────────
    sweep_results = propagation_regime_analysis(
        output_prefix="fig_regime", fixed=None, n_mc=20)

    # ── 6) Cooling design guideline curve (Fig 6) ─────────────────────────────
    print(sep)
    print("  COOLING DESIGN GUIDELINE SWEEP (Fig 6)")
    print(sep)
    hc_sweep_values = np.array([
          2,   10,   20,   40,   60,   80,  100,  120,
        140,  160,  180,  200,  220,  250,  280,  310,
        340,  370,  400,  440,  480,  500
    ], dtype=float)

    hc_base = dict(BASELINE_PARAMS)
    hc_base.update({
        "n_rows": 3, "n_cols": 3,
        "disable_cell_internal_conv": True,
        "radiation_scale": 0.20,
        "cell_spacing": 0.002,
        "k_contact": 0.20, "contact_area": 0.001,
        "vent_energy_per_neighbour": 10_000.0, "vent_duration": 35.0,
        "gap_eps": 1e-6,
        "seed_cfg": {"mode": "temperature", "T_seed_C": 250.0},
        "t_max": 300.0, "dt": 0.2,
    })

    all_outcomes, all_n_trig = run_hc_design_sweep(
        hc_sweep_values, hc_base, n_mc=50, t_max=300.0, dt=0.2,
        seed_offset=9000)

    plot_design_guideline_curve(
        hc_sweep_values, all_outcomes, all_n_trig,
        n_cells=9, save_name="fig6_design_curve")

    export_json(
        {
            "hc_values":      hc_sweep_values.tolist(),
            "p_full":         [float(np.mean(o)) for o in all_outcomes],
            "mean_ntrig":     [float(np.mean(n)) for n in all_n_trig],
            "n_mc_per_point": 50,
        },
        os.path.join(FIG_DIR, "hc_sweep_summary.json"),
    )

    print(SEP)
    print("  ALL FIGURES SAVED TO:", FIG_DIR)
    print(SEP)
