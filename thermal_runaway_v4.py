"""
THERMAL RUNAWAY PROPAGATION SIMULATION — v4.0  (Physics-corrected)
======================================================================
Root-cause analysis of v3.4 over-propagation and fixes applied:

  PROBLEM                         OLD VALUE    NEW VALUE    JUSTIFICATION
  ─────────────────────────────────────────────────────────────────────────
  k_contact (W/m·K)               1.0          0.20         Air-gap dominated;
                                                            lit. range 0.05–0.5
  cell_spacing (m)                0.0015       0.002        Typical 21700 pack gap
  vent_energy_per_nbr (J)         500          10 000       ~10–20 % of cell energy
                                                            to each neighbour
  vent_duration (s)               20           35           TR venting phase
  radiation_scale (–)             0.3          0.20         Conservative view factor
  T_trigger_std (°C)              8            15           Measured mfg. scatter
  H_variation_std (–)             0.05         0.08         Cell-to-cell chemistry
  mass variation (%)              2            3
  Cp variation (%)                1            2

Expected baseline results (air-cooled 3×3, T_seed = 250 °C):
  • Full-propagation probability  ≈ 45–70 %
  • Non-zero variance across MC ensemble
  • Liquid cooling → <5 % propagation probability
"""

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

warnings.filterwarnings("ignore")

# ── numpy trapz compat ────────────────────────────────────────────────────────
try:
    _trapz = np.trapezoid       # NumPy ≥ 2.0
except AttributeError:
    _trapz = np.trapz           # NumPy < 2.0

# ── tqdm fallback ─────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

if not _HAS_TQDM:
    class tqdm:                 # noqa: N801
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

# ── Publication-quality style ──────────────────────────────────────────────────
# Inspired by Nature / Journal of Power Sources figure standards
PUB_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.figsize": (12, 8),
    "font.size": 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "legend.framealpha": 0.90,
    "legend.edgecolor": "0.8",
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "patch.edgecolor": "0.2",
}
matplotlib.rcParams.update(PUB_STYLE)

# Colorblind-safe, Nature-inspired palette
C = {
    "blue":     "#0072B2",
    "orange":   "#E69F00",
    "green":    "#009E73",
    "red":      "#D55E00",
    "purple":   "#CC79A7",
    "sky":      "#56B4E9",
    "yellow":   "#F0E442",
    "black":    "#000000",
    "grey":     "#999999",
}

REGIME_COLORS = {"none": "#4393C3", "partial": "#F4A582", "full": "#D6604D"}
REGIME_LABELS = {
    "none":    "No propagation (≤1 cell)",
    "partial": "Partial (2 – N−1 cells)",
    "full":    "Full propagation (all cells)",
}


def save_figure(fig: plt.Figure, name: str) -> None:
    base = os.path.join(FIG_DIR, name)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    print(f"  Saved → {base}.png  |  {base}.pdf")


def _in_jupyter() -> bool:
    try:
        shell = get_ipython().__class__.__name__   # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except Exception:
        return False


# ── JSON helpers ───────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        return super().default(obj)


def export_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved → {path}")


# =============================================================================
# PART 1 — SINGLE CELL
# =============================================================================
class CellThermalRunaway:
    """
    Lumped single-cell TR model.
    Four exothermic reactions (SEI, anode, cathode, electrolyte) each
    follow dc/dt = −k(T) c with Arrhenius rate k and exact exponential
    integration.  Per-cell enthalpy scatter is applied once at construction.
    """

    def __init__(
        self,
        cell_id: int = 0,
        params: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.cell_id = int(cell_id)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.params: Dict[str, Any] = {
            "mass":   0.070,       # kg  (21700 nominal)
            "Cp":     1100.0,      # J/kg·K
            "area":   0.008,       # m²  surface area for module cooling
            "h_conv": 2.0,         # W/m²·K  (disabled in module; set to 0)
            "T_amb":  25.0,        # °C

            # ── Arrhenius: A [1/s], E [J/mol], H [J/kg_reactant], x [–]
            "A_sei": 4.583e14, "E_sei": 1.35e5, "H_sei": 962_500.0,  "x_sei": 0.15,
            "A_an":  3.404e7,  "E_an":  7.5e4,  "H_an":  1_232_000.0,"x_an":  0.25,
            "A_ca":  2.398e10, "E_ca":  1.0e5,  "H_ca":  1_078_000.0,"x_ca":  0.35,
            "A_el":  1.991e11, "E_el":  1.1e5,  "H_el":  1_540_000.0,"x_el":  0.25,

            # diffusion-limited rate caps
            "k_max_sei": 2.0,
            "k_max_sec": 0.5,

            # trigger distribution — updated std to 15 °C
            "T_trigger_mean": 170.0,
            "T_trigger_std":   15.0,

            # enthalpy scatter — updated to 8 %
            "H_variation_std": 0.08,

            "R": 8.314,
        }
        if params:
            self.params.update(params)

        # apply enthalpy scatter once at construction
        H_std = float(self.params["H_variation_std"])
        if H_std > 0.0:
            for key in ("H_sei", "H_an", "H_ca", "H_el"):
                self.params[key] *= 1.0 + H_std * self.rng.standard_normal()

        self.reset()

    # ──────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.T = float(self.params["T_amb"])
        self.c_sei = self.c_an = self.c_ca = self.c_el = 1.0

        self.time_history:    List[float] = []
        self.T_history:       List[float] = []
        self.Q_gen_history:   List[float] = []
        self.c_sei_history:   List[float] = []
        self.c_an_history:    List[float] = []
        self.c_ca_history:    List[float] = []
        self.c_el_history:    List[float] = []

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
        """Exact exponential integration: c_new = c_old exp(−k h)."""
        m = float(self.params["mass"])
        Q = 0.0
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

    def step(
        self,
        dt: float,
        t_now: float,
        T_ext: float | None = None,
        Q_ext: float = 0.0,
    ) -> dict:
        T_amb_eff = float(T_ext if T_ext is not None else self.params["T_amb"])
        active = (self.T > 150.0 or any(
            c < 0.95 for c in (self.c_sei, self.c_an, self.c_ca, self.c_el)))
        sub_dt = 0.001 if active else 0.01
        n_sub  = max(1, int(np.ceil(dt / sub_dt)))
        h      = dt / n_sub

        Q_gen_accum = 0.0
        t_start = t_now - dt

        for sub_i in range(n_sub):
            T_K     = self.T + 273.15
            k       = self._arrhenius_k(T_K)
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

        return {
            "T":            float(self.T),
            "triggered":    bool(self.triggered),
            "trigger_time": None if self.trigger_time is None
                            else float(self.trigger_time),
            "Q_gen":        float(Q_gen_avg),
        }


# =============================================================================
# PART 2 — MODULE-LEVEL PROPAGATION
# =============================================================================
@dataclass
class ModuleSeedConfig:
    """
    Seeding the centre cell at t = 0.
    mode = 'temperature': force T_seed_C (recommended, deterministic).
    mode = 'pulse':       inject Q_seed_W for pulse_duration_s.
    """
    mode:              str   = "temperature"
    T_seed_C:          float = 250.0
    Q_seed_W:          float = 2000.0
    pulse_duration_s:  float = 3.0


class BatteryModule:
    """
    N×M cylindrical-cell module.
    Heat transfer between neighbours:
      (1) Conduction  — k_contact × A_contact / gap × ΔT
      (2) Radiation   — σ ε F_scale A_lat (T_i⁴ − T_j⁴)
      (3) Vent gas    — vent_energy_per_neighbour over vent_duration from each
                        triggered cell to its direct neighbours (power burst)
    Module-level cooling applied to every cell via h_cooling.
    Cell internal convection is disabled (set h_conv = 0) to avoid double
    counting when h_cooling is used.
    """

    # ── Physics constants (updated v4.0) ──────────────────────────────────────
    # Contact conductance: air gap dominated; literature 0.05–0.5 W/m·K
    _K_CONTACT_DEFAULT          = 0.20   # W/m·K
    _CONTACT_AREA_DEFAULT       = 0.001  # m²
    _CELL_SPACING_DEFAULT       = 0.002  # m  (2 mm gap)
    # Vent gas: ~10–20 % of cell energy (~50 kJ) to each neighbour → 5–10 kJ
    _VENT_ENERGY_DEFAULT        = 10_000.0  # J per neighbour
    _VENT_DURATION_DEFAULT      = 35.0      # s
    _RADIATION_SCALE_DEFAULT    = 0.20

    def __init__(
        self,
        n_rows:                  int   = 3,
        n_cols:                  int   = 3,
        cell_spacing:            float = _CELL_SPACING_DEFAULT,
        cooling:                 str   = "air",
        rng:                     np.random.Generator | None = None,
        disable_cell_internal_conv: bool = True,
        radiation_scale:         float = _RADIATION_SCALE_DEFAULT,
        seed_cfg:                ModuleSeedConfig | None = None,
        gap_eps:                 float = 1e-6,
    ):
        self.n_rows    = int(n_rows)
        self.n_cols    = int(n_cols)
        self.n_cells   = self.n_rows * self.n_cols
        self.cell_spacing           = float(cell_spacing)
        self.cooling                = str(cooling).lower()
        self.rng                    = rng if rng is not None else np.random.default_rng()
        self.disable_cell_internal_conv = bool(disable_cell_internal_conv)
        self.radiation_scale        = float(radiation_scale)
        self.seed_cfg               = seed_cfg if seed_cfg is not None else ModuleSeedConfig()
        self.gap_eps                = float(gap_eps)

        # ── conduction ──────────────────────────────────────────────────────
        self.k_contact    = self._K_CONTACT_DEFAULT
        self.contact_area = self._CONTACT_AREA_DEFAULT

        # ── radiation ───────────────────────────────────────────────────────
        self.sigma        = 5.670374419e-8
        self.emissivity   = 0.80
        self.view_factor  = 0.30
        self.lateral_area = np.pi * 0.021 * 0.070  # ~21700 side area ≈ 4.6 cm²

        # ── module cooling ──────────────────────────────────────────────────
        _h_map = {"air": 2.0, "liquid": 200.0, "none": 0.0}
        self.h_cooling = float(_h_map.get(self.cooling, 0.0))

        # ── venting ─────────────────────────────────────────────────────────
        self.vent_energy_per_neighbour = self._VENT_ENERGY_DEFAULT
        self.vent_duration             = self._VENT_DURATION_DEFAULT
        self.vent_start_time           = np.full(self.n_cells, np.nan, dtype=float)

        # ── build cells (updated variability) ───────────────────────────────
        self.cells: List[CellThermalRunaway] = []
        for i in range(self.n_cells):
            extra: dict = {
                "T_trigger_mean":   170.0,
                "T_trigger_std":     15.0,   # ← increased to 15 °C
                "mass": 0.070 * (1.0 + 0.03 * self.rng.standard_normal()),  # ±3 %
                "Cp":   1100.0 * (1.0 + 0.02 * self.rng.standard_normal()), # ±2 %
                "H_variation_std":    0.08,  # 8 % enthalpy scatter
            }
            if self.disable_cell_internal_conv:
                extra["h_conv"] = 0.0
            self.cells.append(
                CellThermalRunaway(cell_id=i, params=extra, rng=self.rng))

        self.reset()

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        for c in self.cells:
            c.reset()
        self.time                 = 0.0
        self.cell_states:         List[dict] = []
        self.trigger_order:       List[int]  = []
        self.propagation_complete = False
        self.vent_start_time[:]   = np.nan

    def idx(self, r: int, c: int) -> int:
        return r * self.n_cols + c

    def neighbours(self, idx: int) -> List[int]:
        r, c = divmod(idx, self.n_cols)
        out  = []
        if r > 0:               out.append(self.idx(r - 1, c))
        if r < self.n_rows - 1: out.append(self.idx(r + 1, c))
        if c > 0:               out.append(self.idx(r, c - 1))
        if c < self.n_cols - 1: out.append(self.idx(r, c + 1))
        return out

    def _pair_heat(self, T_i: float, T_j: float) -> float:
        gap    = max(self.cell_spacing, self.gap_eps)
        Q_cond = self.k_contact * self.contact_area * (T_i - T_j) / gap
        Q_rad  = 0.0
        if T_i > 300.0 or T_j > 300.0:
            T_iK  = T_i + 273.15;  T_jK = T_j + 273.15
            Q_rad = (self.sigma * self.emissivity * self.view_factor *
                     self.radiation_scale * self.lateral_area *
                     (T_iK**4 - T_jK**4))
        return float(Q_cond + Q_rad)

    def _seed_centre(self) -> None:
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

    def _sync_vent_start_times(self) -> None:
        for i, cell in enumerate(self.cells):
            if cell.triggered and np.isnan(self.vent_start_time[i]):
                tt = cell.trigger_time
                self.vent_start_time[i] = float(tt) if tt is not None else float(self.time)

    def step(self, dt: float, trigger_center: bool = True) -> None:
        self.time += dt
        if trigger_center and len(self.cell_states) == 0:
            self._seed_centre()

        # ── inter-cell heat transfer ─────────────────────────────────────────
        HT_net = np.zeros(self.n_cells, dtype=float)
        for i in range(self.n_cells):
            for j in self.neighbours(i):
                if i < j:
                    Q_ij = self._pair_heat(self.cells[i].T, self.cells[j].T)
                    HT_net[i] -= Q_ij
                    HT_net[j] += Q_ij

        self._sync_vent_start_times()

        # ── vent gas power ────────────────────────────────────────────────────
        Q_vent    = np.zeros(self.n_cells, dtype=float)
        vent_rate = (self.vent_energy_per_neighbour /
                     max(1e-12, self.vent_duration))
        for i in range(self.n_cells):
            if not self.cells[i].triggered:
                continue
            t0 = self.vent_start_time[i]
            if np.isnan(t0):
                continue
            elapsed = self.time - t0
            if 0.0 <= elapsed < self.vent_duration:
                for j in self.neighbours(i):
                    Q_vent[j] += vent_rate

        # ── integrate each cell ───────────────────────────────────────────────
        T_amb       = float(self.cells[0].params["T_amb"])
        centre_idx  = self.idx(self.n_rows // 2, self.n_cols // 2)
        cur_states  = []

        for i, cell in enumerate(self.cells):
            Q_cool = (self.h_cooling * float(cell.params["area"]) *
                      (cell.T - T_amb)) if self.h_cooling > 0.0 else 0.0

            Q_seed = 0.0
            if (self.seed_cfg.mode.lower() == "pulse"
                    and i == centre_idx
                    and self.time <= self.seed_cfg.pulse_duration_s):
                Q_seed = float(self.seed_cfg.Q_seed_W)

            Q_ext = float(HT_net[i] + Q_vent[i] + Q_seed - Q_cool)
            state = cell.step(dt=dt, t_now=self.time, T_ext=T_amb, Q_ext=Q_ext)
            cur_states.append(state)

            if state["triggered"] and i not in self.trigger_order:
                self.trigger_order.append(i)

        self._sync_vent_start_times()
        self.cell_states.append({"time": float(self.time), "states": cur_states})
        if len(self.trigger_order) == self.n_cells:
            self.propagation_complete = True

    def run_simulation(
        self,
        t_max: float = 300.0,
        dt:    float = 0.1,
        trigger_center: bool = True,
        show_progress:  bool = True,
    ) -> Dict[str, Any]:
        self.reset()
        n_steps = int(np.ceil(t_max / dt))
        pbar    = tqdm(total=n_steps, desc="Simulating",
                       disable=not show_progress)
        while self.time < t_max and not self.propagation_complete:
            self.step(dt, trigger_center=trigger_center)
            pbar.update(1)
        if self.propagation_complete and self.time < t_max:
            pbar.set_postfix_str(f"complete at {self.time:.1f} s")
        pbar.close()
        return self.compile_results()

    def compile_results(self) -> Dict[str, Any]:
        times     = np.array([s["time"] for s in self.cell_states], dtype=float)
        T_history = np.zeros((len(times), self.n_cells), dtype=float)
        for t_idx, st in enumerate(self.cell_states):
            for c_idx, cs in enumerate(st["states"]):
                T_history[t_idx, c_idx] = float(cs["T"])
        trigger_times = np.full(self.n_cells, np.nan, dtype=float)
        for idx in self.trigger_order:
            tt = self.cells[idx].trigger_time
            if tt is not None:
                trigger_times[idx] = float(tt)
        return {
            "times":                times,
            "T_history":            T_history,
            "trigger_order":        list(self.trigger_order),
            "trigger_times":        trigger_times,
            "n_cells":              self.n_cells,
            "n_rows":               self.n_rows,
            "n_cols":               self.n_cols,
            "propagation_complete": bool(self.propagation_complete),
            "final_time":           float(self.time),
        }


# =============================================================================
# MULTIPROCESSING WORKER
# =============================================================================
def _run_single_simulation(args: tuple) -> Dict[str, Any]:
    param_dict, t_max, dt, seed = args
    rng = np.random.default_rng(int(seed))

    seed_cfg = param_dict.get("seed_cfg")
    if isinstance(seed_cfg, dict):
        seed_cfg = ModuleSeedConfig(**seed_cfg)
    elif seed_cfg is None:
        seed_cfg = ModuleSeedConfig()

    module = BatteryModule(
        n_rows     = int(param_dict.get("n_rows",    3)),
        n_cols     = int(param_dict.get("n_cols",    3)),
        cell_spacing = float(param_dict.get("cell_spacing",
                                            BatteryModule._CELL_SPACING_DEFAULT)),
        cooling    = str(param_dict.get("cooling",   "air")),
        rng        = rng,
        disable_cell_internal_conv = bool(
            param_dict.get("disable_cell_internal_conv", True)),
        radiation_scale = float(param_dict.get("radiation_scale",
                                               BatteryModule._RADIATION_SCALE_DEFAULT)),
        seed_cfg   = seed_cfg,
        gap_eps    = float(param_dict.get("gap_eps", 1e-6)),
    )
    # allow overrides from param_dict
    module.k_contact               = float(param_dict.get("k_contact",
                                           module.k_contact))
    module.contact_area            = float(param_dict.get("contact_area",
                                           module.contact_area))
    module.h_cooling               = float(param_dict.get("h_cooling",
                                           module.h_cooling))
    module.vent_energy_per_neighbour = float(
        param_dict.get("vent_energy_per_neighbour",
                       module.vent_energy_per_neighbour))
    module.vent_duration           = float(param_dict.get("vent_duration",
                                           module.vent_duration))

    results          = module.run_simulation(
        t_max=float(t_max), dt=float(dt),
        trigger_center=True, show_progress=False)
    results["seed"]  = int(seed)
    return results


# =============================================================================
# PART 3 — SINGLE-CELL MONTE CARLO
# =============================================================================
def run_single_cell_mc(
    n_simulations:  int   = 100,
    heating_power:  float = 50.0,
    t_max:          float = 1200.0,
    dt:             float = 0.1,
    seed:           int   = 0,
) -> Dict[str, Any]:
    trigger_temps  = []
    total_energies = []

    for k in tqdm(range(n_simulations), desc="Single-cell MC"):
        rng    = np.random.default_rng(seed + k)
        params = {
            "mass":             0.070 * (1.0 + 0.03 * rng.standard_normal()),
            "Cp":               1100.0 * (1.0 + 0.02 * rng.standard_normal()),
            "T_trigger_std":    15.0,
            "H_variation_std":  0.08,
        }
        cell = CellThermalRunaway(cell_id=0, params=params, rng=rng)
        t = 0.0
        while t < t_max and not cell.triggered:
            t += dt
            cell.step(dt, t_now=t, Q_ext=heating_power)
        while t < t_max:
            t += dt
            cell.step(dt, t_now=t, Q_ext=0.0)

        energy_J = _trapz(
            np.array(cell.Q_gen_history), np.array(cell.time_history))
        total_energies.append(float(energy_J / 1000.0))
        trigger_temps.append(float(cell.T_trigger_sample))

    return {
        "trigger_temps":   np.array(trigger_temps,  dtype=float),
        "total_energies":  np.array(total_energies, dtype=float),
    }


# =============================================================================
# PART 4 — PUBLICATION-QUALITY FIGURES
# =============================================================================

# ── helper: axis panel label ──────────────────────────────────────────────────
def _panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def plot_single_cell_response(
    cell: CellThermalRunaway,
    t_max: float = 1200.0,
    heating_power: float = 50.0,
    save_name: str = "fig1_single_cell_response",
) -> plt.Figure:
    """Four-panel single-cell TR figure (publication quality)."""
    print("Simulating single cell …")
    dt = 0.1
    cell.reset()
    t = 0.0
    while t < t_max and not cell.triggered:
        t += dt;  cell.step(dt, t_now=t, Q_ext=heating_power)
    while t < t_max:
        t += dt;  cell.step(dt, t_now=t, Q_ext=0.0)

    times = np.array(cell.time_history, dtype=float)
    T_arr = np.array(cell.T_history,    dtype=float)
    Q_arr = np.array(cell.Q_gen_history, dtype=float)
    t_tr  = cell.trigger_time

    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.36)

    # (a) Temperature profile
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(times, T_arr, color=C["red"], lw=1.8, label="Cell temperature")
    ax.axhline(cell.params["T_trigger_mean"], ls="--", color=C["grey"], lw=1.1,
               label=f"Mean T$_{{tr}}$ = {cell.params['T_trigger_mean']:.0f} °C")
    ax.axhline(cell.T_trigger_sample, ls=":", color=C["orange"], lw=1.4,
               label=f"Sampled T$_{{tr}}$ = {cell.T_trigger_sample:.1f} °C")
    if t_tr is not None:
        ax.axvline(t_tr, ls=":", color=C["purple"], lw=1.4,
                   label=f"TR onset  {t_tr:.1f} s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.set_xlim(left=0)
    _panel_label(ax, "(a)")

    # (b) Heat generation (log scale)
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(times, np.maximum(Q_arr, 1e-12), color=C["orange"], lw=1.8)
    if t_tr is not None:
        ax.axvline(t_tr, ls=":", color=C["purple"], lw=1.4,
                   label=f"TR onset  {t_tr:.1f} s")
        ax.legend(fontsize=7.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heat generation rate (W)")
    ax.grid(True, alpha=0.2, lw=0.5, which="both")
    ax.set_xlim(left=0)
    _panel_label(ax, "(b)")

    # (c) Reactant depletion
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(times, cell.c_sei_history, label="SEI",        color=C["blue"],   lw=1.5)
    ax.plot(times, cell.c_an_history,  label="Anode",      color=C["red"],    lw=1.5)
    ax.plot(times, cell.c_ca_history,  label="Cathode",    color=C["green"],  lw=1.5)
    ax.plot(times, cell.c_el_history,  label="Electrolyte",color=C["orange"], lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised reactant fraction (−)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.set_xlim(left=0)
    _panel_label(ax, "(c)")

    # (d) Phase plot Q vs T (time-coloured)
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(T_arr, Q_arr, c=times, s=5, alpha=0.7, cmap="viridis",
                    rasterized=True)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Heat generation rate (W)")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Time (s)", fontsize=9)
    ax.grid(True, alpha=0.2, lw=0.5)
    _panel_label(ax, "(d)")

    fig.suptitle("Single-Cell Thermal Runaway Response — LG M50 21700 (NMC)",
                 fontsize=11, fontweight="bold", y=1.005)
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    return fig


def plot_module_propagation(
    results: Dict[str, Any],
    save_name: str = "fig2_module_propagation",
) -> plt.Figure:
    """Four-panel module propagation figure (publication quality)."""
    times         = results["times"]
    T_hist        = results["T_history"]
    n_cells       = results["n_cells"]
    n_rows        = results["n_rows"]
    n_cols        = results["n_cols"]
    trig_order    = results["trigger_order"]
    trig_times    = results["trigger_times"]

    # ── colour cells by trigger order ────────────────────────────────────────
    cmap_cells = plt.cm.tab10

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.35)

    # (a) Temperature histories
    ax = fig.add_subplot(gs[0, 0])
    for i in range(n_cells):
        bold  = trig_order and i == trig_order[0]
        ax.plot(times, T_hist[:, i],
                lw=2.0 if bold else 0.8,
                alpha=0.95 if bold else 0.50,
                color=cmap_cells(i % 10),
                label=f"Cell {i}" if bold else None)
    for tt in trig_times:
        if not np.isnan(tt):
            ax.axvline(tt, ls=":", alpha=0.18, color="black", lw=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.2, lw=0.5)
    if trig_order:
        ax.legend(fontsize=7.5)
    _panel_label(ax, "(a)")

    # (b) Final temperature heat-map with trigger order overlay
    ax  = fig.add_subplot(gs[0, 1])
    fT  = T_hist[-1, :].reshape(n_rows, n_cols)
    vmax = max(500.0, float(np.nanmax(fT)))
    hot_cmap = LinearSegmentedColormap.from_list(
        "hot_pub", ["#f7f7f7", "#fee08b", "#f46d43", "#a50026"])
    im = ax.imshow(fT, cmap=hot_cmap, vmin=25.0, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    ax.set_title(f"Final temperature map  (t = {times[-1]:.0f} s)",
                 fontsize=9, pad=4)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)", fontsize=9)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx in trig_order:
                lbl   = str(trig_order.index(idx))
                tcol  = "white" if fT[r, c] < (vmax * 0.6) else "black"
                ax.text(c, r, lbl, ha="center", va="center",
                        color=tcol, fontweight="bold", fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Trigger timeline (Gantt-style)
    ax = fig.add_subplot(gs[1, 0])
    if trig_order:
        y    = np.arange(len(trig_order))
        tts  = [trig_times[idx] for idx in trig_order]
        bars = ax.barh(y, tts, color=C["blue"], edgecolor="white", height=0.6,
                       alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([f"Cell {idx}" for idx in trig_order], fontsize=8)
        ax.set_xlabel("Trigger time (s)")
        ax.grid(True, alpha=0.2, lw=0.5, axis="x")
        for bar, val in zip(bars, tts):
            ax.text(val + max(tts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f} s", va="center", fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "No cells triggered",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    _panel_label(ax, "(c)")

    # (d) Propagation speed vs distance
    ax = fig.add_subplot(gs[1, 1])
    if len(trig_order) > 1:
        t0r, t0c = divmod(trig_order[0], n_cols)
        dists, speeds = [], []
        for idx in trig_order[1:]:
            r, c = divmod(idx, n_cols)
            d_m  = np.sqrt((r - t0r)**2 + (c - t0c)**2) * 0.021
            dt_  = trig_times[idx] - trig_times[trig_order[0]]
            if dt_ > 0:
                dists.append(d_m * 1000)
                speeds.append((d_m / dt_) * 1000)
        if speeds:
            ax.plot(dists, speeds, "o", color=C["green"], ms=7,
                    markerfacecolor="white", markeredgewidth=1.5, lw=1.5,
                    linestyle="-")
            mu_s = np.mean(speeds)
            ax.axhline(mu_s, ls="--", color=C["grey"], lw=1.2,
                       label=f"Mean = {mu_s:.2f} mm s⁻¹")
            ax.set_xlabel("Distance from trigger cell (mm)")
            ax.set_ylabel("Propagation speed (mm s⁻¹)")
            ax.legend(fontsize=7.5)
            ax.grid(True, alpha=0.2, lw=0.5)
        else:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Only one cell triggered",
                ha="center", va="center", transform=ax.transAxes)
    _panel_label(ax, "(d)")

    fig.suptitle("Module-Level Thermal Runaway Propagation (3×3, air cooling)",
                 fontsize=11, fontweight="bold", y=1.005)
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    return fig


def plot_comparison_with_experiment(
    sim_trigger,
    sim_energy,
    exp_csv_path: str,
    save_name: str = "fig3_validation_vs_experiment",
) -> plt.Figure:
    """Overlay simulation vs NREL experimental distributions."""
    exp_df      = pd.read_csv(exp_csv_path)
    exp_trigger = pd.to_numeric(
        exp_df["Avg-Cell-Temp-At-Trigger-degC"], errors="coerce").dropna()
    exp_energy  = pd.to_numeric(
        exp_df["Corrected-Total-Energy-Yield-kJ"], errors="coerce").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, exp, sim, xlabel, lbl in zip(
        axes,
        [exp_trigger, exp_energy],
        [sim_trigger, sim_energy],
        ["Trigger temperature (°C)", "Total energy released (kJ)"],
        ["(a)", "(b)"],
    ):
        bins = np.linspace(min(exp.min(), sim.min()),
                           max(exp.max(), sim.max()), 12)
        ax.hist(exp, bins=bins, alpha=0.55, label="Experimental",
                color=C["blue"],   edgecolor="white", density=True)
        ax.hist(sim, bins=bins, alpha=0.55, label="Simulation",
                color=C["orange"], edgecolor="white", density=True)
        ax.axvline(exp.mean(), color=C["blue"],   ls="--", lw=1.4,
                   label=f"Exp mean = {exp.mean():.1f}")
        ax.axvline(float(np.mean(sim)), color=C["orange"], ls="--", lw=1.4,
                   label=f"Sim mean = {np.mean(sim):.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, lw=0.5)
        _panel_label(ax, lbl)

    fig.suptitle(
        "Single-Cell Model Validation — NREL Battery Failure Databank (LG M50 21700)",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    return fig


def monte_carlo_parallel(
    param_dict:     Dict[str, Any],
    n_simulations:  int   = 100,
    t_max:          float = 300.0,
    dt:             float = 0.1,
    max_workers:    int | None = None,
    force_serial:   bool | None = None,
    save_name:      str   = "fig4_monte_carlo_baseline",
    seed:           int   = 0,
) -> Dict[str, Any]:
    """
    Run an ensemble of module simulations and produce a four-panel
    publication-quality summary figure with confidence intervals.
    """
    seeds    = list(range(seed, seed + n_simulations))
    tasks    = [(param_dict, t_max, dt, s) for s in seeds]
    if force_serial is None:
        force_serial = _in_jupyter()

    def _serial() -> List[Dict]:
        out = []
        for s in tqdm(seeds, desc="Monte Carlo (serial)"):
            out.append(_run_single_simulation((param_dict, t_max, dt, s)))
        return out

    results_list: List[Dict[str, Any]] = []
    if force_serial:
        print(f"\nMonte Carlo ({n_simulations} runs) — SERIAL")
        results_list = _serial()
    else:
        try:
            workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
            print(f"\nMonte Carlo ({n_simulations} runs) — PARALLEL ({workers} workers)")
            with mp.Pool(processes=workers) as pool:
                for res in tqdm(
                    pool.imap_unordered(_run_single_simulation, tasks),
                    total=n_simulations, desc="Monte Carlo (parallel)"
                ):
                    results_list.append(res)
        except Exception as exc:
            print(f"\n[Parallel failed → serial fallback] {exc}")
            results_list = _serial()

    r0          = results_list[0]
    total_cells = r0["n_cells"]
    n_rows_mc   = r0["n_rows"]
    n_cols_mc   = r0["n_cols"]

    n_trig_arr   = np.array([len(r["trigger_order"]) for r in results_list], dtype=float)
    trig_counts  = np.zeros(total_cells, dtype=float)
    prop_times   = []
    final_temps  = []

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

    prop_times_arr = np.array(prop_times, dtype=float)
    final_temps_arr = np.array(final_temps, dtype=float)
    valid_prop      = prop_times_arr[~np.isnan(prop_times_arr)]

    prop_prob   = float(np.mean(n_trig_arr == total_cells))
    prob_map    = (trig_counts / n_simulations).reshape(n_rows_mc, n_cols_mc)

    # Bootstrap 95 % CI for propagation probability
    bs_props = []
    rng_bs   = np.random.default_rng(seed + 999)
    for _ in range(2000):
        idx_bs = rng_bs.integers(0, n_simulations, n_simulations)
        bs_props.append(float(np.mean(n_trig_arr[idx_bs] == total_cells)))
    ci_low  = float(np.percentile(bs_props, 2.5))
    ci_high = float(np.percentile(bs_props, 97.5))

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.close("all")
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.38)

    # (a) Histogram of cells triggered
    ax = fig.add_subplot(gs[0, 0])
    bins_trig = np.arange(0.5, total_cells + 1.5, 1.0)
    ax.hist(n_trig_arr, bins=bins_trig, color=C["blue"], edgecolor="white",
            alpha=0.80, density=False)
    ax.axvline(n_trig_arr.mean(), ls="--", color=C["red"], lw=1.5,
               label=f"Mean = {n_trig_arr.mean():.2f} ± {n_trig_arr.std():.2f}")
    ax.set_xlabel("Number of cells triggered")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(1, total_cells + 1))
    ax.set_title(
        f"Full propagation: {prop_prob:.1%}  (95 % CI: {ci_low:.1%}–{ci_high:.1%})",
        fontsize=8.5, pad=4)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.2, lw=0.5, axis="y")
    _panel_label(ax, "(a)")

    # (b) Propagation time distribution (full-propagation runs only)
    ax = fig.add_subplot(gs[0, 1])
    if len(valid_prop) > 0:
        ax.hist(valid_prop, bins=min(15, max(5, len(valid_prop)//5)),
                color=C["red"], edgecolor="white", alpha=0.80)
        ax.axvline(np.mean(valid_prop), ls="--", color="black", lw=1.4,
                   label=f"Mean = {np.mean(valid_prop):.1f} s")
        ax.axvline(np.percentile(valid_prop, 25), ls=":", color=C["grey"], lw=1.0)
        ax.axvline(np.percentile(valid_prop, 75), ls=":", color=C["grey"], lw=1.0,
                   label="IQR")
        ax.set_xlabel("Full propagation time (s)")
        ax.set_ylabel("Frequency")
        ax.set_title("Propagation time — full-cascade runs only",
                     fontsize=8.5, pad=4)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.2, lw=0.5, axis="y")
    else:
        ax.text(0.5, 0.5, "No full-propagation runs\nin this ensemble",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Propagation time distribution", fontsize=8.5, pad=4)
    _panel_label(ax, "(b)")

    # (c) Mean final temperature per cell
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(final_temps_arr, bins=20,
            color="#E69F00", edgecolor="white", alpha=0.80)
    ax.axvline(final_temps_arr.mean(), ls="--", color="black", lw=1.4,
               label=f"Mean = {final_temps_arr.mean():.0f} °C")
    ax.set_xlabel("Mean module temperature at end of simulation (°C)")
    ax.set_ylabel("Frequency")
    ax.set_title("Final mean module temperature", fontsize=8.5, pad=4)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.2, lw=0.5, axis="y")
    _panel_label(ax, "(c)")

    # (d) Spatial trigger probability map
    ax = fig.add_subplot(gs[1, 1])
    cmap_prob = LinearSegmentedColormap.from_list(
        "prob", ["#ffffff", "#fee08b", "#f46d43", "#a50026"])
    im = ax.imshow(prob_map, cmap=cmap_prob, vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    ax.set_title("Cell trigger probability map", fontsize=8.5, pad=4)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Trigger probability", fontsize=9)
    for i in range(n_rows_mc):
        for j in range(n_cols_mc):
            ax.text(j, i, f"{prob_map[i, j]:.2f}",
                    ha="center", va="center",
                    fontweight="bold", fontsize=10,
                    color="white" if prob_map[i, j] > 0.6 else "black")
    _panel_label(ax, "(d)")

    cooling_str = str(param_dict.get("cooling", "air")).upper()
    fig.suptitle(
        f"Monte Carlo Propagation Ensemble  (N = {n_simulations}, "
        f"cooling: {cooling_str}, 3×3 module)",
        fontsize=11, fontweight="bold", y=1.005)
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()

    return {
        "propagation_probability_full":   prop_prob,
        "ci_low_95":                      ci_low,
        "ci_high_95":                     ci_high,
        "mean_triggered_cells":           float(np.mean(n_trig_arr)),
        "std_triggered_cells":            float(np.std(n_trig_arr)),
        "mean_propagation_time_full":     float(np.nanmean(prop_times_arr)),
        "std_propagation_time_full":      float(np.nanstd(prop_times_arr)),
        "mean_final_temp":                float(np.mean(final_temps_arr)),
        "probability_map":                prob_map,
        "raw_n_triggered":                n_trig_arr,
        "raw_prop_times":                 prop_times_arr,
        "raw_final_temps":                final_temps_arr,
    }


def plot_air_vs_liquid(
    mc_air: Dict[str, Any],
    mc_liq: Dict[str, Any],
    n_simulations: int = 100,
    save_name: str = "fig5_air_vs_liquid",
) -> plt.Figure:
    """Side-by-side comparison of air and liquid cooling MC ensembles."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    total_cells = int(round(mc_air["raw_n_triggered"].max()))

    # ── (a) triggered cells comparison ────────────────────────────────────────
    ax   = axes[0]
    bins = np.arange(0.5, total_cells + 1.5, 1.0)
    ax.hist(mc_air["raw_n_triggered"], bins=bins, alpha=0.65,
            color=C["orange"], edgecolor="white",
            label=f"Air  (mean {mc_air['mean_triggered_cells']:.1f})")
    ax.hist(mc_liq["raw_n_triggered"], bins=bins, alpha=0.65,
            color=C["blue"],   edgecolor="white",
            label=f"Liquid (mean {mc_liq['mean_triggered_cells']:.1f})")
    ax.set_xlabel("Cells triggered")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(1, total_cells + 1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, lw=0.5, axis="y")
    _panel_label(ax, "(a)")

    # ── (b) propagation probability bar chart ─────────────────────────────────
    ax = axes[1]
    labels = ["Air cooling", "Liquid cooling"]
    probs  = [mc_air["propagation_probability_full"],
              mc_liq["propagation_probability_full"]]
    errs_lo = [probs[0] - mc_air["ci_low_95"],
               probs[1] - mc_liq["ci_low_95"]]
    errs_hi = [mc_air["ci_high_95"] - probs[0],
               mc_liq["ci_high_95"] - probs[1]]
    colors  = [C["orange"], C["blue"]]
    bars    = ax.bar(labels, probs, color=colors, alpha=0.80, edgecolor="white",
                     width=0.5,
                     yerr=[errs_lo, errs_hi], capsize=6,
                     error_kw={"elinewidth": 1.2, "ecolor": "black"})
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.03,
                f"{p:.0%}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Full-propagation probability")
    ax.set_title("Cooling strategy comparison", fontsize=9, pad=4)
    ax.grid(True, alpha=0.2, lw=0.5, axis="y")
    _panel_label(ax, "(b)")

    # ── (c) final temperature comparison ──────────────────────────────────────
    ax = axes[2]
    t_air = mc_air["raw_final_temps"]
    t_liq = mc_liq["raw_final_temps"]
    all_T = np.concatenate([t_air, t_liq])
    bins2 = np.linspace(all_T.min(), all_T.max(), 20)
    ax.hist(t_air, bins=bins2, alpha=0.65, color=C["orange"], edgecolor="white",
            label=f"Air  (mean {t_air.mean():.0f} °C)")
    ax.hist(t_liq, bins=bins2, alpha=0.65, color=C["blue"],   edgecolor="white",
            label=f"Liquid (mean {t_liq.mean():.0f} °C)")
    ax.set_xlabel("Mean final module temperature (°C)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, lw=0.5, axis="y")
    _panel_label(ax, "(c)")

    fig.suptitle("Effect of Cooling Strategy on Thermal Runaway Propagation",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()
    return fig


# =============================================================================
# PART 5 — PROPAGATION REGIME ANALYSIS
# =============================================================================
def classify_regime(n_triggered: int, n_cells: int) -> int:
    if n_triggered <= 1:   return 0
    if n_triggered < n_cells: return 1
    return 2


def _single_run_regime(args) -> Tuple[int, int]:
    params, seed = args
    r      = _run_single_simulation(
        (params, params.get("t_max", 300.0), params.get("dt", 0.2), seed))
    n_trig = len(r["trigger_order"])
    return n_trig, classify_regime(n_trig, r["n_cells"])


def run_1d_regime_sweep(
    param_name:  str,
    param_values: np.ndarray,
    fixed:       Dict[str, Any],
    n_mc:        int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_frac          = []
    p_none, p_part, p_full = [], [], []
    sem_frac           = []   # standard error on mean fraction

    for val in param_values:
        p = dict(fixed)
        p[param_name] = float(val)
        results  = [_single_run_regime((p, s)) for s in range(n_mc)]
        n_cells  = int(fixed.get("n_rows", 3)) * int(fixed.get("n_cols", 3))
        fracs    = np.array([r[0] / n_cells for r in results])
        regs     = [r[1] for r in results]
        mean_frac.append(float(fracs.mean()))
        sem_frac.append(float(fracs.std() / np.sqrt(n_mc)))
        p_none.append(regs.count(0) / n_mc)
        p_part.append(regs.count(1) / n_mc)
        p_full.append(regs.count(2) / n_mc)

    return (np.array(mean_frac), np.array(sem_frac),
            np.array(p_none), np.array(p_part), np.array(p_full))


def propagation_regime_analysis(
    output_prefix: str = "fig_regime",
    fixed: Optional[Dict[str, Any]] = None,
    n_mc: int = 20,
) -> None:
    print("\n" + "=" * 60)
    print("SECTION — PROPAGATION REGIME ANALYSIS")
    print("=" * 60)

    if fixed is None:
        fixed = dict(
            n_rows=3, n_cols=3,
            cooling="air",
            disable_cell_internal_conv=True,
            radiation_scale  = 0.20,
            cell_spacing     = 0.002,
            k_contact        = 0.20,
            contact_area     = 0.001,
            h_cooling        = 2.0,
            vent_energy_per_neighbour = 10_000.0,
            vent_duration    = 35.0,
            gap_eps          = 1e-6,
            seed_cfg         = {"mode": "temperature", "T_seed_C": 250.0},
            t_max            = 300.0,
            dt               = 0.2,
        )

    # Sweep ranges chosen to bracket realistic operating conditions
    sweeps = {
        "vent_energy_per_neighbour": np.linspace(1_000,  18_000, 12),
        "cell_spacing":              np.linspace(0.001,  0.010,  10),
        "k_contact":                 np.linspace(0.05,   0.80,   10),
        "h_cooling":                 np.linspace(1.0,    500.0,  10),
    }
    xlabels = {
        "vent_energy_per_neighbour": "Vent energy per neighbour  (J)",
        "cell_spacing":              "Cell-to-cell gap  (m)",
        "k_contact":                 "Contact thermal conductance  (W m⁻¹ K⁻¹)",
        "h_cooling":                 "Module cooling coefficient  (W m⁻² K⁻¹)",
    }

    sweep_results = {}
    for pname, pvals in sweeps.items():
        print(f"  Sweeping {pname}  ({len(pvals)} values × {n_mc} MC) …")
        mf, sem, pn, pp, pf = run_1d_regime_sweep(pname, pvals, fixed, n_mc=n_mc)
        sweep_results[pname] = (pvals, mf, sem, pn, pp, pf)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Thermal Runaway Propagation Regime vs Key Physical Parameters",
        fontsize=12, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=REGIME_COLORS["none"],    label=REGIME_LABELS["none"]),
        mpatches.Patch(color=REGIME_COLORS["partial"], label=REGIME_LABELS["partial"]),
        mpatches.Patch(color=REGIME_COLORS["full"],    label=REGIME_LABELS["full"]),
    ]

    for ax, (pname, (pvals, mf, sem, pn, pp, pf)) in zip(
        axes.flat, sweep_results.items()
    ):
        ax.stackplot(
            pvals, pn, pp, pf,
            colors=[REGIME_COLORS["none"],
                    REGIME_COLORS["partial"],
                    REGIME_COLORS["full"]],
            alpha=0.80,
        )
        # mean fraction with ±1 SE shading
        ax.plot(pvals, mf, "k-", lw=1.8, label="Mean triggered fraction", zorder=5)
        ax.fill_between(pvals, mf - sem, mf + sem,
                        color="black", alpha=0.18, label="±1 SE")
        ax.set_xlabel(xlabels[pname], fontsize=9.5)
        ax.set_ylabel("Probability  /  Fraction", fontsize=9.5)
        ax.set_title(pname.replace("_", " ").title(),
                     fontsize=10, fontweight="bold", pad=4)
        ax.set_ylim(0, 1)
        ax.set_xlim(pvals[0], pvals[-1])
        ax.grid(True, alpha=0.20, lw=0.5)

    # shared legend
    fig.legend(handles=legend_patches + [
                   plt.Line2D([0], [0], color="black", lw=1.8,
                              label="Mean triggered fraction")],
               loc="lower center", ncol=4, fontsize=8.5,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, f"{output_prefix}_1d_sweeps")
    plt.show()


# =============================================================================
# BASELINE PARAMETER DICT  (shared across main and workers)
# =============================================================================
BASELINE_PARAMS = dict(
    n_rows   = 3,
    n_cols   = 3,
    cooling  = "air",
    disable_cell_internal_conv = True,
    # ── updated physics ──────────────────────────────────────────────────────
    cell_spacing               = 0.002,      # m  (2 mm gap)
    k_contact                  = 0.20,       # W/m·K  (was 1.0)
    contact_area               = 0.001,      # m²
    h_cooling                  = 2.0,        # W/m²·K (air natural convection)
    radiation_scale            = 0.20,       # (was 0.3)
    vent_energy_per_neighbour  = 10_000.0,   # J  (was 500)
    vent_duration              = 35.0,       # s  (was 20)
    gap_eps                    = 1e-6,
    seed_cfg = {"mode": "temperature", "T_seed_C": 250.0},
)

LIQUID_PARAMS = {**BASELINE_PARAMS, "cooling": "liquid", "h_cooling": 200.0}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")   # headless / file-only rendering

    SEED = 0
    SEP  = "=" * 70
    sep  = "-" * 70

    print(SEP)
    print("  THERMAL RUNAWAY PROPAGATION SIMULATION — v4.0 (Physics-corrected)")
    print(f"  Output figures → {FIG_DIR}")
    print(SEP)

    # ── 1) Single-cell Monte Carlo ────────────────────────────────────────────
    N_SC   = 100
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

    # optional validation plot
    exp_csv = r"E:\Thermal Runaway\lg_m50_failure_data_with_trigger.csv"
    if os.path.exists(exp_csv):
        plot_comparison_with_experiment(
            sc_data["trigger_temps"], sc_data["total_energies"], exp_csv)
    else:
        print(f"\n  [INFO] Experimental CSV not found at {exp_csv!r} — "
              "skipping validation figure.")

    # ── 2) Single deterministic module run ────────────────────────────────────
    print(sep)
    print("  SINGLE MODULE RUN (deterministic seed)")
    print(sep)
    rng    = np.random.default_rng(SEED)
    module = BatteryModule(
        n_rows=3, n_cols=3,
        cell_spacing    = BASELINE_PARAMS["cell_spacing"],
        cooling         = BASELINE_PARAMS["cooling"],
        rng             = rng,
        disable_cell_internal_conv = True,
        radiation_scale = BASELINE_PARAMS["radiation_scale"],
        seed_cfg        = ModuleSeedConfig(mode="temperature", T_seed_C=250.0),
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
        tt = results["trigger_times"]
        o  = results["trigger_order"]
        print(f"  Propagation span    : {tt[o[-1]] - tt[o[0]]:.1f} s")
    export_json(results, os.path.join(FIG_DIR, "module_single_run_results.json"))

    # ── 3) Monte Carlo — air cooling ──────────────────────────────────────────
    print(sep)
    print("  MONTE CARLO — AIR COOLING (100 runs)")
    print(sep)
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

    # ── 4) Monte Carlo — liquid cooling ───────────────────────────────────────
    print(sep)
    print("  MONTE CARLO — LIQUID COOLING (100 runs)")
    print(sep)
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

    # ── 5) Regime analysis ────────────────────────────────────────────────────
    propagation_regime_analysis(output_prefix="fig_regime", fixed=None, n_mc=20)

    print(SEP)
    print("  ALL FIGURES SAVED TO:", FIG_DIR)
    print(SEP)
