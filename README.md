# HPLC Plate Simulation (Tkinter + Matplotlib)

Interactive teaching tool to visualize HPLC band broadening and plate theory from physically-inspired parameters.

This GUI uses:
- Stokes–Einstein diffusion for molecular diffusion coefficients
- A simplified Van Deemter–type equation for plate height
- Ergun equation for pressure drop in packed beds
- A Monte Carlo plate-stepping model to generate chromatograms for two analytes

All of this is exposed through a simple Tkinter interface with a Matplotlib plot.

---

## Features

- **Interactive GUI** (Tkinter) with:
  - Column geometry: ID, length, particle size, porosity  
  - Mobile phase conditions: flow, temperature, %ACN (water/ACN mix)
  - Solute properties: hydrodynamic radius, “move probability” for two analytes
  - Numerical parameters: number of molecules, injection spread, max simulation steps
  - Max allowed pressure (for quick pressure-feasibility check)

- **Physically-motivated models:**
  - Stokes–Einstein diffusion coefficient
  - Water/ACN viscosity vs temperature with log-additive mixing rule
  - Linear velocity from flow and porosity
  - Van Deemter-like relation
  - Ergun equation for pressure drop \(\Delta P\) across the column

- **Chromatogram simulation:**
  - Two analytes (A and B) with independent “move” probabilities (`p_move_A`, `p_move_B`)
  - Plate stepping Monte Carlo dynamics
  - Injection-time dispersion (Gaussian delay in plate steps)
  - Output plotted as **molecules eluted per step vs time** (in minutes)

- **On-plot annotations:**
  - Title summarizing key conditions (L, ID, flow, T, %ACN, viscosity, u, H, N, Dm)
  - Van Deemter equation box
  - Legend for analytes A and B
  - Pressure box with color feedback:
    - Green if `ΔP` < `P max`
    - Red if `ΔP` > `P max`
  - **Optional FWHM box** (toggle via checkbox) with FWHM (in minutes) for A and B

---

## Requirements

- Python 3.8+  
- Packages:
  - `numpy`
  - `matplotlib`
  - `tkinter` (usually included with standard Python installs)

On most systems with a recent Python installation:

```bash
pip install numpy matplotlib
