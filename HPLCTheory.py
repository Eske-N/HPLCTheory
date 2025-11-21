import numpy as np
import matplotlib.pyplot as plt

# Use TkAgg backend for embedding in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont


# Physical constants
k_BOLTZMANN = 1.380649e-23  # J/K


# ---------- Physical helper functions ----------

def stokes_einstein_D(T_K, viscosity_Pa_s, radius_m):
    """
    Stokes–Einstein diffusion coefficient:
    D = k_B * T / (6 * pi * eta * r)
    """
    return k_BOLTZMANN * T_K / (6 * np.pi * viscosity_Pa_s * radius_m)


def linear_velocity(flow_mL_min, inner_diam_mm, porosity=0.6):
    """
    Compute linear velocity u [m/s] from volumetric flow and column geometry.

    u = (flow / (pi * (r^2) * porosity))
    """
    # Convert units
    flow_m3_s = flow_mL_min * 1e-6 / 60.0          # mL/min -> m^3/s
    radius_m = (inner_diam_mm * 1e-3) / 2.0        # mm -> m
    area_m2 = np.pi * radius_m**2

    u = flow_m3_s / (area_m2 * porosity)
    return u  # m/s


def plate_height_vandeemter(u, dp_m, Dm_m2_s,
                            k_A=2.0, gamma=2.0, k_C=0.05):
    """
    Simplified, physically-parameterized Van Deemter-type relation.

    H(u) = A + B/u + C*u
    """
    A = k_A * dp_m
    B = 2.0 * gamma * Dm_m2_s
    C = k_C * dp_m**2 / Dm_m2_s
    return A + B / u + C * u  # m


# ----- Viscosity model: water / acetonitrile mixture -----

def _interp_property(T_C, T_points, values):
    """Simple 1D linear interpolation for properties vs temperature."""
    return np.interp(T_C, T_points, values)


def viscosity_water_cP(T_C):
    """
    Approximate viscosity of pure water [cP] between ~20–60 °C.
    Very rough but OK for teaching.
    """
    T_points = np.array([20.0, 25.0, 40.0, 60.0])
    eta_points = np.array([1.00, 0.89, 0.65, 0.47])  # cP
    return _interp_property(T_C, T_points, eta_points)


def viscosity_acn_cP(T_C):
    """
    Approximate viscosity of pure acetonitrile [cP] between ~20–60 °C.
    Crude linear approximation, good enough pedagogically.
    """
    T_points = np.array([20.0, 25.0, 40.0, 60.0])
    eta_points = np.array([0.38, 0.37, 0.32, 0.29])  # cP, approx
    return _interp_property(T_C, T_points, eta_points)


def viscosity_water_acn_Pa_s(T_C, acn_vol_frac):
    """
    Approximate viscosity [Pa·s] of a water/ACN mixture
    at temperature T_C (°C) and ACN volume fraction (0–1).

    Mixing rule: log(eta_mix) = xw*log(eta_w) + xa*log(eta_a)
    (ideal log-additive, purely for teaching).
    """
    acn_vol_frac = max(0.0, min(1.0, acn_vol_frac))
    x_acn = acn_vol_frac
    x_water = 1.0 - x_acn

    eta_w_cP = viscosity_water_cP(T_C)
    eta_a_cP = viscosity_acn_cP(T_C)

    # log-additive mixing (ideal, pedagogical)
    ln_eta_mix = x_water * np.log(eta_w_cP) + x_acn * np.log(eta_a_cP)
    eta_mix_cP = np.exp(ln_eta_mix)

    return eta_mix_cP * 1e-3  # cP -> Pa·s


# ---------- Pressure drop (Ergun) ----------

def pressure_drop_ergun(flow_mL_min,
                        inner_diam_mm,
                        particle_size_um,
                        porosity,
                        viscosity_Pa_s,
                        length_m,
                        density_kg_m3=1000.0):
    """
    Approximate pressure drop using the Ergun equation for a packed bed.

    ΔP/L = 150 * (1 - ε)^2 / ε^3 * μ * u_s / d_p^2
         + 1.75 * (1 - ε) / ε^3 * ρ * u_s^2 / d_p
    """
    # units
    flow_m3_s = flow_mL_min * 1e-6 / 60.0          # m^3/s
    radius_m = (inner_diam_mm * 1e-3) / 2.0        # m
    area_m2 = np.pi * radius_m**2
    dp_m = particle_size_um * 1e-6                 # m
    eps = porosity

    if eps <= 0 or eps >= 1:
        return 0.0  # avoid nonsense / division by zero

    u_s = flow_m3_s / area_m2  # superficial velocity [m/s]

    term1 = 150.0 * (1.0 - eps)**2 / (eps**3) * viscosity_Pa_s * u_s / (dp_m**2)
    term2 = 1.75 * (1.0 - eps) / (eps**3) * density_kg_m3 * u_s**2 / dp_m
    deltaP_per_m = term1 + term2  # Pa/m

    return deltaP_per_m * length_m  # Pa


# ---------- Monte Carlo simulation using N from H ----------

def simulate_chromatogram_physical(
    n_molecules_A=1000,
    n_molecules_B=1000,
    p_move_A=0.6,
    p_move_B=0.3,
    # Column & packing
    column_length_cm=10.0,
    inner_diam_mm=2.1,
    particle_size_um=3.0,
    porosity=0.6,
    # Mobile phase & solute
    flow_mL_min=0.3,
    temperature_C=30.0,
    acn_vol_frac=0.5,          # fraction ACN (0–1)
    solute_radius_nm=0.5,
    # "Injection plug" dispersion
    injection_spread_steps=0.0,
    # Van Deemter-like empirical constants
    k_A=2.0,
    gamma=2.0,
    k_C=0.05,
    max_steps=1000,
    random_seed=42,
):
    """
    Simulate a chromatogram for 2 analytes.
    The number of plates N is computed from physical-ish parameters
    via a Van Deemter-like expression.

    Viscosity is not an input: it is computed from temperature and
    water/ACN composition.
    """
    rng = np.random.default_rng(random_seed)

    # --- Unit conversions ---
    L_m = column_length_cm / 100.0         # cm -> m
    dp_m = particle_size_um * 1e-6         # um -> m
    T_K = temperature_C + 273.15           # °C -> K
    radius_m = solute_radius_nm * 1e-9     # nm -> m

    # --- Viscosity from T + composition ---
    viscosity_Pa_s = viscosity_water_acn_Pa_s(temperature_C, acn_vol_frac)

    # --- Compute physical parameters ---
    u = linear_velocity(flow_mL_min, inner_diam_mm, porosity=porosity)  # m/s
    Dm = stokes_einstein_D(T_K, viscosity_Pa_s, radius_m)               # m^2/s
    H = plate_height_vandeemter(u, dp_m, Dm, k_A=k_A, gamma=gamma, k_C=k_C)

    N_float = L_m / H
    N_plates = int(round(N_float))
    if N_plates < 1:
        raise ValueError(
            f"Computed N = {N_float:.2f} (< 1). "
            "Check parameters (flow, particle size, etc.)."
        )

    # Pressure drop (Ergun)
    deltaP_Pa = pressure_drop_ergun(
        flow_mL_min=flow_mL_min,
        inner_diam_mm=inner_diam_mm,
        particle_size_um=particle_size_um,
        porosity=porosity,
        viscosity_Pa_s=viscosity_Pa_s,
        length_m=L_m,
        density_kg_m3=1000.0,
    )
    deltaP_bar = deltaP_Pa / 1e5  # 1 bar = 1e5 Pa

    # Time per "plate step": distance per plate / linear velocity
    plate_distance_m = L_m / N_plates
    dt_s = plate_distance_m / u  # time to traverse one plate length if moving

    # --- Initialize molecule positions ---
    positions_A = np.zeros(n_molecules_A, dtype=int)
    positions_B = np.zeros(n_molecules_B, dtype=int)
    active_A = np.ones(n_molecules_A, dtype=bool)
    active_B = np.ones(n_molecules_B, dtype=bool)

    # --- Injection timing dispersion (Gaussian in number of steps) ---
    if injection_spread_steps > 0:
        sigma = injection_spread_steps
        start_delay_A = np.clip(
            rng.normal(loc=0.0, scale=sigma, size=n_molecules_A),
            0, None
        ).astype(int)
        start_delay_B = np.clip(
            rng.normal(loc=0.0, scale=sigma, size=n_molecules_B),
            0, None
        ).astype(int)
    else:
        start_delay_A = np.zeros(n_molecules_A, dtype=int)
        start_delay_B = np.zeros(n_molecules_B, dtype=int)

    eluted_A = np.zeros(max_steps, dtype=int)
    eluted_B = np.zeros(max_steps, dtype=int)

    for step in range(max_steps):
        # --- A: only molecules whose delay has elapsed can move ---
        idx_A = np.where(active_A & (step >= start_delay_A))[0]
        if idx_A.size > 0:
            movers = rng.random(idx_A.size) < p_move_A
            positions_A[idx_A] += movers.astype(int)
            finished = active_A & (positions_A >= N_plates)
            eluted_A[step] = finished.sum()
            active_A[finished] = False

        # --- B ---
        idx_B = np.where(active_B & (step >= start_delay_B))[0]
        if idx_B.size > 0:
            movers = rng.random(idx_B.size) < p_move_B
            positions_B[idx_B] += movers.astype(int)
            finished = active_B & (positions_B >= N_plates)
            eluted_B[step] = finished.sum()
            active_B[finished] = False

        if not active_A.any() and not active_B.any():
            eluted_A = eluted_A[:step + 1]
            eluted_B = eluted_B[:step + 1]
            break

    times_s = np.arange(len(eluted_A)) * dt_s
    return times_s, eluted_A, eluted_B, N_plates, H, N_float, u, Dm, deltaP_bar, viscosity_Pa_s

def compute_fwhm_min(times_s, y):
    """
    Compute FWHM of a peak given times (s) and intensities y.
    Returns width in minutes, or None if no peak.
    """
    y = np.asarray(y)
    if y.size == 0 or y.max() <= 0:
        return None

    half = y.max() / 2.0
    idx = np.where(y >= half)[0]
    if idx.size == 0:
        return None

    left = idx[0]
    right = idx[-1]

    # Left crossing (linear interpolation)
    if left > 0:
        x1, y1 = times_s[left - 1], y[left - 1]
        x2, y2 = times_s[left], y[left]
        t_left = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
    else:
        t_left = times_s[left]

    # Right crossing (linear interpolation)
    if right < len(y) - 1:
        x1, y1 = times_s[right], y[right]
        x2, y2 = times_s[right + 1], y[right + 1]
        t_right = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
    else:
        t_right = times_s[right]

    width_s = t_right - t_left
    if width_s <= 0:
        return None
    return width_s / 60.0  # convert to minutes

# ---------- Tkinter GUI with embedded plot ----------

class HPLCApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HPLC plate simulation")
        self.geometry("1150x650")

        # Increase default font size for all Tk/ttk widgets
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=12)   # try 10–12

        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=12)


        controls = ttk.Frame(self)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ONE place for defaults
        self.param_defaults = {
            "inner_diam_mm": 2.1,
            "column_length_cm": 20.0,
            "flow_mL_min": 1.0,
            "particle_size_um": 10.0,
            "porosity": 0.4,
            "temperature_C": 20.0,
            "acn_percent": 10.0,          # % v/v ACN
            "solute_radius_nm": 6.0,
            "p_move_A": 0.9,
            "p_move_B": 0.85,
            "n_molecules_A": 5000,
            "n_molecules_B": 5000,
            "injection_spread_steps": 0.0,
            "max_steps": 50000,
            "max_pressure_bar": 500.0,
        }

        labels = {
            "inner_diam_mm": "Column ID (mm)",
            "column_length_cm": "L (cm)",
            "flow_mL_min": "Flow (mL/min)",
            "particle_size_um": "Particle size (µm)",
            "porosity": "Porosity",
            "temperature_C": "T (°C)",
            "acn_percent": "% ACN (v/v)",
            "solute_radius_nm": "Solute radius (nm)",
            "p_move_A": "p_move A",
            "p_move_B": "p_move B",
            "n_molecules_A": "N molecules A",
            "n_molecules_B": "N molecules B",
            "injection_spread_steps": "Injection spread (steps)",
            "max_steps": "Max steps",
            "max_pressure_bar": "P max (bar)",
        }

        self.entries = {}
        for i, key in enumerate(self.param_defaults):
            ttk.Label(controls, text=labels[key]).grid(row=i, column=0, sticky="w")
            e = ttk.Entry(controls, width=14)
            e.insert(0, str(self.param_defaults[key]))
            e.grid(row=i, column=1, padx=2, pady=2)
            self.entries[key] = e

        run_button = ttk.Button(controls, text="Run simulation",
                                command=self.run_simulation)
        run_button.grid(row=len(self.param_defaults), column=0, columnspan=2,
                        pady=10, sticky="we")



        # Checkbox: show FWHM on the plot
        self.show_fwhm = tk.BooleanVar(value=False)
        fwhm_check = ttk.Checkbutton(
            controls,
            text="Show FWHM of peaks",
            variable=self.show_fwhm
        )
        fwhm_check.grid(row=len(self.param_defaults)+1,
                        column=0, columnspan=2, sticky="w")

        self.status_var = tk.StringVar(value="")
        status_label = ttk.Label(controls, textvariable=self.status_var,
                                 foreground="red")
        status_label.grid(row=len(self.param_defaults)+2, column=0,
                          columnspan=2, sticky="w")

        status_label.grid(row=len(self.param_defaults)+1, column=0,
                          columnspan=2, sticky="w")

        self.run_simulation()

    def get_float(self, key):
        s = self.entries[key].get()
        try:
            return float(s)
        except ValueError:
            return float(self.param_defaults[key])

    def get_int(self, key):
        s = self.entries[key].get()
        try:
            return int(float(s))
        except ValueError:
            return int(self.param_defaults[key])

    def run_simulation(self):
        self.status_var.set("")
        try:
            inner_diam_mm         = self.get_float("inner_diam_mm")
            column_length_cm      = self.get_float("column_length_cm")
            flow_mL_min           = self.get_float("flow_mL_min")
            particle_size_um      = self.get_float("particle_size_um")
            porosity              = self.get_float("porosity")
            temperature_C         = self.get_float("temperature_C")
            acn_percent           = self.get_float("acn_percent")
            solute_radius_nm      = self.get_float("solute_radius_nm")
            p_move_A              = self.get_float("p_move_A")
            p_move_B              = self.get_float("p_move_B")
            n_molecules_A         = self.get_int("n_molecules_A")
            n_molecules_B         = self.get_int("n_molecules_B")
            injection_spread_steps = self.get_float("injection_spread_steps")
            max_steps             = self.get_int("max_steps")
            max_pressure_bar      = self.get_float("max_pressure_bar")

            acn_vol_frac = acn_percent / 100.0

            k_A = 2.0
            gamma = 2.0
            k_C = 0.05

            (times_s, eA, eB,
             N_plates, H_m, N_float,
             u, Dm, deltaP_bar, viscosity_Pa_s) = \
                simulate_chromatogram_physical(
                    n_molecules_A=n_molecules_A,
                    n_molecules_B=n_molecules_B,
                    p_move_A=p_move_A,
                    p_move_B=p_move_B,
                    column_length_cm=column_length_cm,
                    inner_diam_mm=inner_diam_mm,
                    particle_size_um=particle_size_um,
                    porosity=porosity,
                    flow_mL_min=flow_mL_min,
                    temperature_C=temperature_C,
                    acn_vol_frac=acn_vol_frac,
                    solute_radius_nm=solute_radius_nm,
                    injection_spread_steps=injection_spread_steps,
                    k_A=k_A,
                    gamma=gamma,
                    k_C=k_C,
                    max_steps=max_steps,
                )

            # Update plot
            self.ax.clear()
            self.ax.plot(times_s / 60.0, eA, label="Analyte A")
            self.ax.plot(times_s / 60.0, eB, label="Analyte B")
            self.ax.set_xlabel("Time [min]")
            self.ax.set_ylabel("Molecules eluted per step")

            viscosity_cP = viscosity_Pa_s * 1e3

            title = (
                f"L = {column_length_cm:.1f} cm, ID = {inner_diam_mm:.1f} mm, "
                f"Flow = {flow_mL_min:.2f} mL/min\n"
                f"T = {temperature_C:.1f} °C, %ACN = {acn_percent:.0f} %, "
                f"η ≈ {viscosity_cP:.2f} cP\n"
                f"u = {u:.3e} m/s, H = {H_m:.2e} m, "
                f"N ≈ {N_float:.1f} (used {N_plates}), "
                f"Dm = {Dm:.2e} m²/s"
            )
            self.ax.set_title(title, fontsize=9)

            # --- VAN DEEMTER BOX (top-left) ---
            self.ax.text(
                0.01, 0.99,
                r"$H(u) = k_A d_p + \dfrac{2\gamma D_m}{u} + \dfrac{k_C d_p^2}{D_m}\,u$",
                transform=self.ax.transAxes,
                va="top", ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
            )

            # --- LEGEND (just under Van Deemter), same left alignment ---
            legend = self.ax.legend(
                loc="upper left",
                frameon=True,
                borderaxespad=0.0,  # <- IMPORTANT: no extra padding vs axes
            )
            legend.set_bbox_to_anchor((0.01, 0.86), transform=self.ax.transAxes)

            # homogenize legend style with other boxes
            frame = legend.get_frame()
            frame.set_boxstyle("round")
            frame.set_facecolor("white")
            frame.set_alpha(0.75)
            frame.set_edgecolor("black")
            frame.set_linewidth(1.0)

            # --- PRESSURE BOX (below legend), same left alignment ---
            pressure_color = "red" if deltaP_bar > max_pressure_bar else "green"
            self.ax.text(
                0.01, 0.70,  # just under the legend
                rf"$\Delta P \approx {deltaP_bar:.0f}\ \mathrm{{bar}}$"
                "\n"
                rf"(limite : {max_pressure_bar:.0f} bar)",
                transform=self.ax.transAxes,
                va="top", ha="left",
                fontsize=10,
                color=pressure_color,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            )

            # --- FWHM BOX (optional, below pressure box) ---
            if self.show_fwhm.get():
                fwhmA_min = compute_fwhm_min(times_s, eA)
                fwhmB_min = compute_fwhm_min(times_s, eB)

                lines = []
                if fwhmA_min is not None:
                    lines.append(rf"FWHM A ≈ {fwhmA_min:.3f} min")
                if fwhmB_min is not None:
                    lines.append(rf"FWHM B ≈ {fwhmB_min:.3f} min")

                if lines:
                    fwhm_text = "\n".join(lines)
                    self.ax.text(
                        0.01, 0.54,              # below the pressure box
                        fwhm_text,
                        transform=self.ax.transAxes,
                        va="top", ha="left",
                        fontsize=10,
                        bbox=dict(boxstyle="round",
                                  facecolor="white", alpha=0.75),
                    )

            self.ax.figure.tight_layout()
            self.canvas.draw()


        except Exception as exc:
            self.status_var.set(f"Error: {exc}")



if __name__ == "__main__":
    app = HPLCApp()
    app.mainloop()
