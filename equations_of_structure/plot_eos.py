"""
Equations of State — Tabulated Nuclear EoS Profiles (APR, GNH3, SLy4)
=====================================================================

This script loads tabulated neutron-star equations of state (EoS) from
plain-text files and visualizes their pressure–density relations in
logarithmic space.

The following EoS tables are supported and plotted:
    - APR (Akmal–Pandharipande–Ravenhall, often referred to as APR/APR4)
    - GNH3 (Glendenning)
    - SLy4 (Skyrme Lyon)

For each EoS, the script:
    1) Extracts mass density ρ and pressure p from the corresponding table.
    2) Computes log10(ρ) and log10(p) to handle the wide dynamic range.
    3) Produces two figures:
        - Full profile:  log10(ρ) vs log10(p)
        - Zoomed profile over a selected pressure/density window

The plots are intended for quick validation and comparison of the
stiffness/shape of the EoS tables prior to interpolation (e.g., PCHIP)
and stellar structure integrations (TOV solvers).

Input Data Format
-----------------
Each EoS text file is assumed to be a whitespace-separated table with
(at least) the following columns:
    Column 2 : Mass density ρ [g cm^-3]
    Column 3 : Pressure     p [dyn cm^-2]

Comment lines are allowed and are ignored by the loader functions.

References
----------
    Akmal, A., Pandharipande, V. R., & Ravenhall, D. G. (1998),
    "Equation of state of nucleon matter and neutron star structure",
    Physical Review C, 58(3), 1804–1828.

Units
-----
    - Mass density:  ρ [g cm^-3]
    - Pressure:      p [dyn cm^-2]
    - Log variables: log10(ρ), log10(p)

Dependencies
------------
    - equations_of_structure.extract_data :
        * extract_mass_density_from_eos_txt
        * extract_pressure_from_eos_txt
    - numpy      : log10
    - matplotlib : plotting

Notes
-----
    - The zoom window (xlim/ylim) is chosen to highlight the region where
      the three EoS differ most clearly for typical neutron-star conditions.
    - File paths are given as plain filenames; the working directory must
      contain the EoS text files (or paths must be adjusted accordingly).
"""
# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import log10

from equations_of_structure.extract_data import extract_mass_density_from_eos_txt, extract_pressure_from_eos_txt

# ==========Extract Values==========#
# Mass density from Eos on log10
log10_rho_apr = log10(extract_mass_density_from_eos_txt("eos_akmalpr.txt"))
log10_rho_gnh3 = log10(extract_mass_density_from_eos_txt("eos_glendnh3.txt"))
log10_rho_sly4 = log10(extract_mass_density_from_eos_txt("eos_sly4.txt"))
# Pressure from EoS on log10
log10_p_apr = log10(extract_pressure_from_eos_txt("eos_akmalpr.txt"))
log10_p_gnh3 = log10(extract_pressure_from_eos_txt("eos_glendnh3.txt"))
log10_p_sly4 = log10(extract_pressure_from_eos_txt("eos_sly4.txt"))

# ==========Plot 1: log10(mass density) vs log10(pressure)==========#
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(log10_p_apr, log10_rho_apr, color='goldenrod', linewidth=1.5, label="APR")
ax.plot(log10_p_gnh3, log10_rho_gnh3, color='blue', linewidth=1.5, label="GNH3")
ax.plot(log10_p_sly4, log10_rho_sly4, color='red', linewidth=1.5, label="SLy4")
ax.set_xlabel(r'$\log_{10}(p)\, [\mathrm{dyn\,cm^{-2}}]$')
ax.set_ylabel(r'$\log_{10}(\rho)\, [\mathrm{g\,cm^{-3}}]$')
ax.set_title(r'EoS Profile')
ax.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax.legend(loc="upper left")

# ==========Plot 2: log10(mass density) vs log10(pressure), Zoom==========#
fig_zoom, ax_zoom = plt.subplots(figsize=(7.5, 4.5))
ax_zoom.plot(log10_p_apr, log10_rho_apr, color='goldenrod', linewidth=1.5, label="APR")
ax_zoom.plot(log10_p_gnh3, log10_rho_gnh3, color='blue', linewidth=1.5, label="GNH3")
ax_zoom.plot(log10_p_sly4, log10_rho_sly4, color='red', linewidth=1.5, label="SLy4")
ax_zoom.set_xlabel(r'$\log_{10}(p)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_zoom.set_ylabel(r'$\log_{10}(\rho)\, [\mathrm{g\,cm^{-3}}]$')
ax_zoom.set_title(r'EoS Zoom Profile')
ax_zoom.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_zoom.legend(loc="upper left")
ax_zoom.set_xlim(xmin=32, xmax=36.7)
ax_zoom.set_ylim(ymin=13.5, ymax=15.8)
plt.show()
