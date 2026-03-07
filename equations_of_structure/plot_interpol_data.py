"""
EoS Interpolation Validation — PCHIP vs Tabulated Data
======================================================

This script validates the PCHIP interpolants constructed for several
tabulated neutron-star equations of state (EoS) by comparing the raw
tabulated data against the interpolated curves in log–log space.

The following EoS tables are considered:
    - APR
    - SLy4
    - GNH3

For each EoS, the script plots:

    - Tabulated points:  log10(ρ_mass) vs log10(p)
    - PCHIP interpolation:  log10(ρ_mass(p)) evaluated on a dense grid

The goal is to visually inspect:
    - smoothness of the interpolation,
    - monotonic behaviour,
    - agreement between the interpolant and the original tabulated points,
    - extrapolation behaviour at the boundaries (if present).

Units
-----
The quantities are handled in logarithmic CGS variables:

    - log10(p)     where p is in [dyn cm^-2]
    - log10(ρ_mass) where ρ_mass is in [g cm^-3]

The interpolators rho_pchip_log10_* map:

    log10(p)  ->  log10(ρ_mass)

Dependencies
------------
    - interpol_data :
        * log10_p_* and log10_rho_* arrays (tabulated data in log-space)
        * rho_pchip_log10_* interpolator functions (PCHIP)
    - numpy        : linspace
    - matplotlib   : plotting

Notes
-----
    - A very dense grid (1e6 points) is used for plotting the interpolated
      profiles to ensure smooth curves. This is intended for visualization
      only and may be reduced significantly for faster execution.
"""
# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import linspace

from interpol_data import log10_p_apr, log10_p_gnh3, log10_p_sly4
from interpol_data import log10_rho_apr, log10_rho_gnh3, log10_rho_sly4
from interpol_data import rho_pchip_log10_apr, rho_pchip_log10_gnh3, rho_pchip_log10_sly4

# ==========Parameters==========#
log10_p_plot = linspace(14,37 , int(1e6))

# ==========Plot 1: log10(energy density) vs log10(pressure), APR==========#
fig_apr, ax_apr = plt.subplots(figsize=(7.5, 4.5))
ax_apr.plot(log10_p_apr, log10_rho_apr, color='blue', linewidth=1.5, label="Data")
ax_apr.plot(log10_p_plot, rho_pchip_log10_apr(log10_p_plot), color='red', linestyle='--', linewidth=1.5,
            label="PCHIP interpol")
ax_apr.set_xlabel(r'$\log_{10}(p)$')
ax_apr.set_ylabel(r'$\log_{10}(\rho)$')
ax_apr.set_title(r'EoS Profile APR')
ax_apr.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_apr.legend(loc="upper left")

# ==========Plot 2: log10(energy density) vs log10(pressure), GNH3==========#
fig_gnh3, ax_gnh3 = plt.subplots(figsize=(7.5, 4.5))
ax_gnh3.plot(log10_p_gnh3, log10_rho_gnh3, color='blue', linewidth=1.5, label="Data")
ax_gnh3.plot(log10_p_plot, rho_pchip_log10_gnh3(log10_p_plot), color='red', linestyle='--', linewidth=1.5,
             label="PCHIP interpol")
ax_gnh3.set_xlabel(r'$\log_{10}(p)$')
ax_gnh3.set_ylabel(r'$\log_{10}(\rho)$')
ax_gnh3.set_title(r'EoS Profile GNH3 ')
ax_gnh3.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_gnh3.legend(loc="upper left")

# ==========Plot 3: log10(energy density) vs log10(pressure), SLy4==========#
fig_sly4, ax_sly4 = plt.subplots(figsize=(7.5, 4.5))
ax_sly4.plot(log10_p_sly4, log10_rho_sly4, color='blue', linewidth=1.5, label="Data")
ax_sly4.plot(log10_p_plot, rho_pchip_log10_sly4(log10_p_plot), color='red', linestyle='--', linewidth=1.5,
             label="PCHIP interpol")
ax_sly4.set_xlabel(r'$\log_{10}(p)$')
ax_sly4.set_ylabel(r'$\log_{10}(\rho)$')
ax_sly4.set_title(r'EoS Profile SLY4 ')
ax_sly4.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_sly4.legend(loc="upper left")
plt.show()