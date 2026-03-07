"""
TOV Equation Solver — Radial Stability Analysis of Neutron Stars
=================================================================

This script solves the Tolman–Oppenheimer–Volkoff (TOV) equations for a
family of central pressures using three different equations of state (EoS):
APR, GNH3, and SLy4, in order to analyse the radial stability of neutron stars.

For each EoS and each value of central pressure p_c, the TOV system
is integrated outward from the stellar centre to the surface, yielding
the total gravitational mass M and total radius R of the neutron star.

Radial stability is determined using the condition:

    dM/dp_c > 0  →  Stable configuration
    dM/dp_c < 0  →  Unstable configuration

The derivative dM/dp_c is computed numerically along each mass sequence,
allowing the identification of stable and unstable branches.

Results are presented in three plots:
    1. Total mass  M/M_sun  vs  log10(p_c)  [CGS], showing stable and unstable regions
    2. Total radius  R (km)  vs  log10(p_c)  [CGS], showing stable and unstable regions
    3. Mass-Radius diagram  M/M_sun  vs  R (km), with stability classification

Units
-----
    - Central pressures are sampled in CGS and converted to geometric
      units (GEO) before integration.
    - Radii are converted from GEO (cm) to kilometres after integration.
    - Masses are expressed in units of the solar mass M_sun.

Dependencies
------------
    - equations_of_structure.interpol_data  : interpolated EoS tables
    - utilities.math_methods                : TOV integrator (solve_tov_eos)
    - utilities.physical_data               : unit conversion constants
"""
# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import logspace, log10, gradient, array
from utilities.math_methods import solve_tov_eos

from equations_of_structure.interpol_data import rho_pchip_geo_apr, rho_pchip_geo_gnh3, rho_pchip_geo_sly4
from utilities.physical_data import M_sun, pressure_cgs_to_geo, mass_cgs_to_geo

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_array_cgs = logspace(33.2, 36.5, 600)  # Array of central pressure in CGS
p_central_array_geo = pressure_cgs_to_geo(p_central_array_cgs)  # Array of central pressure in GEO
log10_p_central_array_cgs = log10(p_central_array_cgs)

# ==========Integration==========#
# Initialize total mass and total radius arrays for APR
tot_mass_apr = []
tot_radius_apr = []
# Initialize total mass and total radius arrays for GNH3
tot_mass_gnh3 = []
tot_radius_gnh3 = []
# Initialize total mass and total radius arrays for SLY4
tot_mass_sly4 = []
tot_radius_sly4 = []
# Integrate TOV equations over array of central pressures
for p_c in p_central_array_geo:
    # APR EoS
    r_apr, _, m_apr, _ = solve_tov_eos(p_c, rho_pchip_geo_apr)
    r_apr = r_apr / 1e5  # Convert radial points to Km
    m_apr = m_apr / M_sun_geo  # Convert mass points
    tot_mass_apr.append(m_apr[-1])
    tot_radius_apr.append(r_apr[-1])
    # GNH3 EoS
    r_gnh3, _, m_gnh3, _ = solve_tov_eos(p_c, rho_pchip_geo_gnh3)
    r_gnh3 = r_gnh3 / 1e5  # Convert radial points to Km
    m_gnh3 = m_gnh3 / M_sun_geo  # Convert mass points
    tot_mass_gnh3.append(m_gnh3[-1])
    tot_radius_gnh3.append(r_gnh3[-1])
    # SLy4 EoS
    r_sly4, _, m_sly4, _ = solve_tov_eos(p_c, rho_pchip_geo_sly4)
    r_sly4 = r_sly4 / 1e5  # Convert radial points to Km
    m_sly4 = m_sly4 / M_sun_geo  # Convert mass points
    tot_mass_sly4.append(m_sly4[-1])
    tot_radius_sly4.append(r_sly4[-1])

# ==========Convert to Numpy array==========#
tot_mass_apr = array(tot_mass_apr)
tot_radius_apr = array(tot_radius_apr)
tot_mass_gnh3 = array(tot_mass_gnh3)
tot_radius_gnh3 = array(tot_radius_gnh3)
tot_mass_sly4 = array(tot_mass_sly4)
tot_radius_sly4 = array(tot_radius_sly4)

# ==========Derivatives dM/dpc (stability criteria)==========#
dM_dpc_apr = gradient(tot_mass_apr, p_central_array_cgs)
dM_dpc_gnh3 = gradient(tot_mass_gnh3, p_central_array_cgs)
dM_dpc_sly4 = gradient(tot_mass_sly4, p_central_array_cgs)

stable_apr = dM_dpc_apr > 0
stable_gnh3 = dM_dpc_gnh3 > 0
stable_sly4 = dM_dpc_sly4 > 0

# ==========Slicing==========#
# APR
tot_mass_apr_stable = tot_mass_apr[stable_apr]
tot_radius_apr_stable = tot_radius_apr[stable_apr]
tot_mass_apr_unstable = tot_mass_apr[~stable_apr]
tot_radius_apr_unstable = tot_radius_apr[~stable_apr]
logpc_apr_stable = log10_p_central_array_cgs[stable_apr]
logpc_apr_unstable = log10_p_central_array_cgs[~stable_apr]

# GNH3
tot_mass_gnh3_stable = tot_mass_gnh3[stable_gnh3]
tot_radius_gnh3_stable = tot_radius_gnh3[stable_gnh3]
tot_mass_gnh3_unstable = tot_mass_gnh3[~stable_gnh3]
tot_radius_gnh3_unstable = tot_radius_gnh3[~stable_gnh3]
logpc_gnh3_stable = log10_p_central_array_cgs[stable_gnh3]
logpc_gnh3_unstable = log10_p_central_array_cgs[~stable_gnh3]

# SLy4
tot_mass_sly4_stable = tot_mass_sly4[stable_sly4]
tot_radius_sly4_stable = tot_radius_sly4[stable_sly4]
tot_mass_sly4_unstable = tot_mass_sly4[~stable_sly4]
tot_radius_sly4_unstable = tot_radius_sly4[~stable_sly4]
logpc_sly4_stable = log10_p_central_array_cgs[stable_sly4]
logpc_sly4_unstable = log10_p_central_array_cgs[~stable_sly4]

# ==========Plot 1: Total Mass vs Central Pressure==========#
fig_tot_mass, ax_tot_mass = plt.subplots(figsize=(7.5, 4.5))
ax_tot_mass.plot(logpc_apr_stable, tot_mass_apr_stable, color='goldenrod', linewidth=1.5, linestyle='-',
                 label="APR stable")
ax_tot_mass.plot(logpc_apr_unstable, tot_mass_apr_unstable, color='goldenrod', linewidth=1.5, linestyle='--',
                 label="APR unstable")
ax_tot_mass.plot(logpc_gnh3_stable, tot_mass_gnh3_stable, color='blue', linewidth=1.5, linestyle='-',
                 label="GNH3 stable")
ax_tot_mass.plot(logpc_gnh3_unstable, tot_mass_gnh3_unstable, color='blue', linewidth=1.5, linestyle='--',
                 label="GNH3 unstable")
ax_tot_mass.plot(logpc_sly4_stable, tot_mass_sly4_stable, color='red', linewidth=1.5, linestyle='-',
                 label="SLy4 stable")
ax_tot_mass.plot(logpc_sly4_unstable, tot_mass_sly4_unstable, color='red', linewidth=1.5, linestyle='--',
                 label="SLy4 unstable")
ax_tot_mass.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_tot_mass.set_ylabel(r'$M / M_{\odot}$')
ax_tot_mass.set_title(r'Total Mass')
ax_tot_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_tot_mass.legend(loc="upper left")
# fig_tot_mass.savefig("total_mass_stability.png", dpi=600, bbox_inches="tight")

# ==========Plot 2: Total Radius vs Central Pressure==========#
fig_tot_radius, ax_tot_radius = plt.subplots(figsize=(7.5, 4.5))
ax_tot_radius.plot(logpc_apr_stable, tot_radius_apr_stable, color='goldenrod', linewidth=1.5, linestyle='-',
                   label="APR stable")
ax_tot_radius.plot(logpc_apr_unstable, tot_radius_apr_unstable, color='goldenrod', linewidth=1.5, linestyle='--',
                   label="APR unstable")
ax_tot_radius.plot(logpc_gnh3_stable, tot_radius_gnh3_stable, color='blue', linewidth=1.5, linestyle='-',
                   label="GNH3 stable")
ax_tot_radius.plot(logpc_gnh3_unstable, tot_radius_gnh3_unstable, color='blue', linewidth=1.5, linestyle='--',
                   label="GNH3 unstable")
ax_tot_radius.plot(logpc_sly4_stable, tot_radius_sly4_stable, color='red', linewidth=1.5, linestyle='-',
                   label="SLy4 stable")
ax_tot_radius.plot(logpc_sly4_unstable, tot_radius_sly4_unstable, color='red', linewidth=1.5, linestyle='--',
                   label="SLy4 unstable")
ax_tot_radius.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_tot_radius.set_ylabel(r'$R\ [km]$')
ax_tot_radius.set_title(r'Total Radius')
ax_tot_radius.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_tot_radius.legend(loc="upper right")
# fig_tot_radius.savefig("total_radius_stability.png", dpi=600, bbox_inches="tight")

# ==========Plot 3: Total Mass vs Total Radius==========#
fig_tot_mass_radius, ax_tot_mass_radius = plt.subplots(figsize=(7.5, 4.5))
ax_tot_mass_radius.plot(tot_radius_apr_stable, tot_mass_apr_stable, color='goldenrod', linewidth=1.5, linestyle='-',
                        label="APR stable")
ax_tot_mass_radius.plot(tot_radius_apr_unstable, tot_mass_apr_unstable, color='goldenrod', linewidth=1.5,
                        linestyle='--', label="APR unstable")
ax_tot_mass_radius.plot(tot_radius_gnh3_stable, tot_mass_gnh3_stable, color='blue', linewidth=1.5, linestyle='-',
                        label="GNH3 stable")
ax_tot_mass_radius.plot(tot_radius_gnh3_unstable, tot_mass_gnh3_unstable, color='blue', linewidth=1.5, linestyle='--',
                        label="GNH3 unstable")
ax_tot_mass_radius.plot(tot_radius_sly4_stable, tot_mass_sly4_stable, color='red', linewidth=1.5, linestyle='-',
                        label="SLy4 stable")
ax_tot_mass_radius.plot(tot_radius_sly4_unstable, tot_mass_sly4_unstable, color='red', linewidth=1.5, linestyle='--',
                        label="SLy4 unstable")
ax_tot_mass_radius.set_xlabel(r'$R\ [km]$')
ax_tot_mass_radius.set_ylabel(r'$M / M_{\odot}$')
ax_tot_mass_radius.set_title(r'Total Mass vs Total Radius')
ax_tot_mass_radius.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_tot_mass_radius.legend(loc="upper right")
# fig_tot_mass_radius.savefig("total_mass_radius_stability.png", dpi=600, bbox_inches="tight")
plt.show()
