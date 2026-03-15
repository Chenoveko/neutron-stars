"""
TOV Equation Solver — Fixed-Mass Neutron Star Structure Comparison
===================================================================

This script determines and compares the internal structure of neutron
stars with a fixed gravitational mass of 1.4 M_sun using three different
equations of state (EoS): APR, GNH3, and SLy4.

For each EoS, the central pressure is determined iteratively by scanning
a narrow interval of candidate central pressures and integrating the
TOV equations until a stellar configuration is found whose total mass
matches the target mass within a prescribed tolerance.

Target configuration:
    M_target = 1.4 M_sun
    Allowed tolerance = ±0.01 M_sun

Once the appropriate central pressure is identified for each EoS, the
TOV equations are integrated again to obtain the full radial profiles:

    - Radius r
    - Pressure p(r)
    - Enclosed mass m(r)

From these quantities, the density profile ρ(r) is reconstructed using
the corresponding interpolated equation of state.

The purpose of this script is to compare how different realistic
equations of state affect the internal structure of neutron stars
with the same total mass.

Results are presented in two plots:
    1. Enclosed mass profile  m(r) / M_sun  vs  radius  r (km)
    2. Density profile  log10(ρ(r))  vs  radius  r (km)

Units
-----
    - Central pressures are sampled in CGS and converted to geometrized
      units (G = c = 1) before integration.
    - Radii are converted from cm to kilometres (km).
    - Pressures are converted from geometrized units back to CGS
      (dyn cm^-2) for plotting.
    - Enclosed mass is expressed in units of the solar mass M_sun.
    - Density is expressed in CGS units (g cm^-3).

Dependencies
------------
    - equations_of_structure.interpol_data  : interpolated EoS tables
    - utilities.math_methods                : TOV integrator (solve_tov_eos)
    - utilities.physical_data               : unit conversion constants
    - matplotlib                            : plotting
    - numpy                                 : pressure sampling

Physical Background
-------------------
For a given equation of state, there exists a one-to-one relation between
central pressure and total gravitational mass. By selecting configurations
with identical total mass but different EoS, this script highlights how
microscopic nuclear physics assumptions modify the macroscopic radial
structure (mass distribution and density profile) of neutron stars.
"""
# =====================
# Imports
# =====================
import matplotlib.pyplot as plt

from equations_of_structure.interpol_data import rho_pchip_geo_apr, rho_pchip_geo_gnh3, rho_pchip_geo_sly4
from equations_of_structure.interpol_data import rho_pchip_log10_apr, rho_pchip_log10_gnh3, rho_pchip_log10_sly4
from utilities.tov_solvers import solve_tov_eos
from utilities.physical_data import M_sun, pressure_cgs_to_geo, pressure_geo_to_cgs, mass_cgs_to_geo
from numpy import logspace, log10

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_array_apr_cgs = logspace(35.15, 35.2, 200)  # Array of central pressure in CGS for APR
p_central_array_gnh3_cgs = logspace(34.80, 34.85, 200)  # Array of central pressure in CGS for GNH3
p_central_array_sly4_cgs = logspace(35.1, 35.15, 200)  # Array of central pressure in CGS for APR
p_central_array_apr_geo = pressure_cgs_to_geo(p_central_array_apr_cgs)  # Array of central pressure in GEO for APR
p_central_array_gnh3_geo = pressure_cgs_to_geo(p_central_array_gnh3_cgs)  # Array of central pressure in GEO for GNH3
p_central_array_sly4_geo = pressure_cgs_to_geo(p_central_array_sly4_cgs)  # Array of central pressure in GEO for SLy4
target_mass = 1.4  # target mass in M_sun
mass_threshold = 0.01  # allowed overshoot in M_sun (example: 0.01 M_sun)

# ==========Integration to find central pressure==========#
# Integrate TOV equations over array of central pressures for APR EoS
for p_c in p_central_array_apr_geo:
    _, _, m, _ = solve_tov_eos(p_c, rho_pchip_geo_apr)
    m = m / M_sun_geo  # Convert mass points
    tot_mass = m[-1]
    if target_mass - mass_threshold <= tot_mass <= target_mass + mass_threshold:
        p_central_apr_geo = p_c
        print("central pressure in geo units for APR: ",p_central_apr_geo)
        break
    else:
        continue

# Integrate TOV equations over array of central pressures for GNH3 EoS
for p_c in p_central_array_gnh3_geo:
    _, _, m, _ = solve_tov_eos(p_c, rho_pchip_geo_gnh3)
    m = m / M_sun_geo  # Convert mass points
    tot_mass = m[-1]
    if target_mass - mass_threshold <= tot_mass <= target_mass + mass_threshold:
        p_central_gnh3_geo = p_c
        print("central pressure in geo units for GNH3: ", p_central_gnh3_geo)
        break
    else:
        continue

# Integrate TOV equations over array of central pressures for SLy4 EoS
for p_c in p_central_array_sly4_geo:
    _, _, m, _ = solve_tov_eos(p_c, rho_pchip_geo_sly4)
    m = m / M_sun_geo  # Convert mass points
    tot_mass = m[-1]
    if target_mass - mass_threshold <= tot_mass <= target_mass + mass_threshold:
        p_central_sly4_geo = p_c
        print("central pressure in geo units for SLy4: ", p_central_sly4_geo)
        break
    else:
        continue

# ==========Integration with fixed central pressure==========#
# Integrate TOV equations using APR EoS
r_apr, p_apr, m_apr, status_apr = solve_tov_eos(p_central_apr_geo, rho_pchip_geo_apr)
r_apr = r_apr / 1e5  # Convert radial points to Km
p_apr = pressure_geo_to_cgs(p_apr)  # Convert pressure to CGS
m_apr = m_apr / M_sun_geo  # Convert mass points

# Integrate TOV equations using GNH3 EoS
r_gnh3, p_gnh3, m_gnh3, status_gnh3 = solve_tov_eos(p_central_gnh3_geo, rho_pchip_geo_gnh3)
r_gnh3 = r_gnh3 / 1e5  # Convert radial points to Km
p_gnh3 = pressure_geo_to_cgs(p_gnh3)  # Convert pressure to CGS
m_gnh3 = m_gnh3 / M_sun_geo  # Convert mass points

# Integrate TOV equations using SLY4 EoS
r_sly4, p_sly4, m_sly4, status_sly4l = solve_tov_eos(p_central_sly4_geo, rho_pchip_geo_sly4)
r_sly4 = r_sly4 / 1e5  # Convert radial points to Km
p_sly4 = pressure_geo_to_cgs(p_sly4)  # Convert pressure to CGS
m_sly4 = m_sly4 / M_sun_geo  # Convert mass points

# ==========Plot 1: Mass enclosed ==========#
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r_apr, m_apr, color='goldenrod', linewidth=1.5, label="APR")
ax_mass.plot(r_gnh3, m_gnh3, color='blue', linewidth=1.5, label="GNH3")
ax_mass.plot(r_sly4, m_sly4, color='red', linewidth=1.5, label="SLy4")
ax_mass.set_xlabel(r'$r\ [km]$')
ax_mass.set_ylabel(r'$m(r) / M_{\odot}$')
ax_mass.set_title(rf'Enclosed mass for ${target_mass}\,M_{{\odot}}$')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_mass.legend(loc="upper left")
# fig_mass.savefig("fixed_mass_enclosed.png", dpi=600, bbox_inches="tight")

# ==========Plot 2: Density profile ==========#
fig_density, ax_density = plt.subplots(figsize=(7.5, 4.5))
ax_density.plot(r_apr, rho_pchip_log10_apr(log10(p_apr)), color='goldenrod', linewidth=1.5, label="APR")
ax_density.plot(r_gnh3, rho_pchip_log10_gnh3(log10(p_gnh3)), color='blue', linewidth=1.5, label="GNH3")
ax_density.plot(r_sly4, rho_pchip_log10_sly4(log10(p_sly4)), color='red', linewidth=1.5, label="SLy4")
ax_density.set_xlabel(r'$r\ [km]$')
ax_density.set_ylabel(r'$\log_{10}(\rho)\;[\mathrm{g\,cm^{-3}}]$')
ax_density.set_title(rf'Density profile for ${target_mass}\,M_{{\odot}}$')
ax_density.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_density.legend(loc="lower left")
# fig_density.savefig("fixed_density.png", dpi=600, bbox_inches="tight")
plt.show()
