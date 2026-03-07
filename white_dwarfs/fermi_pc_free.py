# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import logspace, log10, array

from math_methods import solve_tov_eos, solve_newton_eos
from utilities.physical_data import M_sun, pressure_cgs_to_geo, pressure_geo_to_cgs, mass_cgs_to_geo
from eos_fermi import rho_degenerate_fermions_ur_geo

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_array_cgs = logspace(25, 30, 10)  # Array of central pressure in CGS
p_central_array_geo = pressure_cgs_to_geo(p_central_array_cgs)  # Array of central pressure in GEO
log10_p_central_array_cgs = log10(p_central_array_cgs)

# ==========Integration Newton and TOV with UR EoS==========#
# Initialize total mass and total radius arrays
tot_mass_newton = []
tot_radius_newton = []
tot_mass_tov = []
tot_radius_tov = []
# Integrate TOV and Newton equations over array of central pressures
for p_c in p_central_array_geo:
    r_newton, p_newton, m_newton, _ = solve_newton_eos(p_c, rho_degenerate_fermions_ur_geo)
    r_tov, p_tov, m_tov, _ = solve_tov_eos(p_c, rho_degenerate_fermions_ur_geo)
    r_newton = r_newton / 1e5  # Convert radial points to Km
    r_tov = r_tov / 1e5  # Convert radial points to Km
    p_newton = pressure_geo_to_cgs(p_newton)  # Convert pressure to CGS
    p_tov = pressure_geo_to_cgs(p_tov)  # Convert pressure to CGS
    m_newton = m_newton / M_sun_geo  # Convert mass points
    m_tov = m_tov / M_sun_geo  # Convert mass points
    tot_mass_newton.append(m_newton[-1])
    tot_mass_tov.append(m_tov[-1])
    tot_radius_newton.append(r_newton[-1])
    tot_radius_tov.append(r_tov[-1])

# ==========Plot 1: Total Mass vs Central Pressure==========#
fig_tot_mass, ax_tot_mass = plt.subplots(figsize=(7.5, 4.5))
ax_tot_mass.plot(log10_p_central_array_cgs, tot_mass_newton, color='goldenrod', linewidth=1.5, label="Newton")
ax_tot_mass.plot(log10_p_central_array_cgs, tot_mass_tov, color='blue', linewidth=1.5, label="TOV")
ax_tot_mass.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_tot_mass.set_ylabel(r'$M / M_{\odot}$')
ax_tot_mass.set_title(r'Total Mass')
ax_tot_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_tot_mass.legend(loc="upper left")
fig_tot_mass.savefig("total_mass.png", dpi=600, bbox_inches="tight")

# ==========Plot 2: Total Radius vs Central Pressure==========#
fig_tot_radius, ax_tot_radius = plt.subplots(figsize=(7.5, 4.5))
ax_tot_radius.plot(log10_p_central_array_cgs, tot_radius_newton, color='goldenrod', linewidth=1.5, label="Newton")
ax_tot_radius.plot(log10_p_central_array_cgs, tot_radius_tov, color='blue', linewidth=1.5, label="TOV")
ax_tot_radius.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_tot_radius.set_ylabel(r'$R\ [km]$')
ax_tot_radius.set_title(r'Total Radius')
ax_tot_radius.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_tot_radius.legend(loc="upper right")
fig_tot_radius.savefig("total_radius.png", dpi=600, bbox_inches="tight")

# ==========Plot 3: Total Mass vs Total Radius==========#
fig_tot_mass_radius, ax_tot_mass_radius = plt.subplots(figsize=(7.5, 4.5))
ax_tot_mass_radius.plot(tot_radius_newton, tot_mass_newton, color='goldenrod', linewidth=1.5, label="Newton")
ax_tot_mass_radius.plot(tot_radius_tov, tot_mass_tov, color='blue', linewidth=1.5, label="TOV")
ax_tot_mass_radius.set_xlabel(r'$R\ [km]$')
ax_tot_mass_radius.set_ylabel(r'$M / M_{\odot}$')
ax_tot_mass_radius.set_title(r'Total Mass vs Total Radius')
ax_tot_mass_radius.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_tot_mass_radius.legend(loc="upper right")
fig_tot_mass_radius.savefig("total_mass_radius.png", dpi=600, bbox_inches="tight")
plt.show()
