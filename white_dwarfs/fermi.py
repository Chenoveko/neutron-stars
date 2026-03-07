# =====================
# Imports
# =====================
import matplotlib.pyplot as plt

from utilities.physical_functions import rho_degenerate_fermions_ur_geo, rho_degenerate_fermions_nr_geo
from utilities.math_methods import solve_newton_eos
from utilities.physical_data import M_sun, pressure_cgs_to_geo, pressure_geo_to_cgs, mass_cgs_to_geo

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_geo = pressure_cgs_to_geo(1e35)  # Central pressure in CGS

# Integrate TOV equations using Fermi UR EoS
r_ur, p_ur, m_ur, status_ur = solve_newton_eos(p_central_geo, rho_degenerate_fermions_ur_geo)
r_ur = r_ur / 1e5  # Convert radial points to Km
p_ur = pressure_geo_to_cgs(p_ur)  # Convert pressure to CGS
m_ur = m_ur / M_sun_geo  # Convert mass points

# Integrate TOV equations using Fermi NR EoS
r_nr, p_nr, m_nr, status_nr = solve_newton_eos(p_central_geo, rho_degenerate_fermions_nr_geo)
r_nr = r_nr / 1e5  # Convert radial points to Km
p_nr = pressure_geo_to_cgs(p_nr)  # Convert pressure to CGS
m_nr = m_nr / M_sun_geo  # Convert mass points

# ==========Plot 1: Interior Pressure==========#
fig_pressure, ax_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_pressure.plot(r_ur, p_ur, color='goldenrod', linewidth=1.5, label="UR")
ax_pressure.plot(r_nr, p_nr, color='blue', linewidth=1.5, label="NR")
ax_pressure.set_xlabel(r'$r\ (km)$')
ax_pressure.set_ylabel(r'$p(r)\ (dyn\,cm^{-2})$')
ax_pressure.set_title(r'Interior Pressure Fermi EoS')
ax_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_pressure.legend(loc="upper right")

# ==========Plot 2: Mass Enclosed==========#
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r_ur, m_ur, color='goldenrod', linewidth=1.5, label="UR")
ax_mass.plot(r_nr, m_nr, color='blue', linewidth=1.5, label="NR")
ax_mass.set_xlabel(r'$r\ (km)$')
ax_mass.set_ylabel(r'$m(r) / M_{\odot}$')
ax_mass.set_title(r'Enclosed Mass Fermi EoS')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_mass.legend(loc="upper left")

plt.show()