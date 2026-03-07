# =====================
# Imports
# =====================
import matplotlib.pyplot as plt

from eos_fermi import rho_degenerate_fermions_ur_geo
from math_methods import solve_tov_eos, solve_newton_eos
from utilities.physical_data import M_sun, pressure_cgs_to_geo, pressure_geo_to_cgs, mass_cgs_to_geo

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_geo = pressure_cgs_to_geo(5.62e23)  # Central pressure in CGS

# Integrate TOV and Newton equations using Fermi UR EoS
r_newton, p_newton, m_newton, _ = solve_newton_eos(p_central_geo, rho_degenerate_fermions_ur_geo)
r_tov, p_tov, m_tov, _ = solve_tov_eos(p_central_geo, rho_degenerate_fermions_ur_geo)
r_newton = r_newton / 1e5  # Convert radial points to Km
r_tov = r_tov / 1e5  # Convert radial points to Km
p_newton = pressure_geo_to_cgs(p_newton)  # Convert pressure to CGS
p_tov = pressure_geo_to_cgs(p_tov)  # Convert pressure to CGS
m_newton = m_newton / M_sun_geo  # Convert mass points
m_tov = m_tov / M_sun_geo  # Convert mass points

# ==========Plot 1: Interior Pressure==========#
fig_pressure, ax_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_pressure.plot(r_newton, p_newton, color='goldenrod', linewidth=1.5, label="Newton")
ax_pressure.plot(r_tov, p_tov, color='blue', linewidth=1.5, label="TOV")
ax_pressure.set_xlabel(r'$r\ (km)$')
ax_pressure.set_ylabel(r'$p(r)\ (dyn\,cm^{-2})$')
ax_pressure.set_title(r'Interior Pressure Fermi EoS')
ax_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_pressure.legend(loc="upper right")

# ==========Plot 2: Mass Enclosed==========#
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r_newton, m_newton, color='goldenrod', linewidth=1.5, label="Newton")
ax_mass.plot(r_tov, m_tov, color='blue', linewidth=1.5, label="TOV")
ax_mass.set_xlabel(r'$r\ (km)$')
ax_mass.set_ylabel(r'$m(r) / M_{\odot}$')
ax_mass.set_title(r'Enclosed Mass Fermi EoS')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_mass.legend(loc="upper left")

plt.show()

print("Total Mass newton",m_newton[-1])
print("Total Radius newton",r_newton[-1])


print("Total Mass tov",m_tov[-1])
print("Total Radius tov",r_tov[-1])