# =====================
# Imports
# =====================
import matplotlib.pyplot as plt

from equations_of_structure.interpol_data import rho_pchip_geo_apr, rho_pchip_geo_gnh3, rho_pchip_geo_sly4
from utilities.math_methods import solve_tov_eos
from utilities.physical_data import M_sun, pressure_cgs_to_geo, pressure_geo_to_cgs, mass_cgs_to_geo

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_geo = pressure_cgs_to_geo(1.5e35)  # Central pressure in CGS 

# ==========Integration==========#

# Integrate TOV equations using APR EoS
r_apr, p_apr, m_apr, status_apr = solve_tov_eos(p_central_geo, rho_pchip_geo_apr)
r_apr = r_apr / 1e5  # Convert radial points to Km
p_apr = pressure_geo_to_cgs(p_apr)  # Convert pressure to CGS
m_apr = m_apr / M_sun_geo  # Convert mass points

# Integrate TOV equations using GNH3 EoS
r_gnh3, p_gnh3, m_gnh3, status_gnh3 = solve_tov_eos(p_central_geo, rho_pchip_geo_gnh3)
r_gnh3 = r_gnh3 / 1e5  # Convert radial points to Km
p_gnh3 = pressure_geo_to_cgs(p_gnh3)  # Convert pressure to CGS
m_gnh3 = m_gnh3 / M_sun_geo  # Convert mass points

# Integrate TOV equations using SLY4 EoS
r_sly4, p_sly4, m_sly4, status_sly4l = solve_tov_eos(p_central_geo, rho_pchip_geo_sly4)
r_sly4 = r_sly4 / 1e5  # Convert radial points to Km
p_sly4 = pressure_geo_to_cgs(p_sly4)  # Convert pressure to CGS
m_sly4 = m_sly4 / M_sun_geo  # Convert mass points

# ==========Plot 1: Interior Pressure==========#
fig_pressure, ax_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_pressure.plot(r_apr, p_apr, color='goldenrod', linewidth=1.5, label="APR")
ax_pressure.plot(r_gnh3, p_gnh3, color='blue', linewidth=1.5, label="GNH3")
ax_pressure.plot(r_sly4, p_sly4, color='red', linewidth=1.5, label="SLy4")
ax_pressure.set_xlabel(r'$r\ (km)$')
ax_pressure.set_ylabel(r'$p(r)\ (dyn\,cm^{-2})$')
ax_pressure.set_title(r'Interior Pressure')
ax_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_pressure.legend(loc="upper right")

# ==========Plot 2: Zoom Interior Pressure==========#
fig_zoom_pressure, ax_zoom_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_zoom_pressure.plot(r_apr, p_apr, color='goldenrod', linewidth=1.5, label="APR")
ax_zoom_pressure.plot(r_gnh3, p_gnh3, color='blue', linewidth=1.5, label="GNH3")
ax_zoom_pressure.plot(r_sly4, p_sly4, color='red', linewidth=1.5, label="SLy4")
ax_zoom_pressure.set_xlabel(r'$r\ (km)$')
ax_zoom_pressure.set_ylabel(r'$p(r)\ (dyn\,cm^{-2})$')
ax_zoom_pressure.set_title(r'Interior Pressure Surface ')
ax_zoom_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_zoom_pressure.legend(loc="upper right")
ax_zoom_pressure.set_xlim(xmin=10.5, xmax=13)
ax_zoom_pressure.set_ylim(ymin=0, ymax=1e33)

# ==========Plot 3: Mass Enclosed==========#
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r_apr, m_apr, color='goldenrod', linewidth=1.5, label="APR")
ax_mass.plot(r_gnh3, m_gnh3, color='blue', linewidth=1.5, label="GNH3")
ax_mass.plot(r_sly4, m_sly4, color='red', linewidth=1.5, label="SLy4")
ax_mass.set_xlabel(r'$r\ (km)$')
ax_mass.set_ylabel(r'$m(r) / M_{\odot}$')
ax_mass.set_title(r'Enclosed Mass')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_mass.legend(loc="upper left")

# ==========Plot 4: Zoom Mass Enclosed==========#
fig_zoom_mass, ax_zoom_mass = plt.subplots(figsize=(7.5, 4.5))
ax_zoom_mass.plot(r_apr, m_apr, color='goldenrod', linewidth=1.5, label="APR")
ax_zoom_mass.plot(r_gnh3, m_gnh3, color='blue', linewidth=1.5, label="GNH3")
ax_zoom_mass.plot(r_sly4, m_sly4, color='red', linewidth=1.5, label="SLy4")
ax_zoom_mass.set_xlabel(r'$r\ (km)$')
ax_zoom_mass.set_ylabel(r'$m(r) / M_{\odot}$')
ax_zoom_mass.set_title(r'Enclosed Mass Surface')
ax_zoom_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_zoom_mass.legend(loc="upper left")
ax_zoom_mass.set_xlim(xmin=10.8, xmax=13)
ax_zoom_mass.set_ylim(ymin=1, ymax=1.75)

plt.show()