# =====================
# Imports
# =====================
import matplotlib.pyplot as plt

from eos.interpol_data import log10_p_akmalpr
from eos.interpol_data import rho_cs_geo_akmalpr
from utilities.math_methods import rk4_eos_pc_free
from utilities.physical_data import M_sun, pressure_cgs_to_geo, mass_cgs_to_geo

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)

# ==========Results from APR==========#
# Slicing of log1o pressure
log10_p_akmalpr = log10_p_akmalpr[log10_p_akmalpr > 32]
# Converto log10 pressure into CGS
p_akmalpr_cgs = 10 ** log10_p_akmalpr
# Convert CGS pressure into GEO
p_akmalpr_geo = pressure_cgs_to_geo(p_akmalpr_cgs)
# Integrate TOV equations using RK4
results_akmalpr = rk4_eos_pc_free(p_akmalpr_geo, rho_cs_geo_akmalpr)
tot_mass_akmalpr = results_akmalpr[0] / M_sun_geo
tot_radius_akmalpr = results_akmalpr[1] / 1e5  # Convert to Km

# ==========Plot of total mass vs central pressure==========#
fig_tot_mass, ax_tot_mass = plt.subplots(figsize=(7.5, 4.5))
ax_tot_mass.plot(log10_p_akmalpr, tot_mass_akmalpr, color='goldenrod', linewidth=1.5, label="APR")
ax_tot_mass.set_xlabel(r'$log_{10}(p_c)$')
ax_tot_mass.set_ylabel(r'$M / M_{\odot}$')
ax_tot_mass.set_title(r'Total Mass Enclosed')
ax_tot_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)

# ==========Plot of total mass vs radius==========#
fig_tot_radius, ax_tot_radius = plt.subplots(figsize=(7.5, 4.5))
ax_tot_radius.plot(log10_p_akmalpr, tot_radius_akmalpr, color='blue', linewidth=2.0, label="APR")
ax_tot_radius.set_xlabel(r'$log_{10}(p_c)$')
ax_tot_radius.set_ylabel(r'$R\ (km)$')
ax_tot_radius.set_title(r'Total Radius')
ax_tot_radius.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
plt.show()
