# =====================
# Imports
# =====================
from time import perf_counter
from numpy import vectorize, linspace
import matplotlib.pyplot as plt
from tov_solvers import solve_tov_eos
from utilities.physical_data import mass_cgs_to_geo, M_sun
from equations_of_structure.interpol_data import rho_pchip_geo_apr,rho_pchip_geo_gnh3,rho_pchip_geo_sly4
from qnm_functions import axial_potential_in, axial_potential_out

# ==========Log time start==========#
start = perf_counter()
# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_geo_apr = 1.21607989338143e-14
p_central_geo_gnh3 = 5.4257477047488676e-15
p_central_geo_sly4 = 1.103445918598322e-14

# ==========TOV Solutions==========#
# APR
r_apr, p_apr, m_apr, nu_apr, _ = solve_tov_eos(p_central_geo_apr, rho_pchip_geo_apr)
rho_pchip_geo_apr_v = vectorize(rho_pchip_geo_apr)
rho_apr = rho_pchip_geo_apr_v(p_apr)

# GNH3
r_gnh3, p_gnh3, m_gnh3, nu_gnh3, _ = solve_tov_eos(p_central_geo_gnh3, rho_pchip_geo_gnh3)
rho_pchip_geo_gnh3_v = vectorize(rho_pchip_geo_gnh3)
rho_gnh3 = rho_pchip_geo_gnh3_v(p_gnh3)

# SLy4
r_sly4, p_sly4, m_sly4, nu_sly4, _ = solve_tov_eos(p_central_geo_sly4, rho_pchip_geo_sly4)
rho_pchip_geo_sly4_v = vectorize(rho_pchip_geo_sly4)
rho_sly4 = rho_pchip_geo_sly4_v(p_sly4)

# ==========Axial potential==========#
# Vectorize functions
axial_potential_in = vectorize(axial_potential_in)
axial_potential_out = vectorize(axial_potential_out)
# Radial coordinate outside star
r_out_apr = linspace(r_apr[-1], 3*r_apr[-1], 100)
r_out_gnh3 = linspace(r_gnh3[-1], 3*r_gnh3[-1], 100)
r_out_sly4 = linspace(r_sly4[-1], 3*r_sly4[-1], 100)

# Axial potential
axial_potential_apr_in = axial_potential_in(r_apr, m_apr, p_apr,rho_apr,nu_apr)
axial_potential_apr_out = axial_potential_out(r_out_apr, m_apr[-1])

axial_potential_gnh3_in = axial_potential_in(r_gnh3, m_gnh3, p_gnh3,rho_gnh3,nu_gnh3)
axial_potential_gnh3_out = axial_potential_out(r_out_gnh3, m_gnh3[-1])

axial_potential_sly4_in = axial_potential_in(r_sly4, m_sly4, p_sly4,rho_sly4,nu_sly4)
axial_potential_sly4_out = axial_potential_out(r_out_sly4, m_sly4[-1])

# ==========Plots==========#
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(r_apr/1e5, axial_potential_apr_in, color='goldenrod', linewidth=1.5, label="APR in")
ax.plot(r_out_apr/1e5, axial_potential_apr_out, color='goldenrod', linewidth=1.5,linestyle='--', label="APR out")
#ax.plot(r_gnh3/1e5, axial_potential_gnh3_in, color='blue', linewidth=1.5, label="GNH3 in")
#ax.plot(r_out_gnh3/1e5, axial_potential_gnh3_out, color='blue', linewidth=1.5,linestyle='--', label="GNH3 out")
#ax.plot(r_sly4/1e5, axial_potential_sly4_in, color='red', linewidth=1.5, label="SLy4 in")
#ax.plot(r_out_sly4/1e5, axial_potential_sly4_out, color='red', linewidth=1.5,linestyle='--', label="SLy4 out")
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$V(r)$')
ax.set_title(rf'Axial potential')
ax.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax.legend(loc="upper right")
ax.set_yscale('log')
plt.show()
# ==========Log time end==========#
end = perf_counter()
print(f"Elapsed time = {end - start:.3f} s")