# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
import time
from numpy import logspace, log10, vectorize

from equations_of_structure.interpol_data import rho_pchip_geo_apr, rho_pchip_geo_gnh3, rho_pchip_geo_sly4
from utilities.physical_data import M_sun, pressure_cgs_to_geo, mass_cgs_to_geo
from utilities.tov_solvers import solve_tov_eos

# ==========Log time start==========#
start = time.perf_counter()
# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_array_cgs = logspace(34.4, 36.3, 100)  # Array of central pressure in CGS
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
    tot_mass_apr.append(m_apr[-1])
    tot_radius_apr.append(r_apr[-1])
    # GNH3 EoS
    r_gnh3, _, m_gnh3, _ = solve_tov_eos(p_c, rho_pchip_geo_gnh3)
    tot_mass_gnh3.append(m_gnh3[-1])
    tot_radius_gnh3.append(r_gnh3[-1])
    # SLy4 EoS
    r_sly4, _, m_sly4, _ = solve_tov_eos(p_c, rho_pchip_geo_sly4)
    tot_mass_sly4.append(m_sly4[-1])
    tot_radius_sly4.append(r_sly4[-1])

# Estimate QNM
def qnm_real_fit(A:float,B:float,M:float,R:float)->complex:
    """
    Linear fit estimations for rial QNM in GEO units
    :param A: parameter
    :param B: parameter
    :param M: total mass
    :param R: total radius
    return: omega rial [KHz]
    """
    compact = M/R
    R_Km = R/1e5
    return 1/R_Km*(A*compact + B)

def qnm_im_fit(a:float,b:float,c:float,M:float,R:float)->complex:
    """
    Linear fit estimations for imag QNM in GEO units
    :param a: parameter
    :param b: parameter
    :param c: parameter
    :param M: total mass
    :param R: total radius
    return: damping time [nuz]
    """
    compact = M / R
    right = 1/(M/M_sun_geo)*(a*compact**2 + b*compact + c)
    return 1e3/right

qnm_real_fit = vectorize(qnm_real_fit)
qnm_im_fit = vectorize(qnm_im_fit)
# Plot QNM omega real
fig_f, ax_f = plt.subplots(figsize=(7.5, 4.5))
#ax_f.plot(log10_p_central_array_cgs, f_apr, color='goldenrod', linewidth=1.5, label="APR")
#ax_f.plot(log10_p_central_array_cgs, f_gnh3, color='blue', linewidth=1.5, label="GNH3")
ax_f.plot(log10_p_central_array_cgs, qnm_real_fit(-148.7,119.8,tot_mass_sly4,tot_radius_sly4), color='red', linewidth=1.5, label="SLy4")
ax_f.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_f.set_ylabel('f [KHz]')
ax_f.set_title(r'Omega real')
ax_f.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_f.legend(loc="upper right")
print(qnm_real_fit(-148.7,119.8,tot_mass_sly4,tot_radius_sly4)[0])
# Plot QNM damping time
fig_tau, ax_tau = plt.subplots(figsize=(7.5, 4.5))
#ax_tau.plot(log10_p_central_array_cgs, f_apr, color='goldenrod', linewidth=1.5, label="APR")
#ax_tau.plot(log10_p_central_array_cgs, f_gnh3, color='blue', linewidth=1.5, label="GNH3")
ax_tau.plot(log10_p_central_array_cgs, qnm_im_fit(-1221,365.1,21.63,tot_mass_sly4,tot_radius_sly4), color='red', linewidth=1.5, label="SLy4")
ax_tau.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_tau.set_ylabel(r'$\tau$ [$\mu$s]')
ax_tau.set_title(r'Damping time')
ax_tau.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_tau.legend(loc="upper right")
plt.show()
print(qnm_im_fit(-1221,365.1,21.63,tot_mass_sly4,tot_radius_sly4)[0])