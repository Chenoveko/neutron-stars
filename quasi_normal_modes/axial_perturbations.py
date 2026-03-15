# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import logspace, log10, max

from equations_of_structure.interpol_data import rho_pchip_geo_apr, rho_pchip_geo_gnh3, rho_pchip_geo_sly4
from utilities.physical_data import M_sun, pressure_cgs_to_geo, mass_cgs_to_geo
from tov_solvers import solve_tov_eos
from math_methods import muller

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_apr_cgs = logspace(33.2, 36.5, 50)  # Array of central pressure in CGS
p_central_array_geo = pressure_cgs_to_geo(p_central_array_cgs)  # Array of central pressure in GEO
log10_p_central_array_cgs = log10(p_central_array_cgs)
tol =
# ==========Set of initial omegas and angles (free parameter)==========#
omega1 = 5.027e4 + 6.67e-8j
omega2 = 5.030e4 + 6.67e-8j
omega3 = 5.030e4 - 30j
alpha1 = 0.2
alpha2 = 0.2
alpha3 = 0.2
# ==========QNM==========#
# Loop to cover family of central pressure
for p_c in p_central_array_geo:
    # ==========TOV Solutions==========#
    # APR EoS
    r_apr, p_apr, m_apr, nu_apr, status_apr = solve_tov_eos(p_c, rho_pchip_geo_apr)
    r_apr = r_apr / 1e5  # Convert radial points to Km
    m_apr = m_apr / M_sun_geo  # Convert mass points
    # GNH3 EoS
    r_gnh3, p_gnh3, m_gnh3, nu_gnh3, status_gnh3 = solve_tov_eos(p_c, rho_pchip_geo_gnh3)
    r_gnh3 = r_gnh3 / 1e5  # Convert radial points to Km
    m_gnh3 = m_gnh3 / M_sun_geo  # Convert mass points
    # SLy4 EoS
    r_sly4, p_sly4, m_sly4, nu_sly4, status_sly4 = solve_tov_eos(p_c, rho_pchip_geo_sly4)
    r_sly4 = r_sly4 / 1e5  # Convert radial points to Km
    m_sly4 = m_sly4 / M_sun_geo  # Convert mass points
    # ==========Muller Method==========#
    initial_omegas = [omega1, omega2, omega3]
    alphas = [alpha1, alpha2, alpha3]
    for omega in initial_omegas:


