# =====================
# Imports
# =====================
from numpy import array, pi, angle, linspace, meshgrid, vectorize
import numpy as np
from scipy.interpolate import PchipInterpolator
import time
import matplotlib.pyplot as plt
from equations_of_structure.interpol_data import rho_pchip_geo_sly4
from tov_solvers import solve_tov_eos
from utilities.physical_data import M_sun, omega_cgs_to_geo, mass_cgs_to_geo, omega_geo_to_cgs
from muller import muller
from utilities.physical_data import M_sun, pressure_cgs_to_geo, pressure_geo_to_cgs, mass_cgs_to_geo
from qnm_functions import *


# ==========Log time start==========#
start = time.perf_counter()
# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_sly4_geo = 1.103445918598322e-14

# ==========TOV Solutions==========#
r_sly4, p_sly4, m_sly4, nu_sly4, _ = solve_tov_eos(p_central_sly4_geo, rho_pchip_geo_sly4)

# ==========Interpolation TOV Solutions==========#
rho_sly4 = array([rho_pchip_geo_sly4(p) for p in p_sly4])
p_fun = PchipInterpolator(r_sly4, p_sly4)
m_fun = PchipInterpolator(r_sly4, m_sly4)
nu_fun = PchipInterpolator(r_sly4, nu_sly4)
rho_fun = PchipInterpolator(r_sly4, rho_sly4)
dm_fun = m_fun.derivative()
R = r_sly4[-1]
M = m_sly4[-1]
r0 = r_sly4[0]

# ================================
# QNM meshgrid
# ================================
f_real_hz = np.linspace(8.030e3, 8.040e3, 10)      # [Hz]
tau_s = np.linspace(29.25e-6, 29.35e-6, 10)    # [s]

F, TAU = np.meshgrid(f_real_hz, tau_s)

# omega = 2 pi f - i / tau
omega_si = 2.0 * np.pi * F - 1j / TAU # IS
omega_geo = omega_cgs_to_geo(omega_si) # GEO

# condition ->  Im(omega) cos(alpha) - Re(omega) sin(alpha) < 0
alpha_condition = vectorize(alpha_condition)
alpha = np.pi / 4

cond = alpha_condition(alpha, omega_geo)
if np.all(cond):
    print("good")
else:
    print("bad")
# ================================
# Matching Function vectorize
# ================================
f_match = lambda omega: matching_function(omega, r_sly4, m_fun, p_fun, rho_fun, nu_fun, alpha)
f_match = np.vectorize(f_match, otypes=[complex])
# ================================
# Evaluate abs(matching function)
# ================================
f_meshgrid = np.abs(f_match(omega_geo))

# ================================
# Diagnostics of meshgrid
# ================================
idx = np.nanargmin(f_meshgrid)
i_min, j_min = np.unravel_index(idx, f_meshgrid.shape)

f0 = F[i_min, j_min]
tau0 = TAU[i_min, j_min]
omega0 = omega_geo[i_min, j_min]
print("===========Meshgrid search===========")
print(f"Min |F| = {f_meshgrid[i_min, j_min]:.6e}")
print(f"f0 = {f0/1e3:.6f} kHz")
print(f"tau0 = {tau0*1e6:.6f} us")

# ================================
# Muller search
# ================================

omegas_trial = omega_cgs_to_geo([
    2.0 * np.pi * 8.03e3 - 1j / 29.25e-6 ,
    2.0 * np.pi * 8.04e3 - 1j / 29.25e-6 ,
    2.0 * np.pi * 8.03e3 - 1j / 29.35e-6 ,
])

res = muller(f_match, omegas_trial, xtol=1e-10, ftol=1e-10, maxiter=50)

omega_qnm = res.root
omega_qnm_si = omega_geo_to_cgs(omega_qnm)
f_qnm_khz = omega_qnm_si.real / (2.0 * np.pi) / 1e3
tau_qnm_us = -1.0 / omega_qnm_si.imag * 1e6
print("===========Muller search===========")
print(f"QNM: f = {f_qnm_khz:.6f} kHz, tau = {tau_qnm_us:.6f} us")
print(f"Iterations: {res.iterations}")
# ==========Log time end==========#
end = time.perf_counter()
print(f"Elapsed time = {end - start:.3f} s")