# =====================
# Imports
# =====================
import time

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
from numpy import array, pi, real, imag, meshgrid, linspace, log10, nanargmin, unravel_index, vectorize, abs, log
from scipy.constants import c as c_is
from scipy.interpolate import PchipInterpolator as pchip

from equations_of_structure.interpol_data import rho_pchip_geo_sly4
from solve_qnm import solve_qnm_inside, solve_qnm_outside, matching, muller_seed_meshgrid
from tov_solvers import solve_tov_eos
from utilities.physical_data import M_sun, mass_cgs_to_geo
from mullerpy import muller

# ==========Log time start==========#
start = time.perf_counter()
# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_sly4_geo = 1.103445918598322e-14
c_cgs = c_is * 100  # m/s → cm/s

# Omega
f = 8.034e3  # [Hz]
tau = 29.31e-6  # [s]
omega_si = 2.0 * pi * f - 1j / tau  # IS
omega_geo = omega_si / c_cgs  # GEO

# Angle
alpha = pi / 4

# ================================
# Solve TOV
# ================================
r_sly4, p_sly4, m_sly4, nu_sly4, _ = solve_tov_eos(p_central_sly4_geo, rho_pchip_geo_sly4)

# Interpolation -> rho(r), p(r), m(r), nu(r)
rho_sly4 = array([rho_pchip_geo_sly4(p) for p in p_sly4])
p_fun = pchip(r_sly4, p_sly4, extrapolate=False)
m_fun = pchip(r_sly4, m_sly4, extrapolate=False)
nu_fun = pchip(r_sly4, nu_sly4, extrapolate=False)
rho_fun = pchip(r_sly4, rho_sly4, extrapolate=False)
R = r_sly4[-1]
M = m_sly4[-1]
r0 = r_sly4[0]

# Plot  mass enclosed
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r_sly4 / 1e5, m_sly4 / M_sun_geo, color='red', linewidth=1.5, label="SLy4")
ax_mass.set_xlabel(r'$r\ [km]$')
ax_mass.set_ylabel(r'$m(r) / M_{\odot}$')
ax_mass.set_title(rf'Enclosed mass')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_mass.legend(loc="upper left")


# Plot  metric
def nu_ext(r, M):
    return 1 / 2 * log(1 - 2 * M / r)


r_nu_out = linspace(r_sly4[-1], 5 * r_sly4[-1], 300)
fig_metric, ax_metric = plt.subplots(figsize=(7.5, 4.5))
ax_metric.plot(r_sly4 / 1e5, nu_fun(r_sly4), color='red', linewidth=1.5, label="SLy4 in")
ax_metric.plot(r_nu_out / 1e5, nu_ext(r_nu_out, m_sly4[-1]), color='red', linestyle='--', linewidth=1.5,
               label="SLy4 out")
ax_metric.set_xlabel(r'$r\ [km]$')
ax_metric.set_ylabel(r'$\nu(r)$')
ax_metric.set_title(rf'metric')
ax_metric.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_metric.legend(loc="upper left")
# ================================
# Solve RW inside
# ================================
r_rw_in, z_rw_in, z_prime_rw_in = solve_qnm_inside(r0, R, omega_geo, m_fun, p_fun, rho_fun, nu_fun)
g_rw_in_R = z_prime_rw_in[-1] / z_rw_in[-1]  # Value of g_in(R)

# Plot Z RW inside
fig_z_in, ax_z_in = plt.subplots(figsize=(7.5, 4.5))
ax_z_in.plot(r_rw_in / 1e5, real(z_rw_in), color='goldenrod', linewidth=1.5, label="Real Z")
ax_z_in.plot(r_rw_in / 1e5, imag(z_rw_in), color='blue', linewidth=1.5, label="Imag Z")
ax_z_in.set_xlabel(r'$r\ [km]$')
ax_z_in.set_ylabel(r'$Z$')
ax_z_in.set_title(rf'Z profile inside the star')
ax_z_in.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_z_in.legend(loc="upper left")

# ================================
# Solve RW outside
# ================================
t_rw_out, g_rw_out = solve_qnm_outside(M, R, omega_geo, alpha)
phase = g_rw_out - 1j * omega_geo

# Plot Phase Z RW outside
fig_z_in, ax_z_in = plt.subplots(figsize=(7.5, 4.5))
ax_z_in.plot(t_rw_out, real(phase), color='goldenrod', linewidth=1.5, label="Real Z")
ax_z_in.plot(t_rw_out, imag(phase), color='blue', linewidth=1.5, label="Imag Z")
ax_z_in.set_xlabel(r'$r\ [km]$')
ax_z_in.set_ylabel(r'$Z$')
ax_z_in.set_title(rf'Phase Z profile outside the star')
ax_z_in.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_z_in.legend(loc="upper left")
plt.show()

# ================================
# QNM meshgrid
# ================================
f_mesh = linspace(7.98e3, 8.05e3, 5)
tau_mesh = linspace(28.85e-6, 29.2e-6, 5)

F, TAU = meshgrid(f_mesh, tau_mesh)

# omega = 2 pi f - i / tau
OMEGA = (2.0 * pi * F - 1j / TAU) / c_cgs

# ================================
# Matching Function
# ================================
f_match = lambda omega: matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha)

# ================================
# Evaluate abs(matching function)
# ================================
from joblib import Parallel, delayed

omega_flat = OMEGA.flatten()


def eval_match(omega):
    return abs(matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha))


results = Parallel(n_jobs=-1, verbose=1)(  # n_jobs=-1 usa todos los cores
    delayed(eval_match)(omega) for omega in omega_flat
)

f_meshgrid = array(results).reshape(OMEGA.shape)
f_meshgrid_log10 = log10(f_meshgrid)

# ================================
# Minimum search
# ================================
idx = unravel_index(nanargmin(f_meshgrid_log10), f_meshgrid_log10.shape)
f_best = F[idx] / 1e3
tau_best = TAU[idx] * 1e6
print(f"Mínimo → f = {f_best:.3f} kHz | τ = {tau_best:.3f} µs")

# ================================
# Heatmap 2D
# ================================
fig_heat2, ax_heat2 = plt.subplots(figsize=(7.5, 4.5))
pcm = ax_heat2.pcolormesh(F / 1e3, TAU * 1e6, f_meshgrid_log10, shading='gouraud', cmap='viridis')
cbar = fig_heat2.colorbar(pcm, ax=ax_heat2, pad=0.02)
cbar.set_label(r'$\log_{10}|f_{\mathrm{match}}|$', rotation=90)
ax_heat2.set_xlabel('f [KHz]')
ax_heat2.set_ylabel(r'$\tau$ [$\mu$s]')
ax_heat2.set_title(r'Heatmap matching')
ax_heat2.plot(f_best, tau_best, 'ro', ms=4, label='Mínimo')
ax_heat2.legend(loc='upper right', frameon=False)

# ================================
# Heatmap 3D
# ================================
fig_heat3 = plt.figure(figsize=(7.5, 4.5))
ax_heat3 = fig_heat3.add_subplot(111, projection='3d')
surf = ax_heat3.plot_surface(F / 1e3, TAU * 1e6, f_meshgrid_log10, edgecolor='none', antialiased=True, cmap='viridis')
fig_heat3.colorbar(surf, ax=ax_heat3, shrink=0.6, label=r'$\log_{10}|f_{\mathrm{match}}|$')
ax_heat3.set_xlabel('f [KHz]')
ax_heat3.set_ylabel(r'$\tau$ [$\mu$s]')
ax_heat3.set_zlabel(r'$\log_{10}|f_{\mathrm{match}}|$')
ax_heat3.set_title('Matching 3D surface')
ax_heat3.scatter(f_best, tau_best, f_meshgrid_log10[idx], color='red', s=10, label='Mínimo')
ax_heat3.legend()
plt.show()

# ================================
# Muller search
# ================================
p1, p2, p3 = muller_seed_meshgrid(F, TAU, f_meshgrid_log10)

w1 = (2*pi*p1[0] - 1j/p1[1]) / c_cgs
w2 = (2*pi*p2[0] - 1j/p2[1]) / c_cgs
w3 = (2*pi*p3[0] - 1j/p3[1]) / c_cgs

res = muller(f_match, (w1, w2, w3), xtol=1e-10, ftol=1e-10, maxiter=20)
f_muller = res.root.real / (2*pi) * c_cgs / 1e3
tau_muller = -1 / (res.root.imag * c_cgs) * 1e6
print(f"Müller → f = {f_muller:.6f} kHz | τ = {tau_muller:.6f} µs | it = {res.iterations}")

# ==========Log time end==========#
end = time.perf_counter()
print(f"Elapsed time = {end - start:.3f} s")
