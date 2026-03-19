# =====================
# Imports
# =====================
from numpy import array, pi, angle, linspace, meshgrid, vectorize
import numpy as np
from scipy.interpolate import PchipInterpolator
import time
import matplotlib.pyplot as plt
from equations_of_structure.interpol_data import rho_pchip_geo_apr
from solve_qnm import matching_function
from tov_solvers import solve_tov_eos
from utilities.physical_data import M_sun, omega_cgs_to_geo, mass_cgs_to_geo, omega_geo_to_cgs
from mullerpy import muller
from utilities.physical_data import M_sun, pressure_cgs_to_geo, pressure_geo_to_cgs, mass_cgs_to_geo

# ==========Log time start==========#
start = time.perf_counter()
# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_apr_geo = 1.3095524938385862e-16
# p_central_gnh3_geo = 5.4257477047488676e-15
# p_central_sly4_geo = 1.103445918598322e-14
tol_f = 1e-8
tol_omega = 1e-8
l = 2
# ==========TOV Solutions==========#
r_apr, p_apr, m_apr, nu_apr, _ = solve_tov_eos(p_central_apr_geo, rho_pchip_geo_apr)
# r_apr = r_apr / 1e5  # Convert radial points to Km
# p_apr = pressure_geo_to_cgs(p_apr)  # Convert pressure to CGS
# m_apr = m_apr / M_sun_geo  # Convert mass points

# ==========Interpolation TOV Solutions==========#
rho_apr = array([rho_pchip_geo_apr(p) for p in p_apr])
p_fun = PchipInterpolator(r_apr, p_apr)
m_fun = PchipInterpolator(r_apr, m_apr)
nu_fun = PchipInterpolator(r_apr, nu_apr)
rho_fun = PchipInterpolator(r_apr, rho_apr)
dm_fun = m_fun.derivative()
R = r_apr[-1]
M = m_apr[-1]
r0_qnm = r_apr[0]
# ==========Plots==========#
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r_apr, m_apr, color='goldenrod', linewidth=1.5, label="APR")
ax_mass.set_xlabel(r'$r\ [km]$')
ax_mass.set_ylabel(r'$m(r) / M_{\odot}$')
ax_mass.set_title(rf'Enclosed mass')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_mass.legend(loc="upper left")

fig_pressure, ax_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_pressure.plot(r_apr, p_apr, color='goldenrod', linewidth=1.5, label="APR")
ax_pressure.set_xlabel(r'$r\ [km]$')
ax_pressure.set_ylabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_pressure.set_title(rf'Interior pressure')
ax_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_pressure.legend(loc="upper right")

fig_nu, ax_nu = plt.subplots(figsize=(7.5, 4.5))
ax_nu.plot(r_apr, nu_apr, color='goldenrod', linewidth=1.5, label="APR")
ax_nu.set_xlabel(r'$r\ [km]$')
ax_nu.set_ylabel(r'$\nu(r)$')
ax_nu.set_title(rf'Metric function')
ax_nu.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_nu.legend(loc="upper left")


# ================================
# QNM meshgrid
# ================================
f_real_hz = np.linspace(9.7e3, 10.1e3, 50)      # [Hz]
tau_s = np.linspace(1.0e-6, 25.0e-6, 50)    # [s]

F, TAU = np.meshgrid(f_real_hz, tau_s)

# omega = 2 pi f - i / tau
omega_si = 2.0 * np.pi * F - 1j / TAU # IS
omega_geo = omega_cgs_to_geo(omega_si) # GEO
alpha = np.deg2rad(10.0)
# ================================
# Matching Function
# ================================
f_qnm_v = np.vectorize(lambda omega: matching_function(omega, r_apr[0], r_apr[-1], m_fun, p_fun, rho_fun, nu_fun, dm_fun, m_apr[-1], alpha,l))
# ================================
# Evaluate abs(matching function)
# ================================
f_qnm_meshgrid = np.abs(f_qnm_v(omega_geo))

# ================================
# Diagnostics of meshgrid
# ================================
idx = np.nanargmin(f_qnm_meshgrid)
i_min, j_min = np.unravel_index(idx, f_qnm_meshgrid.shape)

f0 = F[i_min, j_min]
tau0 = TAU[i_min, j_min]
omega0 = omega_geo[i_min, j_min]
print("===========Meshgrid search===========")
print(f"Min |F| = {f_qnm_meshgrid[i_min, j_min]:.6e}")
print(f"f0 = {f0/1e3:.6f} kHz")
print(f"tau0 = {tau0*1e6:.6f} us")

# ================================
# Heatmap 2D
# ================================
fig2d, ax2d = plt.subplots(figsize=(8, 6))

# Evita problemas con LogNorm si hubiese ceros
z_plot = np.where(f_qnm_meshgrid <= 0, np.nan, f_qnm_meshgrid)

pcm = ax2d.pcolormesh(
    F / 1e3,               # kHz
    TAU * 1e6,             # us
    z_plot,
    shading='auto',
    norm=LogNorm(vmin=np.nanmin(z_plot), vmax=np.nanmax(z_plot)),
    cmap='viridis'
)

ax2d.scatter(
    f0 / 1e3,
    tau0 * 1e6,
    marker='x',
    s=100,
    linewidths=2,
    color='red',
    label=f"mínimo: {f0/1e3:.4f} kHz, {tau0*1e6:.2f} us"
)

ax2d.set_xlabel(r"$f_{\mathrm{real}}\ \mathrm{[kHz]}$")
ax2d.set_ylabel(r"$\tau\ \mathrm{[\mu s]}$")
ax2d.set_title(r"Heatmap de $|\mathrm{matching\ function}|$")
ax2d.legend(loc='best')

cbar = fig2d.colorbar(pcm, ax=ax2d)
cbar.set_label(r"$|\mathrm{matching\ function}|$")

plt.tight_layout()
plt.show()

# ================================
# Heatmap 3D
# ================================
fig3d = plt.figure(figsize=(9, 7))
ax3d = fig3d.add_subplot(111, projection='3d')

surf = ax3d.plot_surface(
    F / 1e3,
    TAU * 1e6,
    f_qnm_meshgrid,
    cmap='viridis',
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True
)

ax3d.scatter(
    f0 / 1e3,
    tau0 * 1e6,
    z0,
    color='red',
    s=50
)

ax3d.set_xlabel(r"$f_{\mathrm{real}}\ \mathrm{[kHz]}$")
ax3d.set_ylabel(r"$\tau\ \mathrm{[\mu s]}$")
ax3d.set_zlabel(r"$|\mathrm{matching\ function}|$")
ax3d.set_title(r"Superficie 3D de $|\mathrm{matching\ function}|$")

cbar3d = fig3d.colorbar(surf, ax=ax3d, shrink=0.75, pad=0.1)
cbar3d.set_label(r"$|\mathrm{matching\ function}|$")

plt.tight_layout()
plt.show()
# ================================
# Muller search
# ================================
"""
f_qnm = lambda omega: matching_function(omega, r_apr[0], r_apr[-1], m_fun, p_fun, rho_fun, nu_fun, dm_fun, m_apr[-1], np.angle(omega), l)
delta_re = 0.02 * abs(omega0.real) if omega0.real != 0 else 1e-6
delta_im = 0.02 * abs(omega0.imag) if omega0.imag != 0 else 1e-6

omegas_trial = [
    omega0,
    omega0 + delta_re - 1j * delta_im,
    omega0 - delta_re + 1j * delta_im,
]

res = muller(f_qnm, omegas_trial, xtol=1e-10, ftol=1e-10, maxiter=50)

omega_qnm = res.root

omega_qnm_si = omega_geo_to_cgs(omega_qnm)
f_qnm_khz = omega_qnm_si.real / (2.0 * np.pi) / 1e3
tau_qnm_us = -1.0 / omega_qnm_si.imag * 1e6
print("===========Muller search===========")
print(f"QNM: f = {f_qnm_khz:.6f} kHz, tau = {tau_qnm_us:.6f} us")
"""

# ==========Log time end==========#
end = time.perf_counter()
print(f"Elapsed time = {end - start:.3f} s")