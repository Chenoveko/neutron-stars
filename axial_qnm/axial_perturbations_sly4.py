# =====================
# Imports
# =====================
import time

import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
from numpy import array, pi, meshgrid, linspace, log10, nanargmin, unravel_index, abs, interp, logspace
from scipy.constants import c as c_is
from scipy.interpolate import PchipInterpolator as pchip

from equations_of_structure.interpol_data import rho_pchip_geo_sly4
from solve_qnm import matching, muller_seed_meshgrid
from tov_solvers import solve_tov_eos
from utilities.physical_data import M_sun, mass_cgs_to_geo,pressure_cgs_to_geo
from mullerpy import muller

# ==========Log time start==========#
start = time.perf_counter()
# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_array_cgs = logspace(34.4, 36.3, 300)  # Array of central pressure in CGS
p_central_array_geo = pressure_cgs_to_geo(p_central_array_cgs)  # Array of central pressure in GEO
log10_p_central_array_cgs = log10(p_central_array_cgs)
c_cgs = c_is * 100  # m/s → cm/s
alpha = pi / 4 # Angle for CES

# QNM Algorithm
it = 0
f_sly4 = []
tau_sly4 = []
for p_c in p_central_array_geo:
    it += 1
    # Solve TOV
    r_sly4, p_sly4, m_sly4, nu_sly4, _ = solve_tov_eos(p_c, rho_pchip_geo_sly4)
    # Interpolation -> rho(r), p(r), m(r), nu(r)
    rho_sly4 = array([rho_pchip_geo_sly4(p) for p in p_sly4])
    p_fun = pchip(r_sly4, p_sly4, extrapolate=False)
    m_fun = pchip(r_sly4, m_sly4, extrapolate=False)
    nu_fun = pchip(r_sly4, nu_sly4, extrapolate=False)
    rho_fun = pchip(r_sly4, rho_sly4, extrapolate=False)
    # TOV parameters
    R = r_sly4[-1]
    M = m_sly4[-1]
    r0 = r_sly4[0]
    # Matching function
    f_match = lambda omega: matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha)
    # QNM bruteforce search -> meshgrid
    if it ==1:
        f_mesh = linspace(10.44e3, 10.54e3, 100)
        tau_mesh = linspace(15.56e-6, 15.64e-6, 100)
        F, TAU = meshgrid(f_mesh, tau_mesh)
        OMEGA = (2.0 * pi * F - 1j / TAU) / c_cgs
        # Evaluate abs(matching function) in meshgrid
        from joblib import Parallel, delayed
        omega_flat = OMEGA.flatten()
        def eval_match(omega):
            return abs(matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha))
        results = Parallel(n_jobs=-1, verbose=1)(  # n_jobs=-1 usa todos los cores
            delayed(eval_match)(omega) for omega in omega_flat
        )
        f_meshgrid = array(results).reshape(OMEGA.shape)
        f_meshgrid_log10 = log10(f_meshgrid)
        # Minimum search
        idx = unravel_index(nanargmin(f_meshgrid_log10), f_meshgrid_log10.shape)
        f_best = F[idx] / 1e3
        tau_best = TAU[idx] * 1e6
        print(f"Minimum → f = {f_best:.3f} KHz | τ = {tau_best:.3f} µs")
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
        surf = ax_heat3.plot_surface(F / 1e3, TAU * 1e6, f_meshgrid_log10, edgecolor='none', antialiased=True,
                                     cmap='viridis')
        fig_heat3.colorbar(surf, ax=ax_heat3, shrink=0.6, label=r'$\log_{10}|f_{\mathrm{match}}|$')
        ax_heat3.set_xlabel('f [KHz]')
        ax_heat3.set_ylabel(r'$\tau$ [$\mu$s]')
        ax_heat3.set_zlabel(r'$\log_{10}|f_{\mathrm{match}}|$')
        ax_heat3.set_title('Matching 3D surface')
        ax_heat3.scatter(f_best, tau_best, f_meshgrid_log10[idx], color='red', s=10, label='Mínimo')
        ax_heat3.legend()
        plt.show()
        # Muller search
        p1, p2, p3 = muller_seed_meshgrid(F, TAU, f_meshgrid_log10)

        w1 = (2 * pi * p1[0] - 1j / p1[1]) / c_cgs
        w2 = (2 * pi * p2[0] - 1j / p2[1]) / c_cgs
        w3 = (2 * pi * p3[0] - 1j / p3[1]) / c_cgs

        res = muller(f_match, (w1, w2, w3), xtol=1e-10, ftol=1e-10, maxiter=50)
        f_muller = res.root.real / (2 * pi) * c_cgs / 1e3
        tau_muller = -1 / (res.root.imag * c_cgs) * 1e6
        print(f"Müller → f = {f_muller:.6f} kHz | τ = {tau_muller:.6f} µs | it = {res.iterations}")
        f_sly4.append(f_muller)
        tau_sly4.append(tau_muller)
    elif it ==2:
        f_mesh = linspace(10.46e3, 10.54e3, 40)
        tau_mesh = linspace(15.54e-6, 15.64e-6, 40)
        F, TAU = meshgrid(f_mesh, tau_mesh)
        OMEGA = (2.0 * pi * F - 1j / TAU) / c_cgs
        # Evaluate abs(matching function) in meshgrid
        from joblib import Parallel, delayed
        omega_flat = OMEGA.flatten()
        def eval_match(omega):
            return abs(matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha))
        results = Parallel(n_jobs=-1, verbose=1)(  # n_jobs=-1 usa todos los cores
            delayed(eval_match)(omega) for omega in omega_flat
        )
        f_meshgrid = array(results).reshape(OMEGA.shape)
        f_meshgrid_log10 = log10(f_meshgrid)
        # Minimum search
        idx = unravel_index(nanargmin(f_meshgrid_log10), f_meshgrid_log10.shape)
        f_best = F[idx] / 1e3
        tau_best = TAU[idx] * 1e6
        print(f"Minimum → f = {f_best:.3f} KHz | τ = {tau_best:.3f} µs")
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
        surf = ax_heat3.plot_surface(F / 1e3, TAU * 1e6, f_meshgrid_log10, edgecolor='none', antialiased=True,
                                     cmap='viridis')
        fig_heat3.colorbar(surf, ax=ax_heat3, shrink=0.6, label=r'$\log_{10}|f_{\mathrm{match}}|$')
        ax_heat3.set_xlabel('f [KHz]')
        ax_heat3.set_ylabel(r'$\tau$ [$\mu$s]')
        ax_heat3.set_zlabel(r'$\log_{10}|f_{\mathrm{match}}|$')
        ax_heat3.set_title('Matching 3D surface')
        ax_heat3.scatter(f_best, tau_best, f_meshgrid_log10[idx], color='red', s=10, label='Mínimo')
        ax_heat3.legend()
        plt.show()
        # Muller search
        p1, p2, p3 = muller_seed_meshgrid(F, TAU, f_meshgrid_log10)
        w1 = (2 * pi * p1[0] - 1j / p1[1]) / c_cgs
        w2 = (2 * pi * p2[0] - 1j / p2[1]) / c_cgs
        w3 = (2 * pi * p3[0] - 1j / p3[1]) / c_cgs
        res = muller(f_match, (w1, w2, w3), xtol=1e-10, ftol=1e-10, maxiter=50)
        f_muller = res.root.real / (2 * pi) * c_cgs / 1e3
        tau_muller = -1 / (res.root.imag * c_cgs) * 1e6
        print(f"Müller → f = {f_muller:.6f} kHz | τ = {tau_muller:.6f} µs | it = {res.iterations}")
        f_sly4.append(f_muller)
        tau_sly4.append(tau_muller)
    elif it == 3:
        f_extrapol = interp(p_c,p_central_array_geo[:2],f_sly4)
        tau_extrapol = interp(p_c, p_central_array_geo[:2], tau_sly4)
    else:
        break
# ==========Log time end==========#
end = time.perf_counter()
print(f"Elapsed time = {end - start:.3f} s")
