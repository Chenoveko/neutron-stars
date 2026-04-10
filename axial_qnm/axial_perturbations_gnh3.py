# =====================
# Imports
# =====================
import time

import matplotlib.pyplot as plt
from mullerpy import muller
# import matplotlib
# matplotlib.use('TkAgg')
from numpy import array, pi, meshgrid, linspace, log10, nanargmin, unravel_index, abs, concatenate, logspace, nan,savez
from scipy.constants import c as c_is
from scipy.interpolate import PchipInterpolator as pchip

from equations_of_structure.interpol_data import rho_pchip_geo_gnh3
from solve_qnm import matching, muller_seed_meshgrid,muller_seed_from_extrapolation
from tov_solvers import solve_tov_eos
from utilities.physical_data import M_sun, mass_cgs_to_geo, pressure_cgs_to_geo, pressure_geo_to_cgs

# ==========Log time start==========#
start = time.perf_counter()
# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_anchor = 6.56654135036757e+34 # anchor point at 1.4 solar masses for central pressure
p_central_array_cgs_up = logspace(log10(p_anchor), 36.5, 100)  # Array of central pressure in CGS up
p_central_array_cgs_down = logspace(log10(p_anchor), 34.75, 100)  # Array of central pressure in CGS down
p_central_array_geo_up = pressure_cgs_to_geo(p_central_array_cgs_up)  # Array of central pressure in GEO up
p_central_array_geo_down = pressure_cgs_to_geo(p_central_array_cgs_down)  # Array of central pressure in GEO down
log10_p_central_array_cgs_up = log10(p_central_array_cgs_up)
log10_p_central_array_cgs_down = log10(p_central_array_cgs_down)
c_cgs = c_is * 100  # m/s → cm/s
alpha = pi / 4  # Angle for CES

# =====================
# QNM Algorithm
# =====================
# Up search
it_up = 0
f_gnh3_up = []
tau_gnh3_up = []
for p_c in p_central_array_geo_up:
    it_up += 1
    # Solve TOV
    r_gnh3, p_gnh3, m_gnh3, nu_gnh3, _ = solve_tov_eos(p_c, rho_pchip_geo_gnh3)
    # Interpolation -> rho(r), p(r), m(r), nu(r)
    rho_gnh3 = array([rho_pchip_geo_gnh3(p) for p in p_gnh3])
    p_fun = pchip(r_gnh3, p_gnh3, extrapolate=False)
    m_fun = pchip(r_gnh3, m_gnh3, extrapolate=False)
    nu_fun = pchip(r_gnh3, nu_gnh3, extrapolate=False)
    rho_fun = pchip(r_gnh3, rho_gnh3, extrapolate=False)
    # TOV parameters
    R = r_gnh3[-1]
    M = m_gnh3[-1]
    r0 = r_gnh3[0]
    # Matching function
    f_match = lambda omega: matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha)
    # QNM bruteforce search -> meshgrid
    if it_up == 1:
        f_mesh = linspace(6.85e3, 7.10e3, 5)
        tau_mesh = linspace(28.35e-6, 28.5e-6, 5)
        F, TAU = meshgrid(f_mesh, tau_mesh)
        OMEGA = (2.0 * pi * F - 1j / TAU) / c_cgs
        # Evaluate abs(matching function) in meshgrid
        from joblib import Parallel, delayed
        omega_flat = OMEGA.flatten()
        def eval_match(omega):
            return abs(matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha))
        results = Parallel(n_jobs=3, verbose=1)(  # n_jobs=-1 usa todos los cores
            delayed(eval_match)(omega) for omega in omega_flat
        )
        f_meshgrid = array(results).reshape(OMEGA.shape)
        f_meshgrid_log10 = log10(f_meshgrid)
        # Minimum search
        idx = unravel_index(nanargmin(f_meshgrid_log10), f_meshgrid_log10.shape)
        f_best = F[idx] / 1e3
        tau_best = TAU[idx] * 1e6
        print(f"Minimum at first iteration GNH3 → f = {f_best:.3f} KHz | τ = {tau_best:.3f} µs | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        # ================================
        # Heatmap 2D
        # ================================
        fig_heat2, ax_heat2 = plt.subplots(figsize=(7.5, 4.5))
        pcm = ax_heat2.pcolormesh(F / 1e3, TAU * 1e6, f_meshgrid_log10, shading='gouraud', cmap='viridis')
        cbar = fig_heat2.colorbar(pcm, ax=ax_heat2, pad=0.02)
        cbar.set_label(r'$\log_{10}|f_{\mathrm{match}}|$', rotation=90)
        ax_heat2.set_xlabel('f [KHz]')
        ax_heat2.set_ylabel(r'$\tau$ [$\mu$s]')
        ax_heat2.set_title(rf'Heatmap matching GNH3, iteration {it_up}')
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
        ax_heat3.set_title(rf'Matching 3D surface GNH3, iteration {it_up}')
        ax_heat3.scatter(f_best, tau_best, f_meshgrid_log10[idx], color='red', s=10, label='Mínimo')
        ax_heat3.legend()
        plt.show()
        # Muller search
        p1, p2, p3 = muller_seed_meshgrid(F, TAU, f_meshgrid_log10)
        w1 = (2 * pi * p1[0] - 1j / p1[1]) / c_cgs
        w2 = (2 * pi * p2[0] - 1j / p2[1]) / c_cgs
        w3 = (2 * pi * p3[0] - 1j / p3[1]) / c_cgs
        res = muller(f_match, (w1, w2, w3), xtol=1e-12, ftol=1e-12, maxiter=50)
        f_muller = res.root.real / (2 * pi) * c_cgs / 1e3
        tau_muller = -1 / (res.root.imag * c_cgs) * 1e6
        print(f"Müller GNH3 → f = {f_muller:.6f} kHz | τ = {tau_muller:.6f} µs | it = {res.iterations} | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        f_gnh3_up.append(f_muller)
        tau_gnh3_up.append(tau_muller)
    elif it_up == 2:
        f_mesh = linspace(6.85e3, 7.10e3, 5)
        tau_mesh = linspace(28.35e-6, 28.5e-6, 5)
        F, TAU = meshgrid(f_mesh, tau_mesh)
        OMEGA = (2.0 * pi * F - 1j / TAU) / c_cgs
        # Evaluate abs(matching function) in meshgrid
        from joblib import Parallel, delayed
        omega_flat = OMEGA.flatten()
        def eval_match(omega):
            return abs(matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha))
        results = Parallel(n_jobs=3, verbose=1)(  # n_jobs=-1 usa todos los cores
            delayed(eval_match)(omega) for omega in omega_flat
        )
        f_meshgrid = array(results).reshape(OMEGA.shape)
        f_meshgrid_log10 = log10(f_meshgrid)
        # Minimum search
        idx = unravel_index(nanargmin(f_meshgrid_log10), f_meshgrid_log10.shape)
        f_best = F[idx] / 1e3
        tau_best = TAU[idx] * 1e6
        print(f"Minimum at second iteration GNH3 → f = {f_best:.3f} KHz | τ = {tau_best:.3f} µs | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        # ================================
        # Heatmap 2D
        # ================================
        fig_heat2, ax_heat2 = plt.subplots(figsize=(7.5, 4.5))
        pcm = ax_heat2.pcolormesh(F / 1e3, TAU * 1e6, f_meshgrid_log10, shading='gouraud', cmap='viridis')
        cbar = fig_heat2.colorbar(pcm, ax=ax_heat2, pad=0.02)
        cbar.set_label(r'$\log_{10}|f_{\mathrm{match}}|$', rotation=90)
        ax_heat2.set_xlabel('f [KHz]')
        ax_heat2.set_ylabel(r'$\tau$ [$\mu$s]')
        ax_heat2.set_title(rf'Heatmap matching GNH3, iteration {it_up}')
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
        ax_heat3.set_title(rf'Matching 3D surface GNH3, iteration {it_up}')
        ax_heat3.scatter(f_best, tau_best, f_meshgrid_log10[idx], color='red', s=10, label='Mínimo')
        ax_heat3.legend()
        plt.show()
        # Muller search
        p1, p2, p3 = muller_seed_meshgrid(F, TAU, f_meshgrid_log10)
        w1 = (2 * pi * p1[0] - 1j / p1[1]) / c_cgs
        w2 = (2 * pi * p2[0] - 1j / p2[1]) / c_cgs
        w3 = (2 * pi * p3[0] - 1j / p3[1]) / c_cgs
        res = muller(f_match, (w1, w2, w3), xtol=1e-12, ftol=1e-12, maxiter=50)
        f_muller = res.root.real / (2 * pi) * c_cgs / 1e3
        tau_muller = -1 / (res.root.imag * c_cgs) * 1e6
        print(f"Müller GNH3 → f = {f_muller:.6f} kHz | τ = {tau_muller:.6f} µs | it = {res.iterations} | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        f_gnh3_up.append(f_muller)
        tau_gnh3_up.append(tau_muller)
    else :
        (w1, w2, w3), (f0, tau0), seeds = muller_seed_from_extrapolation(
            p_central_array_geo_up[:len(f_gnh3_up)],
            array(f_gnh3_up) * 1e3,  # kHz -> Hz si tus listas están en kHz
            array(tau_gnh3_up) * 1e-6,  # us -> s
            p_c,
            c_cgs,
            f_match=f_match,
            df_rel=1e-4,
            dtau_rel=1e-4,
            local_refine=True,
            n_local=5,
        )
        try:
            res = muller(f_match, (w1, w2, w3), xtol=1e-12, ftol=1e-12, maxiter=50)
        except ValueError:
            f_gnh3_up.append(nan)
            tau_gnh3_up.append(nan)
            continue
        f_muller = res.root.real / (2 * pi) * c_cgs / 1e3
        tau_muller = -1 / (res.root.imag * c_cgs) * 1e6
        print(f"Müller GNH3 → f = {f_muller:.6f} kHz | τ = {tau_muller:.6f} µs | it = {res.iterations} | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        f_gnh3_up.append(f_muller)
        tau_gnh3_up.append(tau_muller)

# Down search
it_down = 0
f_gnh3_down = []
tau_gnh3_down = []
for p_c in p_central_array_geo_down:
    it_down += 1
    # Solve TOV
    r_gnh3, p_gnh3, m_gnh3, nu_gnh3, _ = solve_tov_eos(p_c, rho_pchip_geo_gnh3)
    # Interpolation -> rho(r), p(r), m(r), nu(r)
    rho_gnh3 = array([rho_pchip_geo_gnh3(p) for p in p_gnh3])
    p_fun = pchip(r_gnh3, p_gnh3, extrapolate=False)
    m_fun = pchip(r_gnh3, m_gnh3, extrapolate=False)
    nu_fun = pchip(r_gnh3, nu_gnh3, extrapolate=False)
    rho_fun = pchip(r_gnh3, rho_gnh3, extrapolate=False)
    # TOV parameters
    R = r_gnh3[-1]
    M = m_gnh3[-1]
    r0 = r_gnh3[0]
    # Matching function
    f_match = lambda omega: matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha)
    # QNM bruteforce search -> meshgrid
    if it_down == 1:
        f_mesh = linspace(6.85e3, 7.10e3, 5)
        tau_mesh = linspace(28.35e-6, 28.5e-6, 5)
        F, TAU = meshgrid(f_mesh, tau_mesh)
        OMEGA = (2.0 * pi * F - 1j / TAU) / c_cgs
        # Evaluate abs(matching function) in meshgrid
        from joblib import Parallel, delayed
        omega_flat = OMEGA.flatten()
        def eval_match(omega):
            return abs(matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha))
        results = Parallel(n_jobs=3, verbose=1)(  # n_jobs=-1 usa todos los cores
            delayed(eval_match)(omega) for omega in omega_flat
        )
        f_meshgrid = array(results).reshape(OMEGA.shape)
        f_meshgrid_log10 = log10(f_meshgrid)
        # Minimum search
        idx = unravel_index(nanargmin(f_meshgrid_log10), f_meshgrid_log10.shape)
        f_best = F[idx] / 1e3
        tau_best = TAU[idx] * 1e6
        print(f"Minimum at first iteration GNH3 → f = {f_best:.3f} KHz | τ = {tau_best:.3f} µs | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        # ================================
        # Heatmap 2D
        # ================================
        fig_heat2, ax_heat2 = plt.subplots(figsize=(7.5, 4.5))
        pcm = ax_heat2.pcolormesh(F / 1e3, TAU * 1e6, f_meshgrid_log10, shading='gouraud', cmap='viridis')
        cbar = fig_heat2.colorbar(pcm, ax=ax_heat2, pad=0.02)
        cbar.set_label(r'$\log_{10}|f_{\mathrm{match}}|$', rotation=90)
        ax_heat2.set_xlabel('f [KHz]')
        ax_heat2.set_ylabel(r'$\tau$ [$\mu$s]')
        ax_heat2.set_title(rf'Heatmap matching GNH3, iteration {it_down}')
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
        ax_heat3.set_title(rf'Matching 3D surface GNH3, iteration {it_down}')
        ax_heat3.scatter(f_best, tau_best, f_meshgrid_log10[idx], color='red', s=10, label='Mínimo')
        ax_heat3.legend()
        plt.show()
        # Muller search
        p1, p2, p3 = muller_seed_meshgrid(F, TAU, f_meshgrid_log10)
        w1 = (2 * pi * p1[0] - 1j / p1[1]) / c_cgs
        w2 = (2 * pi * p2[0] - 1j / p2[1]) / c_cgs
        w3 = (2 * pi * p3[0] - 1j / p3[1]) / c_cgs
        res = muller(f_match, (w1, w2, w3), xtol=1e-12, ftol=1e-12, maxiter=50)
        f_muller = res.root.real / (2 * pi) * c_cgs / 1e3
        tau_muller = -1 / (res.root.imag * c_cgs) * 1e6
        print(f"Müller GNH3 → f = {f_muller:.6f} kHz | τ = {tau_muller:.6f} µs | it = {res.iterations} | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        f_gnh3_down.append(f_muller)
        tau_gnh3_down.append(tau_muller)
    elif it_down == 2:
        f_mesh = linspace(6.85e3, 7.10e3, 5)
        tau_mesh = linspace(28.35e-6, 28.5e-6, 5)
        F, TAU = meshgrid(f_mesh, tau_mesh)
        OMEGA = (2.0 * pi * F - 1j / TAU) / c_cgs
        # Evaluate abs(matching function) in meshgrid
        from joblib import Parallel, delayed
        omega_flat = OMEGA.flatten()
        def eval_match(omega):
            return abs(matching(r0, R, omega, m_fun, p_fun, rho_fun, nu_fun, M, alpha))
        results = Parallel(n_jobs=3, verbose=1)(  # n_jobs=-1 usa todos los cores
            delayed(eval_match)(omega) for omega in omega_flat
        )
        f_meshgrid = array(results).reshape(OMEGA.shape)
        f_meshgrid_log10 = log10(f_meshgrid)
        # Minimum search
        idx = unravel_index(nanargmin(f_meshgrid_log10), f_meshgrid_log10.shape)
        f_best = F[idx] / 1e3
        tau_best = TAU[idx] * 1e6
        print(f"Minimum at second iteration GNH3 → f = {f_best:.3f} KHz | τ = {tau_best:.3f} µs | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        # ================================
        # Heatmap 2D
        # ================================
        fig_heat2, ax_heat2 = plt.subplots(figsize=(7.5, 4.5))
        pcm = ax_heat2.pcolormesh(F / 1e3, TAU * 1e6, f_meshgrid_log10, shading='gouraud', cmap='viridis')
        cbar = fig_heat2.colorbar(pcm, ax=ax_heat2, pad=0.02)
        cbar.set_label(r'$\log_{10}|f_{\mathrm{match}}|$', rotation=90)
        ax_heat2.set_xlabel('f [KHz]')
        ax_heat2.set_ylabel(r'$\tau$ [$\mu$s]')
        ax_heat2.set_title(rf'Heatmap matching GNH3, iteration {it_down}')
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
        ax_heat3.set_title(rf'Matching 3D surface GNH3, iteration {it_down}')
        ax_heat3.scatter(f_best, tau_best, f_meshgrid_log10[idx], color='red', s=10, label='Mínimo')
        ax_heat3.legend()
        plt.show()
        # Muller search
        p1, p2, p3 = muller_seed_meshgrid(F, TAU, f_meshgrid_log10)
        w1 = (2 * pi * p1[0] - 1j / p1[1]) / c_cgs
        w2 = (2 * pi * p2[0] - 1j / p2[1]) / c_cgs
        w3 = (2 * pi * p3[0] - 1j / p3[1]) / c_cgs
        res = muller(f_match, (w1, w2, w3), xtol=1e-12, ftol=1e-12, maxiter=50)
        f_muller = res.root.real / (2 * pi) * c_cgs / 1e3
        tau_muller = -1 / (res.root.imag * c_cgs) * 1e6
        print(f"Müller GNH3 → f = {f_muller:.6f} kHz | τ = {tau_muller:.6f} µs | it = {res.iterations} | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        f_gnh3_down.append(f_muller)
        tau_gnh3_down.append(tau_muller)
    else :
        (w1, w2, w3), (f0, tau0), seeds = muller_seed_from_extrapolation(
            p_central_array_geo_down[:len(f_gnh3_down)][::-1],
            array(f_gnh3_down[::-1]) * 1e3,
            array(tau_gnh3_down[::-1]) * 1e-6,
            p_c,
            c_cgs,
            f_match=f_match,
            df_rel=1e-4,
            dtau_rel=1e-4,
            local_refine=True,
            n_local=5,
        )
        try:
            res = muller(f_match, (w1, w2, w3), xtol=1e-12, ftol=1e-12, maxiter=50)
        except ValueError:
            f_gnh3_down.append(nan)
            tau_gnh3_down.append(nan)
            continue
        f_muller = res.root.real / (2 * pi) * c_cgs / 1e3
        tau_muller = -1 / (res.root.imag * c_cgs) * 1e6
        print(f"Müller GNH3 → f = {f_muller:.6f} kHz | τ = {tau_muller:.6f} µs | it = {res.iterations} | log10(p_c [cgs]) = {log10(pressure_geo_to_cgs(p_c)):.6f}")
        f_gnh3_down.append(f_muller)
        tau_gnh3_down.append(tau_muller)


# =====================
# Ploting
# =====================
# Invert
p_down_sorted = p_central_array_geo_down[::-1]
f_down_sorted = array(f_gnh3_down)[::-1]
tau_down_sorted = array(tau_gnh3_down)[::-1]
# Remove duplicate anchor point
p_down_sorted = p_down_sorted[:-1]
f_down_sorted = f_down_sorted[:-1]
tau_down_sorted = tau_down_sorted[:-1]
# Concatenate
p_total_gnh3 = concatenate((p_down_sorted, p_central_array_geo_up))
f_total_gnh3 = concatenate((f_down_sorted, array(f_gnh3_up)))
tau_total_gnh3 = concatenate((tau_down_sorted, array(tau_gnh3_up)))
# Make xlabel
log_p_total_gnh3 = log10(pressure_geo_to_cgs(p_total_gnh3))
# Save results
savez("gnh3_results.npz",log_p=log_p_total_gnh3,f=f_total_gnh3,tau=tau_total_gnh3)

# ==========Plot 1: Frequency vs Central Pressure==========#
fig_f, ax_f = plt.subplots(figsize=(7.5, 4.5))
ax_f.plot(log_p_total_gnh3, array(f_total_gnh3), color='blue', linewidth=1.5, label="GNH3")
ax_f.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_f.set_ylabel('f [KHz]')
ax_f.set_title(r'Frequency')
ax_f.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_f.legend(loc="upper right")
# fig_f.savefig("f_gnh3.png", dpi=600, bbox_inches="tight")

# ==========Plot 2: Damping time vs Central Pressure==========#
fig_tau, ax_tau = plt.subplots(figsize=(7.5, 4.5))
ax_tau.plot(log_p_total_gnh3, array(tau_total_gnh3), color='blue', linewidth=1.5, label="GNH3")
ax_tau.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
ax_tau.set_ylabel(r'$\tau$ [$\mu$s]')
ax_tau.set_title(r'Damping time GNH3')
ax_tau.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_tau.legend(loc="upper left")
# fig_tot_radius.savefig("tau_gnh3.png", dpi=600, bbox_inches="tight")
plt.show()
# ==========Log time end==========#
end = time.perf_counter()
print(f"Elapsed time = {end - start:.3f} s")
