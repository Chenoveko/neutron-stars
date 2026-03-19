# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import numpy as np

from numpy import logspace, log10, max

from equations_of_structure.interpol_data import (
    rho_pchip_geo_apr,
    rho_pchip_geo_gnh3,
    rho_pchip_geo_sly4,
)
from utilities.physical_data import M_sun, pressure_cgs_to_geo, mass_cgs_to_geo
from utilities.tov_solvers import solve_tov_eos


def integrate_one_pressure(args):
    """
    Resuelve las 3 EoS para una presión central dada y devuelve
    las masas y radios finales.
    """
    p_c, M_sun_geo = args

    # APR
    r_apr, _, m_apr, _ = solve_tov_eos(p_c, rho_pchip_geo_apr)
    r_apr_final = r_apr[-1] / 1e5
    m_apr_final = m_apr[-1] / M_sun_geo

    # GNH3
    r_gnh3, _, m_gnh3, _ = solve_tov_eos(p_c, rho_pchip_geo_gnh3)
    r_gnh3_final = r_gnh3[-1] / 1e5
    m_gnh3_final = m_gnh3[-1] / M_sun_geo

    # SLy4
    r_sly4, _, m_sly4, _ = solve_tov_eos(p_c, rho_pchip_geo_sly4)
    r_sly4_final = r_sly4[-1] / 1e5
    m_sly4_final = m_sly4[-1] / M_sun_geo

    return (
        m_apr_final, r_apr_final,
        m_gnh3_final, r_gnh3_final,
        m_sly4_final, r_sly4_final,
    )


def main():
    # ==========Log time start==========#
    start = time.perf_counter()

    # ==========Parameters==========#
    M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
    p_central_array_cgs = logspace(33.2, 36.5, 100)  # Array of central pressure in CGS
    p_central_array_geo = pressure_cgs_to_geo(p_central_array_cgs)  # Array of central pressure in GEO
    log10_p_central_array_cgs = log10(p_central_array_cgs)

    # ==========Parallel integration==========#
    tasks = [(p_c, M_sun_geo) for p_c in p_central_array_geo]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(integrate_one_pressure, tasks)

    results = np.array(results)

    # Columnas:
    # 0: m_apr, 1: r_apr, 2: m_gnh3, 3: r_gnh3, 4: m_sly4, 5: r_sly4
    tot_mass_apr = results[:, 0]
    tot_radius_apr = results[:, 1]
    tot_mass_gnh3 = results[:, 2]
    tot_radius_gnh3 = results[:, 3]
    tot_mass_sly4 = results[:, 4]
    tot_radius_sly4 = results[:, 5]

    # ==========Plot 1: Total Mass vs Central Pressure==========#
    fig_tot_mass, ax_tot_mass = plt.subplots(figsize=(7.5, 4.5))
    ax_tot_mass.plot(log10_p_central_array_cgs, tot_mass_apr, color='goldenrod', linewidth=1.5, label="APR")
    ax_tot_mass.plot(log10_p_central_array_cgs, tot_mass_gnh3, color='blue', linewidth=1.5, label="GNH3")
    ax_tot_mass.plot(log10_p_central_array_cgs, tot_mass_sly4, color='red', linewidth=1.5, label="SLy4")
    ax_tot_mass.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
    ax_tot_mass.set_ylabel(r'$M / M_{\odot}$')
    ax_tot_mass.set_title(r'Total Mass')
    ax_tot_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
    ax_tot_mass.legend(loc="upper left")

    # ==========Plot 2: Total Radius vs Central Pressure==========#
    fig_tot_radius, ax_tot_radius = plt.subplots(figsize=(7.5, 4.5))
    ax_tot_radius.plot(log10_p_central_array_cgs, tot_radius_apr, color='goldenrod', linewidth=1.5, label="APR")
    ax_tot_radius.plot(log10_p_central_array_cgs, tot_radius_gnh3, color='blue', linewidth=1.5, label="GNH3")
    ax_tot_radius.plot(log10_p_central_array_cgs, tot_radius_sly4, color='red', linewidth=1.5, label="SLy4")
    ax_tot_radius.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
    ax_tot_radius.set_ylabel(r'$R\ [km]$')
    ax_tot_radius.set_title(r'Total Radius')
    ax_tot_radius.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
    ax_tot_radius.legend(loc="upper right")

    # ==========Plot 3: Total Mass vs Total Radius==========#
    fig_tot_mass_radius, ax_tot_mass_radius = plt.subplots(figsize=(7.5, 4.5))
    ax_tot_mass_radius.plot(tot_radius_apr, tot_mass_apr, color='goldenrod', linewidth=1.5, label="APR")
    ax_tot_mass_radius.plot(tot_radius_gnh3, tot_mass_gnh3, color='blue', linewidth=1.5, label="GNH3")
    ax_tot_mass_radius.plot(tot_radius_sly4, tot_mass_sly4, color='red', linewidth=1.5, label="SLy4")
    ax_tot_mass_radius.set_xlabel(r'$R\ [km]$')
    ax_tot_mass_radius.set_ylabel(r'$M / M_{\odot}$')
    ax_tot_mass_radius.set_title(r'Total Mass vs Total Radius')
    ax_tot_mass_radius.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
    ax_tot_mass_radius.legend(loc="upper right")

    plt.show()

    # Print Max Mass values
    print(f'Max mass for APR: {max(tot_mass_apr)}')
    print(f'Max mass for GNH3: {max(tot_mass_gnh3)}')
    print(f'Max mass for SLY4: {max(tot_mass_sly4)}')

    # ==========Log time end==========#
    end = time.perf_counter()
    print("Elapsed time = {}s".format(end - start))


if __name__ == "__main__":
    mp.freeze_support()
    main()