# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

from equations_of_structure.interpol_data import (
    rho_pchip_geo_apr, rho_pchip_geo_gnh3, rho_pchip_geo_sly4,
    rho_pchip_log10_apr, rho_pchip_log10_gnh3, rho_pchip_log10_sly4
)
from utilities.tov_solvers import solve_tov_eos
from utilities.physical_data import (
    M_sun, pressure_cgs_to_geo, pressure_geo_to_cgs, mass_cgs_to_geo
)
from numpy import logspace, log10


def find_central_pressure(args):
    """
    Busca la presión central que produce una estrella con masa objetivo
    dentro de la tolerancia dada.
    """
    eos_name, p_central_array_geo, rho_func, target_mass, mass_threshold, M_sun_geo = args

    for p_c in p_central_array_geo:
        _, _, m, _ = solve_tov_eos(p_c, rho_func)
        m = m / M_sun_geo
        tot_mass = m[-1]

        if target_mass - mass_threshold <= tot_mass <= target_mass + mass_threshold:
            return eos_name, p_c

    raise ValueError(
        f"No se encontró presión central para {eos_name} "
        f"en el rango dado con target_mass={target_mass} y threshold={mass_threshold}"
    )


def main():
    # ==========Log time start==========#
    start = time.perf_counter()

    # ==========Parameters==========#
    M_sun_geo = mass_cgs_to_geo(M_sun)

    p_central_array_apr_cgs = logspace(35.15, 35.2, 200)
    p_central_array_gnh3_cgs = logspace(34.80, 34.85, 200)
    p_central_array_sly4_cgs = logspace(35.1, 35.15, 200)

    p_central_array_apr_geo = pressure_cgs_to_geo(p_central_array_apr_cgs)
    p_central_array_gnh3_geo = pressure_cgs_to_geo(p_central_array_gnh3_cgs)
    p_central_array_sly4_geo = pressure_cgs_to_geo(p_central_array_sly4_cgs)

    target_mass = 1.4
    mass_threshold = 0.01

    # ==========Parallel search of central pressure==========#
    tasks = [
        ("APR",  p_central_array_apr_geo,  rho_pchip_geo_apr,  target_mass, mass_threshold, M_sun_geo),
        ("GNH3", p_central_array_gnh3_geo, rho_pchip_geo_gnh3, target_mass, mass_threshold, M_sun_geo),
        ("SLy4", p_central_array_sly4_geo, rho_pchip_geo_sly4, target_mass, mass_threshold, M_sun_geo),
    ]

    with mp.Pool(processes=3) as pool:
        results = pool.map(find_central_pressure, tasks)

    central_pressures = dict(results)

    p_central_apr_geo = central_pressures["APR"]
    p_central_gnh3_geo = central_pressures["GNH3"]
    p_central_sly4_geo = central_pressures["SLy4"]

    print("central pressure in geo units for APR: ", p_central_apr_geo)
    print("central pressure in geo units for GNH3:", p_central_gnh3_geo)
    print("central pressure in geo units for SLy4:", p_central_sly4_geo)

    # ==========Integration with fixed central pressure==========#
    # APR
    r_apr, p_apr, m_apr, status_apr = solve_tov_eos(p_central_apr_geo, rho_pchip_geo_apr)
    r_apr = r_apr / 1e5
    p_apr = pressure_geo_to_cgs(p_apr)
    m_apr = m_apr / M_sun_geo

    # GNH3
    r_gnh3, p_gnh3, m_gnh3, status_gnh3 = solve_tov_eos(p_central_gnh3_geo, rho_pchip_geo_gnh3)
    r_gnh3 = r_gnh3 / 1e5
    p_gnh3 = pressure_geo_to_cgs(p_gnh3)
    m_gnh3 = m_gnh3 / M_sun_geo

    # SLy4
    r_sly4, p_sly4, m_sly4, status_sly4 = solve_tov_eos(p_central_sly4_geo, rho_pchip_geo_sly4)
    r_sly4 = r_sly4 / 1e5
    p_sly4 = pressure_geo_to_cgs(p_sly4)
    m_sly4 = m_sly4 / M_sun_geo

    # ==========Plot 1: Mass enclosed==========#
    fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
    ax_mass.plot(r_apr, m_apr, color='goldenrod', linewidth=1.5, label="APR")
    ax_mass.plot(r_gnh3, m_gnh3, color='blue', linewidth=1.5, label="GNH3")
    ax_mass.plot(r_sly4, m_sly4, color='red', linewidth=1.5, label="SLy4")
    ax_mass.set_xlabel(r'$r\ [km]$')
    ax_mass.set_ylabel(r'$m(r) / M_{\odot}$')
    ax_mass.set_title(rf'Enclosed mass for ${target_mass}\,M_{{\odot}}$')
    ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
    ax_mass.legend(loc="upper left")

    # ==========Plot 2: Density profile==========#
    fig_density, ax_density = plt.subplots(figsize=(7.5, 4.5))
    ax_density.plot(r_apr, rho_pchip_log10_apr(log10(p_apr)), color='goldenrod', linewidth=1.5, label="APR")
    ax_density.plot(r_gnh3, rho_pchip_log10_gnh3(log10(p_gnh3)), color='blue', linewidth=1.5, label="GNH3")
    ax_density.plot(r_sly4, rho_pchip_log10_sly4(log10(p_sly4)), color='red', linewidth=1.5, label="SLy4")
    ax_density.set_xlabel(r'$r\ [km]$')
    ax_density.set_ylabel(r'$\log_{10}(\rho)\;[\mathrm{g\,cm^{-3}}]$')
    ax_density.set_title(rf'Density profile for ${target_mass}\,M_{{\odot}}$')
    ax_density.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
    ax_density.legend(loc="lower left")

    plt.show()

    # ==========Log time end==========#
    end = time.perf_counter()
    print("Elapsed time = {}s".format(end - start))


if __name__ == "__main__":
    mp.freeze_support()  # útil en Windows
    main()