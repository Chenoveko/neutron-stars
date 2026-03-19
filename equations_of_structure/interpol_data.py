"""
Equation of State Interpolation — PCHIP  ρ(p)  for Tabulated EoS
=======================================================================

This module builds smooth interpolants for several tabulated neutron-star
equations of state (EoS) using a Piecewise Cubic Hermite Interpolating
Polynomial (PCHIP). The interpolants provide the mass density as a
function of pressure and are intended for use in stellar structure
integrations (e.g., TOV solvers).

Three EoS tables are supported:
    - APR
    - GNH3
    - SLy4

Input Data Format
-----------------
Each EoS text file is assumed to be a whitespace-separated table with
(at least) the following columns:

    Column 2 : Mass density  ρ_mass  [g cm^-3]
    Column 3 : Pressure      p       [dyn cm^-2]

The helper functions
    - extract_mass_density_from_eos_txt
    - extract_pressure_from_eos_txt
are used to read the relevant columns while ignoring commented lines.

Interpolation Strategy
----------------------
Interpolation is performed in logarithmic variables:

    x = log10(p_cgs)
    y = log10(ρ_mass_cgs)

A PCHIP interpolator y(x) is constructed for each EoS. PCHIP is used
because it is shape-preserving and avoids oscillations typical of high-order
splines, making it well-suited for monotonic EoS tables.

The resulting interpolants are exposed in two forms:

1) Logarithmic interpolants (CGS):
    rho_pchip_log10_* (log10_p)  ->  log10(ρ_mass_cgs)

2) Physical density functions for TOV integration (geometrized units):
    rho_pchip_geo_* (p_geo)  ->  ε_geo

where p_geo is converted to CGS pressure, the interpolant is evaluated in
log-space, and the mass density is converted to energy density via:

    ε_cgs = ρ_mass_cgs * c^2

Finally, ε_cgs is converted to geometrized units for direct use in the
TOV equations.

Units
-----
    - Tabulated pressures are in CGS: p [dyn cm^-2]
    - Tabulated densities are in CGS: ρ_mass [g cm^-3]
    - Energy density is formed as ε = ρ_mass c^2 [erg cm^-3]
    - Returned densities for TOV are in geometrized CGS units (G = c = 1)

Dependencies
------------
    - scipy.interpolate.PchipInterpolator      : monotone cubic interpolator
    - equations_of_structure.extract_data      : EoS column extraction utilities
    - utilities.physical_data                  : unit conversion functions/constants
        * pressure_geo_to_cgs
        * energy_density_cgs_to_geo
        * c
    - numpy                                    : log10, isfinite, nan

Notes
-----
    - Interpolators are constructed with extrapolate=True. Extrapolation
      outside the tabulated pressure range may be unphysical; domain
      checks are applied in the GEO wrappers to ensure log10 is only
      evaluated for positive, finite pressures.
    - The GEO wrapper functions return NaN if the input pressure is not
      finite or not strictly positive.
"""
# =====================
# Imports
# =====================
import time
from numpy import log10, ndarray, nan, isfinite
from scipy.interpolate import PchipInterpolator
from pathlib import Path
from equations_of_structure.extract_data import extract_mass_density_from_eos_txt, extract_pressure_from_eos_txt
from utilities.physical_data import pressure_geo_to_cgs, energy_density_cgs_to_geo
from utilities.physical_data import c
# ==========Log time start==========#
start = time.perf_counter()
BASE_DIR = Path(__file__).resolve().parent

# ==========Extract Values==========#
log10_p_apr = log10(extract_pressure_from_eos_txt(BASE_DIR / "eos_akmalpr.txt"))
log10_rho_apr = log10(extract_mass_density_from_eos_txt(BASE_DIR / "eos_akmalpr.txt"))
log10_p_gnh3 = log10(extract_pressure_from_eos_txt(BASE_DIR / "eos_glendnh3.txt"))
log10_rho_gnh3 = log10(extract_mass_density_from_eos_txt(BASE_DIR / "eos_glendnh3.txt"))
log10_p_sly4 = log10(extract_pressure_from_eos_txt(BASE_DIR / "eos_sly4.txt"))
log10_rho_sly4 = log10(extract_mass_density_from_eos_txt(BASE_DIR / "eos_sly4.txt"))

# ==========Interpolation==========#
pchip_log10_apr = PchipInterpolator(log10_p_apr, log10_rho_apr, extrapolate=True)
pchip_log10_gnh3 = PchipInterpolator(log10_p_gnh3, log10_rho_gnh3, extrapolate=True)
pchip_log10_sly4 = PchipInterpolator(log10_p_sly4, log10_rho_sly4, extrapolate=True)

# =====================
# Rho functions in CGS units (log-space)
# =====================

def rho_pchip_log10_apr(log10_p: ndarray) -> ndarray:
    """
    :param log10_p: One-dimensional NumPy array of log10(p_cgs) values.
    :return:
        - log10_rho_mass_cgs: Interpolated log10(ρ_mass_cgs) [g cm^-3]
    """
    return pchip_log10_apr(log10_p)


def rho_pchip_log10_gnh3(log10_p: ndarray) -> ndarray:
    """
    :param log10_p: One-dimensional NumPy array of log10(p_cgs) values.
    :return:
        - log10_rho_mass_cgs: Interpolated log10(ρ_mass_cgs) [g cm^-3]
    """
    return pchip_log10_gnh3(log10_p)


def rho_pchip_log10_sly4(log10_p: ndarray) -> ndarray:
    """
    :param log10_p: One-dimensional NumPy array of log10(p_cgs) values.
    :return:
        - log10_rho_mass_cgs: Interpolated log10(ρ_mass_cgs) [g cm^-3]
    """
    return pchip_log10_sly4(log10_p)


# =====================
# Rho functions in GEO CGS units (for TOV integration)
# =====================

def rho_pchip_geo_apr(p_geo):
    """
    :param p_geo: Pressure in geometrized CGS units.
    :return:
        - eps_geo: Energy density in geometrized units (ε_geo).
          Returns NaN if p_geo is not finite or not strictly positive.
    """
    p_cgs = pressure_geo_to_cgs(p_geo)

    # Domain guard: log10 requires p_cgs > 0 and finite
    if (not isfinite(p_cgs)) or (p_cgs <= 0):
        return nan

    log10_p = log10(p_cgs)
    log10_rho_mass_cgs = rho_pchip_log10_apr(log10_p)

    rho_mass_cgs = 10.0 ** log10_rho_mass_cgs
    eps_cgs = rho_mass_cgs * c ** 2

    return energy_density_cgs_to_geo(eps_cgs)


def rho_pchip_geo_gnh3(p_geo):
    """
    :param p_geo: Pressure in geometrized CGS units.
    :return:
        - eps_geo: Energy density in geometrized units (ε_geo).
          Returns NaN if p_geo is not finite or not strictly positive.
    """
    p_cgs = pressure_geo_to_cgs(p_geo)

    # Domain guard: log10 requires p_cgs > 0 and finite
    if (not isfinite(p_cgs)) or (p_cgs <= 0):
        return nan

    log10_p = log10(p_cgs)
    log10_rho_mass_cgs = rho_pchip_log10_gnh3(log10_p)

    rho_mass_cgs = 10.0 ** log10_rho_mass_cgs
    eps_cgs = rho_mass_cgs * c ** 2

    return energy_density_cgs_to_geo(eps_cgs)


def rho_pchip_geo_sly4(p_geo):
    """
    :param p_geo: Pressure in geometrized CGS units.
    :return:
        - eps_geo: Energy density in geometrized units (ε_geo).
          Returns NaN if p_geo is not finite or not strictly positive.
    """
    p_cgs = pressure_geo_to_cgs(p_geo)

    # Domain guard: log10 requires p_cgs > 0 and finite
    if (not isfinite(p_cgs)) or (p_cgs <= 0):
        return nan

    log10_p = log10(p_cgs)
    log10_rho_mass_cgs = rho_pchip_log10_sly4(log10_p)

    rho_mass_cgs = 10.0 ** log10_rho_mass_cgs
    eps_cgs = rho_mass_cgs * c ** 2

    return energy_density_cgs_to_geo(eps_cgs)

# ==========Log time end==========#
end = time.perf_counter()
print("Elapsed time = {}s".format(end - start))
