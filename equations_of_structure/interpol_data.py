# =====================
# Imports
# =====================

from numpy import log10, ndarray, nan, isfinite
from scipy.interpolate import PchipInterpolator
from pathlib import Path
from equations_of_structure.extract_data import extract_mass_density_from_eos_txt, extract_pressure_from_eos_txt
from utilities.physical_data import pressure_geo_to_cgs, energy_density_cgs_to_geo
from utilities.physical_data import c

BASE_DIR = Path(__file__).resolve().parent

"""
Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) for equations of state (EoS)
-----------------------------------------------------------------------------------------
From:
    SciPy documentation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
"""
# AKMALPR
log10_p_apr = log10(extract_pressure_from_eos_txt(BASE_DIR / "eos_akmalpr.txt"))
log10_rho_apr = log10(extract_mass_density_from_eos_txt(BASE_DIR / "eos_akmalpr.txt"))

# GLENDNH3
log10_p_gnh3 = log10(extract_pressure_from_eos_txt(BASE_DIR / "eos_glendnh3.txt"))
log10_rho_gnh3 = log10(extract_mass_density_from_eos_txt(BASE_DIR / "eos_glendnh3.txt"))

# SLY4
log10_p_sly4 = log10(extract_pressure_from_eos_txt(BASE_DIR / "eos_sly4.txt"))
log10_rho_sly4 = log10(extract_mass_density_from_eos_txt(BASE_DIR / "eos_sly4.txt"))

# Interpolation using Scipy PCHIP
pchip_log10_apr = PchipInterpolator(log10_p_apr, log10_rho_apr, extrapolate=True)
pchip_log10_gnh3 = PchipInterpolator(log10_p_gnh3, log10_rho_gnh3,extrapolate=True)
pchip_log10_sly4 = PchipInterpolator(log10_p_sly4, log10_rho_sly4, extrapolate=True)

# Rho functions in CGS units
def rho_pchip_log10_apr(log10_p: ndarray) -> ndarray:
    return pchip_log10_apr(log10_p)


def rho_pchip_log10_gnh3(log10_p: ndarray) -> ndarray:
    return pchip_log10_gnh3(log10_p)


def rho_pchip_log10_sly4(log10_p: ndarray) -> ndarray:
    return pchip_log10_sly4(log10_p)

# Rho functions in GEO CGS units
def rho_pchip_geo_apr(p_geo):
    p_cgs = pressure_geo_to_cgs(p_geo)
    # Domain guard: log10 requires p_cgs > 0 and finite
    if (not isfinite(p_cgs)) or (p_cgs <= 0):
        return nan
    log10_p = log10(p_cgs)
    log10_rho_mass_cgs = rho_pchip_log10_apr(log10_p)  # may return nan if out of interpolation range
    rho_mass_cgs = 10.0 ** log10_rho_mass_cgs
    eps_cgs = rho_mass_cgs * c ** 2
    return energy_density_cgs_to_geo(eps_cgs)

def rho_pchip_geo_gnh3(p_geo):
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
    p_cgs = pressure_geo_to_cgs(p_geo)
    # Domain guard: log10 requires p_cgs > 0 and finite
    if (not isfinite(p_cgs)) or (p_cgs <= 0):
        return nan
    log10_p = log10(p_cgs)
    log10_rho_mass_cgs = rho_pchip_log10_sly4(log10_p)
    rho_mass_cgs = 10.0 ** log10_rho_mass_cgs
    eps_cgs = rho_mass_cgs * c ** 2
    return energy_density_cgs_to_geo(eps_cgs)