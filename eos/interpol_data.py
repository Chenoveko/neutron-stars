# =====================
# Imports
# =====================

from numpy import log10, ndarray
from scipy.interpolate import CubicSpline

from eos.extract_data import extract_mass_density_from_eos_txt, extract_pressure_from_eos_txt
from utilities.physical_data import pressure_geo_to_cgs, energy_density_cgs_to_geo
"""
Cubic spline (CS) interpolation of equations of state (EoS)
-----------------------------------------------------------------------------------------
From:
    SciPy documentation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#scipy.interpolate.CubicSpline
"""
# AKMALPR
log10_p_akmalpr = log10(extract_pressure_from_eos_txt("../eos/eos_akmalpr.txt"))
log10_rho_akmalpr = log10(extract_mass_density_from_eos_txt("../eos/eos_akmalpr.txt"))

# GLENDNH3
log10_p_glendnh3 = log10(extract_pressure_from_eos_txt("../eos/eos_glendnh3.txt"))
log10_rho_glendnh3 = log10(extract_mass_density_from_eos_txt("../eos/eos_glendnh3.txt"))

# SLY4
log10_p_sly4 = log10(extract_pressure_from_eos_txt("../eos/eos_sly4.txt"))
log10_rho_sly4 = log10(extract_mass_density_from_eos_txt("../eos/eos_sly4.txt"))

# Interpolation using Scipy
cs_log10_akmalpr = CubicSpline(log10_p_akmalpr, log10_rho_akmalpr, bc_type='not-a-knot', extrapolate=False)
cs_log10_glendnh3 = CubicSpline(log10_p_glendnh3, log10_rho_glendnh3, bc_type='not-a-knot', extrapolate=False)
cs_log10_sly4 = CubicSpline(log10_p_sly4, log10_rho_sly4, bc_type='not-a-knot', extrapolate=True)

# Rho functions
def rho_cs_log10_akmalpr(log10_p: ndarray) -> ndarray:
    return cs_log10_akmalpr(log10_p)


def rho_cs_log10_glendnh3(log10_p: ndarray) -> ndarray:
    return cs_log10_glendnh3(log10_p)


def rho_cs_log10_sly4(log10_p: ndarray) -> ndarray:
    return cs_log10_sly4(log10_p)

# Rho functions in GEO units
def rho_cs_geo_akmalpr(p_geo):
    p_cgs = pressure_geo_to_cgs(p_geo)
    log10_rho_cgs = rho_cs_log10_akmalpr(log10(p_cgs))
    rho_geo = energy_density_cgs_to_geo(10.0**log10_rho_cgs)
    return rho_geo

def rho_cs_geo_glendnh3(p_geo):
    p_cgs = pressure_geo_to_cgs(p_geo)
    log10_rho_cgs = rho_cs_log10_glendnh3(log10(p_cgs))
    rho_geo = energy_density_cgs_to_geo(10.0**log10_rho_cgs)
    return rho_geo

def rho_cs_geo_sly4(p_geo):
    p_cgs = pressure_geo_to_cgs(p_geo)
    log10_rho_cgs = rho_cs_log10_sly4(log10(p_cgs))
    rho_geo = energy_density_cgs_to_geo(10.0**log10_rho_cgs)
    return rho_geo