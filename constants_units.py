# =====================
# Imports
# =====================
from numpy import asarray

"""
Physical and Astronomical Constants
----------------------------------
From:
    Gravity: An Introduction to Einstein's General Relativity
    James B. Hartle

Units:
    Centimetre–Gram–Second system (CGS)
"""
c = 3e10 # Speed of light in vacuum (cm/s)
G = 6.67e-8 # Gravitational constant (dyn cm^2/g^2)
# Masses (g)
M_sun   = 1.99e33
M_earth = 5.97e27
M_moon  = M_earth / 81.3
# Radius (cm)
R_sun = 6.96e10
R_earth = 6.38e8
R_moon = 1.74e8

"""
Conversion functions (CGS <-> geometric)
----------------------------------
From:
    Gravity: An Introduction to Einstein's General Relativity
    James B. Hartle
"""
# -----------------------------
# Helper (vectorizable inputs)
# -----------------------------
def _arr(x):
    return asarray(x, dtype=float)

# -----------------------------
# Time (CGS <-> geometric)
# -----------------------------
def time_cgs_to_geo(t_cgs):
    t_cgs = _arr(t_cgs)
    t_geo = c * t_cgs
    return t_geo

def time_geo_to_cgs(t_geo):
    t_geo = _arr(t_geo)
    t_cgs = t_geo / c
    return t_cgs

# -----------------------------
# Mass (CGS <-> geometric)
# -----------------------------
def mass_cgs_to_geo(m_cgs):
    m_cgs = _arr(m_cgs)
    m_geo = (G / c**2) * m_cgs
    return m_geo

def mass_geo_to_cgs(m_geo):
    m_geo = _arr(m_geo)
    m_cgs = (c**2 / G) * m_geo
    return m_cgs

# -----------------------------
#  Mass Density  (CGS <-> geometric)
# -----------------------------
def mass_density_cgs_to_geo(rho_cgs):
    rho_cgs = _arr(rho_cgs)
    rho_geo = (G / c**2) * rho_cgs
    return rho_geo

def mass_density_geo_to_cgs(rho_geo):
    rho_geo = _arr(rho_geo)
    rho_cgs = (c**2 / G) * rho_geo
    return rho_cgs

# -----------------------------
# Pressure  (CGS <-> geometric)
# -----------------------------
def pressure_cgs_to_geo(p_cgs):
    p_cgs = _arr(p_cgs)
    p_geo = (G / c**4) * p_cgs
    return p_geo

def pressure_geo_to_cgs(p_geo):
    p_geo = _arr(p_geo)
    p_cgs = (c**4 / G) * p_geo
    return p_cgs

# -----------------------------
# Energy density  (CGS <-> geometric)
# -----------------------------
def energy_density_cgs_to_geo(erg_cgs):
    erg_cgs = _arr(erg_cgs)
    erg_geo = (G / c**4) * erg_cgs
    return erg_geo

def energy_density_geo_to_cgs(erg_geo):
    erg_geo = _arr(erg_geo)
    erg_cgs = (c**4 / G) * erg_geo
    return erg_cgs