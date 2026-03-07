"""
Physical and Astronomical Constants — CGS and Geometrized Units
================================================================

This module defines fundamental physical and astronomical constants
in the Centimetre–Gram–Second (CGS) system, together with conversion
functions between CGS units and geometrized units (G = c = 1).

Geometrized units are commonly used in General Relativity, where:

    - Length, time and mass share the same dimension
    - G = 1 and c = 1
"""

# =====================
# Imports
# =====================
from scipy.constants import c as _c, G as _G_SI
from numpy import asarray

# Speed of light (CGS)
c = _c * 100  # m/s → cm/s

# Gravitational constant (CGS)
G = _G_SI * (100**3) / 1000  # SI → cm^3 g^-1 s^-2

M_sun = 1.98847e33  # g (IAU 2015 recommended value)
R_sun = 6.957e10    # cm

# -----------------------------
# Helper (vectorizable inputs)
# -----------------------------
def _arr(x):
    """
    :param x: Scalar or array-like input.
    :return: NumPy array with dtype=float.
    """
    return asarray(x, dtype=float)


# -----------------------------
# Time (CGS <-> geometric)
# -----------------------------
def time_cgs_to_geo(t_cgs):
    """
    :param t_cgs: Time in CGS units [s].
    :return:
        - t_geo: Time in geometrized units.
    """
    t_cgs = _arr(t_cgs)
    t_geo = c * t_cgs
    return t_geo


def time_geo_to_cgs(t_geo):
    """
    :param t_geo: Time in geometrized units.
    :return:
        - t_cgs: Time in CGS units [s].
    """
    t_geo = _arr(t_geo)
    t_cgs = t_geo / c
    return t_cgs


# -----------------------------
# Mass (CGS <-> geometric)
# -----------------------------
def mass_cgs_to_geo(m_cgs):
    """
    :param m_cgs: Mass in CGS units [g].
    :return:
        - m_geo: Mass in geometrized units (length).
    """
    m_cgs = _arr(m_cgs)
    m_geo = (G / c ** 2) * m_cgs
    return m_geo


def mass_geo_to_cgs(m_geo):
    """
    :param m_geo: Mass in geometrized units.
    :return:
        - m_cgs: Mass in CGS units [g].
    """
    m_geo = _arr(m_geo)
    m_cgs = (c ** 2 / G) * m_geo
    return m_cgs


# -----------------------------
# Pressure (CGS <-> geometric)
# -----------------------------
def pressure_cgs_to_geo(p_cgs):
    """
    :param p_cgs: Pressure in CGS units [dyn cm^-2].
    :return:
        - p_geo: Pressure in geometrized units.
    """
    p_cgs = _arr(p_cgs)
    p_geo = (G / c ** 4) * p_cgs
    return p_geo


def pressure_geo_to_cgs(p_geo):
    """
    :param p_geo: Pressure in geometrized units.
    :return:
        - p_cgs: Pressure in CGS units [dyn cm^-2].
    """
    p_geo = _arr(p_geo)
    p_cgs = (c ** 4 / G) * p_geo
    return p_cgs


# -----------------------------
# Energy density (CGS <-> geometric)
# -----------------------------
def energy_density_cgs_to_geo(rho_cgs):
    """
    :param rho_cgs: Energy density in CGS units [erg cm^-3].
    :return:
        - rho_geo: Energy density in geometrized units.
    """
    rho_cgs = _arr(rho_cgs)
    rho_geo = (G / c ** 4) * rho_cgs
    return rho_geo


def energy_density_geo_to_cgs(rho_geo):
    """
    :param rho_geo: Energy density in geometrized units.
    :return:
        - rho_cgs: Energy density in CGS units [erg cm^-3].
    """
    rho_geo = _arr(rho_geo)
    rho_cgs = (c ** 4 / G) * rho_geo
    return rho_cgs