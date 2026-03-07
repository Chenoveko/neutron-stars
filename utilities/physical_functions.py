"""
Relativistic Stellar Structure — TOV and Schwarzschild Interior Solution
----------------------------------------------------------------------------

This module defines:

    - tov_equations: Differential form of the Tolman–Oppenheimer–Volkoff
      (TOV) equations in geometrized units.

    - schwarzschild_solution: Analytical Schwarzschild interior solution
      for a uniform-density relativistic star.

All equations are expressed in geometrized units (G = c = 1).
"""
# =====================
# Imports
# =====================
from numpy import pi, sqrt, array, ndarray


def tov_equations(var: ndarray, r: float, rho: float) -> ndarray:
    """
    TOV equations in GEO units
    :param var: state variables
    :param r: radial coordinate
    :param rho: energy density
    :return: derivatives
    """
    # Unpack variables
    p, m = var
    # Interior pressure equation
    dp_dr = - (rho + p) * (m + 4 * pi * r ** 3 * p) / (r * (r - 2 * m))
    # Enclosed mass equation
    dm_dr = 4.0 * pi * r ** 2 * rho
    return array([dp_dr, dm_dr], float)


def schwarzschild_solution(r: ndarray, M: float, R: float) -> ndarray:
    """
    Analytical Schwarzschild interior solution
    :param r: radial coordinate
    :param M: total mass
    :param R: total radius
    :return: pressure and enclosed mass
    """
    # Buchdahl bound for physical regularity
    if M / R >= 4 / 9:
        raise ValueError("Schwarzschild interior solution requires M/R < 4/9")
    # Uniform energy density
    rho = 3 * M / (4 * pi * R ** 3)
    # Interior pressure
    p = rho * (
            (sqrt(1 - 2 * M * r ** 2 / R ** 3) - sqrt(1 - 2 * M / R)) /
            (3 * sqrt(1 - 2 * M / R) - sqrt(1 - 2 * M * r ** 2 / R ** 3))
    )
    # Enclosed mass
    m = (4.0 / 3.0) * pi * rho * r ** 3
    return array([p, m], float)