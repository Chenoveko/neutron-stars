# =====================
# Imports
# =====================
from numpy import pi, sqrt, array, ndarray, linspace

"""
Tolman–Oppenheimer–Volkoff (TOV) Equations.
-----------------------------------------
Equations of structure for spherically symmetric relativistic stars.

From:
    Gravity: An Introduction to Einstein's General Relativity
    James B. Hartle

Units:
    Geometrized units in CGS
"""


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


"""
Tolman–Oppenheimer–Volkoff (TOV) Equations, Schwarzschild interior solution.
-----------------------------------------
Interior solution for spherically symmetric relativistic stars with uniform-density.

From:
    A first course in general relativity
    Bernard Schutz

Units:
    Geometrized units in CGS
"""


def schwarzschild_solution(r: ndarray, M: float, R: float) -> ndarray:
    """
    :param r: radial coordinate
    :param M: total mass
    :param R: total radius
    :return: pressure and enclosed mass
    """
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
