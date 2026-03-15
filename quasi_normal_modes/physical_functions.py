# =====================
# Imports
# =====================
from numpy import pi, array, ndarray


def tov_equations(var: ndarray, r: float, rho: float) -> ndarray:
    """
    TOV equations in GEO units
    :param var: state variables
    :param r: radial coordinate
    :param rho: energy density
    :return: derivatives
    """
    # Unpack variables
    p, m, nu = var
    # Interior pressure equation
    dp_dr = - (rho + p) * (m + 4 * pi * r ** 3 * p) / (r * (r - 2 * m))
    # Enclosed mass equation
    dm_dr = 4.0 * pi * r ** 2 * rho
    # Metric function
    dnu_dr = (m + 4 * pi * r ** 3 * p) / (r * (r - 2 * m))
    return array([dp_dr, dm_dr, dnu_dr], float)
