# =====================
# Imports
# =====================
from numpy import pi, array, ndarray


def tov_equations(var: ndarray, r: float, rho: float) -> tuple[float, float,float]:
    """
    TOV equations in GEO units
    :param var: state variables
    :param r: radial coordinate
    :param rho: energy density
    :return: derivatives
    """
    # Unpack variables
    p, m ,nu= var[0], var[1],var[2]
    # Enclosed mass equation
    dm_dr = 4 * pi * r ** 2 * rho
    # Metric function
    dnu_dr = 2 * (m + 4 * pi * r**3 * p) / (r * (r - 2 * m))
    # Interior pressure equation
    dp_dr = -(rho + p) * dnu_dr / 2
    return dp_dr, dm_dr, dnu_dr
