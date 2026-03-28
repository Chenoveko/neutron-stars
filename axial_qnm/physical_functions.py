# =====================
# Imports
# =====================
from numpy import pi, ndarray


def tov_equations(var: ndarray, r: float, rho: float) -> tuple[float, float,float]:
    """
    Geometry inside the star in GEO units:
        ds^2 = -e^{2nu(r)} dt^2 + e^{lambda(r)} dr^2 + r^2 (dtheta^2 + sin^2(theta) * dphi^2)
    This gives TOV equations
        dm/dr = 4*pi*r^2*rho(r)
        dp/dr = - [rho(r) + p(r)] * [m(r) + 4*pi*r^3 p(r)] / [r^2 (1 - 2m(r)/r)]
        dnu/dr = [m(r) + 4*pi*r^3 p(r)] / [r^2 (1 - 2m(r)/r)]
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
    dnu_dr = (m + 4 * pi * r**3 * p) / (r * (r - 2 * m))
    # Interior pressure equation
    dp_dr = -(rho + p) * dnu_dr
    return dp_dr, dm_dr, dnu_dr
