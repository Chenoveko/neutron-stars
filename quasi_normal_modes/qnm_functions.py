from numpy import pi, exp, sin,cos
from typing import Callable



#######################################
# Axial perturbations inside the star #
#######################################
def axial_potential_in(r: float, m: Callable, p: Callable, rho: Callable, nu: Callable, l: int = 2) -> float:
    """
    Axial potential inside the star in GEO units
        V(r) = e^{2ν(r)} / r^3 * [ l(l+1) r + 4π r^3 (ρ(r) - p(r)) - 6 m(r) ]
    :param r: radial coordinate
    :param m: mass value m(r)
    :param p: pressure value p(r)
    :param rho: energy density value rho(r)
    :param nu: metric value nu(r)
    :param l: angular momentum number
    :return: axial potential
    """
    return exp(2.0 * nu(r)) / r**3 * (l * (l + 1) * r + 4.0 * pi * r**3 * (rho(r) - p(r)) - 6.0 * m(r))


def regge_wheeler_in(r: float, var, omega: complex, m: Callable, p: Callable, rho: Callable, nu: Callable,
                     l: int = 2) -> tuple[complex, complex]:
    """
    Regge-Wheeler linear ODE inside the star in GEO units,
    Equation:
        d²z/dr_*² + [omega² - V(r)] z = 0
    Using:
        dr*/dr = 1 / f(r)
        f(r) = 1 - 2m(r)/r
        f'(r) = 2m(r)/r^2 - 2m'(r)/r
        m'(r) = 4*pi*r^2*rho(r)
    This gives:
        d²z/dr² = -[f df/dr dz/dr + (omega² - V(r)) z] / f²

    :param r: radial coordinate
    :param var: state variables (z, dz/dr)
    :param omega: angular frequency value
    :param m: mass value
    :param p: pressure value
    :param rho: energy density value
    :param nu: metric value
    :param l: angular momentum number
    :return: derivatives (dz/dr, d²z/dr²)
    """
    z, dzdr = var
    f = 1.0 - 2.0 * m(r) / r
    m_prime = 4.0 * pi * r ** 2 * rho(r)
    f_prime = -2.0 * m_prime / r + 2.0 * m(r) / r ** 2
    v = axial_potential_in(r, m, p, rho, nu, l)
    d2zdr2 = (-(f * f_prime) * dzdr - (omega ** 2 - v) * z) / f ** 2
    return dzdr, d2zdr2


########################################
# Axial perturbations outside the star #
########################################

def axial_potential_out(r: complex, M: float, l: int = 2) -> complex:
    """
    Axial potential outside the star in GEO units
        V(r) = (1 - 2M/r) * [l(l+1)/r^2 - 6M/r^3]
    :param r: radial coordinate
    :param M: total mass value
    :param l: angular momentum number
    :return: axial potential
    """
    return (1.0 - 2.0 * M / r) * (l * (l + 1) / r ** 2 - 6.0 * M / r ** 3)


def compact_rot_coord(t: float, R: float, alpha: float) -> complex:
    """
    Compactified rotated coordinate outside the star
    for Complex Exterior Scaling in GEO units
        r(t) = R + (1-t)/t exp(i*alpha)
    :param t: compactified coordinate t ∈ (0, 1]
    :param omega: angular frequency value
    :param R: total radius
    :param alpha: rotate angle
    :return: complex radial coordinate r(t)
    """
    return R + ((1.0 - t) / t) * exp(1j * alpha)


def regge_wheeler_out(t: float, g:complex, omega: complex, alpha: float, M: float, R: float, l: int = 2) -> complex:
    """
    Regge-Wheeler linear ODE outside the star in GEO units
    Equation:
        d²z/dr_*² + [omega² - V(r)] z = 0
    Using:
        g = z'/z
    This gives a Riccati equation:
        dg/dr_* + g² + omega² - V(r) = 0
    Using:
        dr*/dr = 1 / f(r)
        f(r) = 1 - 2M/r
    This gives:
        f(r)dg/dr + g² + omega² - V(r) = 0
    Using CES:
        r(t) = R + (1-t)/t exp(i*alpha)
        dr/dt = - exp(i*alpha)/t^2
    return:
        dg/dt = dg/dr * dr/dt

    :param t: compactified coordinate t ∈ (0, 1]
    :param g: state variables
    :param omega: angular frequency value
    :param alpha: complex rotation angle
    :param M: total mass
    :param R: total radius
    :param l: angular momentum number
    :return: derivative dg/dt
    """
    if alpha_condition(omega, alpha):
        r = compact_rot_coord(t, R, alpha)
        f = 1.0 - 2.0 * M / r
        v = axial_potential_out(r, M, l)
        drdt = -exp(1j * alpha) / t ** 2
        return drdt * (v - g ** 2 - omega ** 2) / f
    else:
        raise ValueError("Alpha condition for CES is not met")


def alpha_condition(omega: complex, alpha: float ) -> bool:
    """
    Alpha condition for CES
        Im(omega) * cos(alpha) - Re(omega) * sin(alpha) < 0
    :param omega: angular frequency value
    :param alpha: complex rotation angle
    :return: True if alpha condition is met, False otherwise
    """
    if omega.imag * cos(alpha) - omega.real * sin(alpha) < 0:
        return True
    else:
        return False
