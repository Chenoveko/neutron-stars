from numpy import pi, exp, log, sqrt
from scipy.integrate import quad
from typing import Callable


def axial_potential_in(r: float, m: float, p: float, rho: float, nu: float, l: int = 2) -> float:
    """
    Axial potential inside the star in GEO units
    :param r: radial coordinate
    :param m: mass value
    :param p: pressure value
    :param rho: energy density value
    :param nu: metric value
    :param l: angular momentum number
    :return: axial potential
    """
    return exp(2.0 * nu) / r ** 3 * (l * (l + 1) * r + 4.0 * pi * r ** 3 * (rho + p) - 6.0 * m)


def axial_potential_out(r: float, M: float, l: int = 2) -> float:
    """
    Axial potential outside the star in GEO units
    :param r: radial coordinate
    :param M: total mass value
    :param l: angular momentum number
    :return: axial potential
    """
    return (1.0 - 2.0 * M / r) * (l * (l + 1) / r ** 2 - 6.0 * M / r ** 3)


def tortoise_coordinate_in(r: float, m_fun: Callable, nu_fun: Callable) -> float:
    """
    Tortoise coordinate inside the star in GEO units
    :param r: radial coordinate
    :param m_fun: mass function m(r)
    :param nu_fun: metric function nu(r)
    :return: r*
    """

    def integrand(x: float) -> float:
        return exp(-nu_fun(x)) / sqrt(1.0 - 2.0 * m_fun(x) / x)

    return quad(integrand, 0.0, r)[0]


def tortoise_coordinate_out(r: float, M: float) -> float:
    """
    Tortoise coordinate outside in GEO units
    :param r: radial coordinate
    :param M: total mass value
    :return: r*
    """
    return r + 2.0 * M * log(r / (2.0 * M) - 1.0)


def regge_wheeler_in(var, r: float, omega: complex, m: float, p: float, rho: float, nu: float, l: int = 2) -> tuple[
    complex, complex]:
    """
    Regge-Wheeler linear ODE in GEO units
    d²z/dr_*² + [omega² - V(r)] z = 0
    :param var: state variables
    :param r: radial coordinate
    :param omega: angular frequence value
    :param m: mass value
    :param p: pressure value
    :param rho: energy density value
    :param nu: metric value
    :param l: angular momentum number
    :return: derivatives
    """
    z, dzdr_star = var
    d2zd2r_star = -(omega ** 2 - axial_potential_in(r, m, p, rho, nu, l)) * z
    return dzdr_star, d2zd2r_star


def regge_wheeler_out(var, r: float, omega: complex, M: float, l: int = 2) -> tuple[complex, complex]:
    """
    Regge-Wheeler linear ODE in GEO units
    d²z/dr_*² + [omega² - V(r)] z = 0
    :param var: state variables
    :param r: radial coordinate
    :param omega: angular frequence value
    :param M: total mass value
    :param l: angular momentum number
    :return: derivatives
    """
    z, dzdr_star = var
    d2zd2r_star = -(omega ** 2 - axial_potential_out(r, M, l)) * z
    return dzdr_star, d2zd2r_star

def regge_wheeler_g_in(g: complex, r: float, omega: complex, m: float, p: float, rho: float, nu: float,l: int = 2) -> complex:
    """
    Regge-Wheeler Riccati ODE in GEO units inside the star
    dg/dr_* = -g² - omega² + V(r)
    :param g: logarithmic derivative g = z'/z
    :param r: radial coordinate
    :param omega: angular frequence value
    :param m: mass value
    :param p: pressure value
    :param rho: energy density value
    :param nu: metric value
    :param l: angular momentum number
    :return: derivative dg/dr_*
    """
    return -g ** 2 - omega ** 2 + axial_potential_in(r, m, p, rho, nu, l)


def regge_wheeler_g_out(g: complex, r: float, omega: complex, M: float, l: int = 2) -> complex:
    """
    Regge-Wheeler Riccati ODE in GEO units outside the star
    dg/dr_* = -g² - omega² + V(r)
    :param g: logarithmic derivative g = (dz/dr_*) / z
    :param r: radial coordinate
    :param omega: angular frequence value
    :param M: total mass value
    :param l: angular momentum number
    :return: derivative dg/dr_*
    """
    return -g ** 2 - omega ** 2 + axial_potential_out(r, M, l)












def regge_wheeler_g(r, g, omega, m, p, rho, nu, m_prime, l):
    A = 1.0 - 2.0 * m / r
    A_prime = -2.0 * m_prime / r + 2.0 * m / r ** 2
    V = axial_potential(r, m, p, rho, nu, l)
    return -g ** 2 - (A_prime / A) * g - (omega ** 2 - V) / A ** 2


def regge_wheeler_g_in(r, g, omega, m_fun, p_fun, rho_fun, nu_fun, dm_fun, l):
    m = m_fun(r)
    p = p_fun(r)
    rho = rho_fun(r)
    nu = nu_fun(r)
    m_prime = dm_fun(r)
    return regge_wheeler_g(r, g, omega, m, p, rho, nu, m_prime, l)


def s_ext(r, M):
    return 1.0 - 2.0 * M / r


def ds_ext_dr(r, M):
    return 2.0 * M / r ** 2


def nu_ext(r, M):
    return 0.5 * log(s_ext(r, M))


def compactified_radius(t, R, alpha):
    return R + ((1.0 - t) / t) * exp(1j * alpha)


def dr_dt(t, alpha):
    return -exp(1j * alpha) / t ** 2


def regge_wheeler_g_out_t(t, g, omega, R, M, alpha, l=2):
    r = compactified_radius(t, R, alpha)
    s = s_ext(r, M)
    s_prime = ds_ext_dr(r, M)
    V = axial_potential_ext(r, M, l)

    dg_dr = -g ** 2 - (s_prime / s) * g - (omega ** 2 - V) / s ** 2
    return dr_dt(t, alpha) * dg_dr
