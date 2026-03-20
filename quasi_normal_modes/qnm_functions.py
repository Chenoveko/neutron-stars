from numpy import pi, exp, log, sqrt, ndarray, array
from scipy.integrate import quad
from typing import Callable, Any
from scipy.integrate import solve_ivp
import numpy as np


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


def axial_potential_out(r: complex, M: float, l: int = 2) -> complex:
    """
    Axial potential outside the star in GEO units
    :param r: radial coordinate
    :param M: total mass value
    :param l: angular momentum number
    :return: axial potential
    """
    return (1.0 - 2.0 * M / r) * (l * (l + 1) / r ** 2 - 6.0 * M / r ** 3)


def tortoise_coordinate_in(r0: float, r: float, m_fun: Callable, nu_fun: Callable) -> float:
    """
    Tortoise coordinate inside the star in GEO units
    :param r0: initial radius
    :param r: radial coordinate
    :param m_fun: mass function m(r)
    :param nu_fun: metric function nu(r)
    :return: r*
    """

    def integrand(x: float) -> float:
        return exp(-nu_fun(x)) / sqrt(1.0 - 2.0 * m_fun(x) / x)

    return quad(integrand, r0, r)[0]


def tortoise_coordinate_out(r: float, M: float) -> float:
    """
    Tortoise coordinate outside in GEO units
    :param r: radial coordinate
    :param M: total mass value
    :return: r*
    """
    return r + 2.0 * M * log(r / (2.0 * M) - 1.0)


def drdr_star(r: float, m: float, nu: float) -> float:
    """
    dr/dr*
    :param r: radial coordinate
    :param m: mass value
    :param nu: metric value
    :return: dr/dr*
    """
    return exp(nu) * sqrt(1.0 - 2.0 * m / r)


def regge_wheeler_g_in(g: complex, r: float, omega: complex, m: float, p: float, rho: float, nu: float,
                       l: int = 2) -> complex:
    """
    Regge-Wheeler Riccati ODE in GEO units inside the star
    dg/dr_* = -g² - omega² + V(r)
    :param g: logarithmic derivative g = z'/z
    :param r: radial coordinate
    :param omega: angular frequency value
    :param m: mass value
    :param p: pressure value
    :param rho: energy density value
    :param nu: metric value
    :param l: angular momentum number
    :return: derivative dg/dr_*
    """
    return -g ** 2 - omega ** 2 + axial_potential_in(r, m, p, rho, nu, l)


def system_in(r_star: float, var, omega: complex, m_fun, p_fun, rho_fun, nu_fun, l: int = 2) -> tuple[complex, complex]:
    """
    Interior system in GEO units written in terms of the tortoise coordinate
    dr/dr_* = exp(nu) * sqrt(1 - 2m/r)
    dg/dr_* = -g² - omega² + V(r)
    :param r_star: tortoise coordinate
    :param var: state variables
    :param omega: angular frequency value
    :param m_fun: mass function m(r)
    :param p_fun: pressure function p(r)
    :param rho_fun: energy density function rho(r)
    :param nu_fun: metric function nu(r)
    :param l: angular momentum number
    :return: derivatives
    """
    r = float(var[0].real)
    g = var[1]

    m = m_fun(r)
    p = p_fun(r)
    rho = rho_fun(r)
    nu = nu_fun(r)

    dr_dst = drdr_star(r, m, nu)
    dg_dst = regge_wheeler_g_in(g, r, omega, m, p, rho, nu, l)

    return dr_dst, dg_dst


def solve_regge_wheeler_g_inside(r: ndarray, m_fun: Callable, p_fun: Callable, rho_fun: Callable, nu_fun: Callable,
                                 omega: complex, l: int = 2):
    """
    Solve the interior Regge-Wheeler Riccati equation in GEO units using the tortoise coordinate
    :param r: radial coordinate array from TOV
    :param m_fun: mass function m(r)
    :param p_fun: pressure function p(r)
    :param rho_fun: energy density function rho(r)
    :param nu_fun: metric function nu(r)
    :param omega: angular frequency value
    :param l: angular momentum number
    :return: interior solution
    """

    # Initial and final radius
    r0 = r[0]
    R = r[-1]
    # Initial tortoise coordinate and stellar surface in tortoise coordinate
    r_star_0 = 0.0
    r_star_R = tortoise_coordinate_in(r0, R, m_fun, nu_fun)

    # Initial conditions from regularity at the center:
    # Z ~ r^(l+1)  =>  (1/Z) dZ/dr ~ (l+1)/r
    # g = (1/Z) dZ/dr_* = (dr/dr_*) (1/Z) dZ/dr
    m0 = m_fun(r0)
    nu0 = nu_fun(r0)

    g0 = drdr_star(r0, m0, nu0) * (l + 1.0) / r0

    init = array([r0, g0], dtype=complex)

    # Integrate
    sol = solve_ivp(fun=system_in, t_span=(r_star_0, r_star_R), y0=init, args=(omega, m_fun, p_fun, rho_fun, nu_fun, l),
                    method='RK45', max_step=1e3, rtol=1e-6, atol=1e-8)

    return sol


def compact_exterior_complex_path(t: float, R: float, alpha: float) -> complex:
    """
    Compact Exterior complex-scaling radial path
    r(t) = R + (1-t)/t exp(i alpha)
    :param t: exterior integration parameter compactified t ∈ (0, 1]
    :param R: stellar radius
    :param alpha: complex-scaling angle
    :return: complex radial coordinate
    """
    return R + ((1.0 - t) / t) * exp(1j * alpha)


def drdt_out(t: float, alpha: float) -> complex:
    """
    Radial derivative along the compactified ECS contour
    dr/dt = -exp(i alpha) / t^2
    :param t: compactified exterior coordinate
    :param alpha: complex-scaling angle
    :return: dr/dt
    """
    return -exp(1j * alpha) / t ** 2


def s_out(r: complex, M: float) -> complex:
    """
    Exterior Schwarzschild factor
    s = (dr_*/dr)^(-1) = 1 - 2M/r
    :param r: radial coordinate
    :param M: total mass value
    :return: s(r)
    """
    return 1.0 - 2.0 * M / r


def dsdr_out(r: complex, M: float) -> complex:
    """
    Radial derivative of the exterior Schwarzschild factor
    ds/dr = 2M / r^2
    :param r: radial coordinate
    :param M: total mass value
    :return: ds/dr
    """
    return 2.0 * M / r ** 2


def regge_wheeler_g_out(r: complex, g: complex, omega: complex, M: float, l: int = 2) -> complex:
    """
    Regge-Wheeler Riccati ODE in GEO units outside the star written in terms of r

    g = (dZ/dr) / Z

    dg/dr = -g^2 - (s'/s) g - (omega^2 - V(r)) / s^2

    :param r: radial coordinate
    :param g: logarithmic derivative g = (dZ/dr) / Z
    :param omega: angular frequency value
    :param M: total mass value
    :param l: angular momentum number
    :return: derivative dg/dr
    """
    s = s_out(r, M)
    dsdr = dsdr_out(r, M)
    v = axial_potential_out(r, M, l)

    return -g ** 2 - (dsdr / s) * g - (omega ** 2 - v) / s ** 2


def system_out(t: float, var, R: float, alpha: float, omega: complex, M: float,
               l: int = 2) -> tuple[complex]:
    """
    Exterior system in GEO units written along the compactified ECS contour
    dg/dt = (dr/dt) dg/dr
    :param t: compactified exterior coordinate
    :param var: state variables
    :param R: stellar radius
    :param alpha: complex-scaling angle
    :param omega: angular frequency value
    :param M: total mass value
    :param l: angular momentum number
    :return: derivatives
    """
    g = var[0]
    r = compact_exterior_complex_path(t, R, alpha)
    dgdt = drdt_out(t, alpha) * regge_wheeler_g_out(r, g, omega, M, l)
    return (dgdt,)


def solve_regge_wheeler_g_outside(R: float, M: float, omega: complex, alpha: float,
                                  t_inf: float = 1e-4, l: int = 2):
    """
    Solve the exterior Regge-Wheeler Riccati equation in GEO units using compactified ECS
    :param R: stellar radius
    :param M: total mass value
    :param omega: angular frequency value
    :param alpha: complex-scaling angle
    :param t_inf: starting point near infinity
    :param l: angular momentum number
    :return: exterior solution
    """

    # Asymptotic outgoing condition from the notes: g ~ -i omega
    g_inf = -1j * omega
    init = array([g_inf], dtype=complex)

    # Integrate from infinity (t ~ 0) to the surface (t = 1)
    sol = solve_ivp(fun=system_out, t_span=(t_inf, 1.0), y0=init, args=(R, alpha, omega, M, l), method='RK45',
                    rtol=1e-6, atol=1e-8)

    return sol

def matching_function(omega: complex, r: ndarray, m_fun: Callable, p_fun: Callable, rho_fun: Callable,
                      nu_fun: Callable, alpha: float, t_inf: float = 1e-4, l: int = 2) -> complex:
    """
    Matching function at the stellar surface
    f(omega) = g_in(R) - g_out(R)

    where
        g_in(R)  = (1/Z) dZ/dr evaluated from the interior solution
        g_out(R) = (1/Z) dZ/dr evaluated from the exterior solution

    :param omega: angular frequency value
    :param r: radial coordinate array from TOV
    :param m_fun: mass function m(r)
    :param p_fun: pressure function p(r)
    :param rho_fun: energy density function rho(r)
    :param nu_fun: metric function nu(r)
    :param alpha: complex-scaling angle
    :param t_inf: starting point near infinity for the compactified exterior integration
    :param l: angular momentum number
    :return: matching function evaluated at the stellar surface
    """

    # Surface values
    R = r[-1]
    M = m_fun(R)

    # Interior solution: this gives g_* = (1/Z) dZ/dr_*
    sol_in = solve_regge_wheeler_g_inside(r, m_fun, p_fun, rho_fun, nu_fun, omega, l)
    g_in_star_R = sol_in.y[1, -1]

    # Convert interior value to g = (1/Z) dZ/dr
    # Inside the star: dr/dr_* = exp(nu) * sqrt(1 - 2m/r)
    nu_R = nu_fun(R)
    s_in_R = drdr_star(R, M, nu_R)
    g_in_R = g_in_star_R / s_in_R

    # Exterior solution: already written for g = (1/Z) dZ/dr
    sol_out = solve_regge_wheeler_g_outside(R, M, omega, alpha, t_inf=t_inf, l=l)
    g_out_R = sol_out.y[0, -1]

    return g_in_R - g_out_R


def alpha_condition(alpha:float, omega:complex)->bool:
    if omega.imag * np.cos(alpha) - omega.real * np.sin(alpha) <0:
        return True
    else:
        return False
