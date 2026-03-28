# =====================
# Imports
# =====================
from typing import Tuple, Callable

from numpy import array, ndarray, nanargmin, unravel_index
from scipy.integrate import solve_ivp

from qnm_functions import regge_wheeler_in, regge_wheeler_out, compact_rot_coord


def solve_qnm_inside(r0: float, R: float, omega: complex, m: Callable, p: Callable, rho: Callable,
                     nu: Callable, l: int = 2) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Solves the RW equation inside the star in GEO units
    :param r0: Initial radius near the center to avoid numerical singularities.
    :param R: total radius
    :param omega: angular frequency value
    :param m: mass function m(r)
    :param p: pressure function p(r)
    :param rho: energy density function rho(r)
    :param nu: metric function nu(r)
    :param l: angular momentum number
    :return:
        - radial coordinate
        - z
        - dz/dr
    """
    # Initial Conditions z ~ r^(l+1)
    z0 = r0 ** (l + 1)
    dzdr0 = (l + 1) * r0 ** l
    # [Re(z), Im(z), Re(dz/dr), Im(dz/dr)]
    init = array([z0, 0.0, dzdr0, 0.0], dtype=float)
    # Lambda function
    rw_in = lambda r, var: regge_wheeler_in(r, var, omega, m, p, rho, nu, l)
    # Scipy integrator
    sol_in = solve_ivp(rw_in, t_span=(r0, R), y0=init, method="RK45", rtol=1e-10, atol=1e-10)
    # Reconstruct complex solution
    z = sol_in.y[0] + 1j * sol_in.y[1]
    dzdr = sol_in.y[2] + 1j * sol_in.y[3]
    return sol_in.t, z, dzdr


def solve_qnm_outside(M: float, R: float, omega: complex, alpha: float, t_inf: float = 1e-10, l: int = 2) -> Tuple[
    ndarray, ndarray]:
    """
    Solves the RW equation outside the star in GEO units
    :param M: total mass
    :param R: total radius
    :param omega: angular frequency value
    :param alpha: complex rotation angle
    :param t_inf: compactified coordinate -> infinity
    :param l: angular momentum number
    :return:
        - t
        - g
    """
    # Initial Conditions -> outgoing asymptotic condition
    # Z ~ exp(- i omega r_*)  =>  g ~ -i omega
    r_inf = compact_rot_coord(t_inf, R, alpha)
    f_inf = 1.0 - 2.0 * M / r_inf
    g_inf = -1j * omega / f_inf
    init = array([g_inf.real, g_inf.imag], dtype=float)
    # Lambda function
    rw_out = lambda t, g: regge_wheeler_out(t, g, omega, alpha, M, R, l)
    # Scipy integrator
    sol_out = solve_ivp(rw_out, t_span=(t_inf, 1.0), y0=init, method="RK45", max_step=1e-2, rtol=1e-10, atol=1e-10)
    g = sol_out.y[0] + 1j * sol_out.y[1]
    return sol_out.t, g


def matching(r0: float, R: float, omega: complex, m: Callable, p: Callable, rho: Callable, nu: Callable,
             M: float, alpha: float, t_inf: float = 1e-10, l: int = 2) -> complex:
    """
    Matching function in the surface
    f = g_in(R) - g_out(R)
    :param r0: Initial radius near the center to avoid numerical singularities.
    :param R: total radius
    :param omega: angular frequency value
    :param m: mass value
    :param p: pressure value
    :param rho: energy density value
    :param nu: metric value
    :param l: angular momentum number
    :param M: total mass
    :param alpha: complex rotation angle
    :param t_inf: compactified coordinate -> infinity
    :param l: angular momentum number
    :return: matching value in the surface
    """
    # Interior solution
    _, z_rw_in, z_prime_rw_in = solve_qnm_inside(r0, R, omega, m, p, rho, nu)
    g_rw_in_R = z_prime_rw_in[-1] / z_rw_in[-1]  # Value of g_in(R)
    # Outside solution
    _, g_rw_out = solve_qnm_outside(M, R, omega, alpha, t_inf, l)
    g_rw_out_R = g_rw_out[-1]

    return g_rw_in_R - g_rw_out_R


def muller_seed_meshgrid(F, TAU, Z):
    ny, nx = Z.shape
    iy, ix = unravel_index(nanargmin(Z), Z.shape)

    vecinos = []
    for j in range(max(0, iy - 1), min(ny, iy + 2)):
        for i in range(max(0, ix - 1), min(nx, ix + 2)):
            vecinos.append((Z[j, i], F[j, i], TAU[j, i], j, i))

    vecinos.sort(key=lambda x: x[0])  # ordenar por valor de Z

    # 3 mejores puntos del entorno
    p1 = (vecinos[0][1], vecinos[0][2])
    p2 = (vecinos[1][1], vecinos[1][2])
    p3 = (vecinos[2][1], vecinos[2][2])

    return p1, p2, p3
