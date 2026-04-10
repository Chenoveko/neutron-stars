# =====================
# Imports
# =====================
from typing import Tuple, Callable

from numpy import array, ndarray, nanargmin, unravel_index, linspace, pi, sqrt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
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
    sol_in = solve_ivp(rw_in, t_span=(r0, R), y0=init, method="DOP853", rtol=1e-12, atol=1e-12, max_step=1e3)
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
    sol_out = solve_ivp(rw_out, t_span=(t_inf, 1.0), y0=init, method="RK45", rtol=1e-12, atol=1e-12,max_step=1e-3,first_step=1e-9)
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
    """
    Selects 3 Müller seeds from a meshgrid search.
    Uses the minimum of Z and its neighboring points to build the seeds.
    :param F: frequency meshgrid
    :param TAU: damping time meshgrid
    :param Z: log10 of matching function values
    :return: three seed points (p1, p2, p3)
    """
    ny, nx = Z.shape
    iy, ix = unravel_index(nanargmin(Z), Z.shape)

    neighbors = []
    for j in range(max(0, iy - 1), min(ny, iy + 2)):
        for i in range(max(0, ix - 1), min(nx, ix + 2)):
            neighbors.append((Z[j, i], F[j, i], TAU[j, i], j, i))

    neighbors.sort(key=lambda x: x[0])

    p1 = (neighbors[0][1], neighbors[0][2])
    p2 = (neighbors[1][1], neighbors[1][2])
    p3 = (neighbors[2][1], neighbors[2][2])

    return p1, p2, p3


def extrapolate_qnm_point(p_hist, f_hist, tau_hist, p_new):
    """
    Extrapolate (f, tau) at p_new.
    - 2 points  -> linear
    - 3 points  -> linear using last two points
    - >=4 points -> PCHIP
    :param p_hist: array of central pressures
    :param f_hist: array of frequencies [Hz]
    :param tau_hist: array of damping times [s]
    :param p_new: new central pressure
    :return: extrapolated (f0, tau0)
    """
    p_hist, f_hist, tau_hist = array(p_hist, float), array(f_hist, float), array(tau_hist, float);
    n = len(p_hist)
    if n < 2: raise ValueError("At least 2 points required")
    if n <= 3:
        p0, p1 = p_hist[-2], p_hist[-1]; sf = (f_hist[-1] - f_hist[-2]) / (p1 - p0); st = (tau_hist[-1] - tau_hist[
            -2]) / (p1 - p0); f0 = f_hist[-1] + sf * (p_new - p1); tau0 = tau_hist[-1] + st * (p_new - p1)
    else:
        f0 = float(PchipInterpolator(p_hist, f_hist, extrapolate=True)(p_new)); tau0 = float(
            PchipInterpolator(p_hist, tau_hist, extrapolate=True)(p_new))
    return f0, tau0


def muller_seed_from_extrapolation(p_hist, f_hist, tau_hist, p_new, c_cgs, f_match=None, df_rel=2e-4, dtau_rel=2e-4,
                                   local_refine=True, n_local=5):
    """
    Generate 3 Müller seeds from extrapolation.
    Uses local tangent direction and optional micro-refinement to avoid wrong minima.
    :param p_hist: array of central pressures
    :param f_hist: array of frequencies [Hz]
    :param tau_hist: array of damping times [s]
    :param p_new: new central pressure
    :param c_cgs: speed of light [cm/s]
    :param f_match: matching function
    :param df_rel: relative frequency window
    :param dtau_rel: relative tau window
    :param local_refine: enable local search
    :param n_local: local grid size
    :return: (w1,w2,w3),(f0,tau0),(p1,p2,p3)
    """
    p_hist, f_hist, tau_hist = array(p_hist, float), array(f_hist, float), array(tau_hist, float);
    f0, tau0 = extrapolate_qnm_point(p_hist, f_hist, tau_hist, p_new);
    df, dt = f_hist[-1] - f_hist[-2], tau_hist[-1] - tau_hist[-2];
    vf, vt = df / max(abs(f0), 1e-30), dt / max(abs(tau0), 1e-30);
    nrm = sqrt(vf * vf + vt * vt);
    ef, et = (1.0, 0.0) if nrm < 1e-14 else (vf / nrm, vt / nrm);
    nf, nt = -et, ef;
    df0 = max(df_rel * abs(f0), 5.0);
    dt0 = max(dtau_rel * abs(tau0), 5e-10)
    if local_refine and (f_match is not None):
        fg, tg = linspace(f0 - df0, f0 + df0, n_local), linspace(tau0 - dt0, tau0 + dt0, n_local);
        best = (f0, tau0, 1e300)
        for ff in fg:
            for tt in tg:
                if tt <= 0: continue
                try:
                    val = abs(f_match((2 * pi * ff - 1j / tt) / c_cgs)); best = (ff, tt, val) if val < best[2] else best
                except:
                    pass
        f0, tau0 = best[0], best[1]
    p1 = (f0, tau0);
    p2 = (f0 + 0.8 * df0 * ef, tau0 + 0.8 * dt0 * et);
    p3 = (f0 - 0.5 * df0 * ef + 0.35 * df0 * nf, tau0 - 0.5 * dt0 * et + 0.35 * dt0 * nt)
    w1 = (2 * pi * p1[0] - 1j / p1[1]) / c_cgs;
    w2 = (2 * pi * p2[0] - 1j / p2[1]) / c_cgs;
    w3 = (2 * pi * p3[0] - 1j / p3[1]) / c_cgs
    return (w1, w2, w3), (f0, tau0), (p1, p2, p3)
