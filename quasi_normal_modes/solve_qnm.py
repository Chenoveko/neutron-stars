from numpy import array,angle
from scipy.integrate import solve_ivp

from qnm_functions import (
    axial_potential,
    compactified_radius,
    regge_wheeler_g_out_t,
    s_ext,
)
from qnm_functions import regge_wheeler_g_in



def solve_regge_wheeler_inside(omega, r0, R, m_fun, p_fun, rho_fun, nu_fun, dm_fun, l):
    Z0 = r0 ** (l + 1)
    Zp0 = (l + 1) * r0 ** l

    y0 = array([Z0, Zp0], dtype=complex)

    def rhs(r, y):
        Z, Zp = y

        m = m_fun(r)
        p = p_fun(r)
        rho = rho_fun(r)
        nu = 0.5 * nu_fun(r)
        m_prime = dm_fun(r)

        A = 1.0 - 2.0 * m / r
        A_prime = -2.0 * m_prime / r + 2.0 * m / r**2
        V = axial_potential(r, m, p, rho, nu, l)

        Zpp = -(A_prime / A) * Zp - (omega**2 - V) / A**2 * Z

        return array([Zp, Zpp], dtype=complex)

    sol = solve_ivp(
        rhs,
        (r0, R),
        y0,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    Z = sol.y[0, -1]
    Zp = sol.y[1, -1]

    return Zp / Z


def solve_regge_wheeler_g_outside(omega, R, M, alpha, l=2, t0=1e-2):
    r0 = compactified_radius(t0, R, alpha)

    s0 = s_ext(r0, M)
    L = l * (l + 1)

    g0 = -1j * omega / s0 + 1j * L / (2.0 * omega * r0**2)

    y0 = array([g0.real, g0.imag], dtype=float)

    def rhs(t, y):
        g = y[0] + 1j * y[1]
        dg_dt = regge_wheeler_g_out_t(t, g, omega, R, M, alpha, l=l)
        return array([dg_dt.real, dg_dt.imag], dtype=float)

    sol = solve_ivp(
        rhs,
        (t0, 1.0),
        y0,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        max_step=1e-2,
    )

    return sol.y[0, -1] + 1j * sol.y[1, -1]


def matching_function(omega, r0, R, m_fun, p_fun, rho_fun, nu_fun, dm_fun, M,alpha, l=2):
    # Integrate inside the star
    g_in = solve_regge_wheeler_inside(omega, r0, R, m_fun, p_fun, rho_fun, nu_fun, dm_fun, l)
    # Integrate outside the star
    g_out = solve_regge_wheeler_g_outside(omega, R, M, alpha, l)
    return g_in - g_out