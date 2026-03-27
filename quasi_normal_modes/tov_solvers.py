# =====================
# Imports
# =====================
from typing import Callable, Tuple
from numpy import ndarray, array, pi, sqrt, isfinite,log
from scipy.integrate import solve_ivp
from physical_functions import tov_equations

def solve_tov_eos(p_c: float, rho_func: Callable, r0: float = 1e-4, r_max: float = 20e5,
                  method: str = 'RK45') -> Tuple[ndarray, ndarray, ndarray, ndarray, bool]:
    """
    Solves the TOV equations for a given equation of state
    :param p_c: Central pressure in GEO-CGS units
    :param rho_func: Function that returns the energy density as a function of pressure.
    :param r0: Initial radius near the center to avoid numerical singularities.
    :param r_max: Maximum radius up to which the integration is performed.
    :param method: Numerical integration method used by the ODE solver.
    :return:
        - r : Numpy array of radial points
        - p : Numpy array of interior pressure
        - m : Numpy array of enclosed mass
        - status: Stop event bool, 1 if stopped by an event, 0 if reached r_max
    """
    # Integrate until pressure has dropped by 20 orders of magnitude
    p_min = p_c * 1e-15
    # Taylor expansion (order 3) near center to get initial conditions
    rho_c = rho_func(p_c)
    p_r0 = p_c - (2 / 3) * pi * r0 ** 2 * (rho_c + p_c) * (rho_c + 3 * p_c)
    m_r0 = (4 / 3) * pi * rho_c * r0 ** 3
    nu_r0 = 0.0 + (4 / 3) * pi * r0 ** 2 * (rho_c + 3 * p_c)
    # Initial conditions
    init = array([p_r0, m_r0, nu_r0], float)

    # System to use solve_ivp
    def tov_system(r, y):
        p, m, nu = y
        rho_val = rho_func(p)
        return tov_equations(y, r, rho_val)

    # Event to trigger when pressure crosses p_min downward
    def event_pressure_threshold(r, y):
        return y[0] - p_min

    event_pressure_threshold.terminal = True
    event_pressure_threshold.direction = -1

    # Event to trigger when pressure becomes negative
    def event_negative_pressure(r, y):
        return y[0]

    event_negative_pressure.terminal = True
    event_negative_pressure.direction = -1

    # Stop when enclosed mass saturates (dm/dr / m < eps_rel)
    def event_mass_saturation(r, y):
        p, m = y[0], y[1]
        if (not isfinite(p)) or (p <= 0) or (not isfinite(m)) or (m <= 0):
            return 0.0
        rho = rho_func(p)
        if (not isfinite(rho)) or (rho <= 0):
            return 0.0
        return (4.0 * pi * r ** 2 * rho / m) - 1e-8

    event_mass_saturation.terminal = True
    event_mass_saturation.direction = -1

    # Integrate
    sol = solve_ivp(fun=tov_system, t_span=(r0, r_max), y0=init, method=method,
                    events=[event_pressure_threshold, event_negative_pressure,event_mass_saturation], max_step=1e2,
                    rtol=1e-10, atol=1e-10)

    r = sol.t
    p = sol.y[0]
    m = sol.y[1]
    nu = sol.y[2]

    # Surface values
    R = r[-1]
    M = m[-1]

    # Normalize nu to match exterior Schwarzschild:
    # g_tt = -exp(2nu), so nu(R) = 1/2*ln(1 - 2M/R)
    nu = nu - nu[-1] + 0.5 * log(1.0 - 2.0 * M / R)

    return r, p, m, nu, sol.status