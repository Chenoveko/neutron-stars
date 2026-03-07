# =====================
# Imports
# =====================
from typing import Callable, Tuple

from numpy import ndarray, array, pi, sqrt, isfinite
from scipy.integrate import solve_ivp

from utilities.physical_functions import tov_equations, newton_equations



def solve_tov_eos(p_c: float, rho_func: Callable, r0: float = 1e-4, r_max: float = 1e11,
                  method: str = 'RK45') -> Tuple[ndarray, ndarray, ndarray, bool]:
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
    # Integrate until pressure has dropped by 15 orders of magnitude
    p_min = p_c * 1e-15
    # Taylor expansion (order 3) near center to get initial conditions
    rho_c = rho_func(p_c)
    p_r0 = p_c - (2 / 3) * pi * r0 ** 2 * (rho_c + p_c) * (rho_c + 3 * p_c)
    m_r0 = (4 / 3) * pi * rho_c * r0 ** 3
    # Initial conditions
    init = array([p_r0, m_r0], float)

    # System to use solve_ivp
    def tov_system(r, y):
        p, m = y
        rho_val = rho_func(p)
        return tov_equations(y, r, rho_val)

    # Event to trigger when pressure becomes negative
    def event_negative_pressure(r, y):
        return y[0]

    event_negative_pressure.terminal = True
    event_negative_pressure.direction = -1

    # Integrate
    sol = solve_ivp(fun=tov_system, t_span=(r0, r_max), y0=init, method=method,
                    events= event_negative_pressure, max_step=1e6,
                    rtol=1e-4, atol=1e-6)
    r = sol.t
    p = sol.y[0]
    m = sol.y[1]
    status = sol.status
    return r, p, m, status

"""
Numerical integration to solve Newton equations using EoS
-----------------------------------------------------------------------------------------
"""

def solve_newton_eos(p_c: float, rho_func: Callable, r0: float = 1e-4, r_max: float = 1e11,
                  method: str = 'RK45') -> Tuple[ndarray, ndarray, ndarray, bool]:
    """
    Solves the Newton equations for a given equation of state
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
    # Taylor expansion (order 3) near center to get initial conditions
    rho_c = rho_func(p_c)
    p_r0 = p_c - (2 / 3) * pi * r0 ** 2 * (rho_c + p_c) * (rho_c + 3 * p_c)
    m_r0 = (4 / 3) * pi * rho_c * r0 ** 3
    # Initial conditions
    init = array([p_r0, m_r0], float)

    # System to use solve_ivp
    def newton_system(r, y):
        p, m = y
        rho_val = rho_func(p)
        return newton_equations(y, r, rho_val)

    # Event to trigger when pressure becomes negative
    def event_negative_pressure(r, y):
        return y[0]

    event_negative_pressure.terminal = True
    event_negative_pressure.direction = -1

    # Integrate
    sol = solve_ivp(fun=newton_system, t_span=(r0, r_max), y0=init, method=method,
                    events=event_negative_pressure, max_step=1e6,
                    rtol=1e-4, atol=1e-6)
    r = sol.t
    p = sol.y[0]
    m = sol.y[1]
    status = sol.status
    return r, p, m, status