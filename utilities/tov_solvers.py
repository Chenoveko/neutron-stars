"""
Stellar Structure Integrators — TOV Solvers (Uniform Density and Generic EoS)
============================================================================

This module provides numerical solvers for the internal structure of
spherically symmetric relativistic stars by integrating the
Tolman–Oppenheimer–Volkoff (TOV) equations.

Two complementary solvers are implemented:

    1) solve_tov_schwarzschild
       Numerical integration of the TOV system for a uniform-density star
       (Schwarzschild interior model). The energy density is constant and
       determined by the chosen total mass M and radius R.

    2) solve_tov_eos
       Numerical integration of the TOV system using an arbitrary equation
       of state (EoS) provided as a function  ρ = ρ(p).

All integrations are performed with SciPy's solve_ivp ODE integrator.

Units
-----
    - Geometrized units in CGS are assumed (G = c = 1).
    - Radii are handled in centimetres (cm) in the integrators.
    - Pressure p, enclosed mass m, and energy density ρ are handled in
      consistent geometrized CGS units.

Numerical Strategy
------------------
The stellar structure equations are formally singular at r = 0. To avoid
numerical instabilities, integrations start from a small radius r0 > 0.
Initial conditions at r0 are obtained from a third-order Taylor expansion
around the centre:

    p(r0) ≈ p_c - (2/3)π r0^2 (ρ_c + p_c)(ρ_c + 3p_c)
    m(r0) ≈ (4/3)π ρ_c r0^3

This ensures regular behaviour at the centre while preserving the correct
local curvature of the pressure and mass profiles.

--------------------------------------------------------------------------------
Function: solve_tov_schwarzschild
--------------------------------------------------------------------------------
Numerical integration of the TOV equations for a uniform-density star.

This solver corresponds to the Schwarzschild interior model, where the
energy density ρ is constant and is fixed by the global parameters M and R:

    ρ = 3M / (4πR^3)

The central pressure is computed analytically from the Schwarzschild interior
solution and used to generate the initial conditions for the numerical
integration.

A physical compactness condition is enforced:

    M/R < 4/9

which is required for the Schwarzschild uniform-density interior solution to
remain regular.

Parameters
----------
    M : float
        Total gravitational mass (geometrized units).
    R : float
        Total radius (cm, in geometrized CGS length units).
    r0 : float, optional
        Starting radius close to the centre to avoid the singular point r = 0.
    method : str, optional
        Integration method used by solve_ivp (default: 'RK45').

Returns
-------
    ndarray
        Array of shape (3, N) containing:
            - r_points : radial grid
            - p_points : pressure profile p(r)
            - m_points : enclosed mass profile m(r)

Notes
-----
    - The right-hand side is explicitly set to zero if p <= 0 to prevent
      unphysical evolution beyond the surface.
    - The integration span is fixed to (r0, R) since R is prescribed.

--------------------------------------------------------------------------------
Function: solve_tov_eos
--------------------------------------------------------------------------------
Numerical integration of the TOV equations using an equation of state (EoS).

This solver integrates the relativistic stellar structure equations given a
central pressure p_c and an EoS relation provided as a callable:

    rho_func(p) = ρ(p)

The integration begins at r0 with Taylor-expanded initial conditions and
proceeds outward until a surface-like stopping condition is met.

Stopping Criteria (Events)
--------------------------
To robustly terminate the integration and define an effective stellar surface,
three events are implemented:

    1) Pressure threshold:
           p(r) drops below  p_min = p_c * 1e-15
       This avoids requiring p to reach exactly zero, which can be numerically
       problematic for stiff equations of state.

    2) Negative pressure:
           p(r) crosses below 0
       Safety condition to stop unphysical trajectories.

    3) Mass saturation:
           (dm/dr) / m < eps_rel
       Stops when the relative mass growth becomes negligible, indicating that
       the enclosed mass has effectively converged.

Parameters
----------
    p_c : float
        Central pressure in geometrized CGS units.
    rho_func : Callable
        Function returning the energy density ρ as a function of pressure p.
    r0 : float, optional
        Starting radius close to the centre to avoid numerical singularities.
    r_max : float, optional
        Maximum radius (cm) allowed for the integration if no event stops it.
    method : str, optional
        Integration method used by solve_ivp (default: 'RK45').

Returns
-------
    (r, p, m, status) : Tuple[ndarray, ndarray, ndarray, bool]
        r : radial grid (cm)
        p : pressure profile p(r)
        m : enclosed mass profile m(r)
        status : solve_ivp termination flag
            - 1 if stopped by an event (surface condition reached)
            - 0 if the solver reached r_max without triggering an event
"""
# =====================
# Imports
# =====================
from typing import Callable, Tuple
from numpy import ndarray, array, pi, sqrt, isfinite
from scipy.integrate import solve_ivp
from utilities.physical_functions import tov_equations


def solve_tov_schwarzschild(M: float, R: float, r0: float = 1e-8, method: str = 'RK45') -> Tuple[
    ndarray, ndarray, ndarray]:
    """
    :param M: total mass
    :param R: total radius
    :param r0: radial point near center
    :param method: Integration method to use
    :return:
        - r_points: Numpy array of radial coordinate points
        - p_points: Numpy array of interior pressure points
        - m_points: Numpy array of enclosed mass points
    """
    # Buchdahl bound: required for regular Schwarzschild interior solution
    if M / R >= 4 / 9:
        raise ValueError("Schwarzschild interior solution requires M/R < 4/9")
    # Central pressure of schwarzschild_solution
    p_c = (3 * M / (4 * pi * R ** 3)) * (1 - sqrt(1 - 2 * M / R)) / (3 * sqrt(1 - 2 * M / R) - 1)
    # Uniform energy density
    rho = 3 * M / (4 * pi * R ** 3)
    # Taylor expansion (order 3) near center to get initial conditions
    p_r0 = p_c - (2 / 3) * pi * r0 ** 2 * (rho + p_c) * (rho + 3 * p_c)
    m_r0 = (4 / 3) * pi * rho * r0 ** 3

    # System to use solve_ivp
    def tov_system(r, y):
        if y[0] <= 0:
            return [0.0, 0.0]
        return tov_equations(y, r, rho)  # var primero

    # Initial conditions
    init = array([p_r0, m_r0], float)
    # Integrate
    sol = solve_ivp(fun=tov_system, t_span=(r0, R), y0=init, method=method, first_step=10, max_step=20, rtol=1e-4,
                    atol=1e-6)
    return sol.t, sol.y[0], sol.y[1]


def solve_tov_eos(p_c: float, rho_func: Callable, r0: float = 1e-4, r_max: float = 15e5,
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
    # Integrate until pressure drops 15 orders of magnitude below central value
    p_min = p_c * 1e-15
    # 3rd-order Taylor expansion near center for initial conditions
    rho_c = rho_func(p_c)
    p_r0 = p_c - (2.0 / 3.0) * pi * r0 ** 2 * (rho_c + p_c) * (rho_c + 3.0 * p_c)
    m_r0 = (4.0 / 3.0) * pi * rho_c * r0 ** 3
    init = array([p_r0, m_r0], float)

    # Wrapper for solve_ivp — evaluates EOS at each step
    def tov_system(r, y):
        rho_val = rho_func(y[0])
        return tov_equations(y, r, rho_val)  # var primero

    # Stop when pressure crosses p_min downward
    def event_pressure_threshold(r, y):
        return y[0] - p_min

    event_pressure_threshold.terminal = True
    event_pressure_threshold.direction = -1

    # Stop when pressure becomes negative
    def event_negative_pressure(r, y):
        return y[0]

    event_negative_pressure.terminal = True
    event_negative_pressure.direction = -1

    # Stop when enclosed mass saturates (dm/dr / m < eps_rel)
    def event_mass_saturation(r, y):
        p, m = y
        if (not isfinite(p)) or (p <= 0) or (not isfinite(m)) or (m <= 0):
            return 0.0
        rho = rho_func(p)
        if (not isfinite(rho)) or (rho <= 0):
            return 0.0
        return (4.0 * pi * r ** 2 * rho / m) - 1e-8

    event_mass_saturation.terminal = True
    event_mass_saturation.direction = -1
    sol = solve_ivp(fun=tov_system, t_span=(r0, r_max), y0=init, method=method,
                    events=[event_pressure_threshold, event_negative_pressure, event_mass_saturation],
                    max_step=1e3, rtol=1e-10, atol=1e-10)

    return sol.t, sol.y[0], sol.y[1], sol.status
