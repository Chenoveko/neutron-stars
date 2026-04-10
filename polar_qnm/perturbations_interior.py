import numpy as np
from scipy.integrate import solve_ivp


def perturbations_interior(r: float, y: np.typing.NDArray[np.complexfloating],
                           m_function: Callable, p_function: Callable, rho_function: Callable, nu_function: Callable,
                           Gamma1_function: Callable,
                           omega: complex, ell: int = 2
                           ) -> list[complex]:
    """
    Interior polar perturbation equations for relativistic star in GEO units
    :param radial: Radial coordinate
    :param y: H1, K, W and X at r
    :param m_function: mass function m(r)
    :param p_function: pressure function p(r)
    :param rho_function: energy density function rho(r)
    :param nu_function: metric function nu(r)
    :param Gamma1_function: adiabatic index function nu(r)
    :param omega: angular frequency value
    :param ell: angular momentum number
    :return:
        - dH1dr
        - dKdr
        - dWdr
        - dXdr
    """
    H1, K, W, X = y
    # Compute values of m, p, rho, nu, Gamma1 at r
    m = m_function(r)
    p = p_function(r)
    nu = nu_function(r)
    # Compute square root of mode frequency
    omega2 = np.square(omega)

    if p < 0:
        epsilon = np.nan
        Gamma1 = np.nan
    else:
        epsilon = rho_function(r)
        Gamma1 = Gamma1_function(r)

    dmdr = 4 * np.pi * r ** 2 * epsilon
    dnudr = 2 * (m + 4 * np.pi * r ** 3 * p) / (r * (r - 2 * m))
    dpdr = -(epsilon + p) * dnudr / 2

    expnu = np.exp(nu)
    explambda = 1 / (1 - 2 * m / r)
    dlambdadr = 2 * explambda * (dmdr / r - m / r ** 2)

    d2nudr2 = (
            2
            / (r * (r - 2 * m))
            * (dmdr + 4 * np.pi * r ** 2 * (3 * p + r * dpdr) + (m + r * dmdr - r) * dnudr)
    )

    # perturbations
    H0 = (
                 8 * np.pi * r ** 3 / expnu ** (1 / 2) * X
                 - (
                         ell * (ell + 1) / 2 * (m + 4 * np.pi * r ** 3 * p)
                         - omega2 * r ** 3 / (explambda * expnu)
                 )
                 * H1
                 + (
                         (ell + 2) * (ell - 1) / 2 * r
                         - omega2 * r ** 3 / expnu
                         - explambda
                         / r
                         * (m + 4 * np.pi * r ** 3 * p)
                         * (3 * m - r + 4 * np.pi * r ** 3 * p)
                 )
                 * K
         ) / (3 * m + (ell + 2) * (ell - 1) / 2 * r + 4 * np.pi * r ** 3 * p)
    V = (
            expnu ** (1 / 2)
            / omega2
            * (
                    1 / (epsilon + p) * X
                    - dnudr / (2 * r) * (expnu / explambda) ** (1 / 2) * W
                    - expnu ** (1 / 2) / 2 * H0
            )
    )

    dH1dr = -(
            ell + 1 + 2 * m * explambda / r + 4 * np.pi * r ** 2 * explambda * (p - epsilon)
    ) / r * H1 + explambda / r * (H0 + K - 16 * np.pi * (epsilon + p) * V)
    dKdr = (
            1 / r * H0
            + ell * (ell + 1) / (2 * r) * H1
            - ((ell + 1) / r - dnudr / 2) * K
            - 8 * np.pi * (epsilon + p) * explambda ** (1 / 2) / r * W
    )
    dWdr = -(ell + 1) / r * W + r * explambda ** (1 / 2) * (
            1 / (Gamma1 * p * expnu ** (1 / 2)) * X
            - ell * (ell + 1) / r ** 2 * V
            + 1 / 2 * H0
            + K
    )
    dXdr = -ell / r * X + (epsilon + p) * expnu ** (1 / 2) * (
            (1 / r - dnudr / 2) / 2 * H0
            + (r * omega2 / expnu + ell * (ell + 1) / (2 * r)) / 2 * H1
            + (3 * dnudr / 2 - 1 / r) / 2 * K
            - ell * (ell + 1) * dnudr / (2 * r ** 2) * V
            - (
                    4 * np.pi * (epsilon + p) * explambda ** (1 / 2)
                    + omega2 * explambda ** (1 / 2) / expnu
                    + (dnudr * (dlambdadr / 2 + 2 / r) - d2nudr2) / (2 * explambda ** (1 / 2))
            )
            / r
            * W
    )

    return [dH1dr, dKdr, dWdr, dXdr]


def taylor_coefficients(
        Kc: complex, Wc: complex, pc: float,r0:float, rho_function: Callable,
        nu_function: Callable,
        Gamma1_function: Callable,
        omega: complex, ell: int = 2
) -> tuple[complex, complex, complex, complex, complex, complex]:
    """
    Calculate coefficients for Taylor expansion near centre.
    :param Kc: K at stellar centre
    :param Wc: W at stellar centre
    :param pc: central pressure
    :param r0: Initial radius near the center
    :param rho_function: energy density function rho(p)
    :param nu_function: metric function nu(r)
    :param Gamma1_function: adiabatic index function nu(r)
    :param omega: angular frequency value
    :param ell: angular momentum number
    :return:
        - H1c
        - d2H1dr2c
        - d2Kdr2c
        - d2Wdr2c
        - Xc
        - d2Xdr2c
    """
    epsilonc = rho_function(pc)
    expnuc = np.exp(nu_function(r0))
    Gamma1c = Gamma1_function(r0)
    Gammac = Gamma1c
    # Compute square root of mode frequency
    omega2 = np.square(omega)

    # coefficients for background quantities
    p2 = -4 * np.pi / 3 * (epsilonc + pc) * (epsilonc + 3 * pc)
    epsilon2 = p2 * (epsilonc + pc) / (Gammac * pc)
    nu2 = 8 * np.pi / 3 * (epsilonc + 3 * pc)

    p4 = -(
            2 * np.pi / 5 * (epsilonc + pc) * (epsilon2 + 5 * p2)
            + 2 * np.pi / 3 * (epsilon2 + p2) * (epsilonc + 3 * pc)
            + 32 * np.pi ** 2 / 9 * epsilonc * (epsilonc + pc) * (epsilonc + 3 * pc)
    )
    nu4 = 4 * np.pi / 5 * (epsilon2 + 5 * p2) + 64 * np.pi ** 2 / 9 * epsilonc * (
            epsilonc + 3 * pc
    )

    # zeroth-order coefficients
    H1c = (2 * ell * Kc + 16 * np.pi * (epsilonc + pc) * Wc) / (ell * (ell + 1))
    Xc = (
            (epsilonc + pc)
            * expnuc ** (1 / 2)
            * (1 / 2 * Kc + (nu2 / 2 - omega2 / (ell * expnuc)) * Wc)
    )

    # solve for second-order coefficients
    Q0 = (
            4
            / ((ell + 2) * (ell - 1))
            * (
                    8 * np.pi / expnuc ** (1 / 2) * Xc
                    - (8 * np.pi / 3 * epsilonc + omega2 / expnuc) * Kc
                    - (2 * np.pi / 3 * ell * (ell + 1) * (epsilonc + 3 * pc) - omega2 / expnuc)
                    * H1c
            )
    )
    Q1 = (
            2
            / (ell * (ell + 1))
            * (
                    1 / (Gamma1c * pc * expnuc ** (1 / 2)) * Xc
                    + 3 / 2 * Kc
                    + 4 * np.pi / 3 * (ell + 1) * epsilonc * Wc
            )
    )

    b = np.array(
        [
            1 / 4 * nu2 / expnuc ** (1 / 2) * Xc
            + 1 / 4 * (epsilon2 + p2) * Kc
            + 1 / 4 * (epsilonc + pc) * Q0
            + 1 / 2 * omega2 * (epsilonc + pc) / expnuc * Q1
            - (
                    p4
                    - 4 * np.pi / 3 * epsilonc * p2
                    + omega2 / (2 * ell) * (epsilon2 + p2 - (epsilonc + pc) * nu2) / expnuc
            )
            * Wc,
            ###
            4 * np.pi / 3 * (epsilonc + 3 * pc) * Kc
            + 1 / 2 * Q0
            - 4
            * np.pi
            * (epsilon2 + p2 + 8 * np.pi / 3 * epsilonc * (epsilonc + pc))
            * Wc,
            ###
            4 * np.pi * (1 / 3 * (2 * ell + 3) * epsilonc - pc) * H1c
            + 8 * np.pi / ell * (epsilon2 + p2) * Wc
            - 8 * np.pi * (epsilonc + pc) * Q1
            + 1 / 2 * Q0,
            ###
            1
            / 2
            * (epsilon2 + p2 + 1 / 2 * (epsilonc + pc) * nu2)
            * ell
            / (epsilonc + pc)
            * Xc
            + (epsilonc + pc)
            * expnuc ** (1 / 2)
            * (
                    1 / 2 * nu2 * Kc
                    + 1 / 4 * Q0
                    + 1 / 2 * omega2 / expnuc * H1c
                    - 1 / 4 * ell * (ell + 1) * nu2 * Q1
                    + (
                            1 / 2 * (ell + 1) * nu4
                            - 2 * np.pi * (epsilon2 + p2)
                            - 16 * np.pi ** 2 / 3 * epsilonc * (epsilonc + pc)
                            + 1 / 2 * (nu4 - 4 * np.pi / 3 * epsilonc * nu2)
                            + 1 / 2 * omega2 / expnuc * (nu2 - 8 * np.pi / 3 * epsilonc)
                    )
                    * Wc
            ),
        ]
    )

    A = np.array(
        [
            [
                0,
                -1 / 4 * (epsilonc + pc),
                1
                / 2
                * (
                        p2
                        + (epsilonc + pc) * omega2 * (ell + 3) / (ell * (ell + 1) * expnuc)
                ),
                1 / (2 * expnuc ** (1 / 2)),
            ],
            ###
            [
                -1 / 4 * ell * (ell + 1),
                1 / 2 * (ell + 2),
                4 * np.pi * (epsilonc + pc),
                0,
            ],
            ###
            [
                1 / 2 * (ell + 3),
                -1,
                -8 * np.pi * (epsilonc + pc) * (ell + 3) / (ell * (ell + 1)),
                0,
            ],
            ###
            [
                -1 / 8 * ell * (ell + 1) * (epsilonc + pc) * expnuc ** (1 / 2),
                0,
                -(epsilonc + pc)
                * expnuc ** (1 / 2)
                * (
                        1 / 4 * (ell + 2) * nu2
                        - 2 * np.pi * (epsilonc + pc)
                        - 1 / 2 * omega2 / expnuc
                ),
                1 / 2 * (ell + 2),
            ],
        ]
    )

    x = np.linalg.solve(A, b)
    d2H1dr2c, d2Kdr2c, d2Wdr2c, d2Xdr2c = x
    return (H1c, d2H1dr2c, d2Kdr2c, d2Wdr2c, Xc, d2Xdr2c)


def solve_perturbations_interior(
        R:float, omega: complex, ell: int = 2
) -> tuple[
    np.typing.NDArray[np.complexfloating], object, object, object, object, object
]:
    """
    Integrate perturbation equations in interior
    :param R: stellar radius
    :param Wc: W at stellar centre
    :param pc: central pressure
    :param r0: Initial radius near the center
    :param rho_function: energy density function rho(p)
    :param nu_function: metric function nu(r)
    :param Gamma1_function: adiabatic index function nu(r)
    :param omega: angular frequency value
    :param ell: angular momentum number
    :return:
        - H1c
        - d2H1dr2c
        - d2Kdr2c
        - d2Wdr2c
        - Xc
        - d2Xdr2c
    """
    # Compute square root of mode frequency
    omega2 = np.square(omega)
    # starting point
    r0 = 1e-4
    # matching point
    rmatch = R / 2

    # from centre
    # Solution 1
    if isinstance(omega2, complex):
        Kc, Wc = 1 + 0j, 0 + 0j
    else:
        Kc, Wc = 1, 0
    H1c, d2H1dr2c, d2Kdr2c, d2Wdr2c, Xc, d2Xdr2c = taylor_coefficients(
        Kc, Wc, background, ell, omega2
    )
    H10 = H1c + 1 / 2 * r0 ** 2 * d2H1dr2c
    K0 = Kc + 1 / 2 * r0 ** 2 * d2Kdr2c
    W0 = Wc + 1 / 2 * r0 ** 2 * d2Wdr2c
    X0 = Xc + 1 / 2 * r0 ** 2 * d2Xdr2c

    sol1 = solve_ivp(
        perturbations_interior,
        [r0, rmatch],
        [H10, K0, W0, X0],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # Solution 2
    if isinstance(omega2, complex):
        Kc, Wc = 0 + 0j, 1 + 0j
    else:
        Kc, Wc = 0, 1
    H1c, d2H1dr2c, d2Kdr2c, d2Wdr2c, Xc, d2Xdr2c = taylor_coefficients(
        Kc, Wc, background, ell, omega2
    )
    H10 = H1c + 1 / 2 * r0 ** 2 * d2H1dr2c
    K0 = Kc + 1 / 2 * r0 ** 2 * d2Kdr2c
    W0 = Wc + 1 / 2 * r0 ** 2 * d2Wdr2c
    X0 = Xc + 1 / 2 * r0 ** 2 * d2Xdr2c

    sol2 = solve_ivp(
        perturbations_interior,
        [r0, rmatch],
        [H10, K0, W0, X0],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # from surface
    # Solution 3
    if isinstance(omega2, complex):
        H1f, Kf, Wf = 1 + 0j, 0 + 0j, 0 + 0j
        Xf = 0 + 0j
    else:
        H1f, Kf, Wf = 1, 0, 0
        Xf = 0

    sol3 = solve_ivp(
        perturbations_interior,
        [R, rmatch],
        [H1f, Kf, Wf, Xf],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # Solution 4
    if isinstance(omega2, complex):
        H1f, Kf, Wf = 0 + 0j, 1 + 0j, 0 + 0j
        Xf = 0 + 0j
    else:
        H1f, Kf, Wf = 0, 1, 0
        Xf = 0

    sol4 = solve_ivp(
        perturbations_interior,
        [R, rmatch],
        [H1f, Kf, Wf, Xf],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # Solution 5
    if isinstance(omega2, complex):
        H1f, Kf, Wf = 0 + 0j, 0 + 0j, 1 + 0j
        Xf = 0 + 0j
    else:
        H1f, Kf, Wf = 0, 0, 1
        Xf = 0

    sol5 = solve_ivp(
        perturbations_interior,
        [R, rmatch],
        [H1f, Kf, Wf, Xf],
        args=(background, ell, omega2),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )

    # solve for coefficients of general solution
    A = np.zeros((5, 5), dtype=type(omega2))
    A[:4, 0] = sol1.y[:, -1]
    A[:4, 1] = sol2.y[:, -1]
    A[:4, 2] = -sol3.y[:, -1]
    A[:4, 3] = -sol4.y[:, -1]
    A[:4, 4] = -sol5.y[:, -1]
    # normalising at surface
    A[4, :] = [0, 0, sol3.y[2, 0], sol4.y[2, 0], sol5.y[2, 0]]
    b = np.array([0, 0, 0, 0, 1])
    x = np.linalg.solve(A, b)

    return (x, sol1, sol2, sol3, sol4, sol5)



def X(
        r: float,
        y: np.typing.NDArray[np.complexfloating],
        m_function: Callable, p_function: Callable,  nu_function: Callable,
        rho_function: Callable,
        omega: complex, ell: int = 2
) -> complex:
    """
    Lagrangian pressure perturbation function in GEO units
    :param radial: Radial coordinate
    :param y: H1, K, W and V at r
    :param m_function: mass function m(r)
    :param p_function: pressure function p(r)
    :param nu_function: metric function nu(r)
    :param rho_function: energy density function rho(p)
    :param omega: angular frequency value
    :param ell: angular momentum number
    :return: X
    """
    H1, K, W, V = y
    # Compute values of m, p, nu at r
    m = m_function(r)
    p = p_function(r)
    nu = nu_function(r)
    # Compute rho(p)
    epsilon = rho_function(p)
    # Compute square root of mode frequency
    omega2 = np.square(omega)

    explambda = 1 / (1 - 2 * m / r)
    expnu = np.exp(nu)

    dlambdadr = (1 - explambda) / r + 8 * np.pi * r * explambda * epsilon
    dnudr = (explambda - 1) / r + 8 * np.pi * r * explambda * p

    n = (ell + 2) * (ell - 1) / 2

    H0 = (
                 r ** 2 / explambda * (omega2 * r / expnu - (n + 1) * dnudr / 2) * H1
                 + (
                         n * r
                         - omega2 * r ** 3 / expnu
                         - r ** 2 * dnudr / (4 * explambda) * (r * dnudr - 2)
                 )
                 * K
                 + 4
                 * np.pi
                 * r ** 2
                 * (epsilon + p)
                 * (dnudr / explambda ** (1 / 2) * W + 2 * r * omega2 / expnu * V)
         ) / ((n + 1) * r - r / (2 * explambda) * (r * dlambdadr + 2))

    return complex(
        (epsilon + p)
        * (
                omega2 / expnu ** (1 / 2) * V
                + dnudr / (2 * r) * (expnu / explambda) ** (1 / 2) * W
                + expnu ** (1 / 2) / 2 * H0
        )
    )
