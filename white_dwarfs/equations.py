def newton_equations(var: ndarray, r: float, rho: float) -> ndarray:
    """
    Newton equations in GEO units (white dwarfs)
    :param var: state variables
    :param r: radial coordinate
    :param rho: mass density
    :return: derivatives
    """
    # Unpack variables
    p, m = var
    # Interior pressure equation
    dp_dr = - m * rho / (r ** 2)
    # Enclosed mass equation
    dm_dr = 4.0 * pi * (r ** 2) * rho
    return array([dp_dr, dm_dr], float)
