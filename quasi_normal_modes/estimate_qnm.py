from utilities.physical_data import M_sun, mass_cgs_to_geo

M_sun_geo = mass_cgs_to_geo(M_sun)

def qnm_real_fit(A:float,B:float,M:float,R:float)->complex:
    """
    Linear fit estimations for rial QNM in GEO units
    :param A: parameter
    :param B: parameter
    :param M: total mass
    :param R: total radius
    return: omega rial [KHz]
    """
    compact = M/R
    R_Km = R/1e5
    return 1/R_Km*(A*compact + B)

def qnm_im_fit(a:float,b:float,c:float,M:float,R:float)->complex:
    """
    Linear fit estimations for imag QNM in GEO units
    :param a: parameter
    :param b: parameter
    :param c: parameter
    :param M: total mass
    :param R: total radius
    return: damping time [nuz]
    """
    compact = M / R
    right = 1/(M/M_sun_geo)*(a*compact**2 + b*compact + c)
    return 1e3/right


# Estimate SLy4
R = 1126026.005975641
M = 205264.03991190076

print("omega real en KHz: ",qnm_real_fit(-148.7,119.8,M,R))
print("damping time: ",qnm_im_fit(-1221,365.1,21.63,M,R))