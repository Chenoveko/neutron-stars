# =====================
# Imports
# =====================
from numpy import loadtxt, ndarray

"""
Extract pressure data from an Equation of State (EoS) text file
-----------------------------------------
Units: 
    Column 0 : Row index (integer, optional)
    Column 1 : Barionic number density n_B [fm^-3]
    Column 2 : Mass density rho [g/cm^3]
    Column 3 : Pressure p [dyn/cm^2]
"""


def extract_mass_density_from_eos_txt(path: str, comments: str = "#") -> ndarray:
    # Load numerical data from the text file, ignoring commented lines
    np_data = loadtxt(path, comments=comments)
    # Extract the mass density column (slicing)
    mass_density = np_data[:, 2]
    return mass_density


def extract_pressure_from_eos_txt(path: str, comments: str = "#") -> ndarray:
    # Load numerical data from the text file, ignoring commented lines
    np_data = loadtxt(path, comments=comments)
    # Extract the pressure column (slicing)
    pressure = np_data[:, 3]
    return pressure
