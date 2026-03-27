"""
Equation of State (EoS) Data Extraction Utilities
=================================================

This module provides simple helper functions to extract physical
quantities from tabulated Equation of State (EoS) text files.

The expected file format is a whitespace-separated table with columns:

    Column 0 : Row index (integer, optional)
    Column 1 : Baryonic number density  n_B  [fm^-3]
    Column 2 : Mass density  ρ  [g cm^-3]
    Column 3 : Pressure  p  [dyn cm^-2]

Commented lines (starting with '#', by default) are ignored.

These functions are intended to provide clean NumPy arrays for further
processing, interpolation, or construction of pressure–density relations
used in TOV integrations.
"""
# =====================
# Imports
# =====================
from numpy import loadtxt, ndarray
from pathlib import Path
from typing import Union

def extract_mass_density_from_eos_txt(path: Union[str, Path], comments: str = "#") -> ndarray:
    """
    :param path: str or Path to the EoS text file
    :param comments: Character marking comment lines to ignore (default: "#")
    :return:
        - mass_density: One-dimensional NumPy array containing the mass density ρ [g cm^-3]
    """
    # Load numerical data from the text file, ignoring commented lines
    np_data = loadtxt(path, comments=comments)
    # Column 2 corresponds to mass density ρ
    mass_density = np_data[:, 2]
    return mass_density

def extract_pressure_from_eos_txt(path: Union[str, Path], comments: str = "#") -> ndarray:
    """
    :param path: str or Path to the EoS text file
    :param comments: Character marking comment lines to ignore (default: "#")
    :return:
        - pressure: One-dimensional NumPy array containing the pressure values p [dyn cm^-2]
    """
    # Load numerical data from the text file, ignoring commented lines
    np_data = loadtxt(path, comments=comments)
    # Column 3 corresponds to pressure p
    pressure = np_data[:, 3]
    return pressure


def extract_barionic_density_from_eos_txt(path: Union[str, Path], comments: str = "#") -> ndarray:
    """
    :param path: str or Path to the EoS text file
    :param comments: Character marking comment lines to ignore (default: "#")
    :return:
        - pressure: One-dimensional NumPy array containing the barionic density values nB [fm^-3]
    """
    # Load numerical data from the text file, ignoring commented lines
    np_data = loadtxt(path, comments=comments)
    # Column 3 corresponds to pressure p
    pressure = np_data[:, 3]
    return pressure