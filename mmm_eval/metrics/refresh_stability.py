# This file contains the functions for calculating the refresh stability of the MMM framework.

import numpy as np


def calculate_refresh_stability(current_coefficients: np.ndarray, refreshed_coefficients: np.ndarray) -> float:
    """
    Calculate the refresh stability of the MMM framework.
    """
    return np.abs(current_coefficients - refreshed_coefficients)

def calculate_coefficents_have_flipped_signs(current_coefficients: np.ndarray, refreshed_coefficients: np.ndarray) -> bool:
    """
    Calculate if the coefficients have flipped signs.
    """
    return np.any(np.sign(current_coefficients) != np.sign(refreshed_coefficients))