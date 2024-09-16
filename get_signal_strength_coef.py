"""
Returns signal strength estimates for stochastic attacks, based on
the leakage matrix X and estimated base coefficients coef.

Parameters:
X (numpy.ndarray): The leakage matrix of size (nr_traces, nr_points).
coef (numpy.ndarray): The matrix of base coefficients for each point, 
                        of size (nr_bases, nr_points).
base (str): A string specifying which base was used to compute the 
            coefficients. Supported bases are:
            - 'F9': constant power consumption plus individual 8 bits.
            - 'F17': constant power consumption plus individual 16 bits.
            - 'F17xor'
            - 'F17tran'

Returns:
dict: A dictionary with the following data:
        - 'bnorm': The signal strength estimate based on the squared 
                    Euclidean norm of the base vectors.
        - 'bnorm_std': Uses the norm of base vectors and variance of each 
                        sample point.
"""
import numpy as np

def get_signal_strength_coef(X, coef, base):
    """
    - X: numpy.ndarray that is the leakage matrix of size (nr_traces, nr_points).
    - coef: numpy.ndarray matrix of base coefficients for each point, size (nr_bases, nr_points).
    - base: string telling which base was used to compute coefficients (F9, F17, F17xor, F17tran).

    Returns:
    - dict: A dictionary with the following data:
        - 'bnorm': The signal strength estimate based on the squared 
                   Euclidean norm of the base vectors.
        - 'bnorm_std': Uses the norm of base vectors and variance of each sample point.
    """
    nr_points = X.shape[1]

    if base in ['F9', 'F17', 'F17xor', 'F17tran']:
        # Compute bnorm
        bnorm = np.zeros(nr_points)
        for j in range(nr_points):
            bnorm[j] = np.dot(coef[1:, j], coef[1:, j])

        # Compute bnorm_std
        vx = np.var(X, axis=0)
        bnorm_std = np.zeros(nr_points)
        for j in range(nr_points):
            bnorm_std[j] = bnorm[j] / vx[j]

        return {'bnorm': bnorm, 'bnorm_std': bnorm_std}

    raise ValueError(f'Unsupported base: {base}')
