import numpy as np

def get_signal_strength_ssp(M, B, W, nr_traces):
    """
    Returns signal strength estimates based on the matrices M, B, and W.
    
    Parameters:
    - M: Group means matrix of size (nr_groups, nr_points)
    - B: Treatment sum of squares and crossproducts matrix of size (nr_points, nr_points). Can be None.
    - W: Residual sum of squares and crossproducts matrix of size (nr_points, nr_points)
    - nr_traces: Number of traces per group that were used to compute M, B, and W
    
    Returns:
    - curves: Dictionary with signal strength estimates ('dom', 'sosd', 'snr', 'sost', 'ftest')
    """
    nr_groups, nr_points = M.shape

    if B is not None and (B.shape[0] != nr_points or B.shape[1] != nr_points):
        raise ValueError('Incorrect size of B')
    if W.shape[0] != nr_points or W.shape[1] != nr_points:
        raise ValueError('Incorrect size of W')

    curves = {}

    # Compute the difference of means curve (DOM)
    dom = np.zeros(nr_points)
    for i in range(nr_groups - 1):
        for j in range(i + 1, nr_groups):
            ds = M[i, :] - M[j, :]
            dom += np.abs(ds)
    curves['dom'] = dom

    # Compute the difference of means squared (SOSD)
    sosd = np.zeros(nr_points)
    for i in range(nr_groups - 1):
        for j in range(i + 1, nr_groups):
            ds = (M[i, :] - M[j, :]) ** 2
            sosd += ds
    curves['sosd'] = sosd

    # Compute the SNR curve
    V = np.diag(W) / (nr_groups * (nr_traces - 1))
    M_var = np.var(M, axis=0, ddof=0)
    snr = M_var / V
    curves['snr'] = snr

    # Compute the SOST curve
    sost = np.zeros(nr_points)
    for i in range(nr_groups - 1):
        for j in range(i + 1, nr_groups):
            sost += ((M[i, :] - M[j, :]) ** 2) / V
    curves['sost'] = sost

    # Compute the F-test curve
    if B is not None:
        ftest = np.zeros(nr_points)
        for k in range(nr_points):
            ftest[k] = (B[k, k] / (nr_groups - 1)) / (W[k, k] / (nr_groups * (nr_traces - 1)))
        curves['ftest'] = ftest

    return curves
