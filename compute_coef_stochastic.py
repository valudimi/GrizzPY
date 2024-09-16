""" 
Computes the stochastic coefficients beta for a stochastic model based on
the given leakage, input data, and mappings.

Parameters:
- X: A numpy array of shape (nr_traces, nr_points) containing the leakage
    data on some set of points.
- D: A numpy array of length nr_traces, containing the data corresponding to X.
- map_base: A list of functions. If the same mapping bases should be used for
            all points in the data, this should be a list of size u. If different
            mapping bases should be used for each point, it should be a list of
            lists of size (u, nr_points). Each element in the list should be a
            function that takes a value v and returns the mapped value.

Returns:
- coef: A numpy array of shape (u, nr_points) that should be used with the given
        mapping bases in order to model the given leakage using the stochastic approach.
"""

import numpy as np

def compute_coef_stochastic(X, D, map_base):
    """
    Parameters:
    - X: numpy array of shape (nr_traces, nr_points) containing leakage data.
    - D: numpy array of length nr_traces, containing the data corresponding to X.
    - map_base: list of functions of size u or (u, nr_points). Takes value v and returns mapping.

    Returns:
    - coef: numpy array of shape (u, nr_points) containing coefficients for the stochastic model.
    """

    nr_traces, nr_points = X.shape
    D = D.flatten()
    if len(D) != nr_traces:
        raise ValueError('Wrong size of xdata')

    u = len(map_base)
    coef = np.zeros((u, nr_points))

    # print(map_base)

    if isinstance(map_base, list) or map_base.shape[1] == 1:
        # Compute coefficients using same base for all points (fast)
        A = np.zeros((nr_traces, u))
        for k in range(nr_traces):
            for i in range(u):
                # print(i)
                # print(D[k])
                A[k, i] = map_base[i](D[k])
        M = np.linalg.pinv(A)  # A' * A \ A'
        coef = M @ X
    elif map_base.shape[1] != nr_points:
        raise ValueError('Wrong size of map_base')
    else:
        # Compute coefficients for each point using individual bases
        for j in range(nr_points):
            print(f'Computing coefficients for j={j + 1}')
            x = X[:, j]
            A = np.zeros((nr_traces, u))
            for k in range(nr_traces):
                for i in range(u):
                    A[k, i] = map_base[i][j](D[k])
            coef[:, j] = np.linalg.lstsq(A, x, rcond=None)[0] # ((A'*A)\A') * x

    # if isinstance(map_base[0], list) and len(map_base[0]) == nr_points:
    #     # Compute coefficients for each point using individual bases
    #     for j in range(nr_points):
    #         print(f'Computing coefficients for j={j + 1}')
    #         x = X[:, j]
    #         A = np.zeros((nr_traces, u))
    #         for k in range(nr_traces):
    #             for i in range(u):
    #                 A[k, i] = map_base[i][j](D[k])
    #         coef[:, j] = np.linalg.lstsq(A, x, rcond=None)[0]
    # else:
    #     # Compute coefficients using same base for all points (fast)
    #     A = np.zeros((nr_traces, u))
    #     for k in range(nr_traces):
    #         for i in range(u):
    #             A[k, i] = map_base[i](D[k])
    #     M = np.linalg.pinv(A)  # A' * A \ A'
    #     coef = M @ X

    return coef
