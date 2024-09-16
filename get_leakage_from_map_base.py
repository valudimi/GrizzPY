"""
Returns leakage traces using base mappings.

Parameters:
- D: A numpy array of length nr_traces, containing input data for which to return corresponding
        leakage traces.
- coef: A numpy array of shape (nr_bases, nr_points) that will be used together with map_base
        to produce traces of length nr_points.
- map_base: A list of functions. If the same mapping bases should be used for all points,
            this should be a list of size u. If different mapping bases should be used for each point,
            it should be a list of lists of size (u, nr_points). Each element in the list should be a
            function that takes a value v and returns the mapped value.

Returns:
- X: A numpy array of shape (nr_traces, nr_points), having nr_traces of nr_points each.
"""

import numpy as np

def get_leakage_from_map_base(D, coef, map_base):
    """
    - D: numpy array of length nr_traces containing input data to return corresponding traces.
    - coef: numpy array of shape (nr_bases, nr_points) used with map_base to produce traces.
    - map_base: list of functions of size u or (u, nr_points). Takes value v and returns mapping.

    Returns:
    - X: numpy array of shape (nr_traces, nr_points), having nr_traces of nr_points each.
    """

    nr_traces = len(D)
    nr_bases, nr_points = coef.shape

    if len(map_base) != nr_bases:
        raise ValueError('Incompatible sizes between coef and map_base')

    if isinstance(map_base, list) or map_base.shape[1] == 1:
        V = np.zeros((nr_traces, nr_bases))
        for i in range(nr_traces):
            for k in range(nr_bases):
                V[i, k] = map_base[k](D[i])
        X = V @ coef
    elif map_base.shape[1] != nr_bases:
        raise ValueError('Wrong size of map_base')
    else:
        X = np.zeros((nr_traces, nr_points))
        for i in range(nr_traces):
            for j in range(nr_points):
                v = np.zeros(1, nr_bases)
                for k in range(nr_bases):
                    v[k] = map_base[k][j](D[i])
                X[i, j] = np.dot(v, coef[:, j])

    # if isinstance(map_base[0], list) and len(map_base[0]) == nr_points:
    #     # Compute using individual bases (slower)
    #     X = np.zeros((nr_traces, nr_points))
    #     for i in range(nr_traces):
    #         for j in range(nr_points):
    #             v = np.array([map_base[k][j](D[i]) for k in range(nr_bases)])
    #             X[i, j] = np.dot(v, coef[:, j])
    # else:
    #     # Compute using same base (faster)
    #     V = np.zeros((nr_traces, nr_bases))
    #     for i in range(nr_traces):
    #         for k in range(nr_bases):
    #             V[i, k] = map_base[k](D[i])
    #     X = np.dot(V, coef)

    return X
