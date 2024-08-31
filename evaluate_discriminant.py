import numpy as np
from compute_discriminant import compute_discriminant

def evaluate_discriminant(data, groups, miu, sinv, slogdet, prior):
    """
    Evaluates a discriminant score for the given samples and parameters.
    
    Parameters:
    - data: 2D array of shape (nr_samples, nr_points) containing the test samples.
    - groups: 1D array of length nr_test_groups specifying the indices for miu for which the
      discriminant will be computed.
    - miu: 2D array of shape (nr_groups, nr_points) containing the mean vectors for all groups.
    - sinv: 3D array of shape (nr_points, nr_points, nr_groups) or 2D array of shape (nr_points,
      nr_points) if the same covariance is used for all groups.
    - slogdet: 1D array of length nr_groups having the log-determinant of the covariance matrices.
    - prior: 1D array of length nr_groups having the prior probability of each group. If equal
      probability is to be used, pass None or an empty array.
    
    Returns:
    - d: 1D array of shape (nr_test_groups,) with the discriminant scores of the data for each
      group specified by the groups vector.
    """

    # Ensure groups is a 1D array
    groups = groups - 1 # convert to 0 based index
    # print(f'miu: {miu}')
    # print(f'sinv: {sinv}')
    # print(f'slogdet: {slogdet}')
    # print(f'prior: {prior}')

    # Number of slices in sinv
    if sinv is not None or not sinv:
        sinv = np.array(sinv)
        ns = sinv.shape[2] if sinv.ndim == 3 else 1
    else:
        ns = 1
        sinv = None

    if slogdet:
        slogdet = np.array(slogdet)
    else:
        slogdet = None
    if prior:
        prior = np.array(prior)
    else:
        prior = None

    # Compute the output
    if ns > 1:
        d = compute_discriminant(data, miu[groups, :], sinv[:, :, groups], slogdet[groups], prior)
    else:
        # print(f'sinv: {sinv}\nslogdet: {slogdet}\nprior: {prior}\n\n')
        # print(f'miu[groups, :]: {miu[groups, :]}\n\n\n')
        d = compute_discriminant(data, miu[groups, :], sinv, slogdet, prior)
    
    return d

# data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# groups = np.array([1, 2])
# miu = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# sinv = []
# slogdet = None
# prior = None
# d = evaluate_discriminant(data, groups, miu, sinv, slogdet, prior)