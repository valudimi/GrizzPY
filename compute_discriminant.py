import numpy as np

def compute_discriminant(X, miu, sinv, slogdet, prior):
    """
    Computes a discriminant score for the given samples and parameters.

    Parameters:
    - X: 2D array of shape (N, D) containing the test samples.
    - miu: 2D array of shape (nr_groups, D) containing the mean vectors for each group.
    - sinv: 3D array of shape (D, D, nr_groups) or 2D array of shape (D, D) if the same
      covariance is used for all groups.
    - slogdet: 1D array of length nr_groups having the log-determinant of the covariance
      matrices, or empty if not used.
    - prior: 1D array of length nr_groups having the prior probability of each group, or
      empty for equal probability.

    Returns:
    - d: 1D array of length nr_groups with the discriminant scores for each group.
    """

    # Check and initialize parameters
    # In MATLAB:
    # size of X = (1 (sometimes 2), 4), shape of miu = (256, 4)
    # For us, it's a bit different:
    # shape of X = (1, 4), shape of miu = (256, 4)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N, D = X.shape
    nr_groups = miu.shape[0]

    if miu.shape[1] != D:
        raise ValueError('Incompatible size of miu with X')

    if sinv is not None:
        if sinv.ndim == 3 and (sinv.shape[0] != D or sinv.shape[1] != D):
            raise ValueError('Bad dimension sinv')
        if sinv.ndim == 3 and sinv.shape[2] != nr_groups:
            raise ValueError('Incompatible third dimension for sinv')

    if slogdet is not None:
        slogdet = np.asarray(slogdet).flatten()

    if sinv is not None and sinv.ndim == 3 and (slogdet is None or slogdet.size != nr_groups):
        raise ValueError('Incorrect vector slogdet')

    if prior is not None:
        prior = np.asarray(prior).flatten()
        if prior.size != nr_groups:
            raise ValueError('Incorrect length of prior')

    d = np.zeros(nr_groups)

    # Compute discriminant from given parameters
    if sinv is not None and sinv.ndim == 3:
        # Quadratic discriminant when sinv is given for each group
        # print('am ajuns aici 1')
        ct1 = -N / 2
        ct2 = -1 / 2
        for k in range(nr_groups):
            dsum = 0
            for j in range(N):
                x = X[j, :] - miu[k, :]
                dsum += x @ sinv[:, :, k] @ x.T
            d[k] = ct1 * slogdet[k] + ct2 * dsum

    elif sinv is not None and sinv.ndim == 2:
        # Linear discriminant using common covariance
        # print('am ajuns aici 2')
        ct = -N / 2
        xs = np.ones(N) @ X
        for k in range(nr_groups):
            d[k] = miu[k, :] @ sinv @ xs.T + ct * (miu[k, :] @ sinv @ miu[k, :].T)

    elif sinv is None or not sinv:
        # Linear discriminant using identity covariance
        # print('am ajuns aici 3')
        ct = -N / 2
        xs = np.ones(N) @ X
        for k in range(nr_groups):
            d[k] = miu[k, :] @ xs.T + ct * (miu[k, :] @ miu[k, :].T)
            # print(f'd[{k}] = {d[k]}')

    # Add priors to each discriminant if given
    if prior is not None:
        # print('am ajuns aici 4')
        d += N * np.log(prior)
    # print(f'sinv in compute_discriminant: {sinv}')
    return d
