import numpy as np

def evaluate_discriminant(data, groups, miu, sinv, slogdet, prior):
    """
    Evaluates a discriminant score for the given samples and parameters.
    
    Parameters:
    - data: 2D array of shape (nr_samples, nr_points) containing the test samples.
    - groups: 1D array of length nr_test_groups specifying the indices for miu for which the discriminant will be computed.
    - miu: 2D array of shape (nr_groups, nr_points) containing the mean vectors for all groups.
    - sinv: 3D array of shape (nr_points, nr_points, nr_groups) or 2D array of shape (nr_points, nr_points) if the same covariance is used for all groups.
    - slogdet: 1D array of length nr_groups having the log-determinant of the covariance matrices.
    - prior: 1D array of length nr_groups having the prior probability of each group. If equal probability is to be used, pass None or an empty array.
    
    Returns:
    - d: 1D array of shape (nr_test_groups,) with the discriminant scores of the data for each group specified by the groups vector.
    """
    
    # Ensure groups is a 1D array
    groups = np.asarray(groups).flatten()
    
    # Number of slices in sinv
    ns = sinv.shape[2] if sinv.ndim == 3 else 1
    
    # Compute the output
    if ns > 1:
        d = compute_discriminant(data, miu[groups, :], sinv[:, :, groups], slogdet[groups], prior)
    else:
        d = compute_discriminant(data, miu[groups, :], sinv, slogdet, prior)
    
    return d

def compute_discriminant(X, miu, sinv, slogdet, prior):
    """
    Computes a discriminant score.
    
    Parameters:
    - X: 2D array of shape (nr_samples, nr_points) containing the test samples.
    - miu: 2D array of shape (nr_groups, nr_points) containing the mean vectors for all groups.
    - sinv: 3D array of shape (nr_points, nr_points, nr_groups) or 2D array of shape (nr_points, nr_points) if the same covariance is used for all groups.
    - slogdet: 1D array of length nr_groups having the log-determinant of the covariance matrices.
    - prior: 1D array of length nr_groups having the prior probability of each group. If equal probability is to be used, pass None or an empty array.
    
    Returns:
    - d: 1D array of shape (nr_groups,) with the discriminant scores.
    """
    
    # Ensure correct input dimensions
    if miu.shape[1] != X.shape[1]:
        raise ValueError('Incompatible size of miu with X')
    if sinv is not None:
        if sinv.ndim == 3 and sinv.shape[2] != miu.shape[0]:
            raise ValueError('Incompatible third dimension for sinv')
        if sinv.ndim == 2 and sinv.shape[0] != X.shape[1]:
            raise ValueError('Bad dimension sinv')
    if slogdet is not None:
        slogdet = slogdet.flatten()
        if sinv.ndim == 3 and slogdet.size != miu.shape[0]:
            raise ValueError('Incorrect vector slogdet')
    if prior is not None:
        prior = prior.flatten()
        if prior.size != miu.shape[0]:
            raise ValueError('Incorrect length of prior')

    # Initialize discriminant scores
    nr_samples = X.shape[0]
    nr_groups = miu.shape[0]
    d = np.zeros(nr_groups)

    # Compute discriminant from given parameters
    if sinv is not None and sinv.ndim == 3:
        # Quadratic discriminant when sinv is given for each group
        ct1 = -nr_samples / 2
        ct2 = -1 / 2
        for k in range(nr_groups):
            dsum = 0
            for j in range(nr_samples):
                x = X[j, :] - miu[k, :]
                dsum += x @ sinv[:, :, k] @ x.T
            d[k] = ct1 * slogdet[k] + ct2 * dsum
    elif sinv is not None and sinv.ndim == 2:
        # Linear discriminant using common covariance
        ct = -nr_samples / 2
        xs = X.sum(axis=0)
        for k in range(nr_groups):
            d[k] = miu[k, :] @ sinv @ xs + ct * (miu[k, :] @ sinv @ miu[k, :].T)
    else:
        # Linear discriminant using identity covariance
        ct = -nr_samples / 2
        xs = X.sum(axis=0)
        for k in range(nr_groups):
            d[k] = miu[k, :] @ xs + ct * (miu[k, :] @ miu[k, :].T)

    # Add priors to each discriminant if given
    if prior is not None:
        d += nr_samples * np.log(prior)

    return d

# Example usage
# Assuming data, groups, miu, sinv, slogdet, and prior are defined
# d = evaluate_discriminant(data, groups, miu, sinv, slogdet, prior)
