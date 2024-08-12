import numpy as np

def compute_template(X, mle=None):
    """
    Compute template parameters.
    
    Parameters:
    - X: 3-dimensional array of size (nr_samples, nr_interest_points, nr_groups)
         containing the data for which templates should be computed.
    - mle: Optional parameter. If non-zero, the maximum likelihood estimate is used
           for the covariance matrix (divides by nr_samples). If zero or None, the 
           unbiased estimator is used (divides by nr_samples - 1).
    
    Returns:
    - tmiu: A 2-dimensional array of size (nr_groups, nr_interest_points) containing the
            mean vector of each group.
    - tsigma: A 3-dimensional array of size (nr_interest_points, nr_interest_points, nr_groups)
              having the covariance matrix of each group.
    """
    
    # Initialize and check parameters
    nr_samples, nr_interest_points, nr_groups = X.shape
    tmiu = np.zeros((nr_groups, nr_interest_points))
    tsigma = np.zeros((nr_interest_points, nr_interest_points, nr_groups))
    mct = 1 / nr_samples
    
    if mle is None or mle == 0:
        sct = 1 / (nr_samples - 1)
    else:
        sct = mct
    
    # Compute the templates for each group
    for k in range(nr_groups):
        x = X[:, :, k]
        tmiu[k, :] = mct * np.ones(nr_samples) @ x
        xm = x - np.ones((nr_samples, 1)) @ tmiu[k, :][np.newaxis, :]
        tsigma[:, :, k] = sct * (xm.T @ xm)
    
    return tmiu, tsigma