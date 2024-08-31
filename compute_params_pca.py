import numpy as np

def compute_params_pca(M, threshold=0.95, alternate=False):
    """
    Computes PCA parameters.

    Parameters:
    - M: 2D array of shape (nr_groups, nr_trace_points), containing the precomputed mean 
      trace values for each group.
    - threshold: Float, specifying the threshold used to select the first K dimensions for
      dimensionality reduction using the cumulative percentage of total variation. Default is 0.95.
    - alternate: Boolean, if True, uses the alternative method proposed by Standaert et al. Useful
      if the first dimension is small.

    Returns:
    - W: 2D array, the projection matrix containing either all the eigenvectors (if K is returned),
      or just the first K eigenvectors (principal directions) that retain the specified variance.
    - D: 1D array, the eigenvalues corresponding to each eigenvector.
    - xmm: 1D array, the average of all the mean traces, used to normalize input data before
      projection.
    - K: Integer, the number of principal components that retain the specified variance.
    """

    nr_groups, _ = M.shape

    # Compute the average of all the mean traces
    xmm = np.mean(M, axis=0)
    X = M - xmm

    if alternate:
        # Standaert's variant for large data
        UU, S, _ = np.linalg.svd((1 / nr_groups) * (X @ X.T))
        SIQ = np.linalg.pinv(np.sqrt(S))
        U = (1 / np.sqrt(nr_groups)) * (X.T @ UU) @ SIQ
    else:
        # Direct SVD
        U, S, _ = np.linalg.svd((1 / nr_groups) * (X.T @ X))

    # Store the eigenvalues in D
    D = S

    # Find the principal directions that retain the specified threshold of variance
    cumulative_variance = np.cumsum(D) / np.sum(D)
    K = np.searchsorted(cumulative_variance, threshold) + 1

    # Determine if we return all eigenvectors or just the first K
    if 'K' in locals():
        W = U
    else:
        W = U[:, :K]

    return W, D, xmm, K
