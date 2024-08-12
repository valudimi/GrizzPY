import numpy as np
from scipy.linalg import eig, svd

def compute_params_lda(B, S, nr_groups=None, threshold=None, use_eig=False):
    """
    Computes Fisher's LDA parameters.
    
    Parameters:
    - B: Between-class scatter matrix
    - S: Pooled within-class scatter matrix
    - nr_groups: Number of groups/classes used to compute B and S (optional)
    - threshold: Minimum threshold of the total variance (optional)
    - use_eig: Boolean to specify if actual eigenvalues should be used instead of singular values
    
    Returns:
    - A: Eigenvector matrix
    - D: Diagonal matrix containing the eigenvalues
    - K: Number of components needed to reach the specified threshold of the total variance (optional)
    """
    nr_points = B.shape[0]
    
    if B.shape[1] != nr_points:
        raise ValueError('B is not square')
    if B.shape != S.shape:
        raise ValueError('Size of B is different from size of S')
    
    # Compute the eigenvalues
    if use_eig:
        D, A = eig(np.linalg.inv(S).dot(B))
        print(f'Eigenvalues:\n{D}\n\nEigenvector matrix:\n{A}\n\n')
    else:
        A, D, _ = svd(np.linalg.inv(S).dot(B))
        print(f'Eigenvalues:\n{D}\n\nEigenvector matrix:\n{A}\n\n')
    
    D = np.diag(D) # Convert eigenvalues to diagonal form (TODO: check if this is necessary; see prints above)

    # Scale eigenvalues to have e'Se = 1 for each e
    Q = A.T.dot(S).dot(A)
    Z = np.diag(1.0 / np.sqrt(np.diag(Q)))
    A = A.dot(Z)
    print(f'Q should be a diagonal matrix if the eigenvectors are correctly scaled
            :\n{Q}\n\n')
    
    # Return K if needed
    K = None
    if nr_groups is not None and threshold is not None:
        for k in range(1, nr_groups + 1):
            f = np.sum(D[:k]) / np.sum(D)
            if f >= threshold:
                K = k
                break
    
    if K is not None:
        return A, D, K
    else:
        return A, D

# Example usage
# Assuming B, S, nr_groups, and threshold are already defined
# A, D, K = compute_params_lda(B, S, nr_groups, threshold)
