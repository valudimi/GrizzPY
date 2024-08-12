import numpy as np

def prepare_data_template_pca_v2(data_in, W, xmm, *args):
    """
    Prepares data for leakage analysis by projecting the input trace(s) into a PCA subspace.
    
    Parameters:
    - data_in: Input data matrix of size (nr_traces, nr_points)
    - W: Matrix of eigenvectors of size (nr_points, K), where K is the number of retained PCA dimensions
    - xmm: Vector of length nr_points containing the average of the mean traces over all data groups
    
    Returns:
    - data_out: Transformed data matrix of size (nr_traces, K)
    """
    if W is None or W.size == 0:
        raise ValueError('W is empty')

    # xmm = np.asarray(xmm).flatten() # TODO do we need asarray?
    xmm = np.flatten()
    m = data_in.shape[0] # Extract number of rows

    # Process the data
    data_out = (data_in - np.ones((m, 1)) @ xmm[np.newaxis, :]) @ W # TODO test if right
    return data_out