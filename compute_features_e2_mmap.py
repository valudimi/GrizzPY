import numpy as np

def compute_features_e2_mmap(mmap_data, metadata, bindex, func_prepare, pp1, pp2=None, pp3=None, pp4=None, pp5=None, inbytes=None):
    """
    Computes features from sample vectors.
    
    Parameters:
    - mmap_data: Memory-mapped object containing the data.
    - metadata: Metadata object with information about the data.
    - bindex: Vector of indices to select block samples.
    - func_prepare: Function to transform the data (e.g., compress it).
    - pp1, pp2, pp3, pp4, pp5: Parameters for func_prepare.
    - inbytes: Optional vector specifying indices of bytes to compute features for.
    
    Returns:
    - xdata: Matrix of size (nr_rows, nr_features, nr_groups) containing the computed features.
    """
    if inbytes is not None and len(inbytes) > 0:
        nr_groups = len(inbytes)
    else:
        nr_groups = metadata['nr_groups']
        inbytes = np.arange(nr_groups)

    # Extract data from each group leakage matrix
    for k in range(nr_groups):
        kindex = np.where(mmap_data['B'][1, :] == inbytes[k])[0]
        lindex = kindex[bindex]
        # Below note transpose and conversion in case we had integer class
        L = mmap_data['X'][:, lindex].T.astype(float)
        
        # Apply the func_prepare function
        data_out = func_prepare(L, pp1, pp2, pp3, pp4, pp5)
        
        if k == 0:
            nr_traces, nr_features = data_out.shape
            xdata = np.zeros((nr_traces, nr_features, nr_groups))
        
        xdata[:, :, k] = data_out
    
    return xdata