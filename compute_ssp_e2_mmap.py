import numpy as np

def compute_ssp_e2_mmap(map, metadata, bindex, bytes, pts):
    nr_groups = metadata['nr_groups']
    nr_bytes = len(bytes)
    if min(bytes) < 0 or max(bytes) > (nr_groups - 1):
        raise ValueError('Some index in bytes is outside available groups')

    if pts is None:
        nr_points = metadata['nr_points']
        pts = np.arrange(nr_points)
    else:
        pts = np.array(pts).flatten()
        nr_points = len(pts)

    M = np.zeros((nr_groups, nr_points))
    B = np.zeros((nr_points, nr_points))
    W = np.zeros((nr_points, nr_points))
    nr_samples_per_group = len(bindex)

    # Compute the group means and the residual matrix W
    print('Computing group means and residual matrix W...\n')


    for i in range(nr_bytes):
        kindex = np.where(map.data[1]['B'][1, :] == bytes[i])[0]
        lindex = kindex[bindex]
        L = map.data[1]['X'][pts, :][:, lindex].T.astype(np.float64)
        M[i, :] = np.mean(L, axis=0)
        X = L - np.ones((nr_samples_per_group, 1)) * M[i, :]
        W += np.dot(X.T, X)

    mvec = np.mean(M, axis=0)

    print('Computing the treatment matrix B...')
    for k in range(nr_bytes):
        xm = M[k, :] - mvec
        B += np.outer(xm, xm)
    
    B *= nr_samples_per_group

    return M, B, W
