import numpy as np

def compute_ssp_e2_mmap_multi(s_data, inbytes, pts=[], roffset=[]):
    nr_sets = s_data['nr_sets']
    # print(f's_data = {s_data}')
    nr_groups = s_data['metadata'][0]['nr_groups']

    # Check if all sets have the same number of groups
    for k in range(1, nr_sets):
        # In the original code, it was [1] instead of [k]
        if s_data['metadata'][k]['nr_groups'] != nr_groups:
            raise ValueError(f'Set {k+1} has a different number of groups than set 1')

    nr_bytes = len(inbytes)

    # Validate bytes
    if min(inbytes) < 0 or max(inbytes) > (nr_groups - 1):
        raise ValueError('Some index in bytes is outside available groups')

    # Determine the number of points
    # print(f'pts = {pts}')
    # if pts is None or not pts:
    if not pts: # TODO determine whether None or not
        nr_points = s_data['metadata'][0]['nr_points']
        pts = np.arange(nr_points)
    else:
        pts = np.array(pts).flatten() # TODO is flatten necessary?
        nr_points = len(pts)
        print(f'flattened pts = {pts}')

    # Initialize matrices
    M = np.zeros((nr_bytes, nr_points))
    B = np.zeros((nr_points, nr_points))
    W = np.zeros((nr_points, nr_points))
    np_total = 0

    # Calculate total number of points
    for k in range(nr_sets):
        np_total += len(s_data['idx'][k])

    print('Running compute_ssp_e2_mmap_multi()...')

    # Compute the group means and the residual matrix W
    print('Computing the mean vectors M and the residual matrix W...')
    mmap_data = s_data['mmap_data']
    # print(f'length of mmap_data = {len(mmap_data)}')
    # print(mmap_data)

    # if condition to check if there is only one set; this massively cuts
    # down runtime if only one set is used
    if len(mmap_data) == 1:
        X_map = mmap_data[0]['X'][pts, :]
        B_map = mmap_data[0]['B'][1, :]
        i = s_data['idx'][0]

        for k in range(nr_bytes):
            # print('am intrat in primul for')
            kindex = np.where(B_map == inbytes[k])[0]
            lindex = kindex[i]
            L = X_map[:, lindex].T
            if roffset:
                print('am intrat in roffset')
                L += np.outer(roffset[0][:, inbytes[k]], np.ones(nr_points))

            M[k, :] = np.mean(L, axis=0)
            X = L - M[k, :]
            W += np.dot(X.T, X)

    else:
        for k in range(nr_bytes):
            print('am intrat in al doilea for')
            idx = 0
            L = np.zeros((np_total, nr_points))

            for i in range(nr_sets):
                print(i)
            # kindex = np.where(s_data['mmap_data'][i]['data'][0]['B'][1, :] == inbytes[k])[0]
                # kindex = np.where(mmap_data[i]['data'][0]['B'][1, :] == inbytes[k])[0]
                kindex = np.where(mmap_data[i]['B'][1, :] == inbytes[k])[0]
                lindex = kindex[s_data['idx'][i]]
                print(f'kindex = {kindex}\nlindex = {lindex}')
            # L(idx:idx+length(lindex)-1,:) = double(s_data.mmap_data{i}.data(1).X(pts,lindex)');
                # L[idx:idx+len(lindex), :] = mmap_data[i]['data'][0]['X'] \
                L[idx:idx+len(lindex), :] = mmap_data[i]['X'] \
                                        [pts, :][:, lindex].T#.astype(np.float64)
                print('am trecut de L')
                # if roffset is not None or not roffset:
                if roffset:
                    print('am intrat in roffset')
                    L[idx:idx+len(lindex), :] += np.outer(roffset[i][:, inbytes[k]], np.ones(nr_points))

                idx += len(lindex)

            M[k, :] = np.mean(L, axis=0)
            X = L - M[k, :]
            # W += X.T @ X
            W += np.dot(X.T, X)

    mvec = np.mean(M, axis=0)

    print('Computing the treatment matrix B...')
    for k in range(nr_bytes):
        xm = M[k, :] - mvec
        B += np.outer(xm, xm)

    B *= np_total

    print(f'M = {M}\nB = {B}\nW = {W}')
    return M, B, W, np_total # TODO check if np_total should be returned
