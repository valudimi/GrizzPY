import numpy as np

def compute_features_e2_mmap_multi(s_data, func_prepare, pp1, pp2, pp3, pp4, 
                            pp5, inbytes=None, roffset=None):
    """
    TODO
    """
    nr_sets = s_data['nr_sets']

    if inbytes and len(inbytes) > 0:
        nr_groups = len(inbytes)
    else:
        nr_groups = s_data['metadata'][0]['nr_groups']
        inbytes = list(range(nr_groups))

    if min(inbytes) < 0 or max(inbytes) > (nr_groups - 1):
        raise ValueError('Some index in inbytes is outside available groups')

    print(f's_data["idx"]: {s_data["idx"]}')
    np_total = 0
    for k in range(nr_sets):
        np_total += len(s_data['idx'][k]) # TODO probably not gonna work

    nr_points = s_data['metadata'][0]['nr_points']
    if roffset is None:
        roffset = [None] * nr_sets

    print('Running compute_features_e2_mmap_multi()...')

    # Initialize xdata
    xdata = None

    # TODO create an if check just like in the other one to help reduce runtime
    # Extract data from each group leakage matrix
    for k in range(nr_groups):
        idx = 0
        L = np.zeros((np_total, nr_points))

        for i in range(nr_sets):
            # kindex = np.where(s_data['mmap_data'][i]['data'][0]['B'][1, :] == inbytes[k])[0]
            kindex = np.where(s_data['mmap_data'][i]['B'][1, :] == inbytes[k])[0]
            lindex = kindex[s_data['idx'][i]]

            # L[idx:idx + len(lindex), :] = s_data['mmap_data'][i]['data'][0]['X'][:, lindex].T
            L[idx:idx + len(lindex), :] = s_data['mmap_data'][i]['X'][:, lindex].T

            if roffset[i] is not None:
                L[idx:idx + len(lindex), :] += np.outer(roffset[i][:, inbytes[k]], np.ones(nr_points))

            idx += len(lindex)

        print(f'Size of L/data_in: {L.shape}\nSize of pp1/W: {pp1.shape}\nSize of pp2/xmm: {pp2.shape}\nSize of pp3: {pp3}\nSize of pp4: {pp4}\nSize of pp5: {pp5}')
        data_out = func_prepare(L, pp1, pp2, pp3, pp4, pp5)

        if k == 0:
            nr_traces, nr_features = data_out.shape
            xdata = np.zeros((nr_traces, nr_features, nr_groups))

        xdata[:, :, k] = data_out

    return xdata
