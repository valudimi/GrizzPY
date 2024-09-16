import numpy as np

def compute_features_e2_mmap_adapt(map_data, metadata, bindex, s_adapt, func_prepare, pp1, pp2, pp3, pp4, pp5, inbytes=None):
    if inbytes is not None and len(inbytes) > 0:
        nr_groups = len(inbytes)
    else:
        nr_groups = metadata['nr_groups']
        inbytes = list(range(nr_groups))

    print('Running compute_features_e2_mmap_adapt()...')

    xdata = None

    # Extract data from each group leakage matrix
    for k in range(nr_groups):
        print(f'Computing features for data group {inbytes[k]}')
        kindex = np.where(map_data['data'][0]['B'][1, :] == inbytes[k])[0]
        lindex = kindex[bindex]

        L = map_data['data'][0]['X'][:, lindex].T

        if s_adapt['type'] == 'none':
            pass  # Do nothing
        elif s_adapt['type'] == 'offset_median':
            xmm = s_adapt['xmm']
            nr_traces = L.shape[0]
            for i in range(nr_traces):
                offset = np.median(L[i, :]) - np.median(xmm)
                L[i, :] = L[i, :] - offset
        else:
            raise ValueError(f'Adaptation type not recognized: {s_adapt["type"]}')

        data_out = func_prepare(L, pp1, pp2, pp3, pp4, pp5)

        if k == 0:
            nr_traces, nr_features = data_out.shape
            xdata = np.zeros((nr_traces, nr_features, nr_groups))

        xdata[:, :, k] = data_out

    return xdata
