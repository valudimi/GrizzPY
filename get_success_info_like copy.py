import numpy as np

def get_success_info_like(test_data, nr_iter, nr_traces_vec, func_discriminant, *args):
    """
    Computes success rate information from test data using a discriminant function.

    Parameters:
    - test_data: 3D numpy array of shape (nr_traces, nr_interest_points, nr_groups).
    - nr_iter: Integer, number of iterations for random sampling.
    - nr_traces_vec: List or numpy array, specifying the number of traces to test.
    - func_discriminant: Function to compute the discriminant score.
    - *args: Additional arguments to be passed to func_discriminant.

    Returns:
    - success_info: Dictionary containing success rate information.
    """

    nr_traces = test_data.shape[0]
    nr_groups = test_data.shape[2]
    nr_test_groups = len(nr_traces_vec)
    good_class = np.arange(1, nr_groups + 1)

    success_info = {
        'depth': {
            'avg': {},
            'joint': {}
        },
        'rindex': {}
    }

    # Compute success info for each group size
    for i in range(nr_test_groups):
        print(f'Computing success info for group size {nr_traces_vec[i]}')

        # Initialize depth vectors
        success_info['depth']['avg'][f'group{i+1}'] = np.zeros((nr_groups, nr_iter))
        success_info['depth']['joint'][f'group{i+1}'] = np.zeros((nr_groups, nr_iter))

        # Select random traces for all tests and iterations
        rindex = np.random.randint(0, nr_traces, size=(nr_traces_vec[i], nr_iter))
        success_info['rindex'][f'group{i+1}'] = rindex

        # Compute success info for each group
        for group in range(nr_groups):
            # Perform tests for nr_iter iterations
            for count in range(nr_iter):
                # Select data
                data = test_data[rindex[:, count], :, group]
                data_avg = np.mean(data, axis=0)

                # Compute likelihood values
                l_avg = func_discriminant(data_avg[np.newaxis, :], np.arange(1, nr_groups + 1), *args)
                l_joint = func_discriminant(data, np.arange(1, nr_groups + 1), *args)

                # Compute depth vectors
                si_avg = np.argsort(l_avg.flatten())[::-1]
                success_info['depth']['avg'][f'group{i+1}'][group, count] = np.where(si_avg == good_class[group] - 1)[0][0] + 1

                si_joint = np.argsort(l_joint.flatten())[::-1]
                success_info['depth']['joint'][f'group{i+1}'][group, count] = np.where(si_joint == good_class[group] - 1)[0][0] + 1

    return success_info
