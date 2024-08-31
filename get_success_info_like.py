import numpy as np

def get_success_info_like(test_data, nr_iter, nr_traces_vec, func_discriminant, *args):
    """
    Returns success rate information from some data.
    
    Parameters:
    - test_data: The test data, preprocessed by a suitable function, 
                 should be a matrix of size (nr_traces, nr_interest_points, nr_groups).
    - nr_iter: Specifies how many times to run each test on randomly picked samples.
    - nr_traces_vec: A vector of integers where each element represents the size of one test group.
    - func_discriminant: A function that returns a discriminant of a sample vector or matrix.
    - *args: Additional arguments to pass to func_discriminant.
    
    Returns:
    - success_info: A dictionary containing the results of this method.
    """
    nr_traces, _, nr_groups = test_data.shape
    nr_test_groups = len(nr_traces_vec)
    # print(f'Number of test groups: {nr_test_groups}')

    good_class = np.arange(0, nr_groups)
    # print(f'Good class: {good_class}')

    success_info = {
        'depth': {
            'avg': {},
            'joint': {}
        },
        'rindex': {}
    }

    for i in range(nr_test_groups):
        print(f'Computing success info for group size {nr_traces_vec[i]}')

        # Set up the depth vectors
        # TODO: check if dtype=int is necessary and why
        # success_info['depth']['avg'][f'group{i+1}'] = np.zeros((nr_groups, nr_iter), dtype=int)
        success_info['depth']['avg'][f'group{i+1}'] = np.zeros((nr_groups, nr_iter))
        # success_info['depth']['joint'][f'group{i+1}'] = np.zeros((nr_groups, nr_iter), dtype=int)
        success_info['depth']['joint'][f'group{i+1}'] = np.zeros((nr_groups, nr_iter))

        # Select random traces to use for all the tests and iterations
        rindex = np.random.randint(0, nr_traces, size=(nr_traces_vec[i], nr_iter))
        success_info['rindex'][f'group{i+1}'] = rindex

        # Compute success info for each group
        for group in range(nr_groups):
            # Perform the tests for nr_iter
            for count in range(nr_iter):
                # Select data
                data = test_data[rindex[:, count], :, group]
                data_avg = np.mean(data, axis=0)
                # print(f'data: {data}')
                # print(f'data_avg: {data_avg}')

                # Compute likelihood values
                #tmiu, tsigma pentru LDA
                l_avg = func_discriminant(data_avg, np.arange(1, nr_groups + 1), *args)
                l_joint = func_discriminant(data, np.arange(1, nr_groups + 1), *args)
                # print(f'l_avg: {l_avg}')
                # print(f'l_joint: {l_joint}')

                # Compute depth vectors
                # The main assumption here is that the group corresponding to
                # the highest likelihood is the correct group. Therefore make
                # sure that the function you have passed as func_discriminant
                # has this property.
                si_avg = np.argsort(l_avg)[::-1]
                # print(f'si_avg: {si_avg}')

                success_info['depth']['avg'][f'group{i+1}'][group, count] = \
                      np.where(si_avg == good_class[group])[0][0] + 1
                si_joint = np.argsort(l_joint)[::-1]
                # print(f'si_joint: {si_joint}')

                success_info['depth']['joint'][f'group{i+1}'][group, count] = \
                      np.where(si_joint == good_class[group])[0][0] + 1

    return success_info
