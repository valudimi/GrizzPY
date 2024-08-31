import numpy as np

def prepare_data_template(data_in, interest_points, *args):
    """
    Prepares data for leakage analysis by selecting interesting points from the traces.

    Parameters:
    - data_in: Input data matrix of size (nr_traces, nr_points).
    - interest_points: 1D array of length nr_interest_points containing the indices of
      the interesting points to be selected.

    Returns:
    - data_out: Transformed data matrix of size (nr_traces, nr_interest_points).
    """
    if interest_points is None or len(interest_points) == 0:
        raise ValueError('interest_points is empty')

    # Select the interesting points from the input data
    data_out = data_in[:, interest_points]

    return data_out
