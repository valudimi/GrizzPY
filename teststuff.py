import numpy as np

# Example data_out for the first iteration (k = 0)
data_out = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# Example nr_groups value
nr_groups = 5

# First iteration (k = 0)
k = 0
if k == 0:
    nr_traces, nr_features = data_out.shape
    xdata = np.zeros((nr_traces, nr_features, nr_groups))

print("nr_traces:", nr_traces)  # Output: 2
print("nr_features:", nr_features)  # Output: 3
print("xdata shape:", xdata.shape)  # Output: (2, 3, 5)
xdata[:, :, k] = data_out
print(xdata)