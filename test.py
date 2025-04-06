import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Given data points
x = np.array([-66.83586883544922, -63.83588409423828, -60.96198272705078, -58.527774810791016,
              -56.34273910522461, -54.565956115722656, -53.229007720947266, -52.42738342285156,
              -52.19707107543945, -52.186737060546875, -52.186737060546875])

y = np.array([27.998329162597656, 28.006664276123047, 28.037200927734375, 28.573915481567383,
              29.77351951599121, 31.554716110229492, 33.781429290771484, 36.25187301635742,
              38.89096450805664, 42.565128326416016, 42.565128326416016])

# Remove duplicate x values
unique_indices = np.unique(x, return_index=True)[1]
x_unique = x[unique_indices]
y_unique = y[unique_indices]

# Ensure x values are sorted
sorted_indices = np.argsort(x_unique)
x_sorted = x_unique[sorted_indices]
y_sorted = y_unique[sorted_indices]