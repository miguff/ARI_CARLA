import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Original data
X = np.array([-66.83586883544922, -63.83588409423828, -60.96198272705078, -58.527774810791016, -56.34273910522461, -54.565956115722656, -53.229007720947266, -52.42738342285156, -52.19707107543945, -52.186737060546875, -52.186737060546875])
Y = np.array([27.998329162597656, 28.006664276123047, 28.037200927734375, 28.573915481567383, 29.77351951599121, 31.554716110229492, 33.781429290771484, 36.25187301635742, 38.89096450805664, 42.565128326416016, 42.565128326416016])

# Parameterize the curve
t = np.linspace(0, 1, len(X))

# Fit splines to parameterized x and y
tck, u = splprep([X, Y], s=0)

# Generate dense parameter values
u_dense = np.linspace(0, 1, 500)
X_dense, Y_dense = splev(u_dense, tck)

# Plot the original points and the fitted spline
plt.plot(X, Y, 'ro', label='Original Points')
plt.plot(X_dense, Y_dense, 'b-', label='Parametric Spline')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Parametric B-Spline Fitting to Route Segment')
plt.show()
