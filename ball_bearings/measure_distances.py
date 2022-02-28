import sys
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt

filename = sys.argv[1]

data = pd.read_hdf(filename)
points = data.loc[0, ['x', 'y']].values

tree = spatial.KDTree(points)
dists, indices = tree.query(points, 7)
dists = dists[:, 1:].ravel()

plt.hist(dists, bins=1000)
plt.show()

from particletracking import statistics

pc = statistics.PropertyCalculator()
pc.correlations()