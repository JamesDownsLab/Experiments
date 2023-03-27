import numpy as np
import pandas as pd
from collections import deque
from line_profiler_pycharm import profile
import matplotlib.pyplot as plt
import scipy.spatial
from scipy import spatial
from typing import TypeVar
from mpl_toolkits.axes_grid1 import make_axes_locatable

DelaunayTesselation = TypeVar('scipy.spatial.Delaunay')

def plot_clusters(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    scatt1 = ax1.scatter(df.x, df.y, c=np.angle(df.translational_order), cmap='hsv')
    def annotate(a, ax):
        if len(a) > 3:
            hull = spatial.ConvexHull(a[['x', 'y']].values)
            i = hull.vertices
            i = np.append(i, hull.vertices[0])
            ax.plot(a.x.values[i], a.y.values[i], 'r--')
    df.groupby('cluster').apply(annotate, ax1)

    fig.colorbar(scatt1, ax=ax1, shrink=0.6)
    ax1.set_aspect('equal')
    scatt2 = ax2.scatter(df.x, df.y, c=np.abs(df.hexatic_order), cmap='hsv')
    df.groupby('cluster').apply(annotate, ax2)

    ax2.set_aspect('equal')
    fig.colorbar(scatt2, ax=ax2, shrink=0.6)
    plt.show()

def estimate_diameter(points):
    tree = spatial.KDTree(points)
    dists, indices = tree.query(points, 7)
    diameter = np.median(dists)
    return diameter

def find_delaunay_vectors(points, list_indices, point_indices):
    repeat = list_indices[1:] - list_indices[:-1]
    return points[point_indices] - np.repeat(points, repeat, axis=0)

def delaunay_neighbours(points: np.ndarray, distance_threshold: float=None) -> list[np.ndarray]:
    tess: DelaunayTesselation = spatial.Delaunay(points)
    list_indices, points_indices = tess.vertex_neighbor_vertices
    if distance_threshold:
        vectors = find_delaunay_vectors(points, list_indices, points_indices)
        filtered = np.linalg.norm(vectors, axis=1) < distance_threshold

        def check(p_indices, filtered):
            return np.array([p for p, f in zip(p_indices, filtered) if f])

        neighbours = [check(points_indices[i:j], filtered[i:j]) for i, j in
                      zip(list_indices[:-1], list_indices[1:])]
    else:
        neighbours = [points_indices[i:j] for i, j in zip(list_indices[:-1], list_indices[1:])]
    return neighbours


@profile
def add_clusters(frame: pd.DataFrame,
                 angle_threshold: float,
                 hexatic_threshold: float,
                 distance_threshold: float
                 ) -> pd.DataFrame:

    translational_angle: np.ndarray = np.angle(frame.translational_order.values)
    hexatic_abs: np.ndarray = np.abs(frame.hexatic_order.values)
    clusters: np.ndarray = np.zeros(len(frame))
    current_cluster: int = 1
    unused_index: deque = deque([i for i in range(len(frame))])
    current_indexes: deque = deque([])
    neighbours_all_points: list[np.ndarray] = \
        delaunay_neighbours(frame[['x', 'y']].values, distance_threshold)
    while len(unused_index) > 0:
        index = unused_index.popleft()
        if clusters[index] != 0: continue
        current_indexes.append(index)
        while len(current_indexes) > 0:
            index = current_indexes.popleft()
            clusters[index] = current_cluster
            # Check which neighbours have similar angles
            current_angle = translational_angle[index]
            neighbours = neighbours_all_points[index]
            if len(neighbours) == 0: continue
            neighbours_angles = translational_angle[neighbours]
            angle_differences = np.abs(current_angle - neighbours_angles)
            neighbours = neighbours[(angle_differences < angle_threshold) | (angle_differences > 2*np.pi - angle_threshold)]
            if len(neighbours) == 0: continue
            # Check if one of the neighbouring pairs has high hexatic order
            if hexatic_abs[index] > hexatic_threshold:
                pass
            else:
                hexatic_neighbours = hexatic_abs[neighbours]
                neighbours = neighbours[hexatic_neighbours > hexatic_threshold]

            # Remove these from the first queue and add to the second queue
            for n in neighbours:
                if clusters[n] == 0:
                    current_indexes.append(n)
        current_cluster += 1
    frame['cluster'] = clusters
    return frame

if __name__ == '__main__':
    from order_parameters import add_translational_order
    data = pd.read_hdf("/media/data/Data/BallBearing/HIPS/IslandExperiments/1,87mmRepeats/19370064.hdf5")
    data, lattice_vector = add_translational_order(data)
    frame = data.loc[0]
    diameter = estimate_diameter(frame[['x', 'y']].values)
    frame = add_clusters(frame.copy(), 0.2, 0.8, 2*diameter)
    plot_clusters(frame)