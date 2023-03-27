import numpy as np
from scipy import spatial


def get_delaunay_vectors(points: np.ndarray) -> np.ndarray:
    tri = spatial.Delaunay(points)
    triangles = tri.points[tri.simplices]
    vecs = np.array([t-t[[2,2, 0], :] for t in triangles])
    vecs = vecs.reshape((-1, 2))
    return vecs

def get_delaunay_angles(vecs: np.ndarray) -> np.ndarray:
    return np.arctan(vecs[:, 1] / vecs[:, 0])

def get_delaunay_lengths(vecs):
    return np.linalg.norm(vecs, axis=1)

def get_length_and_angle(frame, apothem):
    vecs = get_delaunay_vectors(frame[['x', 'y']].values)
    angles = get_delaunay_angles(vecs)
    if apothem:
        angle = np.median(angles[angles>0])
    else:
        angle = np.median(angles[angles>0.5])
    lengths = get_delaunay_lengths(vecs)
    length = np.median(lengths)
    return length, angle

def get_lattice_vector(length, angle):
    a = angle + np.pi / 2
    cosa = np.cos(a)
    sina = np.sin(a)
    l = 4 * np.pi / (length * np.sqrt(3))
    return np.array((cosa, sina)) * l

def get_length_and_angle_from_lattice_vector(lattice_vector):
    l = np.linalg.norm(lattice_vector)
    length = np.sqrt(3)*l / (4*np.pi)
    angle = np.arccos(lattice_vector[0]/l) - np.pi / 2
    return length, angle

def torder_angle_std(l, a, points):
    G = get_lattice_vector(l, a)
    torder = np.exp(1j*points@G)
    angles = np.angle(torder)
    std = np.std(angles)
    return std

torder_angle_std_vect = np.vectorize(torder_angle_std, excluded=['points'])

def refine_l_and_a(l, a, points, width=0.05):
    lengths = np.linspace((1 - width) * l, (1 + width) * l, 100)
    angles = np.linspace((1 - width) * a, (1 + width) * a, 100)
    lengths, angles = np.meshgrid(lengths, angles)
    stds = np.zeros_like(lengths)
    for li, l in enumerate(lengths):
        for ai, a in enumerate(angles):
            stds[li, ai] = torder_angle_std(l, a, points)
    # stds = torder_angle_std_vect(lengths, angles, points)
    min_index = np.unravel_index(np.argmin(stds, axis=None), stds.shape)
    new_length = lengths[min_index]
    new_angle = angles[min_index]
    return new_length, new_angle

def add_translational_order(df, lattice_vector=None, apothem=False, refine=False):
    frame = df.loc[0]
    if lattice_vector is None:
        length, angle = get_length_and_angle(frame, apothem)
        lattice_vector = get_lattice_vector(length, angle)
    if refine:
        length, angle = get_length_and_angle_from_lattice_vector(lattice_vector)
        length, angle = refine_l_and_a(length, angle, frame[['x', 'y']].values)
        lattice_vector = get_lattice_vector(length, angle)

    df['translational_order'] = np.exp(1j*df[['x', 'y']].values @ lattice_vector)
    return df, lattice_vector


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_hdf("/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/1,95mm/70%/451.hdf5")
    df, lattice_vector = add_translational_order(df)
    print(df.head())




