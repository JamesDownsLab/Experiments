import os.path

import filehandling
import pandas as pd
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set('paper')

refine = True

def main():
    direc = "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,42mm/85%"

    data_files = filehandling.list_files(f'{direc}/*.hdf5')

    G = get_G(data_files[0], order_threshold=0.8)
    plot_G(data_files[0], G)
    add_torder(data_files, G)
    plot_torder(data_files[0])
    plot_torder_and_horder(data_files[0])
    duties = []
    suses_h = []
    suses_t = []
    for file in data_files:
        duty = int(os.path.splitext(os.path.split(file)[1])[0])
        sus_h, sus_t = get_sus(file)
        duties.append(duty)
        suses_h.append(sus_h)
        suses_t.append(sus_t)


    fig, ax1 = plt.subplots()
    color1 = 'tab:red'
    ax1.set_xlabel('Duty')
    ax1.set_ylabel('$\chi_6$', color=color1)
    ax1.plot(duties, suses_h, color=color1)

    color2 = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('$\chi_T$', color=color2)
    ax2.plot(duties, suses_t, color=color2)
    plt.show()

    print(G)

def mean_frame_values(points, param):
    vals = points[param].values
    return np.mean(vals)

def get_sus(file):
    df = pd.read_hdf(file, 'data')
    means_t = df.groupby('frame').apply(mean_frame_values, 'torder')
    sus_t = np.mean(means_t * np.conj(means_t)) - np.mean(means_t)*np.conj(np.mean(means_t))
    means_h = df.groupby('frame').apply(mean_frame_values, 'hexatic_order')
    sus_h = np.mean(means_h * np.conj(means_h)) - np.mean(means_h)*np.conj(np.mean(means_h))
    return sus_h, sus_t

def plot_torder_and_horder(file):
    df = pd.read_hdf(file, 'data')
    x = df.loc[0, 'x']
    y = df.loc[0, 'y']

    t = df.loc[0, 'torder'].values
    u_t = t.real
    v_t = t.imag

    h = df.loc[0, 'hexatic_order'].values
    u_h = h.real
    v_h = h.imag

    cmap = matplotlib.cm.hsv
    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    ax1.quiver(x, y, u_h, v_h,
               color=cmap(norm(np.angle(h))),
               pivot='mid',
               headwidth=3)

    ax2.quiver(x, y, u_t, v_t,
               color=cmap(norm(np.angle(t))),
               pivot='mid',
               headwidth=3)
    plt.show()





def plot_torder(file):
    df = pd.read_hdf(file, 'data')
    print(df['torder'].head())
    points = df.loc[0, ['x', 'y', 'torder']].values
    plt.scatter(points[:, 0], points[:, 1], c=np.angle(points[:, 2]), cmap='hsv')
    plt.show()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    cmap = matplotlib.cm.hsv
    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    x = df.loc[0, 'x']
    y = df.loc[0, 'y']
    t = df.loc[0, 'torder']
    u = np.real(t)
    v = np.imag(t)
    ax.quiver(x, y, u, v,
              color=cmap(norm(np.angle(t))),
              pivot='mid',
              headwidth=3)
    plt.show()

def add_torder(data_files, G):
    for file in data_files:
        df = pd.read_hdf(file, 'data')
        df['torder'] = np.exp(1j*df[['x', 'y']].values@G)
        df.to_hdf(file, 'data')

def plot_G(file, G):
    df = pd.read_hdf(file, 'data')
    points = df.loc[0, ['x', 'y']].values
    plt.plot(*points.T, 'x')
    xmid, ymid = np.mean(points, axis=0)
    plt.arrow(xmid, ymid, G[0]*1000, G[1]*1000, width=2)
    plt.show()

def get_G(df, order_threshold=None):
    """Use the lowest amplitude data file to get G"""
    df = pd.read_hdf(df, 'data')
    frame = df.loc[0]
    if order_threshold is not None:
        frame = frame.loc[np.abs(frame.hexatic_order)>order_threshold]
    vecs = get_delaunay_vectors(frame)
    angles = get_delaunay_angles(vecs)
    angle = get_delaunay_angle(angles)
    lengths = get_delaunay_lengths(vecs)
    length = get_delaunay_length(lengths)
    def torder_angle_std(l, a):
        G = calculate_G(l, a)
        torder = np.exp(1j * frame[['x', 'y']].values @ G)
        angles = np.angle(torder)
        std = np.std(angles)
        return std

    if refine:
        torder_angle_std_vect = np.vectorize(torder_angle_std)
        length, angle = refine_l_and_a(length, angle, torder_angle_std_vect, plot=True, width=0.9)
        length, angle = refine_l_and_a(length, angle, torder_angle_std_vect, plot=True, width=0.05)
    G = calculate_G(length, angle)
    return G





def refine_l_and_a(l, a, func, plot=False, width=0.05):
    lengths = np.linspace((1-width)*l, (1+width)*l, 100)
    angles = np.linspace((1-width)*a, (1+width)*a, 100)
    lengths, angles = np.meshgrid(lengths, angles)
    stds = func(lengths, angles)
    min_index = np.unravel_index(np.argmin(stds, axis=None), stds.shape)
    new_length = lengths[min_index]
    new_angle = angles[min_index]
    if plot:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(stds)
        plt.axvline(min_index[1], c='red')
        plt.axhline(min_index[0], c='red')
        plt.subplot(1, 3, 2)
        plt.plot(lengths[min_index[0], :], stds[min_index[0], :])
        plt.axvline(l, c='red')
        plt.axvline(new_length, c='green')
        plt.xlabel('Length [pix]')
        plt.subplot(1, 3, 3)
        plt.plot(angles[:, min_index[1]], stds[:, min_index[1]])
        plt.axvline(a, c='red')
        plt.axvline(new_angle, c='green')
        plt.xlabel('Angle [rad]')
        plt.show()
    return new_length, new_angle

def calculate_G(length, angle):
    a = angle + np.pi/2
    cosa = np.cos(a)
    sina = np.sin(a)
    l = 4 * np.pi / (length * np.sqrt(3))
    return np.array((cosa, sina))*l


def get_delaunay_length(lengths):
    return np.median(lengths)


def get_delaunay_lengths(vecs):
    lengths = np.linalg.norm(vecs, axis=1)
    return lengths

def get_delaunay_angle(angles):

    angles1 = angles[[(angles > 0) & (angles < np.pi / 3)]]
    angles2 = angles[[(angles >= -np.pi/6)&(angles <= np.pi/6)]]
    plt.subplot(1, 2, 1)
    plt.hist(angles1, bins=np.linspace(-np.pi, np.pi, 100))
    plt.subplot(1, 2, 2)
    plt.hist(angles2, bins=np.linspace(-np.pi, np.pi, 100))
    plt.show()
    choice = input('Is 1 or 2 better')
    if choice == '1':
        print('Choice 1 chosen')
        return np.median(angles1)*180/np.pi
    else:
        print('Choice 2 chosen')
        return np.median(angles2)*180/np.pi


def get_delaunay_vectors(points):
    tri = spatial.Delaunay(points[['x', 'y']])
    triangles = tri.points[tri.simplices]
    vecs = np.array([t-t[[2, 0, 1], :] for t in triangles])
    vecs = vecs.reshape((-1, 2))
    return vecs

def get_delaunay_angles(vecs):
    return np.arctan(vecs[:, 1]/vecs[:, 0])


if __name__ == '__main__':
    main()


