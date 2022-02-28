import os.path

import filehandling
import pandas as pd
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def main(direc):
    data_files = filehandling.list_files(f"{direc}/*.hdf5")[::-1]
    global ax, cmap, norm, G
    G = get_G(data_files[-1])

    fig, ax, cmap, norm = create_animation_window()

    ani = matplotlib.animation.FuncAnimation(fig, next_frame, frames=data_files)
    # for file in data_files:
    #     duty = int(os.path.splitext(os.path.split(file)[1])[0])
    #     x, y, horder, torder = get_data_for_graph(file, G)
    #     update_plot(x, y, horder, torder, duty, ax, cmap, norm)
    return ani

def next_frame(file):
    duty = int(os.path.splitext(os.path.split(file)[1])[0])
    x, y, horder, torder = get_data_for_graph(file, G)
    update_plot(x, y, horder, torder, duty, ax, cmap, norm)

def get_data_for_graph(file, G):
    data = pd.read_hdf(file, '/data')
    data = data.loc[0]
    x = data['x'].values
    y = data['y'].values
    horder = data['hexatic_order'].values
    torder = np.exp(1j*data[['x', 'y']].values@G)
    return x, y, horder, torder



def get_G(df):
    """Use the lowest amplitude data file to get G"""
    df = pd.read_hdf(df, 'data')
    vecs = get_delaunay_vectors(df.loc[0])
    angles = get_delaunay_angles(vecs)
    angle = get_delaunay_angle(angles)
    lengths = get_delaunay_lengths(vecs)
    length = get_delaunay_length(lengths)
    G = calculate_G(length, angle)
    return G

def get_delaunay_vectors(points):
    tri = spatial.Delaunay(points[['x', 'y']])
    triangles = tri.points[tri.simplices]
    vecs = np.array([t-t[[2, 0, 1], :] for t in triangles])
    vecs = vecs.reshape((-1, 2))
    return vecs

def get_delaunay_angles(vecs):
    return np.arctan(vecs[:, 1]/vecs[:, 0])

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

def get_delaunay_lengths(vecs):
    lengths = np.linalg.norm(vecs, axis=1)
    return lengths

def get_delaunay_length(lengths):
    return np.median(lengths)

def calculate_G(length, angle):
    a = (angle+90) * np.pi/180
    cosa = np.cos(a)
    sina = np.sin(a)
    l = 4 * np.pi / (length * np.sqrt(3))
    return np.array((cosa, sina))*l

def create_animation_window():
    fig, ax = plt.subplots(1, 2)
    for a in ax:
        a.set_aspect('equal')
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
    cmap = matplotlib.cm.hsv
    norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    return fig, ax, cmap, norm

def update_plot(x, y, horder, torder, duty, ax, cmap, norm):
    for a in ax:
        for art in a.collections:
            art.remove()
    hu, hv = horder.real, horder.imag
    ax[0].quiver(x, y, hu, hv, color=cmap(norm(np.angle(horder))))
    tu, tv = torder.real, torder.imag
    ax[1].quiver(x, y, tu, tv, color=cmap(norm(np.angle(torder))))
    ax[0].set_title(duty)

if __name__ == '__main__':
    ani = main("/media/data/Data/BallBearing/HIPS/PhaseDiagrams/2,15mm/density70_split")
    ani.save('animation_2,15.mp4')
