from math import pi, atan, sin, cos

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from tqdm import tqdm

import seaborn
seaborn.set()
from Generic import images, filedialogs
from ParticleTracking import dataframes, statistics


def run(direc, lattice_spacing=5):
    files = filedialogs.get_files_directory(direc + '/*.png')
    savename = direc + '/data.hdf5'

    N = len(files)

    # Load images
    ims = [images.load(f, 0) for f in tqdm(files, 'Loading images')]

    # Find Circles
    circles = [images.find_circles(im, 27, 200, 7, 16, 16)
               for im in tqdm(ims, 'Finding Circles')]

    # Save data
    data = dataframes.DataStore(savename, load=False)
    for f, info in tqdm(enumerate(circles), 'Adding Circles'):
        data.add_tracking_data(f, info, ['x', 'y', 'r'])

    # Calculate order parameter
    calc = statistics.PropertyCalculator(data)
    calc.order()

    # Get the course graining width
    cgw = get_cgw(data.df.loc[0]) / 2

    # Create the lattice points
    x = np.arange(0, max(data.df.x), lattice_spacing)
    y = np.arange(0, max(data.df.y), lattice_spacing)
    x, y = np.meshgrid(x, y)

    # Calculate the coarse order fields
    fields = [coarse_order_field(data.df.loc[f], cgw, x, y)
              for f in tqdm(range(N), 'Calculating Fields')]

    # Calculate the field threshold
    field_threshold = get_field_threshold(fields, lattice_spacing, ims[0])

    # Find the contours representing the boundary in each frame
    contours = [find_contours(f, field_threshold)
                for f in tqdm(fields, 'Calculating contours')]

    # Multiply the contours by the lattice spacing
    contours = [c * lattice_spacing for c in contours]

    # Find the angle of the image to rotate the boundary to the x-axis
    a, c, p1, p2 = get_angle(ims[0])

    # Rotate the selection points and the contours by the angle
    p1 = rotate_points(np.array(p1), c, a)
    p2 = rotate_points(np.array(p2), c, a)
    contours = [rotate_points(contour.squeeze(), c, a)
                for contour in contours]

    xmin = int(p1[0])
    xmax = int(p2[0])
    h = int(p1[1])

    # Get the heights of the fluctuations from the straight boundary
    hs = [get_h(contour, ims[0].shape, xmin, xmax, h)
          for contour in tqdm(contours, 'Calculating heights')]

    # Calculate the fourier transforms for all the frames
    L = xmax - xmin
    pixels_to_mms = 195/L
    print('One pixel is {:.2f} mm'.format(pixels_to_mms))

    #convert to mm
    hs = [h * pixels_to_mms for h in hs]
    L = L * pixels_to_mms

    k, yplot = get_fourier(hs, L)

    return k, yplot


def get_cgw(df):
    tree = spatial.cKDTree(df[['x', 'y']].values)
    dists, _ = tree.query(tree.data, 2)
    cgw = np.mean(dists[:, 1])
    return cgw


def coarse_order_field(df, cgw, x, y, no_of_neighbours=20):
    """
    Calculate the coarse-grained field characterising local orientation order
    """

    order = df.order.values

    # Generate the lattice nodes to query
    # x, y = np.meshgrid(x, y)
    r = np.dstack((x, y))

    # Get the positions of all the particles
    particles = df[['x', 'y']].values

    # Generate the tree from the particles
    tree = spatial.cKDTree(particles)

    # Query the tree at all the lattice nodes to find the nearest n particles
    # Set n_jobs=-1 to use all cores
    dists, indices = tree.query(r, no_of_neighbours, n_jobs=-1)

    # Calculate all the coarse-grained delta functions (Katira ArXiv eqn 3
    cg_deltas = np.exp(-dists ** 2 / (2 * cgw ** 2)) / (2 * pi * cgw ** 2)

    # Multiply by the orders to get the summands
    summands = cg_deltas * order[indices]

    # Sum along axis 2 to calculate the field
    field = np.sum(summands, axis=2)

    return field


def get_field_threshold(fields, ls, im):
    # Draw a box around an always ordered region of the image to
    # calculate the phi_o
    fields = np.dstack(fields)
    line_selector = LineSelector(im)
    op1, op2 = line_selector.points
    phi_o = np.mean(
        fields[op1[1] // ls:op2[1] // ls, op1[0] // ls:op2[0] // ls, :])

    # Repeat for disordered
    line_selector = LineSelector(im)
    dp1, dp2 = line_selector.points
    phi_d = np.mean(
        fields[dp1[1] // ls:dp2[1] // ls, dp1[0] // ls:dp2[0] // ls, :])

    field_threshold = (phi_o + phi_d) / 2
    return field_threshold


class LineSelector:
    def __init__(self, im):
        cv2.namedWindow('line', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('line', 960, 540)
        cv2.setMouseCallback('line', self.record)
        self.points = []
        while True:
            cv2.imshow('line', im)
            key = cv2.waitKey(1) & 0xFF
            if len(self.points) == 2:
                break
        cv2.destroyAllWindows()

    def record(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])


def find_contours(f, t):
    t_low = t - 0.02 * t
    t_high = t + 0.02 * 5
    new_f = (f < t_high) * (f > t_low)
    new_f = np.uint8(new_f)
    contours = images.find_contours(new_f)
    contours = images.sort_contours(contours)
    return contours[-1]


def get_angle(im):
    ls = LineSelector(im)
    p1, p2 = ls.points
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    a = -atan(m)
    c = np.array([i // 2 for i in np.shape(im)])[::-1]
    return a, c, p1, p2


def rotate_points(points, center, a):
    rot = np.array(((cos(a), -sin(a)), (sin(a), cos(a))))
    a1 = points - center
    a2 = rot @ a1.T
    a3 = a2.T + center
    return a3


def get_h(contour, shape, xmin, xmax, h):
    xs = []
    ys = []
    im = np.zeros((shape[0] * 2, shape[1] * 2))
    im = cv2.polylines(im, [contour.astype('int32')], True, (255, 255, 255))
    im = images.dilate(im, (3, 3))
    for x in np.arange(xmin, xmax):
        crossings = np.argwhere(im[:, x] == 255)
        dists = crossings - h
        closest = np.argmin((crossings - h) ** 2)
        crossing = crossings[closest]
        ys.append(crossing[0])
        xs.append(x)
    hs = np.array([y - h for y in ys])
    return hs


def get_fourier(hs, L):
    sp = [np.fft.fft(h) for h in hs]
    N = len(hs[0])
    freq = np.fft.fftfreq(N)

    y = np.stack(sp)
    y = np.mean(y, axis=0).squeeze()

    xplot = freq[1:N // 2]
    yplot = L * np.abs(y[1:N // 2]) ** 2
    return xplot, yplot


def plot(k, y):
    p, cov = np.polyfit(np.log(k), np.log(y), 1, cov=True)
    p1 = np.poly1d(p)

    # Plot the results
    plt.figure()
    plt.plot(np.log(k), np.log(y))
    plt.plot(np.log(k), p1(np.log(k)))
    plt.legend(
        ['Data',
         'Fit with gradient ${:.2f} \pm {:.2f}$'.format(p[0],
                                                        cov[0][0] ** 0.5)])
    plt.xlabel('log($k [$mm$^{-1}]$)')
    plt.ylabel('$log( < |\delta h_k|^2 > L [$mm$^3] $)')


    plt.figure()
    plt.loglog(k, yplot, '.')
    plt.loglog(k, np.exp(p1(np.log(k))))
    plt.legend(['Data', 'Fit'])
    plt.xlabel('$k$ [mm$^{-1}$]')
    plt.ylabel(r'$\langle \left| \delta h_k \right|^2L$  [mm$^3$]')


if __name__ == "__main__":
    direc = filedialogs.open_directory()
    k, yplot = run(direc, 5)
    plot(k, yplot)
    # Calculate the best fit line for the fourier
