from math import pi, atan, sin, cos

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from tqdm import tqdm
from shapely import affinity
from shapely.geometry import LineString, Point


from labvision import images
import filehandling
from particletracking import dataframes, statistics


def run():
    direc = "/media/data/Data/FirstOrder/Interfaces/RecordFluctuatingInterfaceJanuary2020/Quick/first_frames"
    savename = f"{direc}/data_new.hdf5"
    files = filehandling.get_directory_filenames(direc + '/*.png')
    ims = [images.load(f, 0) for f in tqdm(files, 'Loading images')]
    ims = [images.bgr_to_gray(im) for im in ims]
    circles = [images.find_circles(im, 27, 200, 7, 16, 16)
               for im in tqdm(ims, 'Finding Circles')]

    data = dataframes.DataStore(savename, load=False)
    for f, info in tqdm(enumerate(circles), 'Adding Circles'):
        data.add_tracking_data(f, info, ['x', 'y', 'r'])

    calc = statistics.PropertyCalculator(data)
    calc.order()

    lattice_spacing = 10
    x = np.arange(0, ims[0].shape[1], lattice_spacing)
    y = np.arange(0, ims[0].shape[0], lattice_spacing)
    x, y = np.meshgrid(x, y)

    # cgw = get_cgw(data.df.loc[0], 1.85) # k=1.85 looked the best
    cgw = get_cgw(data.df.loc[0], 1.85)

    fields = [coarse_order_field(data.df.loc[f], cgw, x, y)
              for f in tqdm(range(len(ims)), 'Calculating Fields')]

    field_threshold = get_field_threshold(fields, lattice_spacing, ims[0])

    contours = [find_contours(f, field_threshold)
                for f in tqdm(fields, 'Calculating contours')]

    # Multiply the contours by the lattice spacing and squeeze
    contours = [c.squeeze() * lattice_spacing for c in contours]

    # Close contours
    contours = [close_contour(c) for c in contours]

    # Convert to LineString
    contours = [LineString(c) for c in contours]

    # Select the line to query the points across
    print("Select the line to query points")
    ls = LineSelector(ims[0])
    p1, p2 = ls.points
    centre_line = get_extended_centre_line(p1, p2)

    # Distance between query points that determines one end of the frequency
    dL = data.df.loc[0].r.mean() / 10
    L = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    N_query = int(L/dL)
    xq, yq = np.linspace(p1[0], p2[0], N_query), np.linspace(p1[1], p2[1],
                                                             N_query)
    dL = np.sqrt((xq[1] - xq[0]) ** 2 + (yq[1] - yq[0]) ** 2)
    dists, crosses = zip(
        *[get_dists(xq, yq, c, centre_line) for c in tqdm(contours)])

    # plot_crosses(crosses, ims)

    # Select points from inside red edge to inside red edge across the centre
    # of the tray which is 200mm to convert pixels to mm
    PIX_2_mm = get_pix_2_mm(ims[0])

    plot_fft(dists, dL, PIX_2_mm, data.df.loc[0].r.mean(), cgw)

def get_cgw(df, k):
    """
    Get the coarse-graining-width as a factor k of the average
    radius of particles in dataframe df
    """
    tree = spatial.cKDTree(df[['x', 'y']].values)
    dists, _ = tree.query(tree.data, 2)
    cgw = np.mean(dists[:, 1])
    return cgw * k

def coarse_order_field(df, cgw, x, y, no_of_neighbours=20):
    """
    Calculate the coarse-grained field characterising local orientation order
    of particles in dataframe df.
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
    print('Click the topmost corner then the bottommost corner of a region \n representing the ordered phase')
    line_selector = LineSelector(im)
    op1, op2 = line_selector.points
    phi_o = np.mean(
        fields[op1[1] // ls:op2[1] // ls, op1[0] // ls:op2[0] // ls, :])

    # Repeat for disordered
    print(
        'Click the topmost corner then the bottommost corner of a region \n representing the disordered phase')
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


def get_extended_centre_line(p1, p2):
    """ Extends the line across the centre of the tray so it can be safely rotated"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    p0 = (p1[0] - dx, p1[1] - dy)
    p3 = (p2[0] + dx, p2[1] + dy)
    return LineString(((p0[0], p0[1]), (p3[0], p3[1])))


def find_contours(f, t):
    t_low = t - 0.02 * t
    t_high = t + 0.02 * 5
    new_f = (f < t_high) * (f > t_low)
    new_f = np.uint8(new_f)
    contours = images.find_contours(new_f)
    contours = images.sort_contours(contours)
    try:
        return contours[-1]
    except IndexError as e:
        print("Only one contour")
        return contours


def close_contour(c):
    """Make the open contour path a loop by adding the first point to the end"""
    c = np.vstack((c, c[0, :]))
    return c

def get_dists(x, y, c, l):
    """
    Calculate the distance from the line l to the contour c for each
    point (x, y) along the line
    """
    dists = []
    crosses = []
    for (xp, yp) in zip(x, y):
        p = Point((xp, yp))
        l_rot = affinity.rotate(l, 90, p)
        cross = c.intersection(l_rot)
        if cross.geom_type == 'Point':
            dist = cross.distance(p)
            cross = cross.x, cross.y
        elif cross.geom_type == 'MultiPoint':
            ds = [c.distance(p) for c in cross]
            dist = np.min(ds)
            cross = cross[np.argmin(ds)]
            cross = cross.x, cross.y
        else:
            dist = 0
            cross = xp, yp
        dists.append(dist)
        crosses.append(cross)
    return dists, crosses


def get_angle(im):
    ls = LineSelector(im)
    p1, p2 = ls.points
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    a = -atan(m)
    c = np.array([i // 2 for i in np.shape(im)])[::-1]
    return a, c, p1, p2


def get_pix_2_mm(im):
    _, _, p1, p2 = get_angle(im)
    L_pix = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    L_mm = 200.0
    return L_mm / L_pix


def plot_fft(dists, dL, pix_2_mm, r, cgw):
    dL *= pix_2_mm
    sp = [np.abs(np.fft.fft(np.array(h) * pix_2_mm)) ** 2 for h in dists]
    N = len(dists[0])
    freq = np.fft.fftfreq(N, dL)[1:N // 2]

    y = (np.stack(sp) * dL * N)[1:N // 2]
    y_mean = np.mean(y, axis=0).squeeze()
    y_err = np.std(y, axis=0, ddof=1).squeeze()

    xplot = freq * 2 * np.pi
    L_x = 2 * np.pi / (dL * N)
    r_x = 2 * np.pi / (r * pix_2_mm)
    cgw_x = 2 * np.pi / (cgw * pix_2_mm)

    xmax = sum(xplot < cgw_x)
    # xmax = len(xplot)

    xplot = np.log10(xplot[5:xmax])
    yplot = np.log10(y_mean[5:xmax])
    yplot_err = 0.434 * y_err[5:xmax] / y_mean[5:xmax]

    coeffs, cov = np.polyfit(xplot, yplot, 1, w=yplot_err, cov=True)
    fit_func = np.poly1d(coeffs)
    yfit = fit_func(xplot)
    m = coeffs[0]
    dm = np.sqrt(cov[0, 0])

    #     m, c, sm, sc = get_fit(xplot, yplot, yplot_err)
    #     yfit = m*xplot + c

    plt.errorbar(xplot, yplot, yerr=yplot_err, fmt='o')
    plt.plot(xplot, yfit, label=f'Fit with gradient {m:.3f} +/- {dm:.3f}')

    plt.axvline(np.log10(L_x), label='L', c='r')
    plt.axvline(np.log10(cgw_x), label='cgw', c='b')
    plt.axvline(np.log10(r_x), label='r', c='g')

    plt.xlabel('log$_{10}(k = 2\pi m/L)$ [mm$^{-1}$]')
    plt.ylabel('log$_{10}(<|\delta h_k|^2>L)$ [mm$^3$]')

    plt.legend()

    plt.show()


def get_pix_2_mm(im):
    _, _, p1, p2 = get_angle(im)
    L_pix = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    L_mm = 200.0
    return L_mm / L_pix


def plot_crosses(crosses, ims):
    for cross, im in zip(crosses, ims):
        cross = np.array(cross)
        plt.figure()
        plt.imshow(im)
        plt.plot(cross[:, 0], cross[:, 1])
        plt.show()

if __name__ == '__main__':
    run()