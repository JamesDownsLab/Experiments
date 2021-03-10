import time
import matplotlib.pyplot as plt
from labvision import camera, images
import numpy as np
from scipy import ndimage
from labequipment import shaker
import os


class TwoIntruders:

    R = 60
    D = 620
    TRACK_RIGHT = True
    SHOW_RIGHT = True
    PLOT_RIGHT = True
    TRACK_LEFT = False
    SHOW_LEFT = False
    PLOT_LEFT = False
    TWO_DIMENSIONS = True
    IMSAVE_INTERVAL = 50
    DATASAVE_INTERVAL = 100
    STEPS = 200000
    DIREC = "/media/data/Data/Orderphobic/TwoIntruders/RailWithThreeSlots"

    RAMP = False
    date = '180221'
    N = 1

    def __init__(self):
        self.setup_directories()
        self.initialise_devices()
        self.get_crops()
        self.start_shaker()
        self.setup_plot()
        self.setup_data()
        self.run()

    def setup_directories(self):
        self.data_save = f"{self.DIREC}/Logging/{self.date}_{self.D}_{self.N}.txt"
        self.data_save_y = f"{self.DIREC}/Logging/{self.date}_{self.D}_{self.N}_y.txt"
        self.plot_save = f"{self.DIREC}/{self.date}_{self.D}_{self.N}.png"
        self.im_save = os.path.splitext(self.data_save)[0]
        if not os.path.exists(self.im_save):
            os.mkdir(self.im_save)

    def initialise_devices(self):
        self.cam = camera.Camera(0)
        self.s = shaker.Shaker()

    def get_crops(self):
        frame = self.get_frame(crop=False)
        crop_result = images.crop_polygon(frame)
        self.crop = crop_result.bbox
        frame = self.get_frame()
        self.left_crop, self.right_crop = self.get_both_bounds(frame)

    def get_both_bounds(self, frame):
        left_crop_result = images.crop_polygon(frame)
        right_crop_result = images.crop_polygon(frame)
        return left_crop_result, right_crop_result

    def start_shaker(self):
        if self.RAMP:
            self.s.ramp(self.D + 60, self.D, 1)
        self.s.change_duty(self.D)

    def get_frame(self, crop=True):
        frame = self.cam.get_frame()
        if self.R != 0:
            frame = ndimage.rotate(frame, self.R, reshape=False)
        if crop:
            frame = images.crop(frame, self.crop)
        return frame

    def get_circles(self, frame):
        red = frame[:, :, 0] - frame[:, :, 2]
        red = images.threshold(red, 65)
        opened = images.opening(red, (31, 31))
        im1 = images.crop(opened, self.left_crop.bbox)
        im2 = images.crop(opened, self.right_crop.bbox)
        if self.TRACK_LEFT:
            m1 = list(images.center_of_mass(im1))
        else:
            m1 = [im1.shape[1] // 2, im1.shape[0] // 2]
        if self.TRACK_RIGHT:
            m2 = list(images.center_of_mass(im2))
        else:
            m2 = [im2.shape[1] // 2, im2.shape[0] // 2]

        m1[0] += self.left_crop.bbox.xmin
        m1[1] += self.left_crop.bbox.ymin
        m2[0] += self.right_crop.bbox.xmin
        m2[1] += self.right_crop.bbox.ymin

        return np.array([[m1[0], m1[1], 50], [m2[0], m2[1], 50]])

    def get_plot_circles(self, circles):
        c1 = plt.Circle((circles[0, 0], circles[0, 1]), circles[0, 2],
                        color='r')
        c2 = plt.Circle((circles[1, 0], circles[1, 1]), circles[1, 2],
                        color='r')
        return c1, c2

    def setup_plot(self):
        frame = self.get_frame()
        circles = self.get_circles(frame)

        plt.ion()
        if self.TWO_DIMENSIONS:
            self.fig, self.ax = plt.subplots(4, 1, sharex=True)
        else:
            self.fig, self.ax = plt.subplots(3, 1, sharex=True)
        self.im_artist = self.ax[0].imshow(frame)
        c1, c2 = self.get_plot_circles(circles)
        if self.SHOW_LEFT:
            self.c1 = self.ax[0].add_artist(c1)
        if self.SHOW_RIGHT:
            self.c2 = self.ax[0].add_artist(c2)
        self.bins = np.linspace(0, frame.shape[1], 100)
        self.hist = self.ax[1].bar(self.bins[:-1], np.zeros_like(self.bins[:-1]),
                         width=self.bins[1] - self.bins[0])
        self.plot_left = self.ax[2].plot([], [])[0]
        self.plot_right = self.ax[2].plot([], [])[0]
        self.fig.subplots_adjust(wspace=0, hspace=0)
        self.ax[1].set_xlabel('x {pix}')
        self.ax[1].set_ylabel('Frequency')

        if self.TWO_DIMENSIONS:
            self.plot_left_y = self.ax[3].plot([], [])[0]
            self.plot_right_y = self.ax[3].plot([], [])[0]
        self.ax[1].set_xlim([0, frame.shape[1]])
        self.fig.show()

    def setup_data(self):
        if os.path.exists(self.data_save):
            if self.TWO_DIMENSIONS:
                self.y = np.loadtxt(self.data_save_y).tolist()
            self.x = np.loadtxt(self.data_save).tolist()
        else:
            self.x = []
            self.y = []
            self.t = []

    def redraw_circles(self, circles):
        if self.SHOW_LEFT:
            self.c1.remove()
        if self.SHOW_RIGHT:
            self.c2.remove()
        c1, c2 = self.get_plot_circles(circles)
        if self.SHOW_LEFT:
            self.c1 = self.ax[0].add_artist(c1)
        if self.SHOW_RIGHT:
            self.c2 = self.ax[0].add_artist(c2)

    def update_image(self, circles, frame):
        current_time = time.time() - self.start_time
        self.ax[0].set_title(
            f"Elapsed Time: {current_time:.1f} s, data: {len(self.x) // 2}")
        self.redraw_circles(circles)
        self.im_artist.set_data(frame)

    def update_hist(self):
        if self.PLOT_LEFT and self.PLOT_RIGHT:
            hist_data = np.array(self.x)
        elif self.PLOT_LEFT and not self.PLOT_RIGHT:
            hist_data = np.array(self.x)[::2]
        elif not self.PLOT_LEFT and self.PLOT_RIGHT:
            hist_data = np.array(self.x)[1::2]
        freq, _ = np.histogram(hist_data, bins=self.bins)
        for rect, h in zip(self.hist, freq):
            rect.set_height(h)
        self.ax[1].set_ylim([0, max(freq)])

    def update_plot(self):
        if self.PLOT_LEFT:
            xl = np.array(self.x)[::2]
            self.plot_left.set_xdata(xl)
            self.plot_left.set_ydata(np.arange(len(xl)))
        if self.PLOT_RIGHT:
            xr = np.array(self.x)[1::2]
            self.plot_right.set_xdata(xr)
            self.plot_right.set_ydata(np.arange(len(xr)))
        self.ax[2].set_ylim([0, len(self.x) // 2])

    def update_y_plot(self):
        if self.PLOT_LEFT:
            yl = np.array(self.y)[::2]
            self.plot_left_y.set_xdata(yl)
            self.plot_left_y.set_ydata(np.arange(len(yl)))
        if self.PLOT_RIGHT:
            yr = np.array(self.y)[1::2]
            self.plot_right_y.set_xdata(yr)
            self.plot_right_y.set_ydata(np.arange(len(yr)))
        self.ax[3].set_ylim([0, len(self.x) // 2])

    def run(self):
        self.start_time = time.time()

        for step in range(self.STEPS):
            frame = self.get_frame()
            if step % self.IMSAVE_INTERVAL == 0:
                images.save(frame, f"{self.im_save}/{step}.png")

            circles = self.get_circles(frame)
            self.x.append(circles[0, 0])
            self.x.append(circles[1, 0])
            if self.TWO_DIMENSIONS:
                self.y.append(circles[0, 1]+250)
                self.y.append(circles[1, 1]+250)

            self.update_image(circles, frame)
            self.update_hist()
            self.update_plot()
            if self.TWO_DIMENSIONS:
                self.update_y_plot()

            self.fig.canvas.flush_events()
            self.fig.canvas.start_event_loop(0.5)

            if step % self.DATASAVE_INTERVAL == 0:
                np.savetxt(self.data_save, self.x)
                if self.TWO_DIMENSIONS:
                    np.savetxt(self.data_save_y, self.y)

            if step % 20 == 0:
                plt.savefig(self.plot_save, dpi=600)

if __name__ == '__main__':
    ti = TwoIntruders()