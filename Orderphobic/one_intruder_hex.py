import time
import matplotlib.pyplot as plt
from labvision import camera, images
import numpy as np
from scipy import ndimage
from labequipment import shaker
import os


class TwoIntruders:

    R = 0
    D = 600
    IMSAVE_INTERVAL = 50
    DATASAVE_INTERVAL = 100
    STEPS = 30000
    DIREC = "/media/data/Data/Orderphobic/TwoIntruders/OneIntruderEverywhere"

    RAMP = False
    date = '100221'
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
        self.data_save_x = f"{self.DIREC}/Logging/{self.date}_{self.D}_{self.N}_x.txt"
        self.data_save_y = f"{self.DIREC}/Logging/{self.date}_{self.D}_{self.N}_y.txt"
        self.plot_save = f"{self.DIREC}/{self.date}_{self.D}_{self.N}.png"
        self.im_save = os.path.splitext(self.data_save_x)[0]
        if not os.path.exists(self.im_save):
            os.mkdir(self.im_save)

    def initialise_devices(self):
        self.cam = camera.Camera(0)
        self.s = shaker.Shaker()

    def get_crops(self):
        frame = self.get_frame(crop=False)
        crop_result = images.crop_polygon(frame)
        self.crop = crop_result.bbox
        self.mask = crop_result.mask
        frame = self.get_frame()

    def start_shaker(self):
        if self.RAMP:
            self.s.ramp(self.D + 60, self.D, 1)
        self.s.change_duty(self.D)

    def get_frame(self, crop=True):
        frame = self.cam.get_frame()
        if crop:
            frame = images.crop_and_mask(frame, self.crop, self.mask)
        return frame

    def get_circles(self, frame):
        red = frame[:, :, 0] - frame[:, :, 2]
        red = images.threshold(red, 120)
        opened = images.opening(red, (31, 31))
        m1 = list(images.center_of_mass(opened))

        return [m1[0], m1[1], 50]

    def get_plot_circles(self, circles):
        c1 = plt.Circle((circles[0], circles[1]), circles[2],
                        color='r')
        return c1

    def setup_plot(self):
        frame = self.get_frame()
        circles = self.get_circles(frame)

        plt.ion()
        self.fig, self.ax = plt.subplots(1, 3)
        self.im_artist = self.ax[0].imshow(frame)
        c1 = self.get_plot_circles(circles)
        self.c1 = self.ax[0].add_artist(c1)
        self.bins = np.linspace(0, frame.shape[1], 10)
        self.hist = self.ax[1].hexbin([], [], bins=20, extent=(0, frame.shape[1], 0, frame.shape[0]))
        self.plot = self.ax[2].plot([], [])[0]
        self.fig.subplots_adjust(wspace=0, hspace=0)
        self.ax[1].set_xlabel('x {pix}')
        self.ax[1].set_ylabel('Frequency')
        self.ax[1].set_xlim([0, frame.shape[1]])
        self.ax[1].set_aspect('equal')
        self.ax[2].set_aspect('equal')
        self.ax[2].set_xlim([0, self.crop.xmax - self.crop.xmin])
        self.ax[2].set_ylim([0, self.crop.ymax - self.crop.ymin])
        self.ax[1].invert_yaxis()
        self.ax[2].invert_yaxis()
        self.fig.show()

    def setup_data(self):
        if os.path.exists(self.data_save_x):
            self.x = np.loadtxt(self.data_save_x).tolist()
            self.y = np.loadtxt(self.data_save_y).tolist()
        else:
            self.x = []
            self.y = []

    def redraw_circles(self, circles):
        self.c1.remove()
        c1 = self.get_plot_circles(circles)
        self.c1 = self.ax[0].add_artist(c1)

    def update_image(self, circles, frame):
        current_time = time.time() - self.start_time
        self.ax[0].set_title(
            f"Elapsed Time: {current_time:.1f} s, data: {len(self.x) // 2}")
        self.redraw_circles(circles)
        self.im_artist.set_data(frame)

    def update_hist(self):

        self.ax[1].hexbin(self.x, self.y, bins=20, extent=(0, self.crop.xmax-self.crop.xmin, 0, self.crop.ymax-self.crop.ymin))

    def update_plot(self):
        self.plot.set_xdata(self.x)
        self.plot.set_ydata(self.y)


    def run(self):
        self.start_time = time.time()

        for step in range(self.STEPS):
            frame = self.get_frame()
            if step % self.IMSAVE_INTERVAL == 0:
                images.save(frame, f"{self.im_save}/{step}.png")

            circles = self.get_circles(frame)
            self.x.append(circles[0])
            self.y.append(circles[1])
            # self.x.append(circles[1, 0])

            self.update_image(circles, frame)
            # self.update_hist()
            self.update_plot()

            self.fig.canvas.flush_events()
            self.fig.canvas.start_event_loop(0.5)

            if step % self.DATASAVE_INTERVAL == 0:
                np.savetxt(self.data_save_x, self.x)
                np.savetxt(self.data_save_y, self.y)

            if step % 20 == 0:
                plt.savefig(self.plot_save, dpi=600)

if __name__ == '__main__':
    ti = TwoIntruders()