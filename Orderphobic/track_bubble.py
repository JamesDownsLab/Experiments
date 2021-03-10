from labequipment import shaker
from labvision import images, camera
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np

class DisplayWindow:

    def __init__(self, frame):
        plt.ion()
        self.fig, (self.ax, self.ax2) = plt.subplots(1, 2, sharey=True)

        self.im_artist = self.ax.imshow(frame)

        self.contour_plot = self.ax.plot([], [])[0]

        self.center_mark = plt.Circle((0, 0), 5, color='r')
        self.center_mark = self.ax.add_artist(self.center_mark)

        self.im_artist2 = self.ax2.imshow(frame)

        self.route_plot = self.ax2.scatter([], [])
        self.ax2.set_xlim([0, frame.shape[1]])
        self.ax2.set_ylim([0, frame.shape[0]])

        self.fig.tight_layout = True


    def flush(self):
        self.fig.canvas.flush_events()
        self.fig.canvas.start_event_loop(0.5)

    def set_frame(self, frame):
        self.im_artist.set_data(frame)
        self.im_artist2.set_data(frame)

    def plot_contour(self, contour):
        self.contour_plot.set_xdata(contour[:, :, 0])
        self.contour_plot.set_ydata(contour[:, :, 1])

    def add_center(self, center):
        self.center_mark.remove()
        self.center_mark = self.ax.add_artist(plt.Circle(center, 5, color='r'))

    def update_positions(self, x, y):
        n = mpl.colors.Normalize(vmin=0, vmax=len(x))
        m = mpl.cm.ScalarMappable(norm=n, cmap=mpl.cm.afmhot)
        self.route_plot.set_offsets(np.c_[x, y])
        # self.route_plot.set_array(np.linspace(0, 255, len(x)))
        self.route_plot.set_facecolor(m.to_rgba(np.arange(len(x))))
        self.route_plot.set_clim(vmin=0, vmax=len(x))

    def save(self, fname):
        self.fig.savefig(fname)


class BubbleTracker:

    figsave = "/media/data/Data/Orderphobic/TrackBubble/120221_bubble_tracking_ramp.png"
    datasave = "/media/data/Data/Orderphobic/TrackBubble/120221_bubble_tracking_ramp"

    def __init__(self):
        self.cam = camera.Camera(0)
        self.shaker = shaker.Shaker()
        self.shaker.change_duty(600)
        self.crop, self.mask = self.get_crop()
        frame = self.get_frame()
        self.window = DisplayWindow(frame)
        self.x = []
        self.y = []
        self.contours = []
        self.run()

    def locate_bubble_contour(self):
        im = self.mean_frame()
        blur = images.gaussian_blur(im, (31, 31))
        threshold = images.threshold(blur, 70)
        contours = images.find_contours(threshold)
        contour = images.sort_contours(contours)[-1]
        return contour

    def get_contour_center(self, c):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY

    def get_frame(self):
        frame = self.cam.get_frame()
        frame = images.crop_and_mask(frame, self.crop, self.mask)
        return frame

    def mean_frame(self):
        ims = [self.get_frame() for i in range(10)]
        im = images.mean(ims)
        im = images.bgr_to_gray(im)
        return im

    def get_crop(self):
        frame = self.cam.get_frame()
        crop_result = images.crop_polygon(frame)
        return crop_result.bbox, crop_result.mask

    def run(self):
        i = 0
        while True:
            i += 1
            self.shaker.ramp(660, 580, 0.2)
            ims = [self.get_frame() for i in range(10)] # Make sure buffered images are used
            im = self.get_frame()
            self.window.set_frame(im)
            contour = self.locate_bubble_contour()
            self.contours.append(contour)
            self.window.plot_contour(contour)

            center = self.get_contour_center(contour)
            self.x.append(center[0])
            self.y.append(center[1])
            self.window.add_center(center)
            self.window.update_positions(self.x, self.y)
            self.window.flush()
            if i % 60 == 0:
                self.window.save(self.figsave)
                self.save_data()

    def save_data(self):
        np.savetxt(self.datasave + '_x.txt', self.x)
        np.savetxt(self.datasave + '_y.txt', self.y)
        np.save(self.datasave + '_c.npy', self.contours)




if __name__ == '__main__':
    bt = BubbleTracker()