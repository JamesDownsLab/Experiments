import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from labvision import camera, images
from scipy import ndimage

cam_num = camera.guess_camera_number()
cam = camera.Camera(cam_num)

frame = cam.get_frame()
# frame = ndimage.rotate(frame, -120, reshape=False)
crop_result = images.crop_polygon(frame)
crop = crop_result.bbox

def get_frame():
    frame = cam.get_frame()
    # frame = ndimage.rotate(frame, -120, reshape=False)
    frame = images.crop(frame, crop)
    return frame

def get_circles(frame):
    red = frame[:, :, 0] - frame[:, :, 2]
    red = images.threshold(red, 65)
    opened = images.opening(red, (31, 31))
    w = opened.shape[1]
    im1, im2 = opened[:, :w//2], opened[:, w//2:]
    images.display(im1)
    m1 = list(images.center_of_mass(im1))
    m2 = list(images.center_of_mass(im2))
    m2[0] += w//2
    return np.array([[m1[0], m1[1], 50], [m2[0], m2[1], 50]])

def get_plot_circles(circles):
    c1 = plt.Circle((circles[0, 0], circles[0, 1]), 25, color='r')
    c2 = plt.Circle((circles[1, 0], circles[1, 1]), 25, color='r')
    return c1, c2


frame = get_frame()
circles = get_circles(frame)


x = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/OffsetRail_temp/Logging/020221_solid_600_foam.txt")
# t = np.loadtxt("/media/data/Data/Orderphobic/TwoIntruders/Logging/261120_liquid_t.txt")

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].imshow(frame)
c1, c2 = get_plot_circles(circles)
c1 = ax[0].add_artist(c1)
c2 = ax[0].add_artist(c2)
bins = np.linspace(0, frame.shape[1], 100)
freq, _ = np.histogram(x, bins=bins)
hist = ax[1].bar(bins[:-1], freq, width=bins[1]-bins[0])
pix_to_mm = 12.5/circles[0, 2]

ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * pix_to_mm))
ax[1].xaxis.set_major_formatter(ticks_x)
fig.subplots_adjust(wspace=0, hspace=0)
ax[1].set_xlabel('x (mm)')
ax[1].set_ylabel('Frequency')
ax[1].set_xlim([0, frame.shape[1]])


plt.hist(x, bins=100)
plt.show()