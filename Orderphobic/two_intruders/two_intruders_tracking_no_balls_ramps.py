import time
import matplotlib.pyplot as plt
from labvision import camera, images
import numpy as np
from scipy import ndimage
from labequipment import shaker




data_save = "/media/data/Data/Orderphobic/TwoIntruders/Logging/301120_liquid_ramps_x.txt"


cam_num = camera.guess_camera_number()
cam = camera.Camera(cam_num)

frame = cam.get_frame()
frame = ndimage.rotate(frame, -120, reshape=False)
crop_result = images.crop_polygon(frame)
crop = crop_result.bbox

s = shaker.Shaker()
s.ramp(670, 630, 1)

def get_frame():
    frame = cam.get_frame()
    frame = ndimage.rotate(frame, -120, reshape=False)
    frame = images.crop(frame, crop)
    return frame

def get_circles(frame):
    frame_blurred = images.gaussian_blur(frame, (15, 15))
    red = frame_blurred[:, :, 2] - frame_blurred[:, :, 0]
    circles = images.find_circles(red, 120, 200, 5, 55, 59)
    if len(circles) > 2:
        if len(circles.shape) == 2:
            return circles[:2, :]
        else:
            return np.stack((circles, circles))
    return circles

def get_circles(frame):
    red = frame[:, :, 0] - frame[:, :, 2]
    opened = images.opening(red, (31, 31))
    w = opened.shape[1]
    im1, im2 = opened[:, :w//2], opened[:, w//2:]
    m1 = list(images.center_of_mass(im1))
    m2 = list(images.center_of_mass(im2))
    m2[0] += w//2
    return np.array([[m1[0], m1[1], 50], [m2[0], m2[1], 50]])

def get_plot_circles(circles):
    c1 = plt.Circle((circles[0, 0], circles[0, 1]), circles[0, 2], color='r')
    c2 = plt.Circle((circles[1, 0], circles[1, 1]), circles[1, 2], color='r')
    return c1, c2


frame = get_frame()
circles = get_circles(frame)

plt.ion()
fig, ax = plt.subplots(2, 1, sharex=True)
im_artist = ax[0].imshow(frame)
c1, c2 = get_plot_circles(circles)
c1 = ax[0].add_artist(c1)
c2 = ax[0].add_artist(c2)
bins = np.linspace(0, frame.shape[1], 100)
hist = ax[1].bar(bins[:-1], np.zeros_like(bins[:-1]), width=bins[1]-bins[0])
fig.subplots_adjust(wspace=0, hspace=0)
ax[1].set_xlabel('x {pix}')
ax[1].set_ylabel('Frequency')
ax[1].set_xlim([0, frame.shape[1]])
fig.show()

import os

if os.path.exists(data_save) and 1 == 2:
    x = np.loadtxt(data_save).tolist()
else:
    x = []
start_time = time.time()
for step in range(200):
    s.ramp(670, 600, 0.1)
    frame = get_frame()
    circles = get_circles(frame)
    x.append(circles[0, 0])
    x.append(circles[1, 0])
    c1.remove()
    c2.remove()
    c1, c2 = get_plot_circles(circles)
    ax[0].add_artist(c1)
    ax[0].add_artist(c2)
    im_artist.set_data(frame)
    ax[0].set_title(f"Ramps: {len(x)//2}")
    freq, _ = np.histogram(x, bins=bins)
    for rect, h, in zip(hist, freq):
        rect.set_height(h)
    ax[1].set_ylim([0, max(freq)])
    fig.canvas.flush_events()
    fig.canvas.start_event_loop(0.5)
    np.savetxt(data_save, x)