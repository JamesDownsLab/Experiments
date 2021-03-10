import time
import matplotlib.pyplot as plt
from labvision import camera, images
import numpy as np
from labequipment import shaker
from scipy import ndimage

import os

D = 600
data_save = "/media/data/Data/Orderphobic/TwoIntruders/OffsetRail_temp/Logging/050221_liquid_600_39.txt"
im_save = os.path.splitext(data_save)[0]
if not os.path.exists(im_save):
    os.mkdir(im_save)



cam_num = camera.guess_camera_number()
cam = camera.Camera(cam_num)

frame = cam.get_frame()
frame = ndimage.rotate(frame, 60, reshape=False)
crop_result = images.crop_polygon(frame)
crop = crop_result.bbox

s = shaker.Shaker()
# s.ramp(650, D, 1)
s.change_duty(D)

def get_frame():
    frame = cam.get_frame()
    frame = ndimage.rotate(frame, 60, reshape=False)
    frame = images.crop(frame, crop)
    return frame

def get_both_bounds(frame):
    left_crop_result = images.crop_polygon(frame)
    right_crop_result = images.crop_polygon(frame)
    return left_crop_result, right_crop_result

def get_circles(frame, left_crop, right_crop):
    red = frame[:, :, 0] - frame[:, :, 2]
    red = images.threshold(red, 65)
    opened = images.opening(red, (31, 31))
    w = opened.shape[1]
    im1 = images.crop(opened, left_crop.bbox)
    im2 = images.crop(opened, right_crop.bbox)
    m1 = list(images.center_of_mass(im1))
    m2 = [0, 0]
    # m2 = list(images.center_of_mass(im2))
    m1[0] += left_crop.bbox.xmin
    m1[1] += left_crop.bbox.ymin
    m2[0] += right_crop.bbox.xmin
    m2[1] += right_crop.bbox.ymin

    return np.array([[m1[0], m1[1], 50], [m2[0], m2[1], 50]])

def get_plot_circles(circles):
    c1 = plt.Circle((circles[0, 0], circles[0, 1]), circles[0, 2], color='r')
    c2 = plt.Circle((circles[1, 0], circles[1, 1]), circles[1, 2], color='r')
    return c1, c2


frame = get_frame()
left_crop, right_crop = get_both_bounds(frame)
circles = get_circles(frame, left_crop, right_crop)

plt.ion()
fig, ax = plt.subplots(2, 1, sharex=True)
im_artist = ax[0].imshow(frame)
c1, c2 = get_plot_circles(circles)
c1 = ax[0].add_artist(c1)
# c2 = ax[0].add_artist(c2)
bins = np.linspace(0, frame.shape[1], 100)
hist = ax[1].bar(bins[:-1], np.zeros_like(bins[:-1]), width=bins[1]-bins[0])
fig.subplots_adjust(wspace=0, hspace=0)
ax[1].set_xlabel('x {pix}')
ax[1].set_ylabel('Frequency')
ax[1].set_xlim([0, frame.shape[1]])
fig.show()

import os

if os.path.exists(data_save):
    x = np.loadtxt(data_save).tolist()
else:
    x = []
    t = []
i = 0
start_time = time.time()
for step in range(5000):
    for repeat in range(20):
        i += 1
        # frame = cam.get_frame()
        # frame = images.crop(frame, crop)
        frame = get_frame()
        if i % 100 == 0:
            images.save(cam.get_frame(), f"{im_save}/{i}.png")
        circles = get_circles(frame, left_crop, right_crop)
        x.append(circles[0, 0])
        x.append(circles[1, 0])
        current_time = time.time() - start_time
        c1.remove()
        # c2.remove()
        c1, c2 = get_plot_circles(circles)
        ax[0].add_artist(c1)
        # ax[0].add_artist(c2)
        im_artist.set_data(frame)
        ax[0].set_title(f"Elapsed Time: {current_time:.1f}, data: {len(x)//2}")
        freq, _ = np.histogram(np.array(x)[::2], bins=bins)
        for rect, h, in zip(hist, freq):
            rect.set_height(h)
        ax[1].set_ylim([0, max(freq)])
        fig.canvas.flush_events()
        fig.canvas.start_event_loop(0.5)

    np.savetxt(data_save, x)

s.change_duty(0)