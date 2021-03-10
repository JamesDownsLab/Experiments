import time
import matplotlib.pyplot as plt
from labvision import camera, images
import numpy as np
from scipy import ndimage
from labequipment import shaker

import os

# EXP_CAM = "/dev/snd/by-id/usb-046d_HD_Pro_Webcam_C920_167C9A7F-02"

R = 60
D = 600
date = '090221'
n = 3
data_save = f"/media/data/Data/Orderphobic/TwoIntruders/OneIntruder/Logging/{date}_{D}_{n}.txt"
im_save = os.path.splitext(data_save)[0]
if not os.path.exists(im_save):
    os.mkdir(im_save)



# cam_num = camera.guess_camera_number()
cam = camera.Camera(0)

frame = cam.get_frame()
if R != 0:
    frame = ndimage.rotate(frame, R, reshape=False)
crop_result = images.crop_polygon(frame)
crop = crop_result.bbox

s = shaker.Shaker()
# s.ramp(650, D, 1)
s.change_duty(D)

def get_frame():
    frame = cam.get_frame()
    if R != 0:
        frame = ndimage.rotate(frame, R, reshape=False)
    frame = images.crop(frame, crop)
    return frame

def get_circles(frame):
    red = frame[:, :, 0] - frame[:, :, 2]
    red = images.threshold(red, 65)
    opened = images.opening(red, (31, 31))
    m = list(images.center_of_mass(opened))
    return np.array([m[0], m[1], 50])

def get_plot_circles(circles):
    c1 = plt.Circle((circles[0], circles[1]), circles[2], color='r')
    return c1


frame = get_frame()
circles = get_circles(frame)

plt.ion()
fig, ax = plt.subplots(3, 1, sharex=True)
im_artist = ax[0].imshow(frame)
c1 = get_plot_circles(circles)
c1 = ax[0].add_artist(c1)
bins = np.linspace(0, frame.shape[1], 100)
hist = ax[1].bar(bins[:-1], np.zeros_like(bins[:-1]), width=bins[1]-bins[0])
fig.subplots_adjust(wspace=0, hspace=0)
ax[1].set_xlabel('x {pix}')
ax[1].set_ylabel('Frequency')
ax[1].set_xlim([0, frame.shape[1]])
plot, = ax[2].plot([], [])
ax[2].set_xlim([0, frame.shape[1]])

fig.show()

import os

if os.path.exists(data_save):
    x = np.loadtxt(data_save).tolist()
else:
    x = []
    t = []
i = 0
start_time = time.time()
for step in range(3000):
    for repeat in range(20):
        frame = get_frame()
        if i % 5 == 0:
            images.save(cam.get_frame(), f"{im_save}/{i}.png")
        circles = get_circles(frame)
        x.append(circles[0])
        current_time = time.time() - start_time
        c1.remove()
        c1 = get_plot_circles(circles)
        ax[0].add_artist(c1)
        im_artist.set_data(frame)
        ax[0].set_title(f"Elapsed Time: {current_time:.1f}, data: {len(x)//2}")
        freq, _ = np.histogram(x, bins=bins)
        for rect, h, in zip(hist, freq):
            rect.set_height(h)
        ax[1].set_ylim([0, max(freq)])
        ax[2].set_ylim([0, len(x)])
        plot.set_xdata(x)
        plot.set_ydata(np.arange(len(x)))
        fig.canvas.flush_events()
        fig.canvas.start_event_loop(0.5)

    np.savetxt(data_save, x)

s.change_duty(0)