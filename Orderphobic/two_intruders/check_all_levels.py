from labequipment import shaker, arduino, stepper
from labvision import camera, images

import numpy as np
import matplotlib.pyplot as plt

import time

data_dir = "/media/data/Data/Orderphobic/TwoIntruders/LevelSampling"
ard = arduino.Arduino("/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_5573532393535190E022-if00")
my_stepper = stepper.Stepper(ard)

my_shaker = shaker.Shaker()
my_shaker.change_duty(600)

cam = camera.Camera(camera.guess_camera_number())
frame = cam.get_frame()
crop_result = images.crop_polygon(frame)
frame = images.crop_and_mask(frame, crop_result.bbox, crop_result.mask)
bins = np.linspace(0, frame.shape[1], 100)
np.savetxt(f"{data_dir}/bins.txt", bins)

def run():
    pos1 = -360
    pos2 = -360
    my_stepper.move_motor(1, 360*15, '-')
    time.sleep(90)
    my_stepper.move_motor(2, 360*15, '-')
    time.sleep(90)

    for j in range(9):
        for i in range(9):
            print(pos1, pos2)
            x = analyse(pos1, pos2)
            plot(x, pos1, pos2)
            pos2 += 90
            my_stepper.move_motor(2, 90*15, '+')
            time.sleep(30)
        pos1 += 90
        my_stepper.move_motor(1, 90*15, '+')
        time.sleep(30)
        pos2 -= 810
        my_stepper.move_motor(2, 810*15, '-')  # This was wrong before, should be motor 2
        time.sleep(120)


def analyse(p1, p2):
    x = []
    t = time.time()
    while (time.time() - t) < 600:
        time.sleep(4)
        try:
            frame = cam.get_frame()
            frame = images.crop_and_mask(frame, crop_result.bbox, crop_result.mask)
            frame_blurred = images.gaussian_blur(frame, (15, 15))
            red = frame_blurred[:, :, 2] - frame_blurred[:, :, 0]
            circles = images.find_circles(red, 25, 200, 5, 31, 31)
            if circles[0, 0] > circles[1, 0]:
                x.append(circles[1, 0])
                x.append(circles[0, 0])
            else:
                x.append(circles[0, 0])
                x.append(circles[1, 0])
        except:
            pass
    x = np.array(x)
    np.savetxt(f"{data_dir}/x_{p1}_{p2}.txt", x)
    return x

def plot(x, p1, p2):
    freq, _ = np.histogram(x, bins=bins)
    plt.figure()
    plt.bar(bins[:-1], freq, width=bins[1]-bins[0])
    plt.xlabel('x[pix]')
    plt.ylabel('frequency')
    plt.title(f"{p1}_{p2}")
    plt.savefig(f"{data_dir}/{p1}_{p2}.png")

run()
my_shaker.change_duty(0)