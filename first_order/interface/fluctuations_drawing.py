import numpy as np
import matplotlib.pyplot as plt
import filehandling
from labvision import images
from math import atan, sin, cos
import cv2

direc = "/media/data/Data/January2020/RecordFluctuatingInterface/Quick/first_frames/"

pixels_to_mms = 0.11797
x = np.loadtxt(direc+'x.txt') / pixels_to_mms
hs = np.loadtxt(direc+'hs.txt') / pixels_to_mms
h = hs[20]

files = filehandling.get_directory_filenames(direc+'/*.png')

im = images.load(files[20])

def rotate_points(points, center, a):
    rot = np.array(((cos(a), -sin(a)), (sin(a), cos(a))))
    a1 = points - center
    a2 = rot @ a1.T
    a3 = a2.T + center
    return a3

p1 = [333, 1318]
p2 = [1784, 528]
midpoint = np.array([p1[0] - p2[0], p1[1]-p2[1]])
h = h + midpoint[1]
x = x + midpoint[0]

m = (p2[1] - p1[1]) / (p2[0] - p1[0])
a = -atan(m)
print(a)
points = np.vstack((x, h)).T
print(points.shape)
points = rotate_points(points, midpoint, -a)
print(points[0])
print(points[1])

points[:, 0] += -points[0, 0] + p1[0]
points[:, 1] += -points[0, 1] + p1[1]

points = np.int32(points)
# points = points.reshape((points.shape[0], 1, points.shape[1]))
# lines = [((a[0], b[0]), (a[1], b[1])) for a, b in zip(points[1:], points[:-1])]
# print(lines)
im = cv2.polylines(im, [points], False, images.YELLOW, 10)
im = images.rotate(im, 30)
# im = images.draw_contours(im, [points], images.YELLOW, 10)
images.display(im)
images.save(im, direc+'boundary.jpeg')
# plt.savefig(direc+'boundary.jpeg', quality=95, dpi=300)
# plt.show()

