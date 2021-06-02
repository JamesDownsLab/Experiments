from labequipment import shaker
from labvision import camera, images
import time

s = shaker.Shaker()
cam_num = camera.guess_camera_number()
cam = camera.Camera(cam_num=cam_num)

save_dir = "/media/data/Data/August2020/balance_test/"

for i in range(20):
    s.ramp(650, 600, 0.2)
    im = cam.get_frame()
    images.save(im, save_dir + '{}.png'.format(i))
    time.sleep(5)
