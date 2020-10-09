from labvision import video, images
from particletracking import dataframes
import os
import numpy as np
import cv2

def mask_from_polygon(im, points):
    msk = np.zeros_like(im, dtype=np.uint8)
    msk = cv2.fillPoly(msk, np.array([points], dtype=np.int32), images.WHITE)
    im = images.mask(im, msk[:, :, 0])
    return im

def crop_then_mask(im):
    im = images.crop(im, metadata['crop'])
    return mask_from_polygon(im, metadata['boundary'])

def get_crop_region(im, corners, factor=0.8):
    center = np.mean(corners, axis=0)
    vectors = corners - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    lengths = np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)
    new_vectors = np.array([np.cos(angles), np.sin(angles)]).T
    new_vectors *= lengths[:, np.newaxis] * factor
    new_corners = center + new_vectors
    return new_corners

vid_file = "/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense/0.1_down_1.MP4"
data_file = os.path.splitext(vid_file)[0] + '.hdf5'

vid = video.ReadVideo(vid_file)
metadata = dataframes.load_metadata(data_file)

print(metadata['boundary'].shape)

start = vid.read_frame(0)
middle = vid.read_frame(vid.num_frames // 2)
end = vid.read_frame(vid.num_frames - 1)

start = crop_then_mask(start)
middle = crop_then_mask(middle)
end = crop_then_mask(end)

save_direc = "/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense/HexagonFigures"

images.save(start, f'{save_direc}/start.png')
images.save(middle, f'{save_direc}/middle.png')
images.save(end, f'{save_direc}/end.png')

new_corners = get_crop_region(start, metadata['boundary'])

start = images.draw_polygon(start, new_corners, color=images.RED, thickness=8)
middle = images.draw_polygon(middle, new_corners, color=images.RED, thickness=8)
end = images.draw_polygon(end, new_corners, color=images.RED, thickness=8)

images.save(start, f'{save_direc}/start_hex.png')
images.save(middle, f'{save_direc}/middle_hex.png')
images.save(end, f'{save_direc}/end_hex.png')
