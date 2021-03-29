import os
import warnings

import filehandling
from particletracking import tracking
import track
from tqdm import tqdm
from labvision import images, video
from particletracking import dataframes, statistics



def get_crop_result(file):
    vid = video.ReadVideo(file)
    frame = vid.read_next_frame()
    return images.crop_polygon(frame)

def go(files, crop_result):
    for i, file in tqdm(enumerate(files)):
        # file = direc + '/' + file
        name, ext = os.path.splitext(file)
        if ext == '.MP4':
            if i == 0 and crop_result is None:
                print('cropping')
                crop_result = get_crop_result(file)
            data_file = name + '.hdf5'
            if not os.path.exists(data_file):
                tracker = tracking.ParticleTracker(file, track.HoughManager(crop_result=crop_result), True)
                tracker.track()
                # tracker.link()
    return crop_result

direc1 = "/media/data/Data/FirstOrder/Susceptibility/Flat2"
files1 = filehandling.get_directory_filenames(direc1+'/*.MP4')
crop_result = go(files1, None)
direc2 = "/media/data/Data/FirstOrder/Susceptibility/Dimpled2"
files2 = filehandling.get_directory_filenames(direc2+'/*.MP4')
go(files2, crop_result)

files1 = filehandling.get_directory_filenames(direc1+'/*.hdf5')
files2 = filehandling.get_directory_filenames(direc2+'/*.hdf5')
files = files1 + files2
for file in files:
    data = dataframes.DataStore(file)
    calculator = statistics.PropertyCalculator(data)
    calculator.order_long()
    calculator.order_nearest_6()
    calculator.density()