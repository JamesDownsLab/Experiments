import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import filehandling

files = filehandling.get_directory_filenames("/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/2,10mm_b/*.hdf5")

for file in tqdm(files):
    data = pd.read_hdf(file)
    if 'down' in file:
        start = 530
        end = 400
    else:
        start = 400
        end = 530
    duties = np.round(np.linspace(start, end, len(data.index)))
    data['duty'] = duties
    data.to_hdf(file, 'data')