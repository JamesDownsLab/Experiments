import filehandling
import numpy as np
from tqdm import tqdm
import pandas as pd

files = filehandling.get_directory_filenames("/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/2,42mm/*.hdf5")
print(files)
for file in tqdm(files):
    data = pd.read_hdf(file)
    data['hex_abs'] = np.abs(data['hexatic_order'])
    mean = data.groupby('duty')['hex_abs'].mean()
    mean.to_hdf(file[:-5]+'_duty_mean.hdf5', 'data')
