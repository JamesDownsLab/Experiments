import filehandling
import numpy as np
from tqdm import tqdm
import pandas as pd

files = filehandling.get_directory_filenames("/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/2,10mm_b/*.hdf5")
print(files)
for file in tqdm(files):
    data = pd.read_hdf(file)
    print(data.head())
    data['correlated_order_abs'] = np.abs(data['correlated_order'])
    mean = data.groupby('duty')['correlated_order_abs'].mean()
    mean.to_hdf(file[:-5]+'_duty_mean_corr.hdf5', 'data')
