import filehandling
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from tqdm import tqdm

files = filehandling.get_directory_filenames("/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/2,10mm_b/*.hdf5")
# files2 = filehandling.get_directory_filenames("/media/data/Data/BallBearing/HIPS/HysteresisExperimentsSeptember2023/2,42mm/*.hdf5")

# files = files1 + files2

def angle_diff(df):
    points = df[['x', 'y']].values
    tree = spatial.cKDTree(points)
    _, indices = tree.query(points, 7)
    orders = df['hexatic_order'].values[indices]
    order = np.mean(orders, axis=1)
    df['correlated_order'] = order
    return df

for file in tqdm(files):
    data = pd.read_hdf(file)
    data = data.groupby('frame').apply(angle_diff)
    data.to_hdf(file, 'data')