import filehandling
import pandas as pd
import numpy as np

directory = "/media/data/Data/BallBearing/HIPS/PhaseDiagramsNewPlate/2,42mm"

densities = ['65%', '75%', '85%']

for density in densities:
    files = filehandling.get_directory_filenames(f"{directory}/{density}/*.hdf5")
    abs_dictionary = {}
    mean_dictionary = {}
    duties = []
    mean_data = []
    error_data = []
    for file in files:
        duty = int(file[-8:-5])
        data = pd.read_hdf(file)
        data['hex_abs'] = np.abs(data.hexatic_order)
        mean_order = np.mean(data.groupby('frame').mean()['hex_abs'])
        std_order = np.std(data.groupby('frame').mean()['hex_abs'])
        # mean_order = np.mean(np.abs(data.hexatic_order))
        # std_order = np.std(np.abs(data.hexatic_order))
        mean_data.append(mean_order)
        error_data.append(std_order)
        duties.append(duty)
    np.savetxt(f"{directory}/{density}_duty.txt", duties)
    np.savetxt(f"{directory}/{density}_order_frame.txt", mean_data)
    np.savetxt(f"{directory}/{density}_order_frame_err.txt", error_data)

