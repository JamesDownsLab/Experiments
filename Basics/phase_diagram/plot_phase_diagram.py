import filehandling
from particletracking import dataframes

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

directory = filehandling.open_directory('/media/data/Data')
files = filehandling.get_directory_filenames(directory+'/*.hdf5')

duties = np.arange(750, 525, 25)
numbers = np.arange(1700, 2400, 50)

counts = {n: {d: 0 for d in duties} for n in numbers}
for file in files:
    filename = os.path.splitext(os.path.split(file)[1])[0]
    number, duty = filename.split('_')
    number = int(number)
    duty = int(duty)
    data = dataframes.DataStore(file)
    order = data.df.order
    count = np.count_nonzero(order > 0.95)/len(order)
    counts[number][duty] = count
    # print(count)
counts = pd.DataFrame(counts)
counts.plot()
plt.show()