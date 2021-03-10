from particletracking import dataframes

data = dataframes.DataStore("/media/NAS/share/James/HighSpeed/test.hdf5")
print(data.df.head())