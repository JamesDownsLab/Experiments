import filehandling
from particletracking import dataframes
import trackpy as tp

# direc1 = filehandling.open_directory(initialdir='/media/data/Data')
# files1 = filehandling.get_directory_filenames(direc1+'/*.hdf5')
#
# direc2 = filehandling.open_directory(initialdir='/media/data/Data')
# files2 = filehandling.get_directory_filenames(direc1+'/*.hdf5')
#
# files = files1 + files2

files = ["/media/data/Data/FirstOrder/PhaseDiagram/FlatPlate2Feb2021/2000.hdf5",]

for file in files:
    with dataframes.DataStore(file) as data:
        d = data.df.copy()
        print(d.head())
    d = d.reset_index()
    print(d.head())
    d = tp.link(d, d.r.mean())
