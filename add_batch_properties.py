import filehandling
from particletracking import statistics, dataframes
import os

direc1 = "/media/data/Data/FirstOrder/Susceptibility/Flat2"
files1 = filehandling.get_directory_filenames(direc1+'/*.hdf5')
direc2 = "/media/data/Data/FirstOrder/Susceptibility/Dimpled2"
files2 = filehandling.get_directory_filenames(direc2+'/*.hdf5')

files = files1 + files2
for file in files:
    # if not file.endswith('hdf5'):
    #     pass
    # file = os.path.join(directory, file)
    print(file)
    with dataframes.DataStore(file) as data:
        calculator = statistics.PropertyCalculator(data)
        calculator.distance()
        data.save()
        # if 'order' not in data.headings:
        #     calculator.order()
        # if 'density' not in data.headings:
        #     calculator.density()
        # if 'order_long' not in data.headings:
        #     calculator.order_long()
        # calculator.density_duty()
        # calculator.order_duty()
        # calculator.order_nearest_6()
        # calculator.density()
        # data.save()



