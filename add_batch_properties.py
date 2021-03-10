import filehandling
from particletracking import statistics, dataframes
import os

# directory1 = "/media/data/Data/FirstOrder/PhaseDiagram/FlatPlate2Feb2021"
# files1 = filehandling.get_directory_filenames(directory1+'/*.hdf5')
# directory2 = "/media/data/Data/FirstOrder/PhaseDiagram/DimpledPlateFeb2021"
# files2 = filehandling.get_directory_filenames(directory2+'/*.hdf5')
# # files = os.listdir(directory)
# # print(files)
directory = "/media/data/Data/FirstOrder/Hysterisis/5Repeats/RedTrayDense"
files = filehandling.get_directory_filenames(directory + '/*.hdf5')

# files = files1 + files2
# files = files2
for file in files:
    # if not file.endswith('hdf5'):
    #     pass
    # file = os.path.join(directory, file)
    print(file)
    with dataframes.DataStore(file) as data:
        calculator = statistics.PropertyCalculator(data)
        # calculator.distance()
        # data.save()
        # if 'order' not in data.headings:
        #     calculator.order()
        # if 'density' not in data.headings:
        #     calculator.density()
        if 'order_long' not in data.headings:
            calculator.order_long()
        # calculator.density_duty()
        # calculator.order_duty()
        data.save()



