import filehandling
from particletracking import statistics, dataframes

directory = filehandling.open_directory('/media/data/Data')
files = filehandling.get_directory_filenames(directory+'/*.hdf5')

params = ['order', 'density']

for file in files:
    print(file)
    data = dataframes.DataStore(file)
    calculator = statistics.PropertyCalculator(data)
    calculator.distance()
    data.save()
    # calculator.order()
    # calculator.density()
    # calculator.density_duty()
    # calculator.order_duty()




