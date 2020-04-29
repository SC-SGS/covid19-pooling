import pysgpp
import numpy as np
from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import precalc_parallel
import pickle
import sys


gridType, dim, degree, test_strategy, qoi, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests, evalType, scale_factor_pop,\
    number_of_instances, lb, ub = getSetup()
# load precalculated data
savePath = "/home/rehmemk/git/covid19-pooling/precalc/"
precalcValuesFileName = savePath + f"precalc_values_{test_strategy}.pkl"
try:
    with open(precalcValuesFileName, 'rb') as f:
        precalculatedValues = pickle.load(f)
    print(f'loaded precalculated evaluations from {precalcValuesFileName}')
except (FileNotFoundError):
    print('could not find precalculated data at {}\nCreating new data file.'.format(
        precalcValuesFileName))
    precalculatedValues = {}

degree = 3
level = 3
grid = pysgpp.Grid_createNakBsplineBoundaryGrid(dim, degree)
grid.getGenerator().regular(level)
points = []
num_to_calculate = 0
gridStorage = grid.getStorage()

for i in range(grid.getSize()):
    point = gridStorage.getPointCoordinates(i)
    for d in range(dim):
        point[d] = lb[d] + (ub[d]-lb[d])*point[d]
    prob_sick = point[0]
    success_rate_test = point[1]
    false_positive_rate = point[2]
    group_size = int(point[3])
    key = tuple([prob_sick, success_rate_test, false_positive_rate,
                 group_size, evalType, number_of_instances])
    if key not in precalculatedValues:
        points.append(point)
        num_to_calculate += 1
print(f' Grid of dim {dim}, level {level} has {grid.getSize()} points, {num_to_calculate} are not yet precalculated')

multiprocessing_dict = precalc_parallel(points, sample_size, test_duration, num_simultaneous_tests,
                                        number_of_instances, scale_factor_pop, test_strategy, evalType)
for key in multiprocessing_dict:
    precalculatedValues[key] = multiprocessing_dict[key]
with open(precalcValuesFileName, "wb") as f:
    pickle.dump(precalculatedValues, f)
print(f"\ncalculated {num_to_calculate} new evaluations")