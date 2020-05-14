import sys
import numpy as np
from sgpp_create_response_surface import getSetup
import pickle
import logging
from sgpp_precalc_parallel import precalc_parallel

numMCPoints = 100

gridType, _, degree, _, qoi, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
    boundaryLevel = getSetup()

test_strategies = [
    'individual-testing',
    'two-stage-testing',
    'binary-splitting',
    'RBS',
    'purim',
    'sobel'
]
group_sizes = list(range(1, 33))

# same points for all methods
np.random.seed(42)
unitpoints = np.random.rand(numMCPoints, 3)
points = []
for i, unitpoint in enumerate(unitpoints):
    for k, group_size in enumerate(group_sizes):
        point = [lb[d] + (ub[d]-lb[d])*unitpoint[d] for d in range(3)] + [group_size]
        points.append(point)

for test_strategy in test_strategies:
    mcData = {}
    multiprocessing_dict = precalc_parallel(points, sample_size, test_duration, num_simultaneous_tests,
                                            number_of_instances, test_strategy)
    regular_dict = {}
    for key in multiprocessing_dict:
        regular_dict[key] = multiprocessing_dict[key]

    filename = f'precalc/values/group_mc{numMCPoints}_{test_strategy}__{number_of_instances}repetitions.pkl'
    with open(filename, 'wb+') as fp:
        pickle.dump(regular_dict, fp)

    print(f'calculated data for {numMCPoints} random points, saved as {filename}')
