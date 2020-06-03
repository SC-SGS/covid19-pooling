import sys
import numpy as np
from sgpp_create_response_surface import getSetup
import pickle
import logging
from sgpp_precalc_parallel import precalc_parallel

numMCPoints = 100

gridType, dim, degree, _, qoi, name, sample_size, num_daily_tests, \
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

# TEMPORARY!
sample_size = 50000
number_of_instances = 20

# same points for all methods
np.random.seed(43)
unitpoints = np.random.rand(numMCPoints, dim)
points = [lb + (ub-lb)*point for point in unitpoints]
for test_strategy in test_strategies:
    mcData = {}
    multiprocessing_dict = precalc_parallel(points, sample_size, test_duration, num_simultaneous_tests,
                                            number_of_instances, test_strategy)
    regular_dict = {}
    for key in multiprocessing_dict:
        regular_dict[key] = multiprocessing_dict[key]

    filename = f'precalc/values/mc{numMCPoints}_{test_strategy}_{dim}dim_{number_of_instances}repetitions_{int(sample_size/1000)}kpop.pkl'
    with open(filename, 'wb+') as fp:
        pickle.dump(regular_dict, fp)

    print(f'calculated data for {numMCPoints} random points, saved as {filename}')
