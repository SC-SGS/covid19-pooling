import sys
import numpy as np
from sgpp_create_response_surface import getSetup
import pickle
import logging
from sgpp_precalc_parallel import precalc_parallel

numMCPoints = 1000

gridType, dim, degree, test_strategy, qoi, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests, evalType, scale_factor_pop,\
    number_of_instances, lb, ub = getSetup()


unitpoints = np.random.rand(numMCPoints, dim)
mcData = {}
points = [lb + (ub-lb)*point for point in unitpoints]
multiprocessing_dict = precalc_parallel(points, sample_size, test_duration, num_simultaneous_tests,
                                        number_of_instances, scale_factor_pop, test_strategy, evalType)
regular_dict = {}
for key in multiprocessing_dict:
    regular_dict[key] = multiprocessing_dict[key]

filename = f'precalc/values/mc{numMCPoints}_{test_strategy}_{dim}dim_{qoi}.pkl'
with open(filename, 'wb+') as fp:
    pickle.dump(regular_dict, fp)

print(f'calculated data for {numMCPoints} random points, saved as {filename}')
