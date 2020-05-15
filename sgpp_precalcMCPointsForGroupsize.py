import sys
import numpy as np
from sgpp_create_response_surface import getSetup
import pickle
import logging
from sgpp_precalc_parallel import precalc_parallel


def calculate_data(points3D):
    '''
    points3D are the points without group sizes
    '''
    numMCPoints = len(points3D)
    print(f'Got {numMCPoints} 3D points')
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
    points = []
    for i, point3D in enumerate(points3D):
        for k, group_size in enumerate(group_sizes):
            point = point3D + [group_size]
            points.append(point)

    print(f'Now calculating {len(points)} * {len(test_strategies)} evaluations')
    for test_strategy in test_strategies:
        multiprocessing_dict = precalc_parallel(points, sample_size, test_duration, num_simultaneous_tests,
                                                number_of_instances, test_strategy)
        regular_dict = {}
        for key in multiprocessing_dict:
            regular_dict[key] = multiprocessing_dict[key]

        filename = f'precalc/values/group_mc{numMCPoints}_{test_strategy}_{number_of_instances}repetitions.pkl'
        with open(filename, 'wb+') as fp:
            pickle.dump(regular_dict, fp)

        print(f'calculated data for {numMCPoints} points, saved as {filename}')


def calcualte_randomly(numMCPoints):
    gridType, _, degree, _, qoi, name, sample_size, num_daily_tests, \
        test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
        boundaryLevel = getSetup()
    np.random.seed(43)
    unitpoints = np.random.rand(numMCPoints, 3)
    points3D = []
    for i, unitpoint in enumerate(unitpoints):
        point = [lb[d] + (ub[d]-lb[d])*unitpoint[d] for d in range(3)]
        points3D.append(point)
    calculate_data(points3D)


def paper_values():
    # Values used in the actual paper
    # !! results in 24 Points and is thus saved as mc24 !!
    success_rates_test = [0.75, 0.99]
    probabilities_sick = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    false_posivite_rate = 0.01
    points3D = []
    for success_rate_test in success_rates_test:
        for probability_sick in probabilities_sick:
            points3D.append([probability_sick, success_rate_test, false_posivite_rate])
    calculate_data(points3D)


if __name__ == "__main__":
    # calcualte_randomly(100)
    paper_values()
