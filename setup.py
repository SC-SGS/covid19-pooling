import logging
import sys
import numpy as np


def getSetup():
    # logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

    gridType = 'nakBsplineBoundary'
    dim = 4
    degree = 3

    #test_strategy = 'individual-testing'
    # test_strategy = 'binary-splitting'
    #test_strategy = 'two-stage-testing'
    # test_strategy = 'RBS'
    test_strategy = 'purim'
    # test_strategy = 'sobel'

    qoi = 'ppt'
    # qoi = 'time'
    # qoi = 'numtests'
    # qoi = 'numconfirmed'

    name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}'

    # reference values. These are defined in sgpp_simStorage::init too.
    # TODO: That's dangerous. Define them only once!
    sample_size = 100000  # 1000000  # 100000
    num_daily_tests = int(sample_size/100)  # 10000
    test_duration = 5
    num_simultaneous_tests = int(num_daily_tests*test_duration/24.0)
    number_of_instances = 10

    prob_sick_range = [0.001, 0.3]
    success_rate_test_range = [0.5, 1.0]
    #success_rate_test_range = [0.5, 0.99]
    false_positive_rate_test_range = [0.0, 0.2]
    #false_positive_rate_test_range = [0.01, 0.2]
    group_size_range = [1, 32]
    lb = np.array([prob_sick_range[0], success_rate_test_range[0],
                   false_positive_rate_test_range[0], group_size_range[0]])
    ub = np.array([prob_sick_range[1], success_rate_test_range[1],
                   false_positive_rate_test_range[1], group_size_range[1]])
    lb = lb[:dim]
    ub = ub[:dim]
    # 1 + how much levels the boundary is coarser than the main axes,
    # 0 means one level finer, 1 means same level, 2 means one level coarser, etc.
    boundaryLevel = 2

    return gridType, dim, degree, test_strategy, qoi, name, sample_size, num_daily_tests,\
        test_duration, num_simultaneous_tests, number_of_instances, lb, ub, boundaryLevel
