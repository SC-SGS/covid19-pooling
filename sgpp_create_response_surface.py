import pysgpp
import numpy as np
import sys
import pickle
import time
import logging
import matplotlib.pyplot as plt
from sgpp_simStorage import sgpp_simStorage, objFuncSGpp


def getSetup():
    # logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

    gridType = 'nakBsplineBoundary'
    dim = 4
    degree = 3

    # test_strategy = 'individual-testing'
    # test_strategy = 'binary-splitting'
    test_strategy = 'two-stage-testing'
    # test_strategy = 'RBS'
    # test_strategy = 'purim'
    # test_strategy = 'sobel'

    qoi = 'time'
    # qoi = 'numtests'
    # qoi = 'numconfirmed'
    #qoi = 'ppt'

    name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}'

    # reference values. These are defined in sgpp_simStorage::init too.
    # TODO: That's dangerous. Define them only once!
    sample_size = 100000
    num_daily_tests = 1000
    test_duration = 5
    num_simultaneous_tests = int(num_daily_tests*test_duration/24.0)
    number_of_instances = 20  # 5

    prob_sick_range = [0.001, 0.3]
    success_rate_test_range = [0.5, 0.99]  # [0.3, 0.99]
    false_positive_rate_test_range = [0.01, 0.2]
    group_size_range = [1, 32]
    lb = np.array([prob_sick_range[0], success_rate_test_range[0],
                   false_positive_rate_test_range[0], group_size_range[0]])
    ub = np.array([prob_sick_range[1], success_rate_test_range[1],
                   false_positive_rate_test_range[1], group_size_range[1]])
    # 1 + how much levels the boundary is coarser than the main axes,
    # 0 means one level finer, 1 means same level, 2 means one level coarser, etc.
    boundaryLevel = 2

    return gridType, dim, degree, test_strategy, qoi, name, sample_size, num_daily_tests,\
        test_duration, num_simultaneous_tests, number_of_instances, lb, ub, boundaryLevel


def getName(refineType, test_strategy, qoi, dim, degree, level, numPoints):
    if refineType == 'regular':
        name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}_level{level}'
    elif refineType == 'adaptive':
        name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}_adaptive{numPoints}'
    return name


def create_reSurf(objFunc, lb, ub, gridType, degree, boundaryLevel, refineType, level, numPoints,
                  initialLevel, numRefine, verbose):
    reSurf = pysgpp.SplineResponseSurface(objFunc, pysgpp.DataVector(lb[:dim]),
                                          pysgpp.DataVector(ub[:dim]), pysgpp.Grid.stringToGridType(gridType),
                                          degree, boundaryLevel)
    start = time.time()
    if refineType == 'regular':
        reSurf.regular(level)
    elif refineType == 'adaptive':
        reSurf.surplusAdaptive(numPoints, initialLevel, numRefine, verbose)
    runtime = time.time()-start
    logging.info('\nDone. Created response surface with {} grid points, took {}s'.format(reSurf.getSize(), runtime))
    objFunc.cleanUp()
    return reSurf


def save_reSurf(reSurf, refineType, test_strategy, qoi, dim, degree, level, numPoints):
    reSurfName = getName(refineType, test_strategy, qoi, dim, degree, level, numPoints)
    path = 'precalc/reSurf'
    # serialize the resposne surface
    # gridStr = reSurf.serializeGrid()
    gridStr = reSurf.getGrid().serialize()
    coeffs = reSurf.getCoefficients()
    # save it to files
    with open(f'{path}/grid_{reSurfName}.dat', 'w+') as f:
        f.write(gridStr)
    # coeffs.toFile('data/coeffs.dat')
    # sgpp DataVector and DataMatrix from File are buggy
    dummyCoeff = np.array([coeffs[i] for i in range(coeffs.getSize())])
    np.savetxt(f'{path}/np_coeff_{reSurfName}.dat', dummyCoeff)
    print(f'saved response surface as {path}/{reSurfName}')


def load_response_Surface(refineType, test_strategy, qoi, dim, degree, level, numPoints, lb, ub):
    name = getName(refineType, test_strategy, qoi, dim, degree, level, numPoints)

    dummyCoeff = np.loadtxt(f'precalc/reSurf/np_coeff_{name}.dat')
    coefficients = pysgpp.DataVector(dummyCoeff)
    grid = pysgpp.Grid.unserializeFromFile(f'precalc/reSurf/grid_{name}.dat')
    precalculatedReSurf = pysgpp.SplineResponseSurface(
        grid, coefficients, pysgpp.DataVector(lb[:dim]), pysgpp.DataVector(ub[:dim]), degree)
    print(f'loaded {test_strategy} response surface for {qoi} with {precalculatedReSurf.getSize()} points')
    return precalculatedReSurf


def calculate_error(reSurf, qoi, numMCPoints, test_strategy, dim, number_of_instances):
    l2error = 0
    nrmse = 0

    # TEMPORARY
    #number_of_instances = 40

    error_reference_data_file = f'precalc/values/mc{numMCPoints}_{test_strategy}_{dim}dim_{number_of_instances}repetitions_{int(sample_size/1000)}kpop.pkl'
    with open(error_reference_data_file, 'rb') as fp:
        error_reference_data = pickle.load(fp)

    num_mc_repetitions = next(iter(error_reference_data))[-1]  # get first key and take its last entry
    logging.info(
        f'loaded {numMCPoints} reference MC values calculated with {num_mc_repetitions} repetitions')

    error_overview = np.zeros((numMCPoints, 6))

    max_val = 0
    min_val = 1e+14
    for i, key in enumerate(error_reference_data):
        try:
            [true_e_time, true_e_num_tests, true_e_num_confirmed_sick_individuals, true_e_num_confirmed_per_test,
                true_e_num_sent_to_quarantine, true_sd_time, true_sd_num_tests, true_sd_num_confirmed_sick_individuals,
                true_sd_num_confirmed_per_test, true_sd_num_sent_to_quarantine, true_e_number_groupwise_tests,
                true_worst_case_number_groupwise_tests, true_e_number_sick_people,
                true_sd_number_sick_people] = error_reference_data[key]

        except:
            [true_e_time, true_e_num_tests, true_e_num_confirmed_sick_individuals, true_e_num_confirmed_per_test,
                true_e_num_sent_to_quarantine, true_sd_time, true_sd_num_tests, true_sd_num_confirmed_sick_individuals,
                true_sd_num_confirmed_per_test, true_sd_num_sent_to_quarantine] = error_reference_data[key]

        if qoi == 'time':
            true_value = true_e_time
        elif qoi == 'numtests':
            true_value = true_e_num_tests
        elif qoi == 'numconfirmed':
            true_value = true_e_num_confirmed_sick_individuals
        elif qoi == 'ppt':
            true_value = true_e_num_confirmed_per_test
        elif qoi == 'sd-time':
            true_value = true_sd_time
        elif qoi == 'sd-numtests':
            true_value = true_sd_num_tests
        elif qoi == 'sd-numconfirmed':
            true_value = true_sd_num_confirmed_sick_individuals
        elif qoi == 'sd-ppt':
            true_value = true_sd_num_confirmed_per_test

        if true_value > max_val:
            max_val = true_value
        if true_value < min_val:
            min_val = true_value

        prob_sick = key[0]
        success_rate_test = key[1]
        false_positive_rate = key[2]
        group_size = key[3]

        if group_size in [1, 32]:
            point = pysgpp.DataVector([prob_sick, success_rate_test, false_positive_rate, group_size])
        else:
            point = pysgpp.DataVector([prob_sick, success_rate_test, false_positive_rate, group_size+1])

        reSurf_value = reSurf.eval(point)
        l2error += (true_value-reSurf_value)**2

        #print(f'{key}    {true_value}    {reSurf_value}     {np.abs(true_value-reSurf_value)}')
        error_overview[i, 0:5] = key
        error_overview[i, 5] = np.abs(true_value-reSurf_value)

    # np.set_printoptions(linewidth=150)
    # print(error_overview)
    # plt.plot(error_overview[:, 3], error_overview[:, 5], 'bx')
    # plt.show()

    l2error = np.sqrt(l2error/numMCPoints)
    if max_val-min_val > 0:
        nrmse = l2error / (max_val-min_val)
    else:
        nrmse = float('NaN')
    return l2error, nrmse


def auxiliary(refineType, test_strategies, qois, dim, degree, lb, ub, level=1, numPoints=1,
              initialLevel=1, numRefine=1, verbose=False):
    l2errors = np.zeros((len(test_strategies), len(qois)))
    nrmses = np.zeros((len(test_strategies), len(qois)))
    gridSizes = np.zeros((len(test_strategies), len(qois)))
    for i, test_strategy in enumerate(test_strategies):
        for j, qoi in enumerate(qois):
            f = sgpp_simStorage(dim, test_strategy, lb, ub, number_of_instances)
            objFunc = objFuncSGpp(f, qoi)
            reSurf = create_reSurf(objFunc, lb, ub, gridType, degree, boundaryLevel, refineType,
                                   level, numPoints, initialLevel, numRefine, verbose)

            if calcError:
                l2errors[i, j], nrmses[i, j] = calculate_error(
                    reSurf, qoi, numMCPoints, test_strategy, dim, number_of_instances)
                gridSizes[i, j] = reSurf.getSize()
            if saveReSurf:
                save_reSurf(reSurf, refineType, test_strategy, qoi, dim, degree, level, numPoints)
    return l2errors, nrmses, gridSizes


if __name__ == "__main__":
    saveReSurf = True
    calcError = True
    plotError = False  # calcError
    numMCPoints = 100

    levels = [1, 2, 3, 4]
    numPointsArray = []  # [10, 100, 200, 400, 800, 1200, 1400, 1600]

    initialLevel = 1    # initial level
    numRefine = 10       # number of grid points refined in each step
    verbose = False  # verbosity of subroutines

    gridType, dim, degree, _, _, name, sample_size, num_daily_tests, \
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
    qois = [
        'ppt',
        # 'sd-ppt',
        # 'time',
        # 'sd-time'
    ]

    regular_l2errors = np.zeros((len(test_strategies), len(qois), len(levels)))
    regular_nrmses = np.zeros((len(test_strategies), len(qois), len(levels)))
    regular_gridSizes = np.zeros((len(test_strategies), len(qois), len(levels)))

    adaptive_l2errors = np.zeros((len(test_strategies), len(qois), len(numPointsArray)))
    adaptive_nrmses = np.zeros((len(test_strategies), len(qois), len(numPointsArray)))
    adaptive_gridSizes = np.zeros((len(test_strategies), len(qois), len(numPointsArray)))

    level = 'dummy'

    refineType = 'regular'
    print('regular')
    for l, level in enumerate(levels):
        print(f'level {level}')
        regular_l2errors[:, :, l], regular_nrmses[:, :, l], regular_gridSizes[:, :, l] = \
            auxiliary(refineType, test_strategies, qois, dim, degree, lb, ub, level)

    refineType = 'adaptive'
    print('adaptive')
    for l, numPoints in enumerate(numPointsArray):
        print(f'num Points {numPoints}')
        adaptive_l2errors[:, :, l], adaptive_nrmses[:, :, l], adaptive_gridSizes[:, :, l]\
            = auxiliary(refineType, test_strategies, qois, dim, degree, lb, ub, level, numPoints,
                        initialLevel, numRefine, verbose)

    if calcError and plotError:
        markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

        plt.figure(figsize=(12, 8))
        plt.title('regular')
        plotindex = 1
        for j, qoi in enumerate(qois):
            plt.subplot(2, 2, plotindex)
            plotindex += 1
            for i, test_strategy in enumerate(test_strategies):
                # plt.plot(regular_gridSizes[i, j, :], regular_l2errors[i, j, :],
                #          label=test_strategy, marker=markers[i], color=colors[i])
                plt.plot(regular_gridSizes[i, j, :], regular_nrmses[i, j, :],
                         label=test_strategy, marker=markers[i], color=colors[i])
                plt.legend()
            plt.title(qoi)
            plt.xlabel('num regular grid points')
            plt.ylabel('NRMSE')
            #plt.ylabel('L2 error')
            plt.gca().set_yscale('log')

        plt.figure(figsize=(12, 8))
        plt.title('adaptive')
        plotindex = 1
        for j, qoi in enumerate(qois):
            plt.subplot(2, 2, plotindex)
            plotindex += 1
            for i, test_strategy in enumerate(test_strategies):
                # plt.plot(adaptive_gridSizes[i, j, :], adaptive_l2errors[i, j, :],
                #          label=test_strategy, marker=markers[i], color=colors[i])
                plt.plot(adaptive_gridSizes[i, j, :], adaptive_nrmses[i, j, :],
                         label=test_strategy, marker=markers[i], color=colors[i])
                plt.legend()
            plt.title(qoi)
            plt.xlabel('num adaptive grid points')
            plt.ylabel('NRMSE')
            #plt.ylabel('L2 error')
            plt.gca().set_yscale('log')
        plt.show()
