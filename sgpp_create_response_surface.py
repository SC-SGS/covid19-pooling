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

    # qoi = 'time'
    # qoi = 'numtests'
    # qoi = 'numconfirmed'
    qoi = 'ppt'

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


def save_reSurf(reSurf, refineType):
    path = 'precalc/reSurf'
    # serialize the resposne surface
    # gridStr = reSurf.serializeGrid()
    gridStr = reSurf.getGrid().serialize()
    coeffs = reSurf.getCoefficients()
    # save it to files
    if refineType == 'regular':
        reSurfName = f'{name}_level{level}'
    elif refineType == 'adaptive':
        reSurfName = f'{name}_adaptive{numPoints}'
    with open(f'{path}/grid_{reSurfName}.dat', 'w+') as f:
        f.write(gridStr)
    # coeffs.toFile('data/coeffs.dat')
    # sgpp DataVector and DataMatrix from File are buggy
    dummyCoeff = np.array([coeffs[i] for i in range(coeffs.getSize())])
    np.savetxt(f'{path}//np_coeff_{reSurfName}.dat', dummyCoeff)
    print(f'saved response surface as {path}/{reSurfName}')


def load_response_Surface(refineType, test_strategy, qoi, dim, degree, level, numPoints, lb, ub):
    if refineType == 'regular':
        name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}_level{level}'
    elif refineType == 'adaptive':
        name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}_adaptive{numPoints}'

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

    error_reference_data_file = f'precalc/values/mc{numMCPoints}_{test_strategy}_{dim}dim_{number_of_instances}repetitions.pkl'
    with open(error_reference_data_file, 'rb') as fp:
        error_reference_data = pickle.load(fp)

    num_mc_repetitions = next(iter(error_reference_data))[-1]  # get first key and take its last entry
    logging.info(
        f'loaded {numMCPoints} reference MC values calculated with {num_mc_repetitions} repetitions')

    max_val = 0
    min_val = 1e+14
    for key in error_reference_data:
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
        point = pysgpp.DataVector([prob_sick, success_rate_test, false_positive_rate, group_size])
        reSurf_value = reSurf.eval(point)
        # print(f'{key}    {true_value}    {reSurf_value}     {np.abs(true_value-reSurf_value)}')
        l2error += (true_value-reSurf_value)**2
    l2error = np.sqrt(l2error/numMCPoints)
    if max_val-min_val > 0:
        nrmse = l2error / (max_val-min_val)
    else:
        nrmse = float('NaN')
    return l2error, nrmse


if __name__ == "__main__":
    saveReSurf = False
    calcError = True
    plotError = True
    numMCPoints = 100

    refineType = 'regular'
    # refineType = 'adaptive'
    levels = [1, 2, 3]
    numPoints = 400  # max number of grid points
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
        'sd-ppt',
        'time',
        'sd-time'
    ]

    l2errors = np.zeros((len(test_strategies), len(qois), len(levels)))
    nrmses = np.zeros((len(test_strategies), len(qois), len(levels)))
    for l, level in enumerate(levels):
        print('\n')
        for i, test_strategy in enumerate(test_strategies):
            for j, qoi in enumerate(qois):
                name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}'
                f = sgpp_simStorage(dim, test_strategy, lb, ub, number_of_instances)
                objFunc = objFuncSGpp(f, qoi)

                reSurf = create_reSurf(objFunc, lb, ub, gridType, degree, boundaryLevel, refineType,
                                       level, numPoints, initialLevel, numRefine, verbose)

                # measure error
                if calcError:
                    l2errors[i, j, l], nrmses[i, j, l] = calculate_error(
                        reSurf, qoi, numMCPoints, test_strategy, dim, number_of_instances)
                    if refineType == 'regular':
                        print(
                            f'{test_strategy:20s}, {qoi:10s} level {level}, {reSurf.getSize()}'
                            f' grid points, l2 error: {l2errors[i,j,l]:.5f}  nrmse: {nrmses[i,j,l]:.5f}')
                    elif refineType == 'adaptive':
                        print(
                            f'{test_strategy:20s}, {qoi:10s} adaptive {reSurf.getSize()} grid'
                            f' points, l2 error: {l2errors[i,j,l]:.5f}  nrmse: {nrmses[i,j,l]:.5f}')

                # save reSurf
                if saveReSurf:
                    save_reSurf(reSurf, refineType)

    if calcError and plotError:
        markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        plt.figure(figsize=(12, 8))
        plotindex = 1
        for j, qoi in enumerate(qois):
            plt.subplot(2, 2, plotindex)
            plotindex += 1
            for i, test_strategy in enumerate(test_strategies):
                plt.plot(levels, nrmses[i, j, :], label=test_strategy, marker=markers[i], color=colors[i])
                plt.legend()
            plt.title(qoi)
            plt.xticks(levels)
            plt.xlabel('level')
            plt.ylabel('NRMSE')
        plt.show()
