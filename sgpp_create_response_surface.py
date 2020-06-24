import pysgpp
import numpy as np
import sys
import pickle
import time
import os
import logging
import matplotlib.pyplot as plt
from Statistics import Corona_Simulation_Statistics
from sgpp_simStorage import sgpp_simStorage, objFuncSGpp
from sgpp_calculate_stochastic_noise import stochastic_noise
from setup import getSetup

# default plot font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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


def calculate_error(reSurf, qoi, sample_size, numMCPoints, test_strategy, dim, number_of_instances):
    l2error = 0
    nrmse = 0

    # TEMPORARY
    number_of_instances = 10
    sample_size = 100000  # 100000
    print(f'Overwriting error sample size to {sample_size} nad instances to {number_of_instances}')

    error_reference_data_file = f'precalc/values/mc{numMCPoints}_{test_strategy}_{dim}dim_{number_of_instances}repetitions_{int(sample_size/1000)}kpop.pkl'
    with open(error_reference_data_file, 'rb') as fp:
        error_reference_data = pickle.load(fp)

    num_mc_repetitions = next(iter(error_reference_data))[-1]  # get first key and take its last entry
    logging.info(
        f'loaded {numMCPoints} reference MC values calculated with {num_mc_repetitions} repetitions')

    error_overview = np.zeros((numMCPoints, 6))

    max_val = 0
    min_val = 1e+14
    worst_error = 0
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

        # if group_size in [1, 32]:
        #     point = [prob_sick, success_rate_test, false_positive_rate, group_size]
        # else:
        #     point = [prob_sick, success_rate_test, false_positive_rate, group_size+1]
        point = [prob_sick, success_rate_test, false_positive_rate, group_size]
        point = pysgpp.DataVector(point[:dim])

        reSurf_value = reSurf.eval(point)
        l2error += (true_value-reSurf_value)**2

        if np.abs(true_value-reSurf_value) > worst_error:
            worst_error = np.abs(true_value-reSurf_value)
            relative_worst_error = worst_error/np.abs(true_value)
            worst_key = key

        #print(f'{key}    {true_value}    {reSurf_value}     {np.abs(true_value-reSurf_value)}')
        #error_overview[i, 0:5] = key
        #error_overview[i, 5] = np.abs(true_value-reSurf_value)

    # np.set_printoptions(linewidth=150)
    # print(error_overview)
    # plt.plot(error_overview[:, 3], error_overview[:, 5], 'bx')
    # plt.show()

    # print(f'{test_strategy:20s}; worst error of {worst_error:.4e} which is relative error {relative_worst_error:.4} for key={worst_key}')

    l2error = np.sqrt(l2error/numMCPoints)
    if max_val-min_val > 0:
        nrmse = l2error / (max_val-min_val)
    else:
        nrmse = float('NaN')
    return l2error, nrmse


def auxiliary(refineType, test_strategies, qois, sample_size, num_daily_tests, test_duration, dim,
              degree, lb, ub, level=1, numPoints=1,
              initialLevel=1, numRefine=1, verbose=False):
    l2errors = np.zeros((len(test_strategies), len(qois)))
    nrmses = np.zeros((len(test_strategies), len(qois)))
    gridSizes = np.zeros((len(test_strategies), len(qois)))
    for i, test_strategy in enumerate(test_strategies):
        for j, qoi in enumerate(qois):
            f = sgpp_simStorage(dim, test_strategy, lb, ub, number_of_instances,
                                sample_size, num_daily_tests, test_duration)
            objFunc = objFuncSGpp(f, qoi)
            reSurf = create_reSurf(objFunc, lb, ub, gridType, degree, boundaryLevel, refineType,
                                   level, numPoints, initialLevel, numRefine, verbose)

            if calcError:
                l2errors[i, j], nrmses[i, j] = calculate_error(
                    reSurf, qoi, sample_size, numMCPoints, test_strategy, dim, number_of_instances)
                gridSizes[i, j] = reSurf.getSize()
            if saveReSurf:
                save_reSurf(reSurf, refineType, test_strategy, qoi, dim, degree, level, numPoints)
    return l2errors, nrmses, gridSizes


if __name__ == "__main__":
    saveReSurf = True
    calcError = True
    plotError = calcError
    plotNoise = plotError
    numMCPoints = 100

    levels = [1, 2, 3, 4]  # , 5 , 6, 7]
    numPointsArray = []  # [10, 100, 200, 400, 800, 1200, 1500]

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
            auxiliary(refineType, test_strategies, qois, sample_size,
                      num_daily_tests, test_duration, dim, degree, lb, ub, level)

    refineType = 'adaptive'
    print('adaptive')
    for l, numPoints in enumerate(numPointsArray):
        print(f'num Points {numPoints}')
        adaptive_l2errors[:, :, l], adaptive_nrmses[:, :, l], adaptive_gridSizes[:, :, l]\
            = auxiliary(refineType, test_strategies, qois, sample_size, num_daily_tests,
                        test_duration, dim, degree, lb, ub, level, numPoints,
                        initialLevel, numRefine, verbose)

    if calcError and plotError:
        markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

        if len(levels) > 0:
            plt.figure(figsize=(8, 6))
            plt.title('regular')
            plotindex = 1
            for j, qoi in enumerate(qois):
                if len(qois) > 1:
                    plt.subplot(2, 2, plotindex)
                    plotindex += 1
                for i, test_strategy in enumerate(test_strategies):
                    plt.plot(regular_gridSizes[i, j, :], regular_l2errors[i, j, :],
                             label=test_strategy, marker=markers[i], color=colors[i])
                    # plt.plot(regular_gridSizes[i, j, :], regular_nrmses[i, j, :],
                    #          label=test_strategy, marker=markers[i], color=colors[i])
                    plt.legend()

                    if plotNoise:
                        numNoisePoints = 100
                        number_outer_repetitions = 10
                        noise = stochastic_noise(test_strategy, qoi, sample_size, number_of_instances,
                                                 numNoisePoints, number_outer_repetitions)
                        plt.plot(regular_gridSizes[i, j, :], [noise]*len(levels),
                                 '--', color=colors[i])  # , marker=markers[i])

                plt.title(qoi)
                plt.xlabel('num regular grid points')
                # plt.ylabel('NRMSE')
                plt.ylabel('L2 error')
                plt.gca().set_yscale('log')
                plt.ylim([1e-3, 1])

        if len(numPointsArray) > 0:
            plt.figure(figsize=(8, 6))
            plt.title('adaptive')
            plotindex = 1
            for j, qoi in enumerate(qois):
                if len(qois) > 1:
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

        plt.tight_layout()
        plt.show()
