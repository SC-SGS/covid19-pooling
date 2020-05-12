import pysgpp
import numpy as np
import sys
import pickle
import time
import logging
from sgpp_simStorage import sgpp_simStorage, objFuncSGpp


def getSetup():
    # logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

    gridType = 'nakBsplineBoundary'
    dim = 4
    degree = 3

    #test_strategy = 'individual-testing'
    #test_strategy = 'binary-splitting'
    test_strategy = 'two-stage-testing'
    #test_strategy = 'RBS'
    #test_strategy = 'purim'
    #test_strategy = 'sobel'

    #qoi = 'time'
    #qoi = 'numtests'
    #qoi = 'numconfirmed'
    qoi = 'ppt'

    name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}'

    # reference values. These are defined in sgpp_simStorage::init too.
    # TODO: That's dangerous. Define them only once!
    sample_size = 100000
    num_daily_tests = 1000
    test_duration = 5
    num_simultaneous_tests = int(num_daily_tests*test_duration/24.0)
    number_of_instances = 5

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


if __name__ == "__main__":
    saveReSurf = False
    calcError = False
    numMCPoints = 1000

    #refineType = 'regular'
    refineType = 'adaptive'
    level = 5
    numPoints = 400  # max number of grid points
    initialLevel = 1    # nitial level
    numRefine = 10       # number of grid points refined in each step
    verbose = False  # verbosity of subroutines

    gridType, dim, degree, _, qoi, _, sample_size, num_daily_tests, \
        test_duration, num_simultaneous_tests, evalType, scale_factor_pop,\
        number_of_instances, lb, ub, boundaryLevel = getSetup()
    test_strategies = [
        'individual-testing',
        'two-stage-testing',
        'binary-splitting',
        'RBS',
        'purim',
        'sobel'
    ]
    for i, test_strategy in enumerate(test_strategies):
        name = name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}'
        f = sgpp_simStorage(dim, test_strategy, qoi, lb, ub)
        objFunc = objFuncSGpp(f)

        reSurf = pysgpp.SplineResponseSurface(
            objFunc, pysgpp.DataVector(lb[:dim]), pysgpp.DataVector(ub[:dim]),
            pysgpp.Grid.stringToGridType(gridType), degree, boundaryLevel)

        logging.info('Begin creating response surface')
        start = time.time()
        if refineType == 'regular':
            reSurf.regular(level)
        elif refineType == 'adaptive':
            reSurf.surplusAdaptive(numPoints, initialLevel, numRefine, verbose)

        runtime = time.time()-start
        logging.info('\nDone. Created response surface with {} grid points, took {}s'.format(reSurf.getSize(), runtime))
        objFunc.cleanUp()

        # measure error
        if calcError:
            error_reference_data_file = f'precalc/values/mc{numMCPoints}_{test_strategy}_{dim}dim.pkl'
            with open(error_reference_data_file, 'rb') as fp:
                error_reference_data = pickle.load(fp)
            l2Error = 0
            max_val = 0
            min_val = 1e+14
            for key in error_reference_data:
                [true_e_time, true_e_num_tests, true_e_num_confirmed_sick_individuals] = error_reference_data[key]
                if qoi == 'time':
                    true_value = true_e_time
                elif qoi == 'numtests':
                    true_value = true_e_num_tests
                elif qoi == 'numconfirmed':
                    true_value = true_e_num_confirmed_sick_individuals
                elif qoi == 'ppt':
                    true_value = true_e_num_confirmed_sick_individuals/true_e_num_tests

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
                #print(f'{key}    {true_value}    {reSurf_value}     {np.abs(true_value-reSurf_value)}')
                l2Error += (true_value-reSurf_value)**2
            l2Error = np.sqrt(l2Error/numMCPoints)
            nrmse = l2Error / (max_val-min_val)
            if refineType == 'regular':
                print(f"{test_strategy}, level {level} {reSurf.getSize()} grid points, l2 error: {l2Error:.5f}  nrmse: {nrmse:.5f}")
            elif refineType == 'adaptive':
                print(f"{test_strategy}, adaptive {reSurf.getSize()} grid points, l2 error: {l2Error:.5f}  nrmse: {nrmse:.5f}")

        if saveReSurf:
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
            logging.info('wrote response surface to /data')
            #print(f'saved response surface as {path}/{reSurfName}')
