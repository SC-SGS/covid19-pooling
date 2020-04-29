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
    # test_strategy = 'binary-splitting'
    test_strategy = 'two-stage-testing'
    # test_strategy = 'RBS
    #test_strategy = 'purim'
    #test_strategy = 'sobel'
    qoi = 'ppt'
    name = name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}'

    sample_size = 100000
    num_daily_tests = 1000
    test_duration = 5
    num_simultaneous_tests = int(num_daily_tests*test_duration/24.0)
    evalType = 'multiMC'
    scale_factor_pop = 1
    number_of_instances = 1
    prob_sick_range = [0.001, 0.3]
    success_rate_test_range = [0.5, 0.99]  # [0.3, 0.99]
    false_positive_rate_test_range = [0.01, 0.2]
    group_size_range = [1, 32]
    lb = np.array([prob_sick_range[0], success_rate_test_range[0],
                   false_positive_rate_test_range[0], group_size_range[0]])
    ub = np.array([prob_sick_range[1], success_rate_test_range[1],
                   false_positive_rate_test_range[1], group_size_range[1]])

    return gridType, dim, degree, test_strategy, qoi, name, sample_size, num_daily_tests,\
        test_duration, num_simultaneous_tests, evalType, scale_factor_pop, number_of_instances, lb, ub


if __name__ == "__main__":
    saveReSurf = True
    gridType, dim, degree, test_strategy, qoi, name, sample_size, num_daily_tests, \
        test_duration, num_simultaneous_tests, evalType, scale_factor_pop,\
        number_of_instances, lb, ub = getSetup()
    f = sgpp_simStorage(dim, test_strategy,  qoi, lb, ub)

    objFunc = objFuncSGpp(f)

    for level in range(3):
        reSurf = pysgpp.SplineResponseSurface(
            objFunc, pysgpp.DataVector(lb[:dim]), pysgpp.DataVector(ub[:dim]),
            pysgpp.Grid.stringToGridType(gridType), degree)

        logging.info('Begin creating response surface')
        start = time.time()
        # create surrogate with regular sparse grid
        reSurf.regular(level)

        # create surrogate with spatially adaptive sparse grid
        # numPoints = 10000  # max number of grid points
        # initialLevel = 1    # nitial level
        # numRefine = 50       # number of grid points refined in each step
        # verbose = False  # verbosity of subroutines
        # reSurf.surplusAdaptive(numPoints, initialLevel, numRefine, verbose)

        runtime = time.time()-start
        logging.info('\nDone. Created response surface with {} grid points, took {}s'.format(reSurf.getSize(), runtime))
        objFunc.cleanUp()

        # measure error
        numMCPoints = 100
        error_reference_data_file = f'precalc/values/mc{numMCPoints}_{test_strategy}_{dim}dim_{qoi}.pkl'
        with open(error_reference_data_file, 'rb') as fp:
            error_reference_data = pickle.load(fp)
        l2Error = 0
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
            prob_sick = key[0]
            success_rate_test = key[1]
            false_positive_rate = key[2]
            group_size = key[3]
            point = pysgpp.DataVector([prob_sick, success_rate_test, false_positive_rate, group_size])
            reSurf_value = reSurf.eval(point)
            # print(f'{key}    {true_value}    {reSurf_value}')
            l2Error += (true_value-reSurf_value)**2
        l2Error = np.sqrt(l2Error)
        print(f"level {level} {reSurf.getSize()} grid points, l2 error: {l2Error}\n")

        if saveReSurf:
            path = 'precalc/reSurf'
            # serialize the resposne surface
            # gridStr = reSurf.serializeGrid()
            gridStr = reSurf.getGrid().serialize()
            coeffs = reSurf.getCoefficients()
            # save it to files
            with open(f'{path}/grid_{name}.dat', 'w+') as f:
                f.write(gridStr)
            # coeffs.toFile('data/coeffs.dat')
            # sgpp DataVector and DataMatrix from File are buggy
            dummyCoeff = np.array([coeffs[i] for i in range(coeffs.getSize())])
            np.savetxt(f'{path}//np_coeff_{name}.dat', dummyCoeff)
            logging.info('wrote response surface to /data')
            # print(f'saved response surface as {path}/{name}')
