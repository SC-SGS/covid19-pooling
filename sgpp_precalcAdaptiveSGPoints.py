import pickle
import pysgpp
import sys
import numpy as np
from setup import getSetup
from sgpp_simStorage import sgpp_simStorage, objFuncSGpp, generateKey
from sgpp_precalc_parallel import calculate_missing_values


def checkPrecalc(reSurf, precalculatedValues, test_strategy, num_simultaneous_tests, test_duration, number_of_instances,
                 sample_size, default_parameters):
    todoPoints = []
    todoPointsDetermined = False
    grid = reSurf.getGrid()
    gridStorage = grid.getStorage()
    lb = reSurf.getLowerBounds()
    ub = reSurf.getUpperBounds()
    for n in range(grid.getSize()):
        point = gridStorage.getPoint(n)
        point_py = np.zeros(4)
        for d in range(dim):
            point_py[d] = lb[d] + (ub[d]-lb[d])*point.getStandardCoordinate(d)
        for d in range(dim, 4):
            point_py[d] = default_parameters[d]
        prob_sick = point_py[0]
        success_rate_test = point_py[1]
        false_positive_rate = point_py[2]
        group_size = int(point_py[3])
        key = generateKey(prob_sick, success_rate_test, false_positive_rate, group_size,
                          test_strategy, num_simultaneous_tests, test_duration, number_of_instances,
                          sample_size)
        if key not in precalculatedValues:
            # if point_py not in todoPoints:
            todoPoints.append(point_py)
            todoPointsDetermined = True

    return todoPointsDetermined, todoPoints


if __name__ == "__main__":
    gridType, dim, degree, _, qoi, name, sample_size, num_daily_tests, \
        test_duration, num_simultaneous_tests, number_of_instances, lb, ub,\
        boundaryLevel = getSetup()

    test_strategies = [
        'individual-testing',
        'two-stage-testing',
        'binary-splitting',
        'RBS',
        'purim',
        'sobel'
    ]
    qoi = 'ppt'
    #qoi = 'time'

    initialLevel = 1
    numRefine = 10
    maxPoints = 1600  # 2500
    verbose = False

    num_total_calculations = 0
    for test_strategy in test_strategies:
        print(f'{test_strategy}:')
        # load precalculated data
        savePath = "/home/rehmemk/git/covid19-pooling/precalc/"
        precalcValuesFileName = savePath + f"precalc_values_{test_strategy}.pkl"
        with open(precalcValuesFileName, 'rb') as fp:
            precalculatedValues = pickle.load(fp)
        #print(f'loaded precalculated evaluations from {precalcValuesFileName}')

        f = sgpp_simStorage(dim, test_strategy, lb, ub, number_of_instances)
        objFunc = objFuncSGpp(f, qoi)
        default_parameters = f.default_parameters

        reSurf = pysgpp.SplineResponseSurface(
            objFunc, pysgpp.DataVector(lb[:dim]), pysgpp.DataVector(ub[:dim]),
            pysgpp.Grid.stringToGridType(gridType), degree, boundaryLevel)
        reSurf.regular(initialLevel)
        todoPointsDetermined = False
        counter = 0
        verbose = False
        while not todoPointsDetermined:
            previousSize = reSurf.getSize()
            if previousSize > maxPoints:
                print(f"nothing to calculate for a maximum of {maxPoints} grid points")
                break
            reSurf.nextSurplusAdaptiveGrid(numRefine, verbose)
            todoPointsDetermined, todoPoints = checkPrecalc(
                reSurf, precalculatedValues, test_strategy, num_simultaneous_tests, test_duration,
                number_of_instances, sample_size, default_parameters)
            if not todoPointsDetermined:
                counter = counter + 1
                print(f"refining ({counter}), grid size: {reSurf.getSize()}")
                reSurf.refineSurplusAdaptive(numRefine, verbose)

        print(f'Now calculating {len(todoPoints)} evaluations')

        num_new_points = calculate_missing_values(dim, todoPoints, sample_size, test_duration,
                                                  num_simultaneous_tests, number_of_instances, test_strategy)
        print(f'Calcualted {num_new_points} new evaluations\n')
        num_total_calculations += num_new_points
    print(f'In total claculated{num_total_calculations} evaluations')
