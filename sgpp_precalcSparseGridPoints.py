import pysgpp
import numpy as np
from sgpp_create_response_surface import getSetup
from argparse import ArgumentParser
from sgpp_precalc_parallel import calculate_missing_values

if __name__ == "__main__":
    parser = ArgumentParser(description='Get a program and run it with input')
    parser.add_argument('--test_strategy', default='RBS', type=str)
    parser.add_argument('--level', default=1, type=int)
    args = parser.parse_args()
    test_strategy = args.test_strategy
    level = args.level

    gridType, dim, degree, _, qoi, name, sample_size, num_daily_tests, \
        test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
        boundaryLevel = getSetup()

    grid = pysgpp.Grid_createNakBsplineBoundaryGrid(dim, degree, boundaryLevel)
    grid.getGenerator().regular(level)
    gridStorage = grid.getStorage()

    points = []
    for i in range(grid.getSize()):
        point = gridStorage.getPointCoordinates(i).array()
        for d in range(dim):
            point[d] = lb[d] + (ub[d]-lb[d])*point[d]
        points.append(point)
    print(f'{test_strategy}; Grid of dim {dim}, level {level} has {grid.getSize()} points')

    num_new_points = calculate_missing_values(points, sample_size, test_duration,
                                              num_simultaneous_tests, number_of_instances, test_strategy)
    print(f'Calcualted {num_new_points} new evaluations\n')
