import pysgpp
import numpy as np
import sys
import pickle
import time
import logging
from sgpp_simStorage import sgpp_simStorage, objFuncSGpp


def getSetup():
    #logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    #logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

    gridType = 'nakBsplineBoundary'
    dim = 1
    degree = 3
#    test_strategy = 'individual-testing'
    test_strategy = 'binary-splitting'
    qoi = 'ppt'
    name = name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}'
    return gridType, dim, degree, test_strategy, qoi, name


if __name__ == "__main__":
    saveReSurf = True
    gridType, dim, degree, test_strategy, qoi, name = getSetup()
    f = sgpp_simStorage(dim, test_strategy,  qoi)

    objFunc = objFuncSGpp(f)
    lb = objFunc.getLowerBounds()
    ub = objFunc.getUpperBounds()

    for level in range(6):
        reSurf = pysgpp.SplineResponseSurface(
            objFunc, lb, ub, pysgpp.Grid.stringToGridType(gridType), degree)

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
        error_reference_data_file = f'precalc/mc{numMCPoints}_{dim}dim_{qoi}.pkl'
        with open(error_reference_data_file, 'rb') as fp:
            error_reference_data = pickle.load(fp)
        l2Error = 0
        for key in error_reference_data:
            true_value = error_reference_data[key]
            reSurf_value = reSurf.eval(pysgpp.DataVector(key))
            #print(f'{key}    {true_value}    {reSurf_value}')
            l2Error += (true_value-reSurf_value)**2
        l2Error = np.sqrt(l2Error)
        print(f"level {level} {reSurf.getSize()} grid points, l2 error: {l2Error}")

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
            print(f'saved response surface as {path}/{name}')
