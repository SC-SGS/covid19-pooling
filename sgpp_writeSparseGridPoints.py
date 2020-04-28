import pysgpp
import numpy as np
from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage


gridType, dim, degree, test_strategy, scale_factor_pop, qoi = getSetup()
f = sgpp_simStorage(dim, test_strategy, scale_factor_pop, qoi)
lb, ub = f.getDomain()

degree = 3
level = 4
grid = pysgpp.Grid_createNakBsplineBoundaryGrid(dim, degree)
grid.getGenerator().regular(level)
points = np.zeros((grid.getSize(), 4))
gridStorage = grid.getStorage()
for i in range(grid.getSize()):
    point = gridStorage.getPointCoordinates(i)
    for d in range(dim):
        points[i, d] = lb[d] + (ub[d]-lb[d])*point[d]

print(f' Grid of level {level} has {grid.getSize()} points')
np.savetxt('precalc/x.dat', points)
