import numpy as np
from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage
import pickle


gridType, dim, degree, test_strategy, qoi, _ = getSetup()
numMCPoints = 100

f = sgpp_simStorage(dim, test_strategy, qoi)
lb_sgpp, ub_sgpp = f.getDomain()
lb = lb_sgpp.array()
ub = ub_sgpp.array()

unitpoints = np.random.rand(numMCPoints, dim)
mcData = {}
for i in range(numMCPoints):
    for d in range(dim):
        point = lb + (ub-lb)*unitpoints[i, :]
        mcData[tuple(point)] = f.eval(point)
filename = f'precalc/values/mc{numMCPoints}_{dim}dim_{qoi}.pkl'
with open(filename, 'wb+') as fp:
    pickle.dump(mcData, fp)

print(f'calculated data for {numMCPoints} random points, saved as {filename}')
