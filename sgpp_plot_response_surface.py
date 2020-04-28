import pysgpp
import numpy as np
import time
from Statistics import Corona_Simulation_Statistics
import matplotlib.pyplot as plt
from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage


gridType, dim, degree, test_strategy, qoi, name = getSetup()
dummyCoeff = np.loadtxt(f'precalc/reSurf/np_coeff_{name}.dat')
coefficients = pysgpp.DataVector(dummyCoeff)
grid = pysgpp.Grid.unserializeFromFile(f'precalc/grid_{name}.dat')

f = sgpp_simStorage(dim, test_strategy,  qoi)
lb, ub = f.getDomain()
precalculatedReSurf = pysgpp.SplineResponseSurface(grid, coefficients, lb, ub, degree)
print(f'precalculated response surface with {precalculatedReSurf.getSize()} points  has been loaded')

# eval parameters:
# probabilities_sick = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
# probabilities_sick = [0.001,  0.1505, 0.3]
probabilities_sick = np.linspace(0.001, 0.3, 21)

# prob_sick = 0.01
success_rate_test = 0.99
false_positive_rate_test = 0.01
test_duration = 5
group_size = 32
tests_per_day = 100000
population = 100000000

if qoi == 'ppt':
    sgpp_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))
    ref_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))

for i, prob_sick in enumerate(probabilities_sick):
    # evaluationPoint = [prob_sick, success_rate_test, false_positive_rate_test,
    #                    group_size,test_duration, tests_per_day, population]
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate_test, group_size]
    sgpp_e_num_confirmed_per_test[i] = precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))
    ref_e_num_confirmed_per_test[i] = f.eval(evaluationPoint)

f.cleanUp()
print('diffs: {}'.format(np.abs(sgpp_e_num_confirmed_per_test-ref_e_num_confirmed_per_test)))
print("l2 err: {}".format(np.linalg.norm(sgpp_e_num_confirmed_per_test-ref_e_num_confirmed_per_test)))

plt.plot(probabilities_sick, sgpp_e_num_confirmed_per_test, '-o', label='SGpp')
plt.plot(probabilities_sick, ref_e_num_confirmed_per_test, '-x', label='ref')
plt.xlabel('infection rate')
plt.ylabel('exp. number of cases per test')
plt.legend()
plt.show()
