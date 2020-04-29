import pysgpp
import numpy as np
import pickle
import time
from Statistics import Corona_Simulation_Statistics
import matplotlib.pyplot as plt
from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import precalc_parallel


gridType, dim, degree, test_strategy, qoi, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests, evalType, scale_factor_pop,\
    number_of_instances, lb, ub = getSetup()

dummyCoeff = np.loadtxt(f'precalc/reSurf/np_coeff_{name}.dat')
coefficients = pysgpp.DataVector(dummyCoeff)
grid = pysgpp.Grid.unserializeFromFile(f'precalc/reSurf/grid_{name}.dat')

precalculatedReSurf = pysgpp.SplineResponseSurface(
    grid, coefficients, pysgpp.DataVector(lb[:dim]), pysgpp.DataVector(ub[:dim]), degree)
print(f'precalculated response surface with {precalculatedReSurf.getSize()} points  has been loaded')

# eval parameters:
#probabilities_sick = np.linspace(0.001, 0.3, 21)
probabilities_sick = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
success_rate_test = 0.64
false_positive_rate = 0.01378
group_size = 2

# precalculations
# load precalculated data
savePath = "/home/rehmemk/git/covid19-pooling/precalc/"
precalcValuesFileName = savePath + f"precalc_values_{test_strategy}.pkl"
try:
    with open(precalcValuesFileName, 'rb') as fp:
        precalculatedValues = pickle.load(fp)
    print(f'loaded precalculated evaluations from {precalcValuesFileName}')
except (FileNotFoundError):
    print('could not find precalculated data at {}\nCreating new data file.'.format(
        precalcValuesFileName))
    precalculatedValues = {}
points = []
num_to_calculate = 0
for prob_sick in probabilities_sick:
    key = tuple([prob_sick, success_rate_test, false_positive_rate,
                 group_size, evalType, number_of_instances])
    if key not in precalculatedValues:
        point = [prob_sick, success_rate_test, false_positive_rate, group_size]
        points.append(point)
        num_to_calculate += 1
print(f'need to calculate {len(points)} new evaluations')
multiprocessing_dict = precalc_parallel(points, sample_size, test_duration, num_simultaneous_tests,
                                        number_of_instances, scale_factor_pop, test_strategy, evalType)
for key in multiprocessing_dict:
    precalculatedValues[key] = multiprocessing_dict[key]
with open(precalcValuesFileName, "wb") as fp:
    pickle.dump(precalculatedValues, fp)

# evaluations
f = sgpp_simStorage(dim, test_strategy,  qoi, lb, ub)
if qoi == 'ppt':
    sgpp_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))
    ref_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))

for i, prob_sick in enumerate(probabilities_sick):
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate, group_size]
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
