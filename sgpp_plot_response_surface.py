import ipdb
import pysgpp
import numpy as np
import pickle
import time
from Statistics import Corona_Simulation_Statistics
import matplotlib.pyplot as plt
from sgpp_create_response_surface import load_response_Surface
from setup import getSetup
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import calculate_missing_values


np.set_printoptions(linewidth=150)

gridType, dim, degree, test_strategy, _, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
    boundaryLevel = getSetup()

refineType = 'regular'  # 'adaptive'
numPoints = 800
level = 7

qoi = 'ppt'

precalculatedReSurf = load_response_Surface(
    refineType, test_strategy, qoi, dim, degree, level, numPoints, lb, ub)
print(f'precalculated response surface with {precalculatedReSurf.getSize()} points  has been loaded')

gridStorage = precalculatedReSurf.getGrid().getStorage()
gridPoints1D = [gridStorage.getPointCoordinate(i, 0)for i in range(gridStorage.getSize())]
#print(f"grid points 1d: {gridPoints1D}")
scaledGridPoints1D = np.sort([lb[0]+(ub[0]-lb[0])*p for p in gridPoints1D])

#print(f"scaled grid points 1d: {scaledGridPoints1D}")

# eval parameters:
#probabilities_sick = np.linspace(0.001, 0.3, 66)
probabilities_sick = np.linspace(0.14, 0.16, 21)
#probabilities_sick = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
probabilities_sick = [0.1111, 0.1734, 0.2784]
success_rate_test = 1.0  # 0.99
false_positive_rate = 0.0  # 0.01
group_size = 8

todoPoints = []
for i, prob_sick in enumerate(probabilities_sick):
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate, group_size]
    todoPoints.append(evaluationPoint)

num_new_points = calculate_missing_values(dim, todoPoints, sample_size, test_duration,
                                          num_simultaneous_tests, number_of_instances, test_strategy)
print(f'Calcualted {num_new_points} new evaluations\n')

# evaluations
f = sgpp_simStorage(dim, test_strategy, lb, ub,  number_of_instances,
                    sample_size, num_daily_tests, test_duration)
sgpp_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))
ref_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))

for i, prob_sick in enumerate(probabilities_sick):
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate, group_size]
    sgpp_e_num_confirmed_per_test[i] = precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint[:dim]))
    ref_e_num_confirmed_per_test[i] = f.eval(evaluationPoint, qoi, recalculate=False)

gridPointValues = np.zeros(len(scaledGridPoints1D))
for i, prob_sick in enumerate(scaledGridPoints1D):
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate, group_size]
    gridPointValues[i] = f.eval(evaluationPoint, qoi, recalculate=False)

f.cleanUp()
print('diffs: {}'.format(np.abs(sgpp_e_num_confirmed_per_test-ref_e_num_confirmed_per_test)))
print("l2 err: {:.5e}".format(np.linalg.norm(sgpp_e_num_confirmed_per_test-ref_e_num_confirmed_per_test)))

# print("\n")
# print(f"prob sick: {probabilities_sick}")
# #print(f"SGpp: {sgpp_e_num_confirmed_per_test}")
# print(f"true: {ref_e_num_confirmed_per_test}")
# print(f"| prob_sick - true | {abs(probabilities_sick-ref_e_num_confirmed_per_test)}")

#plt.plot(probabilities_sick, sgpp_e_num_confirmed_per_test, '-s', label='SGpp')
plt.plot(probabilities_sick, ref_e_num_confirmed_per_test, 'x', label='ref')
plt.plot(scaledGridPoints1D, gridPointValues, 'o-', label='grid points')

plt.xlabel('infection rate')
plt.ylabel('exp. number of cases per test')
plt.legend()
plt.show()
