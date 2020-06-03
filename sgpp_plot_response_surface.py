import pysgpp
import numpy as np
import pickle
import time
from Statistics import Corona_Simulation_Statistics
import matplotlib.pyplot as plt
from sgpp_create_response_surface import getSetup, load_response_Surface
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import calculate_missing_values


gridType, dim, degree, test_strategy, qoi, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
    boundaryLevel = getSetup()

refineType = 'adaptive'
numPoints = 800
level = 1

precalculatedReSurf = load_response_Surface(
    refineType, test_strategy, qoi, dim, degree, level, numPoints, lb, ub)
print(f'precalculated response surface with {precalculatedReSurf.getSize()} points  has been loaded')

# eval parameters:
#probabilities_sick = np.linspace(0.001, 0.3, 21)
probabilities_sick = [0.01, 0.3]  # [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
success_rate_test = 0.99
false_positive_rate = 0.01

todoPoints = []
for i, prob_sick in enumerate(probabilities_sick):
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate, group_size]
    todoPoints.append(evaluationPoint)

num_new_points = calculate_missing_values(todoPoints, sample_size, test_duration,
                                          num_simultaneous_tests, number_of_instances, test_strategy)
print(f'Calcualted {num_new_points} new evaluations\n')

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
