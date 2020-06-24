import pysgpp
import numpy as np
import time
from Statistics import Corona_Simulation_Statistics
import matplotlib.pyplot as plt
from setup import getSetup
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import calculate_missing_values


gridType, dim, degree, _, _, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
    boundaryLevel = getSetup()

test_strategies = [
    # 'individual-testing',
    'two-stage-testing',
    # 'binary-splitting',
    # 'RBS',
    # 'purim',
    # 'sobel'
]
# eval parameters:
probabilities_sick = np.linspace(0.01, 0.3, 101)  # [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
success_rate_test = 1.0  # 0.99
false_positive_rate_test = 0.0  # 0.01
group_size = 8  # 32

qoi = 'ppt'
points = []
for i, test_strategy in enumerate(test_strategies):
    for j, prob_sick in enumerate(probabilities_sick):
        point = [prob_sick, success_rate_test, false_positive_rate_test, group_size]
        points.append(point)
    num_new_points = calculate_missing_values(dim, points, sample_size, test_duration,
                                              num_simultaneous_tests, number_of_instances, test_strategy)
    print(f'Calcualted {num_new_points} new evaluations\n')

    f = sgpp_simStorage(dim, test_strategy, lb, ub, number_of_instances,
                        reference_sample_size=sample_size, reference_num_daily_tests=num_daily_tests,
                        reference_test_duration=5)
    values = np.zeros((len(test_strategies), len(probabilities_sick)))
    for j, prob_sick in enumerate(probabilities_sick):
        point = [prob_sick, success_rate_test, false_positive_rate_test, group_size]
        values[i, j] = f.eval(point, qoi, recalculate=False)


f.cleanUp()
plt.plot(probabilities_sick, values[i, :])
plt.xlabel('infection rate')


plt.legend()
plt.show()
