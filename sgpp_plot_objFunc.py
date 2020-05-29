import pysgpp
import numpy as np
import time
from Statistics import Corona_Simulation_Statistics
import matplotlib.pyplot as plt
from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage


gridType, dim, degree, _, _, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
    boundaryLevel = getSetup()
test_strategies = [
    # 'individual-testing',
    # 'two-stage-testing',
    'binary-splitting',
    # 'RBS',
    # 'purim',
    # 'sobel'
]
# eval parameters:
probabilities_sick = [0.01]  # [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
success_rate_test = 0.99
false_positive_rate_test = 0.01
group_size = 3

qoi = 'groupwise_tests'
values = np.zeros((len(test_strategies), len(probabilities_sick)))
worst = np.zeros((len(test_strategies), len(probabilities_sick)))
for i, test_strategy in enumerate(test_strategies):
    f = sgpp_simStorage(dim, test_strategy, lb, ub, number_of_instances)
    for j, prob_sick in enumerate(probabilities_sick):
        point = [prob_sick, success_rate_test, false_positive_rate_test, group_size]
        values[i, j] = f.eval(point, qoi, recalculate=True)
        worst[i, j] = f.eval(point, 'worst_groupwise_tests', recalculate=True)

#     plt.plot(probabilities_sick, values[i, :])
#     plt.xlabel('infection rate')
# plt.legend()
# plt.show()
print(values)
print(worst)
