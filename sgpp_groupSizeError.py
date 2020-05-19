import sys
import numpy as np
import pickle
import pysgpp
import logging
from sgpp_create_response_surface import getSetup, load_response_Surface
from sgpp_simStorage import sgpp_simStorage
import matplotlib.pyplot as plt

group_sizes = list(range(1, 33))
test_strategies = [
    # 'individual-testing',
    # 'two-stage-testing',
    # 'binary-splitting',
    # 'RBS',
    # 'purim',
    'sobel'
]
gridType, dim, degree, _, qoi, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
    boundaryLevel = getSetup()
qoi = 'time'
# refineType = 'regular'
refineType = 'adaptive'
level = 3
numPoints = 2000  # max number of grid points for adaptively refined grid

# 24 for paper data
numMCPoints = 100  # 24

time_errors = np.zeros((len(test_strategies), numMCPoints, len(group_sizes)))

for i, test_strategy in enumerate(test_strategies):
    ref_e_times = {}
    reSurf_e_times = {}

    reSurf = load_response_Surface(refineType, test_strategy, qoi, dim, degree, level, numPoints, lb, ub)
    filename = f'precalc/values/group_mc{numMCPoints}_{test_strategy}_{number_of_instances}repetitions.pkl'
    with open(filename, 'rb') as fp:
        mcdata = pickle.load(fp)
    for key in mcdata:
        newkey = tuple([key[0], key[1], key[2]])
        group_size = key[3]
        e_time = mcdata[key][0]
        if newkey not in ref_e_times:
            ref_e_times[newkey] = np.zeros(32)
        ref_e_times[newkey][group_size-1] = e_time
    # now we have a dict with arrays of lenght 32 as entries, each corresponding to a group size

    for newkey in ref_e_times:
        for j, group_size in enumerate(group_sizes):
            prob_sick = newkey[0]
            success_rate_test = newkey[1]
            false_positive_rate = newkey[2]
            # TODO there is a mismatch of 1 in the group sizes, I don't know where it comes from
            # Without correction the error plots look random. With correction they look meaningful
            if group_size in [1, 32]:
                evaluationPoint = [prob_sick, success_rate_test, false_positive_rate, group_size+1]
            else:
                evaluationPoint = [prob_sick, success_rate_test, false_positive_rate, group_size+1]
            e_time = reSurf.eval(pysgpp.DataVector(evaluationPoint))
            if newkey not in reSurf_e_times:
                reSurf_e_times[newkey] = np.zeros(32)
            reSurf_e_times[newkey][group_size-1] = e_time
    # now we have the same thing for response surface values

    ref_optimal_group_sizes = np.zeros(numMCPoints)
    ref_optimal_times = np.zeros(numMCPoints)
    reSurf_optimal_group_sizes = np.zeros(numMCPoints)
    reSurf_optimal_times = np.zeros(numMCPoints)

    for j, newkey in enumerate(ref_e_times):
        time_errors[i, j, :] = ref_e_times[newkey]-reSurf_e_times[newkey]

        ref_optimal_group_sizes[j] = int(np.argmin(ref_e_times[newkey])+1)
        ref_optimal_times[j] = np.min(ref_e_times[newkey])
        reSurf_optimal_group_sizes[j] = int(np.argmin(reSurf_e_times[newkey])+1)
        reSurf_optimal_times[j] = np.min(reSurf_e_times[newkey])

        # DEBUGGING
        print(ref_e_times[newkey])
        print(f'ref optimal {ref_optimal_group_sizes[j]} with {ref_optimal_times[j]}')
        print(reSurf_e_times[newkey])
        print(f'reSurf optimal {reSurf_optimal_group_sizes[j]} with {reSurf_optimal_times[j]}')
        print("\n")
        plt.plot(range(1, 33), ref_e_times[newkey], label='true', color='k')
        plt.plot([1, 32], [ref_optimal_times[j]]*2, 'k--',)
        plt.plot(range(1, 33), reSurf_e_times[newkey], label='Kennfeld', color='C0')
        plt.plot([1, 32], [reSurf_optimal_times[j]]*2, 'C0--')
        plt.legend()
        plt.xlabel('group size')
        plt.ylabel('expected time to test all (days)')
        plt.show()

    # baseline are 100 days to test all individually
    print('average error  {:10s} {:.5f} days'.format(str(np.mean(np.abs(reSurf_optimal_group_sizes - ref_optimal_group_sizes))),
                                                     np.mean(np.abs(reSurf_optimal_times-ref_optimal_times))))
    print('worst case     {:10s} {:.5f} days\n'.format(str(np.max(np.abs(reSurf_optimal_group_sizes - ref_optimal_group_sizes))),
                                                       np.max(np.abs(reSurf_optimal_times-ref_optimal_times))))

# average_time_errors = np.zeros((len(test_strategies), len(group_sizes)))
# for i in range(len(test_strategies)):
#     for j in range(len(group_sizes)):
#         average_time_errors[i, j] = np.mean(time_errors[i, :, j])

# print(average_time_errors)

# for i, test_strategy in enumerate(test_strategies):
#     plt.plot(range(1, 33), average_time_errors[i, :])
# plt.show()
