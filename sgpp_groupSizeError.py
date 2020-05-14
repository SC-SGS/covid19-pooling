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
    'individual-testing',
    'two-stage-testing',
    # 'binary-splitting',
    # 'RBS',
    # 'purim',
    # 'sobel'
]
gridType, dim, degree, _, qoi, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
    boundaryLevel = getSetup()
qoi = 'time'
refineType = 'regular'
level = 3
numPoints = 400  # max number of grid points for adaptively refined grid

numMCPoints = 2
ref_e_times = {}
reSurf_e_times = {}

for i, test_strategy in enumerate(test_strategies):
    reSurf = load_response_Surface(refineType, test_strategy, qoi, dim, degree, level, numPoints, lb, ub)
    filename = f'precalc/values/group_mc{numMCPoints}_{test_strategy}__{number_of_instances}repetitions.pkl'
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
        evaluationPoint = [prob_sick, success_rate_test, false_positive_rate, group_size]
        e_time = reSurf.eval(pysgpp.DataVector(evaluationPoint))
        if newkey not in reSurf_e_times:
            reSurf_e_times[newkey] = np.zeros(32)
        reSurf_e_times[newkey][group_size-1] = e_time
# now we have the same thing for response surface values

ref_optimal_group_sizes = np.zeros(numMCPoints)
reSurf_optimal_group_sizes = np.zeros(numMCPoints)
for i, newkey in enumerate(ref_e_times):
    ref_optimal_group_sizes[i] = int(np.argmin(ref_e_times[newkey])+1)
    ref_optimal_time = np.min(ref_e_times[newkey])
    reSurf_optimal_group_sizes[i] = int(np.argmin(reSurf_e_times[newkey])+1)
    reSurf_optimal_time = np.min(reSurf_e_times[newkey])

    # DEBUGGING
    print(ref_e_times[newkey])
    print(f'ref optimal {ref_optimal_group_sizes[i]} with {ref_optimal_time}')
    print(reSurf_e_times[newkey])
    print(f'reSurf optimal {reSurf_optimal_group_sizes[i]} with {reSurf_optimal_time}')
    print("\n")
    plt.plot(range(1, 33), ref_e_times[newkey], label='ref', color='k')
    plt.plot([1, 32], [ref_optimal_time]*2, 'k--',)
    plt.plot(range(1, 33), reSurf_e_times[newkey], label='reSurf', color='C0')
    plt.plot([1, 32], [reSurf_optimal_time]*2, 'C0--')
    plt.legend()
    plt.show()
