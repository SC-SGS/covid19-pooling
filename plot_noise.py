import ipdb
from sgpp_calculate_stochastic_noise import stochastic_noise
from setup import getSetup
import numpy as np
import matplotlib.pyplot as plt
from sgpp_create_response_surface import auxiliary

# default plot font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

gridType, dim, degree, _, _, name, _, _, \
    test_duration, num_simultaneous_tests,    _, lb, ub,\
    boundaryLevel = getSetup()
qoi = 'ppt'
test_strategies = [
    'individual-testing',
    'two-stage-testing',
    'binary-splitting',
    'RBS',
    'purim',
    'sobel'
]
markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
pop_rep = [[1000, 1], [1000, 10], [10000, 1], [10000, 5],  [10000, 10], [100000, 1], [100000, 10]]

# CALCULATE NOISE
X = range(len(pop_rep))
noises = np.zeros((len(test_strategies), len(pop_rep)))
for i, test_strategy in enumerate(test_strategies):
    for j, [pop, rep] in enumerate(pop_rep):
        numNoisePoints = 100
        number_outer_repetitions = 10
        noises[i, j] = stochastic_noise(test_strategy, qoi, pop, rep,
                                        numNoisePoints, number_outer_repetitions)

# CALCULATE BEST AVAILABLE SURROGATE
#  currently adaptive 800pts
gridType = 'nakBsplineBoundary'
reSurf_pop_rep = [[1000, 10],   [10000, 10], [100000, 10]]
adaptive_nrmses = np.zeros((len(test_strategies), len(reSurf_pop_rep), 1))
for j, [pop, rep] in enumerate(reSurf_pop_rep):
    if rep == 10:
        sample_size = pop
        num_daily_tests = int(pop/100)
        number_of_instances = rep
        print(f'calcualting error for {pop}/{rep}')
        _, adaptive_nrmses[:, j, :], _\
            = auxiliary('adaptive', test_strategies, [qoi], sample_size, num_daily_tests,
                        test_duration, dim, number_of_instances, gridType, degree, boundaryLevel, lb, ub,
                        'dummy', 800, 1, 10, verbose=False, calcError=True, numMCPoints=1000,
                        saveReSurf=False)

# PLOT
plt.figure(figsize=[18, 9])
for i, test_strategy in enumerate(test_strategies):
    plt.subplot(2, 3, i+1)
    # hard coded X because I couldn't find a nice way of quickly getting what i want
    plt.plot([1, 4, 6], adaptive_nrmses[i, :, 0], '-', color=colors[i], marker=markers[i], label=test_strategy)
    plt.plot(X, noises[i, :], '--', color=colors[i],  label=test_strategy)

    # sqrt(N), Monte Carlo convergence h^(-1/2)
    plt.plot([1, 4], [5e-2, 1e-2], 'grey', 'o', legend='h^{-1/2}')

    labels = [f'{int(pop/1000)}k/{rep}' for [pop, rep] in pop_rep]
    plt.xticks(range(len(pop_rep)), labels)
    plt.gca().set_yscale('log')
    plt.ylim([1e-3, 1e-1])
    plt.title(f'{test_strategy}')
# plt.legend()
plt.tight_layout()
plt.xlabel('population/repetitions')
plt.ylabel('approx. noise')
plt.show()
