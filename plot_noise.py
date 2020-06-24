from sgpp_calculate_stochastic_noise import stochastic_noise
from setup import getSetup
import numpy as np
import matplotlib.pyplot as plt

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

gridType, dim, degree, _, _, name, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests,    number_of_instances, lb, ub,\
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
pop_rep = [[1000, 1], [1000, 10], [10000, 1], [10000, 5],  [10000, 10], [100000, 1]]
X = range(len(pop_rep))
for i, test_strategy in enumerate(test_strategies):
    noises = np.zeros(len(pop_rep))
    for j, [pop, rep] in enumerate(pop_rep):
        numNoisePoints = 100
        number_outer_repetitions = 10
        noises[j] = stochastic_noise(test_strategy, qoi, pop, rep,
                                     numNoisePoints, number_outer_repetitions)
    plt.plot(X, noises, color=colors[i], marker=markers[i], label=test_strategy)

labels = [f'{int(pop/1000)}k/{rep}' for [pop, rep] in pop_rep]
plt.xticks(range(len(pop_rep)), labels)
plt.gca().set_yscale('log')
plt.ylim([1e-3, 1e-1])
plt.legend()
plt.xlabel('population/repetitions')
plt.ylabel('approx. noise')
plt.show()
