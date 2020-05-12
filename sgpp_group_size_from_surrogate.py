import matplotlib.pyplot as plt
import pysgpp
import numpy as np
import pickle
from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import calculate_missing_values


# DEBUGGING
calculate_comparison = True

# WEBSITE INPUT
input_prob_sick = 0.25
input_success_rate_test = 0.93
input_false_positive_rate = 0.017
input_population = 32824000
input_daily_tests = 146000

group_sizes = list(range(1, 33))
test_strategies = [
    # 'individual-testing',
    'two-stage-testing',
    'binary-splitting',
    'RBS',
    'purim',
    'sobel'
]

# load precalculated response surface
gridType, dim, degree, _, _, _, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests, evalType, scale_factor_pop,\
    number_of_instances, lb, ub, boundaryLevel = getSetup()
qoi = 'time'
refineType = 'regular'
level = 5
numPoints = 400  # max number of grid points for adaptively refined grid


scaled_e_times = np.zeros((len(test_strategies), len(group_sizes)))
ref_scaled_e_times = np.zeros((len(test_strategies), len(group_sizes)))

for i, test_strategy in enumerate(test_strategies):
    evaluationPoints = []
    # load qoi reSurf
    if refineType == 'regular':
        name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}_level{level}'
    elif refineType == 'adaptive':
        name = f'{test_strategy}_{qoi}_dim{dim}_deg{degree}_adaptive{numPoints}'
    dummyCoeff = np.loadtxt(f'precalc/reSurf/np_coeff_{name}.dat')
    coefficients = pysgpp.DataVector(dummyCoeff)
    grid = pysgpp.Grid.unserializeFromFile(f'precalc/reSurf/grid_{name}.dat')
    precalculatedReSurf = pysgpp.SplineResponseSurface(
        grid, coefficients, pysgpp.DataVector(lb[:dim]), pysgpp.DataVector(ub[:dim]), degree)
    print(f'precalculated {test_strategy} response surface with {precalculatedReSurf.getSize()} points  has been loaded')

    for j, group_size in enumerate(group_sizes):
        evaluationPoint = [input_prob_sick, input_success_rate_test, input_false_positive_rate, group_size]
        evaluationPoints.append(evaluationPoint)
        e_time = precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))
        scaled_e_times[i, j] = e_time*num_daily_tests/input_daily_tests * input_population/sample_size

    if calculate_comparison:
        calculate_missing_values(evaluationPoints, sample_size, test_duration, num_simultaneous_tests,
                                 number_of_instances, scale_factor_pop, test_strategy, evalType)

        f = sgpp_simStorage(dim, test_strategy,  qoi, lb, ub)
        for j, group_size in enumerate(group_sizes):
            evaluationPoint = [input_prob_sick, input_success_rate_test, input_false_positive_rate, group_size]
            ref_e_time = f.eval(evaluationPoint)
            ref_scaled_e_times[i, j] = ref_e_time*num_daily_tests/input_daily_tests * input_population/sample_size

optimal_group_sizes = np.zeros(len(test_strategies))
ref_optimal_group_sizes = np.zeros(len(test_strategies))
print('\nOptimal group sizes:')
for i, test_strategy in enumerate(test_strategies):
    optimal_group_sizes[i] = group_sizes[np.argmin(scaled_e_times[i, :])]
    ref_optimal_group_sizes[i] = group_sizes[np.argmin(ref_scaled_e_times[i, :])]
    print(f'{test_strategy}:    {optimal_group_sizes[i]}    [{ref_optimal_group_sizes[i]}]')

markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
linestyles = ['-', '-', '-', '-', '-', '--']
labels = ['Individual testing', '2-level pooling',
          'Binary splitting', 'Recursive binary splitting', 'Purim', 'Sobel-R1']
plt.figure()
for i, test_strategy in enumerate(test_strategies):
    plt.plot(group_sizes, scaled_e_times[i, :], label=labels[i],
             marker=markers[i], color=colors[i], linestyle=linestyles[i])
    plt.plot(group_sizes, ref_scaled_e_times[i, :], label=labels[i],
             marker=markers[i], color=colors[i], linestyle=linestyles[i], alpha=0.6)
    plt.xlabel('group size')
    plt.ylabel('expected time to test pop. [days]')


plt.legend()
plt.show()
