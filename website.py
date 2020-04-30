from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage
import matplotlib.pyplot as plt
import pysgpp
from sgpp_precalc_parallel import precalc_parallel
import numpy as np
import pickle

# DEBUGGING
calculate_comparison = True

# WEBSITE INPUT
input_prob_sick = 0.02
input_success_rate_test = 0.93
input_false_positive_rate = 0.017
input_group_size = 25
input_population = 32824000
input_daily_tests = 146000

probabilities_sick = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
probabilities_sick.append(input_prob_sick)
probabilities_sick = sorted(set(probabilities_sick))

# load precalculated response surface
gridType, dim, degree, _, _, _, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests, evalType, scale_factor_pop,\
    number_of_instances, lb, ub, boundaryLevel = getSetup()
qoi = 'ppt'

#refineType = 'regular'
refineType = 'adaptive'
level = 5
numPoints = 400  # max number of grid points for adaptively refined grid

test_strategies = [
    'individual-testing',
    'two-stage-testing',
    'binary-splitting',
    'RBS',
    'purim',
    'sobel'
]

ppt = np.zeros((len(test_strategies), len(probabilities_sick)))
scaled_e_times = np.zeros(len(test_strategies))
ref_ppt = np.zeros((len(test_strategies), len(probabilities_sick)))
ref_scaled_e_times = np.zeros(len(test_strategies))

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

    # load time reSurf

    # TODO THIS IS ONLY FOR DEBUGGING
    refineType = 'regular'

    if refineType == 'regular':
        name_time = f'{test_strategy}_time_dim{dim}_deg{degree}_level{level}'
    elif refineType == 'adaptive':
        name_time = f'{test_strategy}_time_dim{dim}_deg{degree}_adaptive{numPoints}'

    dummyCoeff_time = np.loadtxt(f'precalc/reSurf/np_coeff_{name_time}.dat')
    coefficients_time = pysgpp.DataVector(dummyCoeff_time)
    grid_time = pysgpp.Grid.unserializeFromFile(f'precalc/reSurf/grid_{name_time}.dat')
    precalculatedReSurf_time = pysgpp.SplineResponseSurface(
        grid_time, coefficients_time, pysgpp.DataVector(lb[:dim]), pysgpp.DataVector(ub[:dim]), degree)

    for j, prob_sick in enumerate(probabilities_sick):
        evaluationPoint = [prob_sick, input_success_rate_test, input_false_positive_rate, input_group_size]
        evaluationPoints.append(evaluationPoint)
        ppt[i, j] = precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))
        if prob_sick == input_prob_sick:
            e_time = precalculatedReSurf_time.eval(pysgpp.DataVector(evaluationPoint))
            # scale it to given poipulation and number of daily tests
            # I use 1000 daily tests, so in total i needed e_time*1000 tests
            # for M available tests instead of 1000 we'd need e_time*1000/M tests
            # for N population instead of 100,000 we'd need e_time*1000/M * N/100000 days
            scaled_e_times[i] = e_time*num_daily_tests/input_daily_tests * input_population/sample_size
    if calculate_comparison:
        # load precalculated data
        savePath = "/home/rehmemk/git/covid19-pooling/precalc/"
        precalcValuesFileName = savePath + f"precalc_values_{test_strategy}.pkl"
        with open(precalcValuesFileName, 'rb') as fp:
            precalculatedValues = pickle.load(fp)
        todoPoints = []
        for point in evaluationPoints:
            key = tuple([point[0], point[1], point[2], point[3], evalType, number_of_instances])
            if key not in precalculatedValues:
                todoPoints.append(point)

        print(f"\ncalculating {len(todoPoints)} new evaluations")
        multiprocessing_dict = precalc_parallel(todoPoints, sample_size, test_duration, num_simultaneous_tests,
                                                number_of_instances, scale_factor_pop, test_strategy, evalType)
        for key in multiprocessing_dict:
            precalculatedValues[key] = multiprocessing_dict[key]
        with open(precalcValuesFileName, "wb") as fp:
            pickle.dump(precalculatedValues, fp)
        f = sgpp_simStorage(dim, test_strategy,  qoi, lb, ub)
        f_time = sgpp_simStorage(dim, test_strategy,  'time', lb, ub)
        for j, prob_sick in enumerate(probabilities_sick):
            evaluationPoint = [prob_sick, input_success_rate_test, input_false_positive_rate, input_group_size]
            ref_ppt[i, j] = f.eval(evaluationPoint)
            if prob_sick == input_prob_sick:
                ref_e_time = f_time.eval(evaluationPoint)
                ref_scaled_e_times[i] = ref_e_time*num_daily_tests/input_daily_tests * input_population/sample_size

for i, test_strategy in enumerate(test_strategies):
    print(f'Using {test_strategy} would have taken  {scaled_e_times[i]:.0f} days')
    print(f'Using {test_strategy} would have taken {ref_scaled_e_times[i]:.0f} days (REFERENCE)')

markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
linestyles = ['-', '-', '-', '-', '-', '--']
labels = ['Individual testing', '2-level pooling',
          'Binary splitting', 'Recursive binary splitting', 'Purim', 'Sobel-R1']
plt.figure()
for i, test_strategy in enumerate(test_strategies):
    plt.plot(probabilities_sick, ppt[i, :], label=labels[i],
             marker=markers[i], color=colors[i], linestyle=linestyles[i])
    if calculate_comparison:
        plt.plot(probabilities_sick, ref_ppt[i, :], label=None,
                 marker=markers[i], color=colors[i], linestyle=linestyles[i], alpha=0.5)


plt.legend()
plt.show()
