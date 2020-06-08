import matplotlib.pyplot as plt
import pysgpp
import numpy as np
import pickle
from sgpp_create_response_surface import getSetup, load_response_Surface
from sgpp_simStorage import sgpp_simStorage

# load precalculated response surface
gridType, dim, degree, _, _, _, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests, \
    number_of_instances, lb, ub, boundaryLevel = getSetup()

degree = 1

#refineType = 'adaptive'
refineType = 'regular'
numPoints = 800  # max number of grid points for adaptively refined grid
level = 4

test_strategies = [
    'individual-testing',
    'two-stage-testing',
    'binary-splitting',
    'RBS',
    'purim',
    'sobel'
]

test_strategy = 'two-stage-testing'
precalculatedReSurf = load_response_Surface(refineType, test_strategy, 'ppt',
                                            dim, degree, level, numPoints, lb, ub)

# default parameters
prob_sick = 0.2317
success_rate_test = 0.84
false_positive_rate_test = 0.11
group_size = 32

# 1D cuts
plt.figure(figsize=(12, 12))
for cut_through_dim in range(4):
    plt.subplot(2, 2, cut_through_dim+1)
    X = np.linspace(lb[cut_through_dim], ub[cut_through_dim], 1000)
    Y = np.zeros(len(X))
    point = [prob_sick, success_rate_test, false_positive_rate_test, group_size]
    for i, x in enumerate(X):
        point[cut_through_dim] = x
        Y[i] = precalculatedReSurf.eval(pysgpp.DataVector(point))
    plt.plot(X, Y)
    # dim 0: prob_sick
    # dim 1 success rate test
    # dim 2: false positive rate
    # dim 3: group size
    plt.title(f'cut through dim {cut_through_dim}')

plt.show()
