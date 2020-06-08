import matplotlib.pyplot as plt
import pysgpp
import numpy as np
import pickle
from sgpp_create_response_surface import getSetup, load_response_Surface
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import calculate_missing_values

# load precalculated response surface
gridType, dim, degree, _, _, _, sample_size, num_daily_tests, \
    test_duration, num_simultaneous_tests, \
    number_of_instances, lb, ub, boundaryLevel = getSetup()

refineType = 'adaptive'
numPoints = 800  # max number of grid points for adaptively refined grid
dummyLevel = 3

test_strategies = [
    # 'individual-testing',
    'two-stage-testing',
    'binary-splitting',
    'RBS',
    'purim',
    'sobel'
]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

plt.figure(figsize=(12, 12))
for plot_dim in range(4):
    plt.subplot(2, 2, plot_dim+1)
    for i, test_strategy in enumerate(test_strategies):
        precalculatedReSurf = load_response_Surface(
            refineType, test_strategy, 'ppt', dim, degree, dummyLevel, numPoints, lb, ub)
        grid = precalculatedReSurf.getGrid()
        gridStorage = grid.getStorage()
        for p in range(gridStorage.getSize()):
            point = gridStorage.getPointCoordinates(p).array()
            plt.plot(lb[plot_dim]+(ub[plot_dim]-lb[plot_dim])*point[plot_dim],
                     [0.1*i], 'x', color=colors[i], label=test_strategy)
# unique labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
