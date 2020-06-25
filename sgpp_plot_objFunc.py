import pysgpp
import numpy as np
import time
from Statistics import Corona_Simulation_Statistics
import matplotlib.pyplot as plt
from setup import getSetup
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import calculate_missing_values
from mpl_toolkits import mplot3d


def plot1D(test_strategies, qoi, success_rate_test, false_positive_rate_test,
           group_size, dim, sample_size, test_duration, num_simultaneous_tests, number_of_instances,
           lb, ub, num_daily_tests):
    probabilities_sick = np.linspace(0.01, 0.3, 101)
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


def plot2D(test_strategy, qoi, false_positive_rate_test, group_size, dim, sample_size, test_duration,
           num_simultaneous_tests, number_of_instances, lb, ub, num_daily_tests):
    probabilities_sick = np.linspace(0.01, 0.3, 101)
    success_rates_test = np.linspace(0.5, 1.0, 101)
    X, Y = np.meshgrid(probabilities_sick, success_rates_test)
    points = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            prob_sick = X[i, j]
            success_rate_test = Y[i, j]
            point = [prob_sick, success_rate_test, false_positive_rate_test, group_size]
            points.append(point)
    num_new_points = calculate_missing_values(dim, points, sample_size, test_duration,
                                              num_simultaneous_tests, number_of_instances, test_strategy)
    print(f'Calcualted {num_new_points} new evaluations\n')

    f = sgpp_simStorage(dim, test_strategy, lb, ub, number_of_instances,
                        reference_sample_size=sample_size, reference_num_daily_tests=num_daily_tests,
                        reference_test_duration=5)
    Z = np.zeros(np.shape(X))
    for i in range(len(X)):
        for j in range(len(X[0])):
            prob_sick = X[i, j]
            success_rate_test = Y[i, j]
            point = [prob_sick, success_rate_test, false_positive_rate_test, group_size]
            Z[i, j] = f.eval(point, qoi, recalculate=False)

    f.cleanUp()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    plt.xlabel('infection rate')
    plt.ylabel('succes rate')
    plt.legend()


if __name__ == "__main__":

    gridType, dim, degree, _, _, name, sample_size, num_daily_tests,\
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
    success_rate_test = 1.0  # 0.99
    false_positive_rate_test = 0.0  # 0.01
    group_size = 8  # 32

    qoi = 'ppt'

    # plot1D(test_strategies, qoi, success_rate_test, false_positive_rate_test,
    #        group_size, dim, sample_size, test_duration, num_simultaneous_tests, number_of_instances,
    #        lb, ub, num_daily_tests)

    test_strategy = 'two-stage-testing'
    plot2D(test_strategy, qoi, false_positive_rate_test, group_size, dim, sample_size, test_duration,
           num_simultaneous_tests, number_of_instances, lb, ub, num_daily_tests)

    plt.show()
