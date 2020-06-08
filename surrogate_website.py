import matplotlib.pyplot as plt
import pysgpp
import numpy as np
import pickle
from sgpp_create_response_surface import getSetup, load_response_Surface
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import calculate_missing_values


if __name__ == "__main__":

    # DEBUGGING
    calculate_comparison = True

    # WEBSITE INPUT
    input_prob_sick = 0.2
    input_success_rate_test = 0.92
    input_false_positive_rate = 0.05
    input_population = 32824000
    input_daily_tests = 146000

    probabilities_sick = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    probabilities_sick.append(input_prob_sick)
    probabilities_sick = sorted(set(probabilities_sick))

    # load precalculated response surface
    gridType, dim, degree, _, _, _, sample_size, num_daily_tests, \
        test_duration, num_simultaneous_tests, \
        number_of_instances, lb, ub, boundaryLevel = getSetup()

    # refineType = 'regular'
    refineType = 'adaptive'
    level = 1
    numPoints = 800  # max number of grid points for adaptively refined grid

    test_strategies = [
        'individual-testing',
        'two-stage-testing',
        'binary-splitting',
        'RBS',
        'purim',
        'sobel'
    ]

    group_sizes = range(1, 33)

    # results
    ppt = np.zeros((len(test_strategies), len(probabilities_sick)))
    sd_ppt = np.zeros((len(test_strategies), len(probabilities_sick)))
    scaled_e_times = np.zeros(len(test_strategies))

    ref_ppt = np.zeros((len(test_strategies), len(probabilities_sick)))
    ref_sd_ppt = np.zeros((len(test_strategies), len(probabilities_sick)))
    ref_scaled_e_times = np.zeros(len(test_strategies))

    optimal_group_sizes = np.zeros((len(test_strategies), len(probabilities_sick)))
    ref_optimal_group_sizes = np.zeros((len(test_strategies), len(probabilities_sick)))

    # auxiliary data for group size optimization
    scaled_e_group_times = np.zeros((len(test_strategies), len(group_sizes)))

    for i, test_strategy in enumerate(test_strategies):
        ref_evaluationPoints = []
        group_evaluationPoints = []
        ppt_precalculatedReSurf = load_response_Surface(
            refineType, test_strategy, 'ppt', dim, degree, level, numPoints, lb, ub)
        # sd_ppt_precalculatedReSurf = load_response_Surface(
        #     refineType, test_strategy, 'sd-ppt', dim, degree, level, numPoints, lb, ub)
        time_precalculatedReSurf = load_response_Surface(
            refineType, test_strategy, 'time', dim, degree, level, numPoints, lb, ub)

        for j, prob_sick in enumerate(probabilities_sick):
            # optimize group size for each prob_sick
            for k, group_size in enumerate(group_sizes):
                evaluationPoint = [prob_sick, input_success_rate_test, input_false_positive_rate, group_size]
                group_evaluationPoints.append(evaluationPoint)
                e_time = time_precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))
                scaled_e_group_times[i, k] = e_time*num_daily_tests/input_daily_tests * input_population/sample_size
                optimal_group_sizes[i, j] = group_sizes[np.argmin(scaled_e_group_times[i, :])]

            evaluationPoint = [prob_sick, input_success_rate_test, input_false_positive_rate, optimal_group_sizes[i, j]]
            ppt[i, j] = ppt_precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))
            ref_evaluationPoints.append(evaluationPoint)

            # FIXING NON-OPTIMAL GROUP SIZES w.r.t. individual-testing
            if ppt[i, j] < ppt[0, j]:
                ppt[i, j] = ppt[0, j]

            # TODO: Currently i skip sd
            #sd_ppt[i, j] = sd_ppt_precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))
            if prob_sick == input_prob_sick:
                e_time = time_precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))
                # scale it to given poipulation and number of daily tests
                # I use 1000 daily tests, so in total i needed e_time*1000 tests
                # for M available tests instead of 1000 we'd need e_time*1000/M tests
                # for N population instead of 100,000 we'd need e_time*1000/M * N/100000 days
                scaled_e_times[i] = e_time*num_daily_tests/input_daily_tests * input_population/sample_size

        if calculate_comparison:
            # this is a simplification. But it's just way to expensive to actually calculate all
            # these optimla group sizes.
            # Calculate the error w.r.t. the optimal group size somewehre else and verify that using
            # the approximative results here is good enough
            ref_optimal_group_sizes = optimal_group_sizes

            # TEMPORARY USE SMALL WEBSITE POPULATIONS
            ref_sample_size = 10000  # sample_size
            reference_num_daily_tests = 100
            reference_test_duration = 5
            ref_num_simultaneous_tests = int(reference_num_daily_tests *
                                             reference_test_duration/24.0)  # num_simultaneous_tests
            ref_number_of_instances = 5  # number_of_instances

            calculate_missing_values(dim, ref_evaluationPoints, ref_sample_size, test_duration, ref_num_simultaneous_tests,
                                     ref_number_of_instances, test_strategy)
            print('precalculated missing values')
            f = sgpp_simStorage(dim, test_strategy,  lb, ub, ref_number_of_instances)

            for j, prob_sick in enumerate(probabilities_sick):
                evaluationPoint = [prob_sick, input_success_rate_test,
                                   input_false_positive_rate, ref_optimal_group_sizes[i, j]]
                ref_ppt[i, j] = f.eval(evaluationPoint, 'ppt')

                # FIXING NON-OPTIMAL GROUP SIZES w.r.t. individual-testing
                if ref_ppt[i, j] < ref_ppt[0, j]:
                    ref_ppt[i, j] = ref_ppt[0, j]

                ref_sd_ppt[i, j] = f.eval(evaluationPoint, 'sd-ppt')
                if prob_sick == input_prob_sick:
                    ref_e_time = f.eval(evaluationPoint, 'time')
                    ref_scaled_e_times[i] = ref_e_time*num_daily_tests/input_daily_tests * input_population/sample_size

            # If this is actually needed something went wrong and calculate_missing_values
            # did not cover everything
            # f.cleanUp()

    print("-------------------------------------------")
    print('expected test time')
    for i, test_strategy in enumerate(test_strategies):
        print(f'Using {test_strategy} would have taken  {scaled_e_times[i]:.0f} days'
              f'    [{ref_scaled_e_times[i]:.0f},   diff = {np.abs(scaled_e_times[i]-ref_scaled_e_times[i]):.1f}]')

    # print("\noptimal group size")
    # for i, test_strategy in enumerate(test_strategies):
    #     print(f'{test_strategy}:    {optimal_group_sizes[i]}    [{ref_optimal_group_sizes[i]}]')

    markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    linestyles = ['-', '-', '-', '-', '-', '--']
    print('\n')

    # plot ppt
    plt.figure(figsize=(12, 8))
    for i, test_strategy in enumerate(test_strategies):
        #plt.subplot(2, 3, i+1)
        plt.plot(probabilities_sick, ppt[i, :], label=test_strategy,
                 marker=markers[i], color=colors[i], linestyle=linestyles[i])
        plt.errorbar(probabilities_sick, ppt[i, :], yerr=sd_ppt[i, :], ecolor='k',
                     linestyle='None', capsize=5)
        if calculate_comparison:
            plt.plot(probabilities_sick, ref_ppt[i, :], label=None,
                     marker=markers[i], color=colors[i], linestyle=linestyles[i], alpha=0.5)
            plt.errorbar(probabilities_sick, ref_ppt[i, :], yerr=ref_sd_ppt[i, :], ecolor='grey',
                         linestyle='None', capsize=5)
            print(f'error in plot of {test_strategy}    {np.linalg.norm(ppt[i,:]-ref_ppt[i,:]):.5f}')
        plt.xlabel('infection rate')
        plt.ylabel('ppt')
        plt.legend()

    # plot optimal group sizes
    # plt.figure(figsize=(12, 8))
    # for i, test_strategy in enumerate(test_strategies):
    #     plt.subplot(2, 3, i+1)
    #     plt.plot(group_sizes, scaled_e_group_times[i, :], label=labels[i],
    #              marker=markers[i], color=colors[i], linestyle=linestyles[i])
    #     plt.plot([group_sizes[0], group_sizes[-1]], [np.min(scaled_e_group_times[i, :])]*2, 'k')
    #     if calculate_comparison:
    #         plt.plot(group_sizes, ref_scaled_e_group_times[i, :], label=labels[i],
    #                  marker=markers[i], color=colors[i], linestyle=linestyles[i], alpha=0.6)
    #         plt.plot([group_sizes[0], group_sizes[-1]], [np.min(ref_scaled_e_group_times[i, :])]*2, 'k', alpha=0.6)

    #     plt.xlabel('group size')
    #     plt.ylabel('expected time to test pop. [days]')
    #     plt.legend()

    plt.show()
