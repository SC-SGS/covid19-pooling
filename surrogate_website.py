import matplotlib.pyplot as plt
import pysgpp
import numpy as np
import pickle
from sgpp_create_response_surface import getSetup, load_response_Surface, getReSurfEvaluationPoint
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import calculate_missing_values
from sgpp_calculate_stochastic_noise import stochastic_noise

if __name__ == "__main__":

    # DEBUGGING
    calculate_comparison = False

    # WEBSITE INPUT
    # input_prob_sick = 0.27
    # input_success_rate_test = 0.55
    # input_false_positive_rate = 0.17
    # input_population = 32824000
    # input_daily_tests = 146000

    # PAPER
    input_prob_sick = 0.001  # dummy value, all used prob_sick are already chosen
    input_success_rate_test = 0.99
    input_false_positive_rate = 0.01
    # US
    input_population = 328240000
    input_daily_tests = 146000
    # DE
    # input_population = 83150000
    # input_daily_tests = 123000
    # UK
    # input_population = 67890000
    # input_daily_tests = 12000
    # IT
    # input_population = 60310000
    # input_daily_tests = 46000
    # SG
    # input_population = 5640000
    # input_daily_tests = 2900

    probabilities_sick = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    probabilities_sick.append(input_prob_sick)
    probabilities_sick = sorted(set(probabilities_sick))

    # load precalculated response surface
    gridType, dim, degree, _, _, _, _, num_daily_tests, \
        test_duration, num_simultaneous_tests, \
        _, lb, ub, boundaryLevel = getSetup()

    # refineType = 'regular'
    refineType = 'adaptive'
    level = 4
    numPoints = 1500  # max number of grid points for adaptively refined grid
    reSurf_sample_size = 100000
    reSurf_number_of_instances = 10

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
    scaled_e_times = np.zeros(len(test_strategies))

    ref_ppt = np.zeros((len(test_strategies), len(probabilities_sick)))
    ref_scaled_e_times = np.zeros(len(test_strategies))

    optimal_group_sizes = np.zeros((len(test_strategies), len(probabilities_sick)))
    ref_optimal_group_sizes = np.zeros((len(test_strategies), len(probabilities_sick)))

    # auxiliary data for group size optimization
    scaled_e_group_times = np.zeros((len(test_strategies), len(group_sizes)))

    for i, test_strategy in enumerate(test_strategies):
        ref_evaluationPoints = []
        group_evaluationPoints = []
        ppt_precalculatedReSurf = load_response_Surface(
            refineType, test_strategy, 'ppt', dim, degree, level, numPoints, lb, ub, reSurf_sample_size,
            reSurf_number_of_instances)
        time_precalculatedReSurf = load_response_Surface(
            refineType, test_strategy, 'time', dim, degree, level, numPoints, lb, ub, reSurf_sample_size,
            reSurf_number_of_instances)

        for j, prob_sick in enumerate(probabilities_sick):
            # optimize group size for each prob_sick
            for k, group_size in enumerate(group_sizes):
                evaluationPoint = [prob_sick, input_success_rate_test, input_false_positive_rate, group_size]
                group_evaluationPoints.append(evaluationPoint)
                e_time = time_precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))
                scaled_e_group_times[i, k] = e_time*num_daily_tests / \
                    input_daily_tests * input_population/reSurf_sample_size
                optimal_group_sizes[i, j] = group_sizes[np.argmin(scaled_e_group_times[i, :])]

            ref_evaluationPoints.append([prob_sick, input_success_rate_test, input_false_positive_rate,
                                         optimal_group_sizes[i, j]])

            evaluationPoint = getReSurfEvaluationPoint(prob_sick, input_success_rate_test,
                                                       input_false_positive_rate, optimal_group_sizes[i, j])
            ppt[i, j] = ppt_precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))

            # FIXING NON-OPTIMAL GROUP SIZES w.r.t. individual-testing
            if ppt[i, j] < ppt[0, j]:
                ppt[i, j] = ppt[0, j]

            if prob_sick == input_prob_sick:
                e_time = time_precalculatedReSurf.eval(pysgpp.DataVector(evaluationPoint))
                # scale it to given poipulation and number of daily tests
                # I use 1000 daily tests, so in total i needed e_time*1000 tests
                # for M available tests instead of 1000 we'd need e_time*1000/M tests
                # for N population instead of 100,000 we'd need e_time*1000/M * N/100000 days
                scaled_e_times[i] = e_time*num_daily_tests/input_daily_tests * input_population/reSurf_sample_size

        if calculate_comparison:
            # this is a simplification. But it's just way to expensive to actually calculate all
            # these optimal group sizes.
            # Calculate the error w.r.t. the optimal group size somewehre else and verify that using
            # the approximative results here is good enough
            ref_optimal_group_sizes = optimal_group_sizes

            # TEMPORARY USE SMALL WEBSITE POPULATIONS
            reference_sample_size = 100000  # 10000
            reference_num_daily_tests = int(reference_sample_size/100)
            reference_test_duration = 5
            ref_num_simultaneous_tests = int(reference_num_daily_tests * reference_test_duration/24.0)
            ref_number_of_instances = 10  # 5

            calculate_missing_values(dim, ref_evaluationPoints, reference_sample_size, test_duration,
                                     ref_num_simultaneous_tests, ref_number_of_instances, test_strategy)
            print('precalculated missing values')
            f = sgpp_simStorage(dim, test_strategy,  lb, ub, ref_number_of_instances,
                                reference_sample_size, reference_num_daily_tests, reference_test_duration)

            for j, prob_sick in enumerate(probabilities_sick):
                evaluationPoint = [prob_sick, input_success_rate_test,
                                   input_false_positive_rate, ref_optimal_group_sizes[i, j]]
                ref_ppt[i, j] = f.eval(evaluationPoint, 'ppt')

                # FIXING NON-OPTIMAL GROUP SIZES w.r.t. individual-testing
                if ref_ppt[i, j] < ref_ppt[0, j]:
                    ref_ppt[i, j] = ref_ppt[0, j]

                if prob_sick == input_prob_sick:
                    ref_e_time = f.eval(evaluationPoint, 'time')
                    ref_scaled_e_times[i] = ref_e_time*reference_num_daily_tests / \
                        input_daily_tests * input_population/reference_sample_size

    print("-------------------------------------------")
    print('expected test time')
    for i, test_strategy in enumerate(test_strategies):
        numNoisePoints = 100
        number_outer_repetitions = 10
        noise = stochastic_noise(test_strategy, 'time', reference_sample_size,
                                 ref_number_of_instances, numNoisePoints, number_outer_repetitions)
        print(f'Using {test_strategy:20s} takes  {str(scaled_e_times[i]):.7s} days'
              f' [true={str(ref_scaled_e_times[i]):.6s}, diff={str(np.abs(scaled_e_times[i]-ref_scaled_e_times[i])):.4s},'
              f' rel diff={str(np.abs(scaled_e_times[i]-ref_scaled_e_times[i])/ref_scaled_e_times[i]):.5s},'
              f' noise {noise:.2f}]')

    print(f'Verification: individual-testing should need {input_population/input_daily_tests:.1f} days')

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
        #plt.errorbar(probabilities_sick, ppt[i, :], yerr=sd_ppt[i, :], ecolor='k',linestyle='None', capsize=5)
        if calculate_comparison:
            plt.plot(probabilities_sick, ref_ppt[i, :], label=None,
                     marker=markers[i], color=colors[i], linestyle=linestyles[i], alpha=0.5)
            print(f'error in plot of {test_strategy}    {np.linalg.norm(ppt[i,:]-ref_ppt[i,:]):.5f}')
        plt.xlabel('infection rate')
        plt.ylabel('ppt')
        plt.legend()

    # plot ppt error
    # if calculate_comparison:
    #     plt.figure(figsize=(12, 8))
    #     for i, test_strategy in enumerate(test_strategies):
    #         #plt.subplot(2, 3, i+1)
    #         plt.plot(probabilities_sick, abs(ppt[i, :]-ref_ppt[i, :]), label=test_strategy,
    #                  marker=markers[i], color=colors[i], linestyle=linestyles[i])
    #         plt.xlabel('infection rate')
    #         plt.ylabel('ppt - diff to reference')
    #         plt.legend()

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

#    plt.show()
