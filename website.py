import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from Statistics import Corona_Simulation_Statistics


def simulate(prob_sick, success_rate_test, false_positive_rate,
             group_size, test_strategy, reference_sample_size,
             reference_num_simultaneous_tests, reference_test_duration, reference_number_of_instances):
    '''
    perform one simulation on fixed reference population
    In ocntrast to the paper, the group size is fixed to the given value for all infection
    rates and NOT optimized
    Parameters:
    prob_sick                           the probability of sickness / infection rate
    success_rate_test                   sensitivity of the test
    false_positive_rate                 false positive rate of the test
    group_size                          group size 
    test_strategy                       choice of pooling algorithm
    reference_sample_size               population size of the reference population  
    reference_num_simultaneous_tests    number of simultaneous tests of the reference population
    reference_test_duration             duration of a test for the reference population
    reference_number_of_instances       number of repetitions of the simulation for stochastical values
    '''
    stat_test = Corona_Simulation_Statistics(prob_sick, success_rate_test,
                                             false_positive_rate, test_strategy,
                                             reference_test_duration, group_size)
    start = time.time()
    stat_test.statistical_analysis(reference_sample_size, reference_num_simultaneous_tests,
                                   reference_number_of_instances)
    runtime = time.time()-start
    print(f'runtime of one simulation: {runtime}s')

    e_time = stat_test.e_time
    e_num_confirmed_per_test = stat_test.e_num_confirmed_per_test

    sd_time = stat_test.sd_time
    sd_num_confirmed_per_test = stat_test.sd_num_confirmed_per_test

    return e_time, e_num_confirmed_per_test, \
        sd_time, sd_num_confirmed_per_test


def generateData(input_prob_sick, input_success_rate_test, input_false_positive_rate, input_group_size,
                 input_population, input_daily_tests):
    '''
    generate the dataset which will be plotted on the website, given the input parameters
    The input_population and inpu_daily_tests are NOT actually used.
    Instead the simulation is performed on a reference population of meaningful size, which balances
    runtime and stochastical neccessity
    Parameters
    input_prob_sick             the probability of sickness / infection rate
    input_success_rate_test     sensitivity of the test
    input_false_positive_rate   false positive rate of the test
    input_group_size            group size
    input_population            population size
    input_daily_tests           number of daily tests
    '''

    # THESE PARAMETERS DETERMINE RUNTIME AND ACCURACY OF THE SIMULATION
    reference_sample_size = 50000
    reference_num_daily_tests = 1000
    reference_number_of_instances = 5
    reference_test_duration = 5
    reference_num_simultaneous_tests = int(reference_num_daily_tests*reference_test_duration/24.0)

    # probabilities_sick = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    probabilities_sick = [0.001,  0.1]
    probabilities_sick.append(input_prob_sick)
    probabilities_sick = sorted(set(probabilities_sick))

    test_strategies = [
        'individual-testing',
        'two-stage-testing',
        'binary-splitting',
        'RBS',
        'purim',
        'sobel'
    ]

    e_ppt = np.zeros((len(test_strategies), len(probabilities_sick)))
    sd_ppt = np.zeros((len(test_strategies), len(probabilities_sick)))
    expected_times = np.zeros(len(test_strategies))

    for i, test_strategy in enumerate(test_strategies):
        for j, prob_sick in enumerate(probabilities_sick):
            e_time, e_num_confirmed_per_test, sd_time, \
                sd_num_confirmed_per_test = simulate(prob_sick, input_success_rate_test,
                                                     input_false_positive_rate, input_group_size,
                                                     test_strategy, reference_sample_size,
                                                     reference_num_simultaneous_tests,
                                                     reference_test_duration,
                                                     reference_number_of_instances)
            e_ppt[i, j] = e_num_confirmed_per_test
            sd_ppt[i, j] = sd_num_confirmed_per_test

            if prob_sick == input_prob_sick:
                # scale it to given poipulation and number of daily tests
                # I use 1000 daily tests, so in total i needed e_time*1000 tests
                # for M available tests instead of 1000 we'd need e_time*1000/M tests
                # for N population instead of 100,000 we'd need e_time*1000/M * N/100000 days
                expected_times[i] = e_time*reference_num_daily_tests / \
                    input_daily_tests * input_population/reference_sample_size

    # THIS IS FOR DEBUGGING / TESTING THE WEBSITE SIMULATION
    for i, test_strategy in enumerate(test_strategies):
        print(f'Using {test_strategy} would have taken  {expected_times[i]:.2f} days')
    markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    linestyles = ['-', '-', '-', '-', '-', '--']
    labels = ['Individual testing', '2-level pooling',
              'Binary splitting', 'Recursive binary splitting', 'Purim', 'Sobel-R1']
    plt.figure()
    for i, test_strategy in enumerate(test_strategies):
        plt.plot(probabilities_sick, e_ppt[i, :], label=labels[i],
                 marker=markers[i], color=colors[i], linestyle=linestyles[i])
        plt.errorbar(probabilities_sick, e_ppt[i, :],
                     yerr=sd_ppt[i, :], ecolor='k', linestyle='None', capsize=5)
    plt.legend()
    plt.show()

    return probabilities_sick, expected_values, standard_deviations, expected_times


if __name__ == "__main__":
    # WEBSITE INPUT
    input_prob_sick = 0.02
    input_success_rate_test = 0.93
    input_false_positive_rate = 0.017
    input_group_size = 25
    input_population = 32824000
    input_daily_tests = 146000

    probabilities_sick, expected_values, standard_deviations, \
        expected_times = generateData(input_prob_sick, input_success_rate_test, input_false_positive_rate, input_group_size,
                                      input_population, input_daily_tests)
