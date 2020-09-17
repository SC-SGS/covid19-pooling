import sys
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from CoronaTestingSimulation import Corona_Simulation
from Statistics import Corona_Simulation_Statistics
import multiprocessing


'''
Scenario 1
Test all individuals of a population
'''

# whether to print plotdata
PRINTPLOTDATA = True

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


# name for the data dump and plots
def getName(scale_factor_pop, scale_factor_test, success_rate_test=0.99):
    name = 'PLOS_scenario1_scalepop{}_scaletest{}'.format(scale_factor_pop, scale_factor_test)
    if success_rate_test != 0.99:
        name += '_{}'.format(success_rate_test)
    return name


def worker(return_dict, sample_size, prob_sick, success_rate_test, false_posivite_rate, test_strategy,
           num_simultaneous_tests, test_duration, group_size, scale_factor_pop,
           tests_repetitions, test_result_decision_strategy, number_of_instances, country):
    '''
    worker function for multiprocessing
    performs the same test tests_repetitions many times and returns expected valkues and standard deviations
    '''

    stat_test = Corona_Simulation_Statistics(prob_sick, success_rate_test,
                                             false_posivite_rate, test_strategy,
                                             test_duration, group_size,
                                             tests_repetitions, test_result_decision_strategy,
                                             scale_factor_pop)
    stat_test.statistical_analysis(sample_size, num_simultaneous_tests, number_of_instances)
    print('Calculated {} for {} prob sick {}'.format(test_strategy, country, prob_sick))
    print('scaled to {} population and {} simulataneous tests\n'.format(sample_size, num_simultaneous_tests))
    # gather results
    worker_dict = {}
    worker_dict['e_num_tests'] = stat_test.e_number_of_tests*scale_factor_pop
    worker_dict['e_time'] = stat_test.e_time*scale_factor_pop
    worker_dict['e_num_confirmed_sick_individuals'] = stat_test.e_num_confirmed_sick_individuals*scale_factor_pop
    worker_dict['e_false_positive_rate'] = stat_test.e_false_positive_rate
    worker_dict['e_ratio_of_sick_found'] = stat_test.e_ratio_of_sick_found
    worker_dict['e_num_confirmed_per_test'] = stat_test.e_num_confirmed_per_test
    worker_dict['e_num_sent_to_quarantine'] = stat_test.e_num_sent_to_quarantine
    worker_dict['sd_num_tests'] = stat_test.sd_number_of_tests*scale_factor_pop
    worker_dict['sd_time'] = stat_test.sd_time*scale_factor_pop
    worker_dict['sd_false_positive_rate'] = stat_test.sd_false_positive_rate
    worker_dict['sd_ratio_of_sick_found'] = stat_test.sd_ratio_of_sick_found
    worker_dict['sd_num_confirmed_per_test'] = stat_test.sd_num_confirmed_per_test
    worker_dict['sd_num_sent_to_quarantine'] = stat_test.sd_num_sent_to_quarantine

    return_dict['{}_{}_{}'.format(test_strategy, country, prob_sick)] = worker_dict


def calculation():
    start = time.time()
    randomseed = 19
    np.random.seed(randomseed)

    probabilities_sick = [0.01]
    success_rate_test = 0.99
    false_posivite_rate = 0.01
    tests_repetitions = 1
    test_result_decision_strategy = 'max'
    number_of_instances = 10
    test_duration = 5

    # optimal group sizes in order individual, two level, binary splitting, RBS, purim, sobel
    optimal_group_sizes = {}
    if success_rate_test == 0.99:
        optimal_group_sizes[0.001] = [1, 32, 32, 32, 32, 32]
        optimal_group_sizes[0.0025] = [1, 23, 32, 32, 32, 32]
        optimal_group_sizes[0.005] = [1, 16, 32, 32, 32, 32]
        optimal_group_sizes[0.0075] = [1, 12, 32, 32, 32, 32]
        optimal_group_sizes[0.01] = [1, 10, 32, 32, 27, 31]
        optimal_group_sizes[0.025] = [1, 7, 16, 30, 14, 30]
        optimal_group_sizes[0.05] = [1, 5, 8, 15, 10, 27]
        optimal_group_sizes[0.1] = [1, 4, 4, 8, 7, 20]
        optimal_group_sizes[0.15] = [1, 3, 4, 6, 6, 32]
        optimal_group_sizes[0.2] = [1, 3, 2, 1, 5, 30]
        optimal_group_sizes[0.25] = [1, 3, 2, 1, 5, 28]
        optimal_group_sizes[0.3] = [1, 3, 1, 1, 1, 19]
        optimal_group_sizes[0.5] = [1, 3, 1, 1, 1, 19]
    elif success_rate_test == 0.75:
        optimal_group_sizes[0.001] = [1, 32, 32, 32, 32, 32]
        optimal_group_sizes[0.0025] = [1, 21, 32, 32, 32, 32]
        optimal_group_sizes[0.005] = [1, 18, 32, 32, 32, 32]
        optimal_group_sizes[0.0075] = [1, 15, 32, 32, 32, 32]
        optimal_group_sizes[0.01] = [1, 12, 32, 32, 31, 32]
        optimal_group_sizes[0.025] = [1, 8, 32, 32, 18, 30]
        optimal_group_sizes[0.05] = [1, 6, 32, 32, 12, 32]
        optimal_group_sizes[0.1] = [1, 5, 32, 31, 8, 8]
        optimal_group_sizes[0.15] = [1, 4, 32, 32, 7, 6]
        optimal_group_sizes[0.2] = [1, 4, 32, 32, 32, 4]
        optimal_group_sizes[0.25] = [1, 4, 32, 32, 32, 3]
        optimal_group_sizes[0.3] = [1, 30, 32, 32, 32, 32]

    # strings identifiying the test strategies
    test_strategies = [
        'individual-testing',
        'two-stage-testing',
        'binary-splitting',
        'RBS',
        'purim',
        'sobel'
    ]

    # use scale_factor_pop = 10 for the original results in the paper
    # use scale_factor_pop = 100 for much faster calculation and  little loss of accuracy
    countries = {}
    countries['DE'] = {'population': 10000, 'tests_per_day': 10000,
                       'scale_factor_pop': 1, 'scale_factor_test': 1}

    num_countries = len(countries.keys())

    print('ref values for individual testing:')
    for country in countries:
        print('{}   {}'.format(country, int(countries[country]['population'] / countries[country]['tests_per_day'])))
    print('\n')

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    e_num_tests = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    e_time = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    e_false_positive_rate = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    e_num_confirmed_sick_individuals = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    e_ratio_of_sick_found = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    e_num_confirmed_per_test = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    e_num_sent_to_quarantine = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))

    sd_num_tests = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    sd_time = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    sd_false_positive_rate = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    sd_ratio_of_sick_found = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    sd_num_confirmed_per_test = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))
    sd_num_sent_to_quarantine = np.zeros((len(test_strategies), num_countries, len(probabilities_sick)))

    jobs = []

    for i, test_strategy in enumerate(test_strategies):
        for j, country in enumerate(countries.keys()):
            for k, prob_sick in enumerate(probabilities_sick):
                group_size = optimal_group_sizes[prob_sick][i]
                scale_factor_pop = countries[country]['scale_factor_pop']
                scale_factor_test = countries[country]['scale_factor_test']
                sample_size = int(countries[country]['population'] / scale_factor_pop / scale_factor_test)
                num_simultaneous_tests = int(
                    np.ceil(countries[country]['tests_per_day']/scale_factor_test*test_duration/24.0))
                p = multiprocessing.Process(target=worker, args=(return_dict, sample_size, prob_sick,
                                                                 success_rate_test, false_posivite_rate, test_strategy, num_simultaneous_tests,
                                                                 test_duration, group_size, scale_factor_pop, tests_repetitions, test_result_decision_strategy,
                                                                 number_of_instances, country))
                jobs.append(p)
                p.start()
    for proc in jobs:
        proc.join()

    # gather results
    for i, test_strategy in enumerate(test_strategies):
        for j, country in enumerate(countries.keys()):
            for k, prob_sick in enumerate(probabilities_sick):
                worker_dict = return_dict['{}_{}_{}'.format(test_strategy, country, prob_sick)]

                e_num_tests[i, j, k] = worker_dict['e_num_tests']
                e_time[i, j, k] = worker_dict['e_time']
                e_num_confirmed_sick_individuals[i, j, k] = worker_dict['e_num_confirmed_sick_individuals']
                e_false_positive_rate[i, j, k] = worker_dict['e_false_positive_rate']
                e_ratio_of_sick_found[i, j, k] = worker_dict['e_ratio_of_sick_found']
                e_num_confirmed_per_test[i, j, k] = worker_dict['e_num_confirmed_per_test']
                e_num_sent_to_quarantine[i, j, k] = worker_dict['e_num_sent_to_quarantine']
                sd_num_tests[i, j, k] = worker_dict['sd_num_tests']
                sd_time[i, j, k] = worker_dict['sd_time']
                sd_false_positive_rate[i, j, k] = worker_dict['sd_false_positive_rate']
                sd_ratio_of_sick_found[i, j, k] = worker_dict['sd_ratio_of_sick_found']
                sd_num_confirmed_per_test[i, j, k] = worker_dict['sd_num_confirmed_per_test']
                sd_num_sent_to_quarantine[i, j, k] = worker_dict['sd_num_sent_to_quarantine']

    sample_sizes = [countries[country]['population'] for country in countries.keys()]
    daily_tests_per_1m = [countries[country]['tests_per_day']/countries[country]
                          ['population']*1000000 for country in countries.keys()]
    print('daily_test_per_1m {}'.format(daily_tests_per_1m))

    runtime = time.time()-start
    print('calculating took {}s'.format(runtime))

    # save data to allow plotting without doing the whole calculation again.
    data = {
        'randomseed': randomseed,
        'probabilities_sick': probabilities_sick,
        'success_rate_test ': success_rate_test,
        'false_posivite_rate': false_posivite_rate,
        'tests_repetitions': tests_repetitions,
        'test_result_decision_strategy': test_result_decision_strategy,
        'test_strategies': test_strategies,
        'countries': countries,
        'number_of_instances': number_of_instances,
        'test_duration': test_duration,
        'group_size': group_size,
        'e_num_tests ': e_num_tests,
        'e_time': e_time,
        'e_false_positive_rate': e_false_positive_rate,
        'e_num_confirmed_sick_individuals': e_num_confirmed_sick_individuals,
        'e_ratio_of_sick_found': e_ratio_of_sick_found,
        'e_num_confirmed_per_test': e_num_confirmed_per_test,
        'e_num_sent_to_quarantine': e_num_sent_to_quarantine,
        'sd_num_tests': sd_num_tests,
        'sd_time': sd_time,
        'sd_false_positive_rate': sd_false_positive_rate,
        'sd_ratio_of_sick_found': sd_ratio_of_sick_found,
        'sd_num_confirmed_per_test': sd_num_confirmed_per_test,
        'sd_num_sent_to_quarantine': sd_num_sent_to_quarantine,
        'sample_sizes': sample_sizes,
        'daily_tests_per_1m': daily_tests_per_1m,
        'runtime': runtime,
    }
    filename = getName(countries['DE']['scale_factor_pop'],
                       countries['DE']['scale_factor_test'],
                       success_rate_test)
    path = 'data/{}.pkl'.format(filename)
    with open(path, 'wb+') as fp:
        pickle.dump(data, fp)
    print('saved data as {}'.format(path))
    return filename


def plotting(filename, prob_sick_plot_index, saveFig=0):
    # load data
    datapath = 'data/{}.pkl'.format(filename)
    with open(datapath, 'rb') as fp:
        data = pickle.load(fp)
    figpath = 'plots/{}'.format(filename)

    # extract relevant parameters from data
    test_strategies = data['test_strategies']
    daily_tests_per_1m = data['daily_tests_per_1m']
    countries = data['countries']
    probabilities_sick = data['probabilities_sick']
    e_time = data['e_time']
    sd_time = data['sd_time']
    e_num_confirmed_per_test = data['e_num_confirmed_per_test']
    sd_num_confirmed_per_test = data['sd_num_confirmed_per_test']

    # plotting
    markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    linestyles = ['-', '-', '-', '-', '-', '--']
    labels = ['Individual testing', '2-level pooling',
              'Binary splitting', 'Recursive binary splitting', 'Purim', 'Sobel-R1']

    ######## prob sick / sick persons per test ########
    plt.figure()
    for i, test_strategy in enumerate(test_strategies):
        for j in [0]:  # it's the same for all countries
            plt.plot(probabilities_sick, e_num_confirmed_per_test[i, j, :],
                     label=labels[i], marker=markers[i], color=colors[i], linestyle=linestyles[i])
            plt.errorbar(probabilities_sick, e_num_confirmed_per_test[i, j, :],
                         yerr=sd_num_confirmed_per_test[i, j, :], ecolor='k', linestyle='None', capsize=5)
    plt.xlabel('infection rate')
    plt.ylabel('exp. number of identified cases per test')
    plt.xticks([0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], [
               '0.1%    ', '    1%', '5%', '10%', '15%', '20%', '25%', '30%', ])
    plt.legend(loc='lower right', fontsize=11)
    if PRINTPLOTDATA:
        print(figpath+'psi{}_probsick_ppt.pdf'.format(prob_sick_plot_index))
        print("%20s" % "probabilities_sick", "".join(map(lambda x: "%7.4f " % x, probabilities_sick)))
        for i, test_strategy in enumerate(test_strategies):
            print("%20s" % test_strategy, "".join(map(lambda x: "%7.5f " % x, e_num_confirmed_per_test[i, 0, :])))
    if saveFig:
        plt.savefig(figpath+'psi{}_probsick_ppt.pdf'.format(prob_sick_plot_index), bbox_inches='tight')

    ######## prob sick / sick persons per test (Zoomed)########
    plt.figure()
    for i, test_strategy in enumerate(test_strategies):
        for j in [0]:  # it's the same for all countries
            plt.plot(probabilities_sick[:7], e_num_confirmed_per_test[i, j, :][:7],
                     label=labels[i], marker=markers[i], color=colors[i], linestyle=linestyles[i])
            plt.errorbar(probabilities_sick[:7], e_num_confirmed_per_test[i, j, :][:7],
                         yerr=sd_num_confirmed_per_test[i, j, :][:7], ecolor='k', linestyle='None', capsize=5)
    plt.xlabel('infection rate')
    plt.ylabel('exp. number of identified cases per test')
    plt.xticks([0.001, 0.01, 0.025, 0.05], ['0.1%', '1%', '2.5%', '5%'])
    if PRINTPLOTDATA:
        print(figpath+'psi{}_probsick_ppt.pdf'.format(prob_sick_plot_index))
        print(probabilities_sick)
        for i, test_strategy in enumerate(test_strategies):
            print(test_strategy, e_num_confirmed_per_test[i, 0, :])
    if saveFig:
        plt.savefig(figpath+'psi{}_probsick_ppt_zoomed.pdf'.format(prob_sick_plot_index), bbox_inches='tight')

    ######## test per 1M / expected time to test all ########
    plt.figure()
    for i, test_strategy in enumerate(test_strategies):
        plt.plot(daily_tests_per_1m, e_time[i, :, prob_sick_plot_index],
                 label=labels[i], marker=markers[i], color=colors[i], linestyle=linestyles[i])
        plt.errorbar(daily_tests_per_1m, e_time[i, :, prob_sick_plot_index],
                     yerr=sd_time[i, :, prob_sick_plot_index], ecolor='k',
                     linestyle='None', capsize=5)
    plt.xticks(daily_tests_per_1m, ['{} {}'.format(country, int(daily_tests_per_1m[i]))
                                    for i, country in enumerate(countries)], rotation=55)
    plt.ylabel('exp. time to test population [days]')
    plt.xlabel('daily tests / 1M population.')
    plt.legend(loc='upper right')
    if PRINTPLOTDATA:
        print(figpath+'psi{}_testsper1M_time.pdf'.format(prob_sick_plot_index))
        print("infection rate: %7.4f" % probabilities_sick[prob_sick_plot_index])
        print(" "*20, "".join(map(lambda x: "%7s " % x, countries)))
        print("%20s" % "daily_test_per_1m", "".join(map(lambda x: "%7.2f " % x, daily_tests_per_1m)))
        for i, test_strategy in enumerate(test_strategies):
            print("%20s" % test_strategy, "".join(map(lambda x: "%7.2f " % x, e_time[i, :, prob_sick_plot_index])))
    if saveFig:
        plt.savefig(figpath+'psi{}_testsper1M_time.pdf'.format(prob_sick_plot_index), bbox_inches='tight')

    ######## test per 1M / expected time to test all (Zoomed)########
    plt.figure()
    for i, test_strategy in enumerate(test_strategies):
        plt.plot(daily_tests_per_1m, e_time[i, :, prob_sick_plot_index],
                 label=labels[i], marker=markers[i], color=colors[i], linestyle=linestyles[i])
        plt.errorbar(daily_tests_per_1m, e_time[i, :, prob_sick_plot_index],
                     yerr=sd_time[i, :, prob_sick_plot_index], ecolor='k',
                     linestyle='None', capsize=5)
    plt.xticks(daily_tests_per_1m, ['{} {}'.format(country, int(daily_tests_per_1m[i]))
                                    for i, country in enumerate(countries)], rotation=55)
    plt.ylabel('exp. time to test population [days]')
    plt.xlabel('daily tests / 1M population.')
    if PRINTPLOTDATA:
        print(figpath+'psi{}_testsper1M_time.pdf'.format(prob_sick_plot_index))
        print(daily_tests_per_1m)
        for i, test_strategy in enumerate(test_strategies):
            print(test_strategy, e_time[i, :, prob_sick_plot_index])
    plt.ylim([0, 1250])
    if saveFig:
        plt.savefig(figpath+'psi{}_testsper1M_time_zoomed.pdf'.format(prob_sick_plot_index), bbox_inches='tight')


if __name__ == "__main__":
    recalculate = True
    if recalculate:
        # either do calculations
        filename = calculation()
    else:
        # or use precalculated data
        scale_factor_pop = 10
        scale_factor_test = 100
        filename = getName(scale_factor_pop, scale_factor_test)

#    saveFig = 1
#    prob_sick_plot_index = 4  # 4 -> 0.01
#    # out of [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
#    plotting(filename, prob_sick_plot_index, saveFig)
#    if saveFig == 0:
#        plt.show()
