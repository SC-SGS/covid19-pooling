import multiprocessing
import sys
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from CoronaTestingSimulation import Corona_Simulation
from Statistics import Corona_Simulation_Statistics
import subprocess

'''
Groupsizes
Determine optimal groupsizes for all methods depending on the infection rate
'''

# default plot font sizes
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def getName(success_rate_test=0.99):
    # name for the data dump and plots
    name = 'groupsizes'
    if success_rate_test != 0.99:
        name += '_{}'.format(success_rate_test)
    return name


def worker(return_dict, sample_size, prob_sick, success_rate_test, false_posivite_rate, test_strategy,
           num_simultaneous_tests, test_duration, group_size,
           tests_repetitions, test_result_decision_strategy, number_of_instances):
    '''
    worker function for multiprocessing
    performs the same test tests_repetitions many times and returns expected valkues and standard deviations
    '''

    stat_test = Corona_Simulation_Statistics(sample_size, prob_sick, success_rate_test,
                                             false_posivite_rate, test_strategy,
                                             num_simultaneous_tests, test_duration, group_size,
                                             tests_repetitions, test_result_decision_strategy)
    stat_test.statistical_analysis(number_of_instances)
    print('Calculated {} for {} prob sick {}'.format(test_strategy, group_size, prob_sick))
    print('scaled to {} population and {} simulataneous tests\n'.format(sample_size, num_simultaneous_tests))
    # gather results
    worker_dict = {}
    worker_dict['e_num_tests'] = stat_test.e_number_of_tests
    worker_dict['e_time'] = stat_test.e_time
    worker_dict['e_num_confirmed_sick_individuals'] = stat_test.e_num_confirmed_sick_individuals
    worker_dict['e_false_positive_rate'] = stat_test.e_false_positive_rate
    worker_dict['e_ratio_of_sick_found'] = stat_test.e_ratio_of_sick_found
    worker_dict['sd_num_tests'] = stat_test.sd_number_of_tests
    worker_dict['sd_time'] = stat_test.sd_time
    worker_dict['sd_false_positive_rate'] = stat_test.sd_false_positive_rate
    worker_dict['sd_ratio_of_sick_found'] = stat_test.sd_ratio_of_sick_found

    return_dict['{}_{}_{}'.format(test_strategy, group_size, prob_sick)] = worker_dict


def calculation():
    start = time.time()
    randomseed = 19
    np.random.seed(randomseed)

    probabilities_sick = [0.01, 0.05, 0.1, 0.15, 0.2]
    group_sizes = list(range(1, 33))
    success_rate_test = 0.99
    false_posivite_rate = 0.01
    tests_repetitions = 1
    test_result_decision_strategy = 'max'
    test_strategies = [
        'individual testing',
        'two stage testing',
        'binary splitting',
        'RBS',
        'purim',
        'sobel'
    ]

    sample_size = 50000
    num_simultaneous_tests = 100
    number_of_instances = 10
    test_duration = 5

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    e_num_tests = np.zeros((len(test_strategies), len(group_sizes), len(probabilities_sick)))
    e_time = np.zeros((len(test_strategies), len(group_sizes), len(probabilities_sick)))
    e_false_positive_rate = np.zeros((len(test_strategies), len(group_sizes), len(probabilities_sick)))
    e_num_confirmed_sick_individuals = np.zeros((len(test_strategies), len(group_sizes), len(probabilities_sick)))
    e_ratio_of_sick_found = np.zeros((len(test_strategies), len(group_sizes), len(probabilities_sick)))

    sd_num_tests = np.zeros((len(test_strategies), len(group_sizes), len(probabilities_sick)))
    sd_time = np.zeros((len(test_strategies), len(group_sizes), len(probabilities_sick)))
    sd_false_positive_rate = np.zeros((len(test_strategies), len(group_sizes), len(probabilities_sick)))
    sd_ratio_of_sick_found = np.zeros((len(test_strategies), len(group_sizes), len(probabilities_sick)))

    jobs = []

    for i, test_strategy in enumerate(test_strategies):
        for j, group_size in enumerate(group_sizes):
            for k, prob_sick in enumerate(probabilities_sick):
                p = multiprocessing.Process(target=worker, args=(return_dict, sample_size, prob_sick,
                                                                 success_rate_test, false_posivite_rate, test_strategy, num_simultaneous_tests,
                                                                 test_duration, group_size, tests_repetitions, test_result_decision_strategy,
                                                                 number_of_instances))
                jobs.append(p)
                p.start()
    for proc in jobs:
        proc.join()

    # gather results
    for i, test_strategy in enumerate(test_strategies):
        for j, group_size in enumerate(group_sizes):
            for k, prob_sick in enumerate(probabilities_sick):
                worker_dict = return_dict['{}_{}_{}'.format(test_strategy, group_size, prob_sick)]

                e_num_tests[i, j, k] = worker_dict['e_num_tests']
                e_time[i, j, k] = worker_dict['e_time']
                e_num_confirmed_sick_individuals[i, j, k] = worker_dict['e_num_confirmed_sick_individuals']
                e_false_positive_rate[i, j, k] = worker_dict['e_false_positive_rate']
                e_ratio_of_sick_found[i, j, k] = worker_dict['e_ratio_of_sick_found']
                sd_num_tests[i, j, k] = worker_dict['sd_num_tests']
                sd_time[i, j, k] = worker_dict['sd_time']
                sd_false_positive_rate[i, j, k] = worker_dict['sd_false_positive_rate']
                sd_ratio_of_sick_found[i, j, k] = worker_dict['sd_ratio_of_sick_found']

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
        'number_of_instances': number_of_instances,
        'test_duration': test_duration,
        'group_sizes': group_sizes,
        'e_num_tests ': e_num_tests,
        'e_time': e_time,
        'e_false_positive_rate': e_false_positive_rate,
        'e_num_confirmed_sick_individuals': e_num_confirmed_sick_individuals,
        'e_ratio_of_sick_found': e_ratio_of_sick_found,
        'sd_num_tests': sd_num_tests,
        'sd_time': sd_time,
        'sd_false_positive_rate': sd_false_positive_rate,
        'sd_ratio_of_sick_found': sd_ratio_of_sick_found,
        'sample_size': sample_size,
        'runtime': runtime,
        'num_simultaneous_tests': num_simultaneous_tests,
    }
    filename = getName(success_rate_test)
    path = 'data/{}.pkl'.format(filename)
    with open(path, 'wb+') as fp:
        pickle.dump(data, fp)
    print('saved data as {}'.format(path))
    return filename


def plotting(filename, saveFig=0):
    # load data
    datapath = 'data/{}.pkl'.format(filename)
    with open(datapath, 'rb') as fp:
        data = pickle.load(fp)
    figpath = 'plots/{}'.format(filename)

    # extract relevant parameters from data
    test_strategies = data['test_strategies']
    probabilities_sick = data['probabilities_sick']
    group_sizes = data['group_sizes']
    e_time = data['e_time']
    sd_time = data['sd_time']

    # plotting
    markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    # optimal group sizes
    print('optimal group sizes:')
    for i, test_strategy in enumerate(test_strategies):
        for k, probability_sick in enumerate(probabilities_sick):
            optsize = group_sizes[np.argmin(e_time[i, :, k])]
            print('{} {} {}'.format(test_strategy, probability_sick, optsize))

    ######## poolsize / expected time ########
    labels = ['individual testing (IT)', 'two stage testing (2LT)',
              'binary splitting (BS)', 'recursive binary splitting (RBS)',
              'purim', 'sobel']
    for k, probability_sick in enumerate(probabilities_sick):
        plt.figure()
        plt.title('infection rate: {}%'.format(int(probability_sick*100)), fontsize=BIGGER_SIZE)
        for i, test_strategy in enumerate(test_strategies):
            plt.plot(group_sizes, e_time[i, :, k],
                     label=labels[i], marker=markers[i], color=colors[i])
            plt.fill_between(group_sizes, e_time[i, :, k]-sd_time[i, :, k],
                             e_time[i, :, k] + sd_time[i, :, k], color=colors[i], alpha=0.4)

        plt.xlabel('pool size')
        if k == 5:
            plt.legend(loc='upper right')
        plt.ylim([0, 130])
        if saveFig:
            plt.savefig(figpath+'_{}.pdf'.format(probability_sick), bbox_inches='tight')

    fig = plt.figure()
    plt.ylabel('expected time to test pop. [days]')
    plt.plot([0], [0], color='white')
    if saveFig:
        plt.savefig(figpath+'ylabel.pdf', bbox_inches='tight')


if __name__ == "__main__":
    # either do claculations
    filename = calculation()

    # or use precalculated data
    # filename = getName()

    saveFig = 1
    plotting(filename, saveFig)
    if saveFig == 0:
        plt.show()
