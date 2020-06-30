import multiprocessing
from Statistics import Corona_Simulation_Statistics
from setup import getSetup
from sgpp_simStorage import simulate
import numpy as np
import pickle
import warnings
import os


def getPath(sample_size, number_of_instances, numMCPoints, number_outer_repetitions):
    name = f'stochastic_noise_raw_data_{int(sample_size/1000)}k_{number_of_instances}rep_{numMCPoints}points_{number_outer_repetitions}times.pkl'
    savePath = os.path.join('/home/rehmemk/git/covid19-pooling/precalc/noise', name)
    return savePath


def worker(return_dict, i, j, sample_size, prob_sick, success_rate_test, false_positive_rate,
           test_duration, group_size, num_simultaneous_tests, number_of_instances,
           test_strategy):
    '''
    worker function for multiprocessing
    '''

    key = tuple([test_strategy, i, j])
    return_dict[key] = simulate(sample_size, prob_sick, success_rate_test, false_positive_rate,
                                test_duration, group_size, num_simultaneous_tests, number_of_instances,
                                test_strategy)


def stochastic_noise_evaluations(points, number_outer_repetitions,
                                 sample_size, number_of_instances, test_duration):

    pointwise_results = np.zeros((len(points), 3))

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    test_strategies = [
        'individual-testing',
        'two-stage-testing',
        'binary-splitting',
        'RBS',
        'purim',
        'sobel'
    ]
    for test_strategy in test_strategies:
        for i in range(number_outer_repetitions):
            for j, point in enumerate(points):
                prob_sick = point[0]
                success_rate_test = point[1]
                false_positive_rate = point[2]
                group_size = int(point[3])

                proc = multiprocessing.Process(target=worker, args=(return_dict, i, j, sample_size,
                                                                    prob_sick, success_rate_test,
                                                                    false_positive_rate, test_duration,
                                                                    group_size, num_simultaneous_tests,
                                                                    number_of_instances, test_strategy))
                jobs.append(proc)
                proc.start()

    for proc in jobs:
        proc.join()
    return return_dict


def stochastic_noise(test_strategy, qoi, sample_size, number_of_instances, numMCPoints,
                     number_outer_repetitions):
    savePath = getPath(sample_size, number_of_instances, numMCPoints, number_outer_repetitions)
    with open(savePath, "rb") as fp:
        results_dict = pickle.load(fp)

    results = np.zeros((number_outer_repetitions, numMCPoints))
    for i in range(number_outer_repetitions):
        for j in range(numMCPoints):
            key = tuple([test_strategy, i, j])
            [e_time, e_num_tests, e_num_confirmed_sick_individuals, e_num_confirmed_per_test,
             e_num_sent_to_quarantine, sd_time, sd_num_tests, sd_num_confirmed_sick_individuals,
             sd_num_confirmed_per_test, sd_num_sent_to_quarantine, e_number_groupwise_tests,
             worst_case_number_groupwise_tests, e_number_sick_people, sd_number_sick_people] = results_dict[key]
            if qoi == 'ppt':
                results[i, j] = e_num_confirmed_per_test
            elif qoi == 'time':
                results[i, j] = e_time
            elif qoi == 'num_confirmed_sick_individuals':
                results[i, j] = e_num_confirmed_sick_individuals
            elif qoi == 'num_sent_to_quarantine':
                results[i, j] = e_num_sent_to_quarantine
            else:
                print(f'noise function for qoi {qoi} does not yet exist')

    pointwise_results = np.zeros((numMCPoints, 3))
    for j in range(numMCPoints):
        pointwise_results[j, 0] = np.max(results[:, j])-np.min(results[:, j])
        pointwise_results[j, 1] = np.mean(results[:, j])
        pointwise_results[j, 2] = np.std(results[:, j])

    worst_diff = np.max(pointwise_results[:, 0])
    average_diff = np.mean(pointwise_results[:, 0])
    expected_std = np.mean(pointwise_results[:, 2])
    # print(f"{test_strategy}: worst diff {worst_diff}    expected std {expected_std}")

    return worst_diff


if __name__ == "__main__":

    numMCPoints = 100
    number_outer_repetitions = 10

    np.random.seed(44)
    gridType, dim, degree, _, _, _, sample_size, num_daily_tests, \
        test_duration, num_simultaneous_tests, \
        number_of_instances, lb, ub, boundaryLevel = getSetup()

    unitpoints = np.random.rand(numMCPoints, dim)
    points = [lb + (ub-lb)*point for point in unitpoints]

    worst_diffs = {}

    results_dict = stochastic_noise_evaluations(points, number_outer_repetitions,
                                                sample_size, number_of_instances, test_duration)
    saveDict = {}
    for key in results_dict:
        saveDict[key] = results_dict[key]

    savePath = getPath(sample_size, number_of_instances, numMCPoints, number_outer_repetitions)
    with open(savePath, "wb+") as fp:
        pickle.dump(saveDict, fp)
    print(f"Saved data as {savePath}")
