import multiprocessing
import pickle
import logging
from sgpp_simStorage import simulate, generateKey


def worker(return_dict, sample_size, prob_sick, success_rate_test, false_positive_rate,
           test_duration, group_size, num_simultaneous_tests, number_of_instances,
           test_strategy):
    '''
    worker function for multiprocessing
    '''

    key = generateKey(prob_sick, success_rate_test, false_positive_rate, group_size, number_of_instances)
    return_dict[key] = simulate(sample_size, prob_sick, success_rate_test, false_positive_rate,
                                test_duration, group_size, num_simultaneous_tests, number_of_instances,
                                test_strategy)
    print(f'Calculated key={key}')


def precalc_parallel(points, sample_size, test_duration, num_simultaneous_tests,
                     number_of_instances, test_strategy):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for point in points:
        prob_sick = point[0]
        success_rate_test = point[1]
        false_positive_rate = point[2]
        group_size = int(point[3])
        proc = multiprocessing.Process(target=worker, args=(return_dict, sample_size, prob_sick, success_rate_test, false_positive_rate,
                                                            test_duration, group_size, num_simultaneous_tests, number_of_instances,
                                                            test_strategy))
        jobs.append(proc)
        proc.start()
    print(f'set up list with {len(jobs)} jobs')

    for proc in jobs:
        proc.join()
    return return_dict


def calculate_missing_values(evaluationPoints, sample_size, test_duration, num_simultaneous_tests,
                             number_of_instances, test_strategy):
    # load precalculated data
    savePath = "/home/rehmemk/git/covid19-pooling/precalc/"
    precalcValuesFileName = savePath + f"precalc_values_{test_strategy}.pkl"
    try:
        with open(precalcValuesFileName, 'rb') as fp:
            precalculatedValues = pickle.load(fp)
    except (FileNotFoundError):
        print('could not find precalculated data at {}\nCreating new data file.'.format(
            precalcValuesFileName))
        precalculatedValues = {}

    todoPoints = []
    for point in evaluationPoints:
        prob_sick = point[0]
        success_rate_test = point[1]
        false_positive_rate = point[2]
        group_size = int(point[3])
        key = generateKey(prob_sick, success_rate_test, false_positive_rate, group_size, number_of_instances)
        if key not in precalculatedValues:
            todoPoints.append(point)

    print(f"\ncalculating {len(todoPoints)} new evaluations")
    multiprocessing_dict = precalc_parallel(todoPoints, sample_size, test_duration, num_simultaneous_tests,
                                            number_of_instances, test_strategy)
    for key in multiprocessing_dict:
        print(f'DID {key=}')
        precalculatedValues[key] = multiprocessing_dict[key]
    with open(precalcValuesFileName, "wb") as fp:
        pickle.dump(precalculatedValues, fp)

    num_new_points = len(todoPoints)
    return num_new_points
