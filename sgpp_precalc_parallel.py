from sgpp_simStorage import simulate
import multiprocessing


def worker(return_dict, sample_size, prob_sick, success_rate_test, false_positive_rate,
           test_duration, group_size, num_simultaneous_tests, number_of_instances,
           scale_factor_pop, test_strategy, evalType):
    '''
    worker function for multiprocessing
    '''

    e_time, e_num_tests, e_num_confirmed_sick_individuals = simulate(sample_size, prob_sick, success_rate_test, false_positive_rate,
                                                                     test_duration, group_size, num_simultaneous_tests, number_of_instances,
                                                                     scale_factor_pop, test_strategy, evalType)
    key = tuple([prob_sick, success_rate_test, false_positive_rate,
                 group_size, evalType, number_of_instances])
    return_dict[key] = [e_time, e_num_tests, e_num_confirmed_sick_individuals]
    print(f'Calculated key={key}')


def precalc_parallel(points, sample_size, test_duration, num_simultaneous_tests,
                     number_of_instances, scale_factor_pop, test_strategy, evalType):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i, point in enumerate(points):
        prob_sick = point[0]
        success_rate_test = point[1]
        false_positive_rate = point[2]
        group_size = int(point[3])
        p = multiprocessing.Process(target=worker, args=(return_dict, sample_size, prob_sick, success_rate_test, false_positive_rate,
                                                         test_duration, group_size, num_simultaneous_tests, number_of_instances,
                                                         scale_factor_pop, test_strategy, evalType))
        jobs.append(p)
        p.start()
    print(f'set up list with {len(jobs)} jobs')
    for proc in jobs:
        proc.join()

    return return_dict
