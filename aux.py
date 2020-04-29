#!/usr/bin/env ipython
# -*- coding: utf-8 -*-

import random as ran
import math
import numpy as np

"""Define auxiliary functions for Corona Testing Simulation."""


def _make_test(testlist, current_success_rate, false_posivite_rate, prob_sick,
               tests_repetitions=1, test_result_decision_strategy='max'):
    """ 
        Function for performing one test. 
        Input:
            testlist - list of probabilities (of being sick) of individuals
            current_success_rate - current probability of a test being successful
            false_positive rate - probability of a false positive
            prob_sick - probability that an individual is sick
            optional:
            tests_repetitions - perform given number of multiple tests 
            test_result_decision_strategy - when using multiple tests decide either for 'max' or 'majority'
    """

    outcomes = [0]*tests_repetitions
    for t in range(tests_repetitions):
        # Define a random parameter for the test
        random_parameter = ran.random()
        # Check, whether the list contains a sick person
        sick_person_exists = 0
        for individual_probability in testlist:
            if individual_probability <= prob_sick:
                sick_person_exists = 1

        # Perform the test
        if (sick_person_exists == 1 and random_parameter <= current_success_rate):
            outcomes[t] = 1
        # elif (sick_person_exists == 1 and random_parameter > current_success_rate):
        #     print("aux.py DEBUG. FALSE POSITIVE")
        elif (sick_person_exists == 0 and random_parameter <= false_posivite_rate):
            outcomes[t] = 1
        else:
            outcomes[t] = 0

    if test_result_decision_strategy == 'max':
        return np.max(outcomes)
    elif test_result_decision_strategy == 'majority':
        if outcomes.count(0) > outcomes.count(1):
            return 0
        else:
            return 1


def _split_groups(active_groups):
    """ Function to perform a binary tree search test on our sample. """
    size_chosen_instance = len(active_groups[1])
    middle = size_chosen_instance//2
    # split the first active group in two equal size groups and then remove the instance from the list of active groups
    test_group = [[active_groups[0][0:middle], active_groups[1][0:middle]]
                  ] + [[active_groups[0][middle:], active_groups[1][middle:]]]
    return test_group


def generate_data(sample_size, prob_sick):
    """ 
    Function to generate data of consecutively numbered individuals which are infected 
    with chance prob_sick
    """
    rawdata = []
    sick_list = []
    number_sick_people = 0
    sick_list_indices = []
    # Generate a sample of raw data: a list of sample_size instances, equally distributed between 0 and 1
    for i in range(sample_size):
        rawdata += [np.random.rand()]  # [ran.random()]

    # Decide, who is infected
    for i in range(sample_size):
        if rawdata[i] <= prob_sick:
            sick_list += [1]
            sick_list_indices.append(i)
            number_sick_people += 1
        else:
            sick_list += [0]
    if number_sick_people == 0:
        print(
            'There would have been zero infected (probably sample_size is quite small). For Debugging purposes one infection has been added')
        infected_individual_index = 0
        rawdata[infected_individual_index] = 0
        sick_list[infected_individual_index] = 1
        sick_list_indices.append(infected_individual_index)
        number_sick_people = 1
    # print('generated data with {} sick people among total {}'.format(number_sick_people, sample_size))
    # print('they are {}\n----\n'.format(sick_list_indices))
    return rawdata, sick_list, number_sick_people
