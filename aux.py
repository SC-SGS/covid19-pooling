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

# === Generate Model parameters ===
# def _estimate_sample_size(prob_sick, confidence_sick_exists):
#     estimation = math.log(confidence_sick_exists, 1-prob_sick)
#     return estimation


# def _model_success_rate(sample_size, success_rate_test):
#     current_success_rate = success_rate_test * \
#         (1/2)**(math.log(sample_size, 2))
#     return current_success_rate


# def _find_number_of_permutations(current_success_rate, success_rate_test):
#     number_of_permutations = math.ceil(
#         math.log(success_rate_test, 1-current_success_rate))
#     return number_of_permutations


# def _generate_permutations(individuals, number_of_permutations):
#     list_of_permutations_of_individuals = []
#     for i in range(number_of_permutations):
#         while len(list_of_permutations_of_individuals) <= i:
#             candidate = list(np.random.permutation(individuals))
#             if candidate not in list_of_permutations_of_individuals:
#                 list_of_permutations_of_individuals.append(candidate)
#     return list_of_permutations_of_individuals
