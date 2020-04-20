#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Corona Testing Simulation."""

import random as ran
import math
import sys
import numpy as np
import warnings
import aux


class Corona_Simulation(object):
    """ Class for Corona Testing Simulation. """

    # === Creating the object ===

    def __init__(self, sample_size, prob_sick=0.1, success_rate_test=0.99, false_posivite_rate=0.1, tests_repetitions=1,
                 test_result_decision_strategy='max'):
        """ Create object for Corona testing simulation.
            Call:
                test = Corona_Simulation(sample_size[, reported_cases, factor_unreported_cases, population, prob_sick])
            Input:
                sample_size - Size of the dataset to be tested
                optional:
                reported_cases - number of reported cases. Default: 39.5
                factor_unreported_cases - factor to take unreported cases into account. Default: 7
                population - total population. Default: 82790
                prob_sick - probability for an individual in population to be sick. Default: None
                success_rate_test - probability of recognizing a positive case. Default: 0.95
                false_posivite_rate - probability of obtaining a false positive. Default: 0.1
                tests_repetitions - perform given number of multiple tests
                test_result_decision_strategy - when using multiple tests decide either for 'max' or 'majority'
        """

        self.sample_size = sample_size
        self.prob_sick = prob_sick

        # Define the success rate of our test (i.e., the chance to recognize a positive case),
        # and the chance to obtain a false positive
        self.success_rate_test = success_rate_test
        self.false_posivite_rate_test = false_posivite_rate

        self.tests_repetitions = tests_repetitions
        self.test_result_decision_strategy = test_result_decision_strategy

        # total time needed
        self.total_time = 0

        # initialize lists and counters
        self.rawdata = []
        self.sick_list = []
        self.number_sick_people = 0
        self.sick_list_indices = []

        self.number_of_rounds = 0
        self.number_of_tests = 0
        self.sick_individuals = []
        self.confirmed_sick_individuals = []
        self.false_positive_individuals = []
        self.success_rate = 0
        self.number_false_positives = 0
        self.false_posivite_rate = 0

    def generate_data(self):
        """ Function to generate data. """

        # Generate a sample of raw data: a list of sample_size instances, equally distributed between 0 and 1
        for i in range(self.sample_size):
            self.rawdata += [np.random.rand()]  # [ran.random()]

        # Decide, who is infected
        for i in range(self.sample_size):
            if self.rawdata[i] <= self.prob_sick:
                self.sick_list += [1]
                self.sick_list_indices.append(i)
                self.number_sick_people += 1
            else:
                self.sick_list += [0]
        if self.number_sick_people == 0:
            print(
                'There would have been zero infected (probably sample_size is quite small). For Debugging purposes one infection has been added')
            infected_individual_index = 0
            self.rawdata[infected_individual_index] = 0
            self.sick_list[infected_individual_index] = 1
            self.sick_list_indices.append(infected_individual_index)
            self.number_sick_people = 1
        # print('generated data with {} sick people among total {}'.format(self.number_sick_people, self.sample_size))
        # print('they are {}\n----\n'.format(self.sick_list_indices))

        self.data = (self.rawdata, self.sick_list,
                     self.number_sick_people, self.sick_list_indices)

    # auxiliary routine for binary_splitting_time_dependent
    # The 'data' is seen as a stack and to create groups entries are popped from this stack
    def get_next_group_from_data(self, group_size):
        if len(self.rawdata) == 0:
            return
        if len(self.rawdata) >= group_size:
            new_group = [self.rawdata.pop(0) for j in range(group_size)]
        elif len(self.rawdata) < group_size and len(self.rawdata) > 0:
            new_group = [self.rawdata.pop(0) for j in range(len(self.rawdata))]
        self.active_groups += [[list(range(self.continuousIndex,
                                           self.continuousIndex+len(new_group))), new_group]]
        self.continuousIndex += len(new_group)

    def update_sick_lists_and_success_rate(self):
        for index in self.sick_individuals:
            if self.sick_list[index] == 1:
                if index not in self.confirmed_sick_individuals:
                    self.confirmed_sick_individuals.append(index)
            elif self.sick_list[index] == 0:
                if index not in self.false_positive_individuals:
                    self.false_positive_individuals.append(index)

        self.number_false_positives = len(self.false_positive_individuals)
        self.false_posivite_rate = self.number_false_positives / self.sample_size
        if self.number_sick_people > 0:
            self.success_rate = len(self.confirmed_sick_individuals) / \
                self.number_sick_people
        else:
            warnings.warn('Zero sick people. Set success rate to one')
            self.success_rate = 1

    # one binary splitting step
    # get current testgroup and appends (potentially) newly created groups to active_groups
    def binary_splitting_step(self, testgroup):
        self.number_of_rounds += 1

        # instantiate test results
        result_test = [0, 0]

        # split in left and right branch
        for i in range(2):
            if len(testgroup[i][0]) != 0:
                result_test[i] += aux._make_test(
                    testgroup[i][1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                    self.tests_repetitions, self.test_result_decision_strategy)
                # Adjust the counter for the number of tests
                self.number_of_tests += self.tests_repetitions
                # Determine the outcome of the finding
                if result_test[i] == 1:
                    if len(testgroup[i][1]) == 1:
                        self.sick_individuals.append(testgroup[i][0][0])
                    else:
                        self.active_groups += [testgroup[i]]

    def _RBS_DIG(self, contaminated_set):
        """ Binary splitting algorithm.
            Input: contaminated set
            Output: - single contaminated item in input set
                    - list of items declared healthy
        """
        healthy_set = [[], []]
        while len(contaminated_set[0]) >> 1:
            test_group = aux._split_groups(contaminated_set)
            if aux._make_test(test_group[0][1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                              self.tests_repetitions, self.test_result_decision_strategy) == 1:
                contaminated_set = test_group[0]
            else:
                contaminated_set = test_group[1]
                for i in range(2):
                    for item in test_group[0][i]:
                        healthy_set[i] += [item]
            self.number_of_tests += self.tests_repetitions
        # single sick individual returned
        if contaminated_set == [[], []]:
            return None, healthy_set
        else:
            return [contaminated_set[0][0], contaminated_set[1][0]], healthy_set

    def RBS_time_step(self):
        for i in range(min(len(self.active_groups), self.num_simultaneous_tests)):
            testgroup = self.active_groups[0]
            self.active_groups = self.active_groups[1:]
            if len(testgroup[0]) == 1:
                if not self.indicator:
                    if aux._make_test(testgroup[1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                                      self.tests_repetitions, self.test_result_decision_strategy) == 1:
                        self.sick_individuals.append(testgroup[0][0])
                    self.number_of_tests += self.tests_repetitions
                    #self.number_groupwise_tests[int(np.floor(testgroup[0][0] / self.group_size))] += 1
                    return
                else:
                    self.sick_individuals.append(testgroup[0][0])
                    return
            else:
                if not self.indicator:
                    testresult = aux._make_test(
                        testgroup[1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                        self.tests_repetitions, self.test_result_decision_strategy)
                    self.number_of_tests += self.tests_repetitions
                    #self.number_groupwise_tests[int(np.floor(testgroup[0][0] / self.group_size))] += 1
                else:
                    testresult = 1

                if testresult == 1:
                    testgroup = aux._split_groups(testgroup)

                    # instantiate test results
                    result_test = [0, 0]

                    # split in left and right branch
                    for i in range(2):
                        result_test[i] += aux._make_test(
                            testgroup[i][1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                            self.tests_repetitions, self.test_result_decision_strategy)
                        self.number_of_tests += self.tests_repetitions
                        #self.number_groupwise_tests[int(np.floor(testgroup[0][0] / self.group_size))] += 1
                        # Determine the outcome of the finding
                    if result_test[0] == 0 and result_test[1] == 1:
                        sick_ind, healthy_ind = self._RBS_DIG(testgroup[1])
                        # remove healthy individuals found by DIG from testgroup
                        for i in range(2):
                            for item in healthy_ind[i]:
                                testgroup[1][i].remove(item)
                        if sick_ind is not None:
                            # add found infected individual
                            self.sick_individuals += [sick_ind[0]]
                            # remove found sick individual from testgroup
                            testgroup[1][0].remove(sick_ind[0])
                            testgroup[1][1].remove(sick_ind[1])
                            # update active groups
                            self.active_groups += [testgroup[1]]
                        else:
                            # update active groups
                            self.active_groups += [testgroup[1]]
                        self.indicator = False
                        return
                    elif result_test[0] == 1 and result_test[1] == 0:
                        sick_ind, healthy_ind = self._RBS_DIG(testgroup[0])
                        # remove healthy individuals found by DIG from testgroup
                        for i in range(2):
                            for item in healthy_ind[i]:
                                testgroup[0][i].remove(item)
                        if sick_ind is not None:
                            # add found sick individual
                            self.sick_individuals += [sick_ind[0]]
                            # remove found sick individual from testgroup
                            testgroup[0][0].remove(sick_ind[0])
                            testgroup[0][1].remove(sick_ind[1])
                            # update active groups
                            self.active_groups += [testgroup[0]]
                        else:
                            # update active groups
                            self.active_groups += [testgroup[0]]
                        self.indicator = False
                        return
                    elif result_test[0] == 1 and result_test[1] == 1:
                        self.active_groups += testgroup
                        self.indicator = True
                        return
                    else:
                        return

    def sobel_candG(self, q, m, n, k, G):
        """
        auxiliary function: candidates for minimization
        corresponds to eq (6)&(7) in (Sobel,Groll 1959)
        q - probability of healtyh (P('test gives negative result')) = 1-prob_sick
        m - size of defective set (a set which has been tested positive)
        m - (n-m) is the size of the binomial set
        k - goes through potential defective set sizes 1..m
        G - expected number of group tests remaining to be performed
        """
        if m != 0:
            # G-Situation
            pSuccess = (q**k - q**m) / (1 - q**m)
            return pSuccess * G[(m - k, n - k)] + (1 - pSuccess) * G[(k, n)]
        else:
            # H-Situation
            pSuccess = q**k
            return pSuccess * G[(0, n - k)] + (1 - pSuccess) * G[(k, n)]

    def sobel_computeGx(self, q, nMax):
        """
        Calculate G and x as dicts
        Entries of G and x are minima of the candidates from sobel_candG
        G(m,n) - expected number of group tests remaining to be performed for defective set of
                size m and binomial set of size n-m
                (For m=0 G is called H in the paper)
        x      - the size of the very next group test

        q  - probability of healtyh (P('test gives negative result')) = 1-prob_sick
        nMax - Values for (m,n) with 0<=m<=n<=nMax are precalculated
        """
        G = {}
        x = {}
        G[(0, 0)] = 0
        for n in range(1, nMax + 1):
            G[(1, n)] = G[(0, n - 1)]
            for m in range(2, n + 1):
                cand = [(k, self.sobel_candG(q, m, n, k, G)) for k in range(1, m)]
                opt = min(cand, key=lambda t: t[1])
                x[(m, n)] = opt[0]
                G[(m, n)] = 1 + opt[1]
            # H(n)
            cand = [(k, self.sobel_candG(q, 0, n, k, G)) for k in range(1, n + 1)]
            opt = min(cand, key=lambda t: t[1])
            x[(0, n)] = opt[0]
            G[(0, n)] = 1 + opt[1]
        return G, x

    def sobel_step(self, m, n):
        """
        For given (m,n) determine size k of the next test and new (m,n) depending on the test result
        'Success' (test negative)
        'Failure' (test positive)
        """
        k = self.x[(m, n)]
        if m != 0:
            # G-situation
            mSuccess = m - k
            nSuccess = n - k
            mFailure = k
            nFailure = n
        else:
            # H-situation
            mSuccess = 0
            nSuccess = n - k
            mFailure = k
            nFailure = n
        # for m==1 we do not need to test
        if mSuccess == 1:
            mSuccess = 0
            nSuccess -= 1
        if mFailure == 1:
            mFailure = 0
            nFailure -= 1
        return (mSuccess, nSuccess), (mFailure, nFailure)

    ####################################
    ######## Pooling Algorithms ########
    ####################################

    def sobel_main(self, num_simultaneous_tests, test_duration, maxGroupsize):
        '''
        The algorithm R1 from (Sobel, Groll 1959)
        Sobel, Milton, and Phyllis A. Groll.
        "Group testing to eliminate efficiently all defectives in a binomial sample."
        Bell System Technical Journal 38.5 (1959): 1179-1252.
        '''
        self.num_simultaneous_tests = num_simultaneous_tests
        self.test_duration = test_duration
        self.continuousIndex = 0
        # counter for the number of tests which are performed on one initial group of
        # maxGroupsize in total
        self.number_groupwise_tests = np.zeros(int(np.ceil(self.sample_size/maxGroupsize)))
        self.number_groupwise_tests_counter = -1

        self.G, self.x = self.sobel_computeGx(1-self.prob_sick, maxGroupsize)

        # initialize active groups
        self.active_groups = []
        self.confirmed_sick_individuals = []
        # this is for indexing the individuals
        while (len(self.rawdata) > 0):
            self.number_groupwise_tests_counter += 1
            # initial groups have maximal size
            # TODO: inefficient way of getting groups
            self.get_next_group_from_data(maxGroupsize)
            binomial_set = self.active_groups[0]
            self.active_groups = self.active_groups[1:]

            # size of current defective group
            sobel_m = 0
            # size of current other group
            sobel_n = maxGroupsize

            testgroup = [[], []]
            defective_set = [[], []]
            while(sobel_m != 0 or sobel_n != 0):
                k = self.x[(sobel_m, sobel_n)]
                (mSuccess, nSuccess), (mFailure, nFailure) = self.sobel_step(sobel_m, sobel_n)
                if sobel_m == 0:
                    # H-situation
                    # take k individuals from binomial set into testgroup
                    testgroup = [binomial_set[0][:k], binomial_set[1][:k]]
                    binomial_set = [binomial_set[0][k:], binomial_set[1][k:]]
                    testresult = aux._make_test(
                        testgroup[1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                        self.tests_repetitions, self.test_result_decision_strategy)
                    self.number_of_tests += self.tests_repetitions
                    self.number_groupwise_tests[self.number_groupwise_tests_counter] += 1
                    if testresult == 1:
                        # infected
                        if len(testgroup[0]) == 1:
                            # print("single infected index {} identified".format(testgroup[0][0]))
                            self.sick_individuals.append(testgroup[0][0])
                            testgroup = [[], []]
                        else:
                            defective_set = testgroup  # TODO make a copy here!?
                    elif testresult == 0:
                        # clean
                        testgroup = [[], []]
                        if len(defective_set[0]) == 1:
                            # print("identified index {} by conlusion".format(defective_set[0][0]))
                            self.sick_individuals.append(defective_set[0][0])
                            defective_set = [[], []]

                elif sobel_m > 0:
                    # G situation
                    # take k individuals from defective set into testgroup
                    testgroup = [defective_set[0][:k], defective_set[1][:k]]
                    defective_set = [defective_set[0][k:], defective_set[1][k:]]
                    testresult = aux._make_test(
                        testgroup[1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                        self.tests_repetitions, self.test_result_decision_strategy)
                    self.number_of_tests += self.tests_repetitions
                    self.number_groupwise_tests[self.number_groupwise_tests_counter] += 1
                    if testresult == 1:
                        # pinfected
                        if len(testgroup[0]) == 1:
                            # print("single infected index {} identified".format(testgroup[0][0]))
                            self.sick_individuals.append(testgroup[0][0])
                            testgroup = [[], []]
                        binomial_set = [binomial_set[0]+defective_set[0], binomial_set[1]+defective_set[1]]
                        defective_set = testgroup

                    elif testresult == 0:
                        # clean
                        testgroup = [[], []]
                        if len(defective_set[0]) == 1:
                            # print("identified index {} by conlusion".format(defective_set[0][0]))
                            self.sick_individuals.append(defective_set[0][0])
                            defective_set = [[], []]
                if testresult == 1:
                    sobel_m = mFailure
                    sobel_n = nFailure
                elif testresult == 0:
                    sobel_m = mSuccess
                    sobel_n = nSuccess

        # simplified time measure, because individual time tracking is a bit complicated
        self.total_time = self.number_of_tests*self.test_duration * self.tests_repetitions/self.num_simultaneous_tests
        self.update_sick_lists_and_success_rate()

    # parent function which calls the binary_splitting_step recursively
    # This is a time independent version which was not used in the paper

    def binary_splitting(self):
        # initiate all active test groups, which is initially the list of all people
        # with index and sick indicator
        self.active_groups = [[range(len(self.rawdata)), self.rawdata]]
        self.sick_individuals = []
        self.number_of_tests = 0
        self.number_of_rounds = 0
        self.confirmed_sick_individuals = []
        while (len(self.active_groups) != 0 and self.number_of_rounds < self.sample_size):
            testgroup = aux._split_groups(self.active_groups[0])
            self.active_groups = self.active_groups[1:]
            self.binary_splitting_step(testgroup)

    # one test per individual, no pooling
    # performed with respect to time and number of simultaneously processable tests
    # By default group size is 1
    # If group size is larger than one, then ALL individuals in a positively tested group are
    # immediately set positive
    def individual_testing_time_dependent(self, num_simultaneous_tests, test_duration, group_size=1):
        self.num_simultaneous_tests = num_simultaneous_tests
        self.test_duration = test_duration
        self.number_groupwise_tests = np.ones(int(np.ceil(self.sample_size/group_size)))

        # initialize active groups
        self.active_groups = []
        self.confirmed_sick_individuals = []
        # this is for indexing the individuals
        self.continuousIndex = 0

        for i in range(self.num_simultaneous_tests):
            self.get_next_group_from_data(group_size)

        while (len(self.active_groups) > 0):
            # Caution: binary_splitting_step adds new groups to active_groups every time it is
            # called. However, we can only process the ones that existed at the beginning of this
            # loop iteration
            self.number_of_rounds += 1
            for i in range(min(len(self.active_groups), self.num_simultaneous_tests)):
                testgroup = self.active_groups[0]
                self.active_groups = self.active_groups[1:]
                testresult = aux._make_test(
                    testgroup[1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                    self.tests_repetitions, self.test_result_decision_strategy)
                self.number_of_tests += self.tests_repetitions

                if testresult == 1:
                    for individual in testgroup[0]:
                        self.sick_individuals.append(individual)

            if len(self.active_groups) < self.num_simultaneous_tests:
                # groups have been fully processed. Add next group from data
                for i in range(self.num_simultaneous_tests-len(self.active_groups)):
                    self.get_next_group_from_data(group_size)

            self.total_time += self.test_duration * self.tests_repetitions
        self.update_sick_lists_and_success_rate()

    # Step 1: Test group. Step 2: If sample is positively tested, test all individuals
    def two_stage_testing_time_dependent(self, num_simultaneous_tests, test_duration, group_size):
        # number of tests which can be performed at once
        self.num_simultaneous_tests = num_simultaneous_tests
        # duration of one test [h]
        self.test_duration = test_duration

        self.number_groupwise_tests = np.zeros(int(np.ceil(self.sample_size/group_size)))

        # initialize active groups
        self.active_groups = []
        self.confirmed_sick_individuals = []
        # this is for indexing the individuals
        self.continuousIndex = 0

        for i in range(self.num_simultaneous_tests):
            self.get_next_group_from_data(group_size)

        while (len(self.active_groups) > 0):
            # Caution: new groups are added to active_groups every time a group is positively tested
            # However, we can only process the ones that existed at the beginning of this loop iteration
            for i in range(min(len(self.active_groups), self.num_simultaneous_tests)):
                testgroup = self.active_groups[0]
                self.active_groups = self.active_groups[1:]
                testresult = aux._make_test(
                    testgroup[1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                    self.tests_repetitions, self.test_result_decision_strategy)
                self.number_of_tests += self.tests_repetitions
                self.number_groupwise_tests[int(np.floor(testgroup[0][0] / group_size))] += 1

                if testresult == 1:
                    if len(testgroup[1]) == 1:
                        self.sick_individuals.append(testgroup[0][0])
                    else:
                        for j in range(len(testgroup[1])):
                            self.number_groupwise_tests[int(np.floor(testgroup[0][0] / group_size))] += 1
                            self.active_groups += [[[testgroup[0]
                                                     [j]], [testgroup[1][j]]]]

            if len(self.active_groups) < self.num_simultaneous_tests:
                # groups have been fully processed. Add next group from data
                for i in range(self.num_simultaneous_tests-len(self.active_groups)):
                    self.get_next_group_from_data(group_size)
            self.total_time += self.test_duration
        self.update_sick_lists_and_success_rate()

    # parent function which performs binary splitting with respect to time and number
    # of simultaneously processable tests
    def binary_splitting_time_dependent(self, num_simultaneous_tests, test_duration, group_size):
        # in contrast to 'binary_splitting' this does not start with one huge group of all individuals
        # but with as many groups as tests are available, each group a reasonable size
        # when there are less groups than simultaneously processable tests new groups are added to
        # active_groups

        # number of tests which can be performed at once
        self.num_simultaneous_tests = num_simultaneous_tests
        # duration of one test [h]
        self.test_duration = test_duration

        self.number_groupwise_tests = np.zeros(int(np.ceil(self.sample_size/group_size)))

        # initialize active groups
        self.active_groups = []
        self.confirmed_sick_individuals = []
        # this is for indexing the individuals
        self.continuousIndex = 0

        for i in range(self.num_simultaneous_tests):
            self.get_next_group_from_data(group_size)

        while (len(self.active_groups) > 0):
            # Caution: binary_splitting_step adds new groups to active_groups every time it is
            # called. However, we can only process the ones that existed at the beginning of this
            # loop iteration
            self.number_of_rounds += 1
            # print('---')
            for i in range(min(len(self.active_groups), self.num_simultaneous_tests)):
                testgroup = self.active_groups[0]
                self.active_groups = self.active_groups[1:]
                testresult = aux._make_test(
                    testgroup[1], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                    self.tests_repetitions, self.test_result_decision_strategy)
                self.number_of_tests += self.tests_repetitions
                self.number_groupwise_tests[int(np.floor(testgroup[0][0] / group_size))] += 1

                if testresult == 1:
                    # if only one individual in group
                    if len(testgroup[1]) == 1:
                        self.sick_individuals.append(testgroup[0][0])
                    else:
                        new_groups = aux._split_groups(testgroup)
                        self.active_groups.append(new_groups[0])
                        self.active_groups.append(new_groups[1])

            if len(self.active_groups) < self.num_simultaneous_tests:
                # groups have been fully processed. Add next group from data
                for i in range(self.num_simultaneous_tests-len(self.active_groups)):
                    self.get_next_group_from_data(group_size)

            self.total_time += self.test_duration * self.tests_repetitions
        self.update_sick_lists_and_success_rate()

    def RBS_time_dependent(self, num_simultaneous_tests, test_duration, group_size):
        '''
        Recursive binary splitting algorithm as in
        Cheng, Yongxi, Ding-Zhu Du, and Feifeng Zheng.
        "A new strongly competitive group testing algorithm with small sequentiality."
        Annals of Operations Research 229.1 (2015): 265-286.
        '''

        # number of tests which can be performed at once
        self.num_simultaneous_tests = num_simultaneous_tests
        # duration of one test [h]
        self.test_duration = test_duration
        self.group_size = group_size

        self.number_groupwise_tests = np.zeros(int(np.ceil(self.sample_size/group_size)))

        # initialize active groups
        self.active_groups = []
        self.confirmed_sick_individuals = []
        # this is for indexing the individuals
        self.continuousIndex = 0

        self.indicator = False

        for i in range(self.num_simultaneous_tests):
            self.get_next_group_from_data(group_size)

        while (len(self.active_groups) > 0):
            self.number_of_rounds += 1

            self.RBS_time_step()

            if len(self.active_groups) < self.num_simultaneous_tests:
                # groups have been fully processed. Add next group from data
                for i in range(self.num_simultaneous_tests-len(self.active_groups)):
                    self.get_next_group_from_data(group_size)

        # simplified time measure, because individual time tracking is a bit complicated for RBS
        self.total_time = self.number_of_tests*self.test_duration * self.tests_repetitions/self.num_simultaneous_tests
        self.update_sick_lists_and_success_rate()

    def purim_time_dependent(self, num_simultaneous_tests, test_duration, group_size):
        '''
        Purim matrix based testing algorithm as in
        Fargion, Benjamin Isac, et al. 
        "Purim: a rapid method with reduced cost for massive detection of CoVid-19." 
        arXiv preprint arXiv:2003.11975 (2020).
        '''
        self.num_simultaneous_tests = num_simultaneous_tests
        self.test_duration = test_duration

        # initialize active groups
        self.active_groups = []
        self.confirmed_sick_individuals = []
        # this is for indexing the individuals
        self.continuousIndex = 0

        self.number_groupwise_tests = np.zeros(int(np.ceil(self.sample_size/(group_size**2))))
        self.number_groupwise_tests_counter = -1

        for i in range(self.num_simultaneous_tests):
            self.get_next_group_from_data(group_size**2)

        while (len(self.active_groups) > 0):
            self.number_of_rounds += 1
            for i in range(min(len(self.active_groups), self.num_simultaneous_tests)):
                self.number_groupwise_tests_counter += 1
                testgroup = self.active_groups[0]
                self.active_groups = self.active_groups[1:]
                nearest_square = round(np.sqrt(len(testgroup[1])))**2
                diff = nearest_square - len(testgroup[1])
                if len(testgroup[1]) <= 4:
                    # if the number of people in a group is 2 or less, do individual testing
                    for j in range(len(testgroup[1])):
                        result = aux._make_test([testgroup[1][j]], self.success_rate_test, self.false_posivite_rate_test,
                                                self.prob_sick, self.tests_repetitions, self.test_result_decision_strategy)
                        self.number_of_tests += self.tests_repetitions
                        self.number_groupwise_tests[self.number_groupwise_tests_counter] += 1
                        if result == 1:
                            self.sick_individuals += [testgroup[0][j]]
                elif int(diff) < 0:
                    # write into nearest_square+1 x nearest_square+1 array
                    testarray = np.ones(int(nearest_square+2*np.sqrt(nearest_square)+1))
                    for l in range(len(testgroup[1])):
                        testarray[l] = testgroup[1][l]
                    testarray = testarray.reshape((int(np.sqrt(nearest_square)+1), int(np.sqrt(nearest_square)+1)))
                    testarray_index = np.ones(int(nearest_square+2*np.sqrt(nearest_square)+1))
                    for l in range(len(testgroup[0])):
                        testarray_index[l] = testgroup[0][l]
                    # testarray_index = np.append(np.asarray(testgroup[0]),np.ones(int(-diff)+1))
                    testarray_index = testarray_index.reshape(
                        (int(np.sqrt(nearest_square)+1), int(np.sqrt(nearest_square)+1)))

                    columns = []
                    rows = []
                    for k in range(int(np.sqrt(nearest_square)+1)):
                        result = aux._make_test(testarray[k, :], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                                                self.tests_repetitions, self.test_result_decision_strategy)
                        self.number_of_tests += self.tests_repetitions
                        self.number_groupwise_tests[self.number_groupwise_tests_counter] += 1
                        if result == 1:
                            columns += [k]
                    for j in range(int(np.sqrt(nearest_square)+1)):
                        result = aux._make_test(testarray[:, j], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                                                self.tests_repetitions, self.test_result_decision_strategy)
                        self.number_of_tests += self.tests_repetitions
                        self.number_groupwise_tests[self.number_groupwise_tests_counter] += 1
                        if result == 1:
                            rows += [j]
                    for k in columns:
                        for j in rows:
                            result = aux._make_test([testarray[k, j]], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                                                    self.tests_repetitions, self.test_result_decision_strategy)
                            self.number_of_tests += self.tests_repetitions
                            self.number_groupwise_tests[self.number_groupwise_tests_counter] += 1
                            if result == 1:
                                self.sick_individuals += [int(testarray_index[k, j])]
                else:
                    testarray = np.ones(int(nearest_square))
                    for l in range(len(testgroup[1])):
                        testarray[l] = testgroup[1][l]
                    testarray = testarray.reshape((int(np.sqrt(nearest_square)), int(np.sqrt(nearest_square))))
                    testarray_index = np.ones(int(nearest_square))
                    for l in range(len(testgroup[0])):
                        testarray_index[l] = testgroup[0][l]
                    # testarray_index = np.append(np.asarray(testgroup[0]),np.ones(int(diff)))
                    testarray_index = testarray_index.reshape(
                        (int(np.sqrt(nearest_square)), int(np.sqrt(nearest_square))))

                    columns = []
                    rows = []
                    for k in range(int(np.sqrt(nearest_square))):
                        result = aux._make_test(testarray[k, :], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                                                self.tests_repetitions, self.test_result_decision_strategy)
                        self.number_of_tests += self.tests_repetitions
                        self.number_groupwise_tests[self.number_groupwise_tests_counter] += 1
                        if result == 1:
                            columns += [k]
                    for j in range(int(np.sqrt(nearest_square))):
                        result = aux._make_test(testarray[:, j], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                                                self.tests_repetitions, self.test_result_decision_strategy)
                        self.number_of_tests += self.tests_repetitions
                        self.number_groupwise_tests[self.number_groupwise_tests_counter] += 1
                        if result == 1:
                            rows += [j]
                    for k in columns:
                        for j in rows:
                            result = aux._make_test([testarray[k, j]], self.success_rate_test, self.false_posivite_rate_test, self.prob_sick,
                                                    self.tests_repetitions, self.test_result_decision_strategy)
                            self.number_of_tests += self.tests_repetitions
                            self.number_groupwise_tests[self.number_groupwise_tests_counter] += 1
                            if result == 1:
                                self.sick_individuals += [int(testarray_index[k, j])]

            if len(self.active_groups) < self.num_simultaneous_tests:
                # groups have been fully processed. Add next group from data
                for i in range(self.num_simultaneous_tests-len(self.active_groups)):
                    self.get_next_group_from_data(group_size**2)

        # simplified time measure, because individual time tracking is a bit complicated for RBS
        self.total_time = self.number_of_tests*self.test_duration/self.num_simultaneous_tests
        self.update_sick_lists_and_success_rate()
