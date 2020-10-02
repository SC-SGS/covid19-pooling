import numpy as np
from CoronaTestingSimulation import Corona_Simulation
from aux import generate_data
import warnings


class Corona_Simulation_Statistics():
    """ Class for statistical analysis of Corona_Simulation.
        test_strategy:          which poooling algorithm to use

        the following arguments are not available for all strategies
        num_simultaneous_tests: number of tests which can be performed at once
        test_duration:          duration of one test in h
        group_size   :          how many samples can be pooled
    """

    def __init__(self, prob_sick,
                 success_rate_test, false_posivite_rate, test_strategy,
                 test_duration=6, group_size=8,
                 tests_repititions=1, test_result_decision_strategy='max', scale_factor_pop=1):

        # scenario settings which will be handed over to Corona_Simulation
        self.prob_sick = prob_sick
        self.success_rate_test = success_rate_test
        self.false_posivite_rate = false_posivite_rate
        self.tests_repititions = tests_repititions
        self.test_result_decision_strategy = test_result_decision_strategy
        self.group_size = group_size
        self.test_duration = test_duration
        self.test_strategy = test_strategy
        self.scale_factor_pop = scale_factor_pop

    def perform_test(self, sample_size, num_simultaneous_tests, rawdata, sick_list, number_sick_people):
        self.test_instance = Corona_Simulation(sample_size, rawdata, sick_list, number_sick_people, self.prob_sick,
                                               self.success_rate_test, self.false_posivite_rate,
                                               self.tests_repititions, self.test_result_decision_strategy)
        if self.test_strategy == 'binary-splitting':
            self.test_instance.binary_splitting_time_dependent(
                num_simultaneous_tests, self.test_duration, self.group_size)
        elif self.test_strategy == 'individual-testing':
            self.test_instance.individual_testing_time_dependent(
                num_simultaneous_tests, self.test_duration)
        elif self.test_strategy == 'two-stage-testing':
            self.test_instance.two_stage_testing_time_dependent(
                num_simultaneous_tests, self.test_duration, self.group_size)
        elif self.test_strategy == 'RBS':
            self.test_instance.RBS_time_dependent(
                num_simultaneous_tests, self.test_duration, self.group_size)
        elif self.test_strategy == 'purim':
            self.test_instance.purim_time_dependent(num_simultaneous_tests, self.test_duration, self.group_size)
        elif self.test_strategy == 'sobel':
            self.test_instance.sobel_main(num_simultaneous_tests, self.test_duration, self.group_size)
        else:
            warnings.warn(f'test strategy {self.test_strategy} unknown')

    def statistical_analysis(self, sample_size, num_simultaneous_tests, number_of_instances):
        '''
        Calculates means and standard deviations of qois for population of sample_size.
        The stochastical values are calculated by using number_of_instances many repetitions of the simulation
        '''
        # result containers
        self.number_of_tests = np.zeros(number_of_instances)
        self.test_times = np.zeros(number_of_instances)
        self.ratios_of_sick_found = np.zeros(number_of_instances)
        self.false_positive_rates = np.zeros(number_of_instances)
        self.number_sick_people = np.zeros(number_of_instances)
        self.num_confirmed_sick_individuals = np.zeros(number_of_instances)
        self.num_sent_to_quarantine = np.zeros(number_of_instances)
        # self.number_groupwise_tests = {}
        self.num_confirmed_per_test = np.zeros(number_of_instances)

        # Generate test data for the desired number of instances.
        for i in range(number_of_instances):
            rawdata, sick_list, number_sick_people = generate_data(sample_size, self.prob_sick)
            self.perform_test(sample_size, num_simultaneous_tests, rawdata, sick_list, number_sick_people)
            self.number_of_tests[i] = self.test_instance.number_of_tests
            self.test_times[i] = self.test_instance.total_time / 24.0
            self.ratios_of_sick_found[i] = self.test_instance.success_rate
            self.false_positive_rates[i] = self.test_instance.false_posivite_rate
            self.number_sick_people[i] = self.test_instance.number_sick_people
            self.num_confirmed_sick_individuals[i] = len(
                self.test_instance.confirmed_sick_individuals)
            self.num_sent_to_quarantine[i] = len(self.test_instance.sick_individuals)
            # self.number_groupwise_tests[i] = self.test_instance.number_groupwise_tests

            # derived metrics
            self.num_confirmed_per_test[i] = self.num_confirmed_sick_individuals[i] / self.number_of_tests[i]

        # Perform statistical analysis
        # means
        self.e_number_of_tests = np.mean(self.number_of_tests)
        self.e_time = np.mean(self.test_times)
        self.e_ratio_of_sick_found = np.mean(self.ratios_of_sick_found)
        self.e_false_positive_rate = np.mean(self.false_positive_rates)
        self.e_number_sick_people = np.mean(self.number_sick_people)
        self.e_num_confirmed_sick_individuals = np.mean(self.num_confirmed_sick_individuals)
        self.e_num_sent_to_quarantine = np.mean(self.num_sent_to_quarantine)
        self.e_num_confirmed_per_test = np.mean(self.num_confirmed_per_test)

        # standard deviations
        self.sd_number_of_tests = np.std(self.number_of_tests)
        self.sd_time = np.std(self.test_times)
        self.sd_ratio_of_sick_found = np.std(self.ratios_of_sick_found)
        self.sd_false_positive_rate = np.std(self.false_positive_rates)
        self.sd_number_sick_people = np.std(self.number_sick_people)
        self.sd_num_confirmed_sick_individuals = np.std(self.num_confirmed_sick_individuals)
        self.sd_num_sent_to_quarantine = np.std(self.num_sent_to_quarantine)
        self.sd_num_confirmed_per_test = np.std(self.num_confirmed_per_test)

        # self.worst_case_number_groupwise_tests = 0.0
        # self.average_number_groupwise_tests = np.zeros(len(self.number_groupwise_tests[0]))
        # for key in self.number_groupwise_tests:
        #     self.average_number_groupwise_tests += self.number_groupwise_tests[key]
        #     self.worst_case_number_groupwise_tests = max(self.worst_case_number_groupwise_tests,
        #                                                  max(self.number_groupwise_tests[key]))
        # self.average_number_groupwise_tests /= len(self.number_groupwise_tests)
        # self.e_number_groupwise_tests = np.mean(self.average_number_groupwise_tests)
