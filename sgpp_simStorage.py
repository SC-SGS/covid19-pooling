import numpy as np
import pickle
from Statistics import Corona_Simulation_Statistics
import warnings
import pysgpp
import logging
import sys
import time


class objFuncSGpp(pysgpp.ScalarFunction):

    def __init__(self, objFunc):
        self.dim = objFunc.getDim()
        self.objFunc = objFunc
        super(objFuncSGpp, self).__init__(self.dim)

    def eval(self, x):
        return self.objFunc.eval(x)

    def getDim(self):
        return self.dim

    def getLowerBounds(self):
        lb, _ = self.objFunc.getDomain()
        return lb

    def getUpperBounds(self):
        _, ub = self.objFunc.getDomain()
        return ub

    def cleanUp(self):
        try:
            self.objFunc.cleanUp()
        except:
            warnings.warn('This objective function does not have a cleanUp routine')

    def getPrecalcData(self):
        try:
            return self.objFunc.getPrecalcData()
        except:
            warnings.warn('This objective function does not have precalculated data')


class sgpp_simStorage():
    def __init__(self, dim, test_strategy, qoi):
        self.dim = dim
        self.test_strategy = test_strategy
        self.qoi = qoi
        self.number_of_instances = 3  # 10

        # set ranges for all 7 parameters
        # tests_per_day and population are combined into one parameter, so effectively there are only 6 dimensions
        # Note for myself: Never go to 7D when creating the response surface!
        # the population parameter is always scaled, so it would increase runtimes without any gain.
        prob_sick_range = [0.001, 0.3]
        success_rate_test_range = [0.3, 0.99]
        false_positive_rate_test_range = [0.01, 0.2]
        group_size_range = [1, 32]
        test_duration_range = [0.1, 10]
        tests_per_day_range = [100, 1000000]
        population_range = [10000, 400000000]
        # for dimensionality < 7 the remaining parameters get default values
        self.default_parameters = [0.1, 0.99, 0.01, 8, 5, 1000, 100000]

        lb = [prob_sick_range[0], success_rate_test_range[0], false_positive_rate_test_range[0],
              group_size_range[0], test_duration_range[0], tests_per_day_range[0], population_range[0]]
        ub = [prob_sick_range[1], success_rate_test_range[1], false_positive_rate_test_range[1],
              group_size_range[1], test_duration_range[1], tests_per_day_range[1], population_range[1]]
        self.lowerBounds = pysgpp.DataVector(lb[:dim])
        self.upperBounds = pysgpp.DataVector(ub[:dim])

        # load precalculated data
        savePath = "/home/rehmemk/git/covid19-pooling/precalc/"
        self.precalcValuesFileName = savePath + \
            f"precalc_values_{self.test_strategy}.pkl"
        try:
            with open(self.precalcValuesFileName, 'rb') as f:
                self.precalculatedValues = pickle.load(f)
            print(f'loaded precalculated evaluations from {self.precalcValuesFileName}')
        except (FileNotFoundError):
            print('could not find precalculated data at {}\nCreating new data file.'.format(
                self.precalcValuesFileName))
            self.precalculatedValues = {}
        self.numNew = 0

    def getDim(self):
        return self.dim

    def getDomain(self):
        return self.lowerBounds, self.upperBounds

    def cleanUp(self):
        with open(self.precalcValuesFileName, "wb") as f:
            pickle.dump(self.precalculatedValues, f)
        print(f"\ncalculated {self.numNew} new evaluations")
        if self.numNew > 0:
            print(
                f"saved them to {self.precalcValuesFileName}, which now contains {len(self.precalculatedValues)} entries")

    def eval(self, x):
        # lists are not allowed as keys, but tuples are
        prob_sick = self.default_parameters[0]
        success_rate_test = self.default_parameters[1]
        false_positive_rate = self.default_parameters[2]
        group_size = self.default_parameters[3]
        test_duration = self.default_parameters[4]
        tests_per_day = self.default_parameters[5]
        population = self.default_parameters[6]
        if self.dim > 0:
            prob_sick = x[0]
        if self.dim > 1:
            success_rate_test = x[1]
        if self.dim > 2:
            false_positive_rate = x[2]
        if self.dim > 3:
            group_size = int(np.ceil(x[3]))
        if self.dim > 4:
            test_duration = x[4]
        if self.dim > 5:
            tests_per_day = x[5]
        if self.dim > 6:
            population = x[6]

        # The reference population always consists of 100,000 individuals
        # Given tests per day and population are combined into one variable
        # num_simultaneous_tests
        sample_size = 100000
        tests_per_capita = tests_per_day/population*sample_size
        num_simultaneous_tests = int(tests_per_capita/test_duration)
        scale_factor_pop = population/sample_size

        key = tuple([prob_sick, success_rate_test, false_positive_rate,
                     group_size, test_duration, num_simultaneous_tests,
                     self.number_of_instances])
        if key in self.precalculatedValues:
            [e_time, e_num_tests, e_num_confirmed_sick_individuals] = self.precalculatedValues[key]
        else:
            # relict parameters. We won't touch these in the near future
            tests_repetitions = 1
            test_result_decision_strategy = 'max'

            print(f'Calculating for {sample_size=}   {num_simultaneous_tests=}')
            print(f'{key=}\n')
            stat_test = Corona_Simulation_Statistics(sample_size, prob_sick, success_rate_test,
                                                     false_positive_rate, self.test_strategy,
                                                     num_simultaneous_tests, test_duration, group_size,
                                                     tests_repetitions, test_result_decision_strategy)
            start = time.time()
            stat_test.statistical_analysis(self.number_of_instances)
            runtime = time.time()-start
            logging.info(f'runtime: {runtime}s\n')

            e_time = stat_test.e_time*scale_factor_pop
            e_num_tests = stat_test.e_number_of_tests*scale_factor_pop
            e_num_confirmed_sick_individuals = stat_test.e_num_confirmed_sick_individuals*scale_factor_pop
            self.precalculatedValues[key] = [e_time, e_num_tests, e_num_confirmed_sick_individuals]
            self.numNew += 1
        if self.qoi == 'time':
            return 1  # e_time
        elif self.qoi == 'numtests':
            return e_num_tests
        elif self.qoi == 'numconfirmed':
            return e_num_confirmed_sick_individuals
        elif self.qoi == 'ppt':
            return e_num_confirmed_sick_individuals/e_num_tests
        else:
            warnings.warn('unknown qoi')
