import numpy as np
import pickle
import warnings
import pysgpp
import logging
import sys
import time
from Statistics import Corona_Simulation_Statistics


class objFuncSGpp(pysgpp.ScalarFunction):
    def __init__(self, objFunc):
        self.dim = objFunc.getDim()
        self.objFunc = objFunc
        super(objFuncSGpp, self).__init__(self.dim)

    def eval(self, x):
        return self.objFunc.eval(x, recalculate=False)

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


def generateKey(prob_sick, success_rate_test, false_positive_rate, group_size, number_of_instances):
    return tuple([prob_sick, success_rate_test, false_positive_rate, group_size, number_of_instances])


def simulate(sample_size, prob_sick, success_rate_test, false_positive_rate, test_duration,
             group_size, num_simultaneous_tests, number_of_instances, test_strategy):
    # relict parameters. We won't touch these in the near future
    tests_repetitions = 1
    test_result_decision_strategy = 'max'

    stat_test = Corona_Simulation_Statistics(prob_sick, success_rate_test,
                                             false_positive_rate, test_strategy,
                                             test_duration, group_size,
                                             tests_repetitions, test_result_decision_strategy)
    start = time.time()
    stat_test.statistical_analysis(sample_size, num_simultaneous_tests, number_of_instances)
    runtime = time.time()-start
    logging.info(f'runtime: {runtime}s\n')

    e_time = stat_test.e_time
    e_num_tests = stat_test.e_number_of_tests
    e_num_confirmed_sick_individuals = stat_test.e_num_confirmed_sick_individuals
    e_num_confirmed_per_test = stat_test.e_num_confirmed_per_test
    e_num_sent_to_quarantine = stat_test.e_num_sent_to_quarantine

    sd_time = stat_test.sd_time
    sd_num_tests = stat_test.sd_number_of_tests
    sd_num_confirmed_sick_individuals = stat_test.sd_num_confirmed_sick_individuals
    sd_num_confirmed_per_test = stat_test.sd_num_confirmed_per_test
    sd_num_sent_to_quarantine = stat_test.sd_num_sent_to_quarantine

    return [e_time, e_num_tests, e_num_confirmed_sick_individuals, e_num_confirmed_per_test,
            e_num_sent_to_quarantine, sd_time, sd_num_tests, sd_num_confirmed_sick_individuals,
            sd_num_confirmed_per_test, sd_num_sent_to_quarantine]


class sgpp_simStorage():
    def __init__(self, dim, test_strategy, qoi, lb, ub):
        self.dim = dim
        self.test_strategy = test_strategy
        self.qoi = qoi

        # The reference population always consists of 100,000 individuals and 1000 tests
        # testing times can simply be scaled accordingly
        self.reference_sample_size = 100000
        self.reference_num_daily_tests = 1000
        self.reference_test_duration = 5
        self.reference_num_simultaneous_tests = int(self.reference_num_daily_tests *
                                                    self.reference_test_duration/24.0)

        self.default_parameters = [0.1, 0.99, 0.01, 8, ]

        self.lowerBounds = pysgpp.DataVector(lb[:dim])
        self.upperBounds = pysgpp.DataVector(ub[:dim])

        # load precalculated data
        savePath = "/home/rehmemk/git/covid19-pooling/precalc/"
        self.precalcValuesFileName = savePath + f"precalc_values_{self.test_strategy}.pkl"
        try:
            with open(self.precalcValuesFileName, 'rb') as f:
                self.precalculatedValues = pickle.load(f)
        except (FileNotFoundError):
            print(f'could not find precalculated data at { self.precalcValuesFileName}\nCreating new data file.')
            self.precalculatedValues = {}
        self.numNew = 0

    def getDim(self):
        return self.dim

    def getDomain(self):
        return self.lowerBounds, self.upperBounds

    def cleanUp(self):
        with open(self.precalcValuesFileName, "wb") as f:
            pickle.dump(self.precalculatedValues, f)
        if self.numNew > 0:
            print(f"\ncalculated {self.numNew} new evaluations")
        if self.numNew > 0:
            print(
                f"saved them to {self.precalcValuesFileName}, which now contains {len(self.precalculatedValues)} entries")

    def eval(self, x, recalculate=False,  number_of_instances=5):
        # lists are not allowed as keys, but tuples are
        prob_sick = self.default_parameters[0]
        success_rate_test = self.default_parameters[1]
        false_positive_rate = self.default_parameters[2]
        group_size = self.default_parameters[3]
        if self.dim > 0:
            prob_sick = x[0]
        if self.dim > 1:
            success_rate_test = x[1]
        if self.dim > 2:
            false_positive_rate = x[2]
        if self.dim > 3:
            group_size = int(x[3])

        key = generateKey(prob_sick, success_rate_test, false_positive_rate, group_size, number_of_instances)
        if key not in self.precalculatedValues or recalculate == True:
            print(f'Calculating key={key}')
            self.precalculatedValues[key] = simulate(self.reference_sample_size, prob_sick, success_rate_test,
                                                     false_positive_rate, self.reference_test_duration, group_size,
                                                     self.reference_num_simultaneous_tests, number_of_instances,
                                                     self.test_strategy)
            self.numNew += 1
            logging.info(f'so far {self.numNew} new evaluations')

        [e_time, e_num_tests, e_num_confirmed_sick_individuals, e_num_confirmed_per_test,
            e_num_sent_to_quarantine, sd_time, sd_num_tests, sd_num_confirmed_sick_individuals,
            sd_num_confirmed_per_test, sd_num_sent_to_quarantine] = self.precalculatedValues[key]

        if self.qoi == 'time':
            return e_time
        elif self.qoi == 'numtests':
            return e_num_tests
        elif self.qoi == 'numconfirmed':
            return e_num_confirmed_sick_individuals
        elif self.qoi == 'ppt':
            return e_num_confirmed_per_test
        elif self.qoi == 'quarantined':
            return e_num_sent_to_quarantine
        elif self.qoi == 'sd-time':
            return sd_time
        elif self.qoi == 'sd-numtests':
            return sd_num_tests
        elif self.qoi == 'sd-numconfirmed':
            return sd_num_confirmed_sick_individuals
        elif self.qoi == 'sd-ppt':
            return sd_num_confirmed_per_test
        elif self.qoi == 'sd-quarantined':
            return sd_num_sent_to_quarantine
        else:
            warnings.warn('unknown qoi')
