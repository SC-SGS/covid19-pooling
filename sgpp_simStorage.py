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
        return self.objFunc.eval(x, recalculate=False, evalType='multiMC')

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


def simulate(sample_size, prob_sick, success_rate_test, false_positive_rate,
             test_duration, group_size, num_simultaneous_tests, number_of_instances,
             scale_factor_pop, test_strategy, evalType):
    # relict parameters. We won't touch these in the near future
    tests_repetitions = 1
    test_result_decision_strategy = 'max'

    stat_test = Corona_Simulation_Statistics(prob_sick, success_rate_test,
                                             false_positive_rate, test_strategy,
                                             test_duration, group_size,
                                             tests_repetitions, test_result_decision_strategy)
    start = time.time()
    if evalType == 'multiMC':
        logging.info(f'multi MC')
        stat_test.multilevel_MonteCarlo([10000, 25000, 100000],
                                        [num_simultaneous_tests]*3,
                                        [16, 8, 4])
    elif evalType == 'MC':
        logging.info(f'MC')
        stat_test.statistical_analysis(sample_size, num_simultaneous_tests, number_of_instances)
    runtime = time.time()-start
    logging.info(f'runtime: {runtime}s\n')

    e_time = stat_test.e_time*scale_factor_pop
    e_num_tests = stat_test.e_number_of_tests*scale_factor_pop
    e_num_confirmed_sick_individuals = stat_test.e_num_confirmed_sick_individuals*scale_factor_pop
    return e_time, e_num_tests, e_num_confirmed_sick_individuals


class sgpp_simStorage():
    def __init__(self, dim, test_strategy, qoi, lb, ub):
        self.dim = dim
        self.test_strategy = test_strategy
        self.qoi = qoi

        self.default_parameters = [0.1, 0.99, 0.01, 8, ]

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

    def eval(self, x, recalculate=False, evalType='multiMC', number_of_instances=1):
        # lists are not allowed as keys, but tuples are
        prob_sick = self.default_parameters[0]
        success_rate_test = self.default_parameters[1]
        false_positive_rate = self.default_parameters[2]
        group_size = self.default_parameters[3]
        # test_duration = self.default_parameters[4]
        # tests_per_day = self.default_parameters[5]
        # population = self.default_parameters[6]
        if self.dim > 0:
            prob_sick = x[0]
        if self.dim > 1:
            success_rate_test = x[1]
        if self.dim > 2:
            false_positive_rate = x[2]
        if self.dim > 3:
            group_size = int(x[3])
        # if self.dim > 4:
        #     test_duration = x[4]
        # if self.dim > 5:
        #     tests_per_day = x[5]
        # if self.dim > 6:
        #     population = x[6]

        # The reference population always consists of 100,000 individuals and 1000 tests
        # testing times can simply be scaled accordingly
        sample_size = 100000
        num_daily_tests = 1000
        test_duration = 5
        num_simultaneous_tests = int(num_daily_tests*test_duration/24.0)

        key = tuple([prob_sick, success_rate_test, false_positive_rate,
                     group_size, evalType, number_of_instances])
        if key in self.precalculatedValues and recalculate == False:
            [e_time, e_num_tests, e_num_confirmed_sick_individuals] = self.precalculatedValues[key]
        else:
            print(f'Calculating key={key}')
            scale_factor_pop = 1
            e_time, e_num_tests, e_num_confirmed_sick_individuals = simulate(sample_size, prob_sick, success_rate_test, false_positive_rate,
                                                                             test_duration, group_size, num_simultaneous_tests, number_of_instances,
                                                                             scale_factor_pop, self.test_strategy, evalType)
            self.precalculatedValues[key] = [e_time, e_num_tests, e_num_confirmed_sick_individuals]
            self.numNew += 1
            logging.info(f'so far {self.numNew} new evaluations')

        if self.qoi == 'time':
            return e_time
        elif self.qoi == 'numtests':
            return e_num_tests
        elif self.qoi == 'numconfirmed':
            return e_num_confirmed_sick_individuals
        elif self.qoi == 'ppt':
            return e_num_confirmed_sick_individuals/e_num_tests
        else:
            warnings.warn('unknown qoi')
