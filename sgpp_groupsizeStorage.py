import numpy as np
import pickle
import logging
from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage
from sgpp_precalc_parallel import calculate_missing_values


def generateGroupSizeKey(test_strategy, prob_sick, success_rate_test, false_positive_rate):
    return tuple([test_strategy, prob_sick, success_rate_test, false_positive_rate])


def calculateOptimalGroupSize(test_strategy, probabilities_sick, success_rate_test, false_positive_rate):
    sample_size = 50000
    num_simultaneous_tests = 100
    number_of_instances = 10
    test_duration = 5
    group_sizes = list(range(1, 33))

    _, _, _, _, _, _, _, _,  _, _,  _, lb, ub, _ = getSetup()

    f = sgpp_simStorage(4, test_strategy,  lb, ub)

    evaluationPoints = []
    for j, prob_sick in probabilities_sick:
        for k, group_size in enumerate(group_sizes):
            evaluationPoint = [prob_sick, success_rate_test, false_positive_rate, group_size]
            evaluationPoints.append(evaluationPoint)
    calculate_missing_values(dim, evaluationPoints, sample_size, test_duration, num_simultaneous_tests,
                             number_of_instances, test_strategy)

    e_times = np.zeros((len(group_sizes), len(probabilities_sick)))
    optimal_group_sizes = np.zeros(len(probabilities_sick))
    for j, prob_sick in probabilities_sick:
        for k, group_size in enumerate(group_sizes):
            e_times[j, k] = f.eval(evaluationPoint, 'time')
        optimal_group_sizes[j] = group_sizes[np.argmin(e_times[j, :])]

    return optimal_group_sizes


class optimalGroupSizeStorage():
    def __init__(self):

        # load precalculated data
        self.precalcValuesFileName = "/home/rehmemk/git/covid19-pooling/precalc/optimal_groupsizes.pkl"
        try:
            with open(self.precalcValuesFileName, 'rb') as f:
                self.precalculatedValues = pickle.load(f)
        except (FileNotFoundError):
            print(
                f'could not find precalculated optimal group sizes at { self.precalcValuesFileName}\nCreating new data file.')
            self.precalculatedValues = {}
        self.numNew = 0

    def cleanUp(self):
        with open(self.precalcValuesFileName, "wb") as f:
            pickle.dump(self.precalculatedValues, f)
        if self.numNew > 0:
            print(f"\ncalculated {self.numNew} new optimal group sizes")
        if self.numNew > 0:
            print(
                f"saved them to {self.precalcValuesFileName}, which now contains {len(self.precalculatedValues)} optimal group sizes")

    # TODO
    # When done, getOptimalGroupSize simply calls this
    # def getOptimalGroupSizes(self, test_strategy, probabilities_sick, success_rate_test, false_positive_rate):
    #     key = generateGroupSizeKey(test_strategy, prob_sick, success_rate_test, false_positive_rate)
    #     if key not in self.precalculatedValues:
    #         print(f'Calculating group sizes for key={key}')
    #         self.precalculatedValues[key] = calculateOptimalGroupSize(
    #             test_strategy, prob_sick, success_rate_test, false_positive_rate)
    #         self.numNew += 1
    #         logging.info(f'so far {self.numNew} new optimal group sizes')
    #     return self.precalculatedValues[key]

    def getOptimalGroupSize(self, test_strategy, prob_sick, success_rate_test, false_positive_rate):
        key = generateGroupSizeKey(test_strategy, prob_sick, success_rate_test, false_positive_rate)
        if key not in self.precalculatedValues:
            print(f'Calculating group sizes for key={key}')
            self.precalculatedValues[key] = calculateOptimalGroupSize(
                test_strategy, [prob_sick], success_rate_test, false_positive_rate)
            self.numNew += 1
            logging.info(f'so far {self.numNew} new optimal group sizes')
        return self.precalculatedValues[key][0]
