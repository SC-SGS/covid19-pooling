import pysgpp
import numpy as np
import time
from Statistics import Corona_Simulation_Statistics
import matplotlib.pyplot as plt
from sgpp_create_response_surface import getSetup
from sgpp_simStorage import sgpp_simStorage


gridType, dim, degree, test_strategy, qoi, name = getSetup()
f = sgpp_simStorage(dim, test_strategy,  qoi)
lb, ub = f.getDomain()

# eval parameters:
# probabilities_sick = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
# probabilities_sick = [0.001,  0.1505, 0.3]
probabilities_sick = np.linspace(0.001, 0.04, 11)

success_rate_test = 0.99
false_positive_rate_test = 0.01
test_duration = 5
group_size = 32
tests_per_day = 0.8  # 100000
population = 100000000


MC1_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))
MC2_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))
multiMC_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))
refMC_e_num_confirmed_per_test = np.zeros(len(probabilities_sick))

start = time.time()
for i, prob_sick in enumerate(probabilities_sick):
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate_test, group_size, test_duration, tests_per_day]
    MC1_e_num_confirmed_per_test[i] = f.eval(evaluationPoint, recalculate=True, evalType='MC',
                                             number_of_instances=1)
MC1time = time.time()-start

start = time.time()
for i, prob_sick in enumerate(probabilities_sick):
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate_test, group_size, test_duration, tests_per_day]
    MC2_e_num_confirmed_per_test[i] = f.eval(evaluationPoint, recalculate=True, evalType='MC',
                                             number_of_instances=2)
MC2time = time.time()-start

start = time.time()
for i, prob_sick in enumerate(probabilities_sick):
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate_test, group_size, test_duration, tests_per_day]
    multiMC_e_num_confirmed_per_test[i] = f.eval(evaluationPoint, recalculate=True, evalType='multiMC')
multiMCtime = time.time()-start

start = time.time()
for i, prob_sick in enumerate(probabilities_sick):
    evaluationPoint = [prob_sick, success_rate_test, false_positive_rate_test, group_size, test_duration, tests_per_day]
    refMC_e_num_confirmed_per_test[i] = f.eval(evaluationPoint, recalculate=True, evalType='MC',
                                               number_of_instances=10)
refMCtime = time.time()-start
# f.cleanUp()
print(f'MC1 took {MC1time}s')
print(f'MC2 took {MC2time}s')
print(f'multiMC took {multiMCtime}s')
print(f'refMC took {refMCtime}s')

np.savetxt('/home/rehmemk/git/covid19-pooling/data/temp/mc1.txt', MC1_e_num_confirmed_per_test)
np.savetxt('/home/rehmemk/git/covid19-pooling/data/temp/mc2.txt', MC2_e_num_confirmed_per_test)
np.savetxt('/home/rehmemk/git/covid19-pooling/data/temp/multimc.txt', multiMC_e_num_confirmed_per_test)
np.savetxt('/home/rehmemk/git/covid19-pooling/data/temp/refmc.txt', refMC_e_num_confirmed_per_test)

print(f'mc1 diff: {np.linalg.norm(refMC_e_num_confirmed_per_test-MC1_e_num_confirmed_per_test)}')
print(f'mc2 diff: {np.linalg.norm(refMC_e_num_confirmed_per_test-MC2_e_num_confirmed_per_test)}')
print(f'multi mc diff: {np.linalg.norm(refMC_e_num_confirmed_per_test-multiMC_e_num_confirmed_per_test)}')
# print(ref_e_num_confirmed_per_test)

plt.plot(probabilities_sick, MC1_e_num_confirmed_per_test, '-o', label='MC1')
plt.plot(probabilities_sick, MC2_e_num_confirmed_per_test, '-<', label='MC2')
plt.plot(probabilities_sick, multiMC_e_num_confirmed_per_test, '-x', label='multiMC')
plt.plot(probabilities_sick, refMC_e_num_confirmed_per_test, '-^', label='refMC')
plt.xlabel('infection rate')
plt.ylabel('exp. number of cases per test')
plt.legend()
plt.show()
