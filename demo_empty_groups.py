from aux import generate_data
from Statistics import Corona_Simulation_Statistics
from CoronaTestingSimulation import Corona_Simulation

# Definiere nötige Parameter
sample_size = 1000
num_simultaneous_tests = 2
test_duration = 5
tests_repetitions = 1
test_result_decision_strategy = 'max'
scale_factor_pop = 1
number_of_instances = 1

# Corona Statistics Instanz für RBS mit Parametern von denen ich weiß dass sie in
# leeren Gruppen resultieren
test_strategy = 'RBS'
prob_sick = 0.22525
success_rate_test = 0.75
false_posivite_rate = 0.2
group_size = 16
stat_test = Corona_Simulation_Statistics(prob_sick, success_rate_test,
                                         false_posivite_rate, test_strategy,
                                         test_duration, group_size,
                                         tests_repetitions, test_result_decision_strategy,
                                         scale_factor_pop)
stat_test.statistical_analysis(sample_size, num_simultaneous_tests, number_of_instances)

# Um zu verifizieren, dass alles durchläuft, hole ich noch ein Ergebnis
e_num_tests = stat_test.e_number_of_tests*scale_factor_pop
print(f'dummy result: e_num_tests={e_num_tests}')
