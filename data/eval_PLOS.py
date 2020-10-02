#!/usr/bin/python3

import sys, pickle

if not len(sys.argv) > 1:
    print("call {} pickle-filename".format(sys.argv[0]))
    sys.exit()

fd = open(sys.argv[1], 'rb')
obj = pickle.load(fd)
#print(obj)

pop = obj['countries']['DE']['population']
print("population", pop)
print("group_size", obj['group_size'])


for (i, strategy) in enumerate(obj['test_strategies']):
    print()
    print("############ {:20} ############".format(strategy))

    print("\nWhole population:")
    e_num_tests = obj['e_num_tests '][i][0][0]
    print("{:1.1f} tests".format(e_num_tests))
    e_num_confirmed = obj['e_num_confirmed_sick_individuals'][i][0][0]
    print("{:1.1f} confirmed (tp)".format(e_num_confirmed))
    e_num_sent_to_quarantine = obj['e_num_sent_to_quarantine'][i][0][0]
    print("{:1.1f} quarantined".format(e_num_sent_to_quarantine))
    e_fp = e_num_sent_to_quarantine - e_num_confirmed
    print("{:1.1f} wrong quarantined (fp)".format(e_fp))
    e_ratio_of_sick_found = obj['e_ratio_of_sick_found'][i][0][0]
    e_missed = e_num_confirmed * (1-e_ratio_of_sick_found)/e_ratio_of_sick_found
    print("{:1.1f} missed (fn)".format(e_missed))
    e_released = pop - e_num_confirmed - e_fp - e_missed
    print("{:1.1f} released (tn)".format(e_released))
   


    print("\nFor 100,000 tests:")
    scaling = 100000/e_num_tests
    print("scaling factor: {:1.2f}".format(scaling))
    print("{:1.1f} tested".format(pop*scaling))
    print("{:1.1f} confirmed (tp)".format(e_num_confirmed*scaling))
    print("{:1.1f} wrong quarantined (fp)".format(e_fp*scaling))
    print("{:1.1f} missed (fn)".format(e_missed*scaling))
    print("{:1.1f} released (tn)".format(e_released*scaling))
