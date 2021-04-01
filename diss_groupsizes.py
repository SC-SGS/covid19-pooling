import multiprocessing
import sys
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from CoronaTestingSimulation import Corona_Simulation
from Statistics import Corona_Simulation_Statistics
import subprocess

sys.path.append('/home/rehmemk/git/diss/gfx/py/helper')  # nopep8
from figure import Figure  # nopep8

'''
Groupsizes
Determine optimal groupsizes for all methods depending on the infection rate
'''

# default plot font sizes
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def getName(success_rate_test=0.99):
    # name for the data dump and plots
    name = 'groupsizes'
    if success_rate_test != 0.99:
        name += '_{}'.format(success_rate_test)
    return name


def plotting(filename, saveFig=0):
    # load data
    datapath = 'data/{}.pkl'.format(filename)
    with open(datapath, 'rb') as fp:
        data = pickle.load(fp)
    figpath = '/home/rehmemk/git/diss/gfx/pre/gfx_8_covid/{}'.format(filename)

    # extract relevant parameters from data
    test_strategies = data['test_strategies']
    probabilities_sick = data['probabilities_sick']
    group_sizes = data['group_sizes']
    e_time = data['e_time']
    sd_time = data['sd_time']

    # plotting
    markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    # optimal group sizes
    print('optimal group sizes:')
    for i, test_strategy in enumerate(test_strategies):
        for k, probability_sick in enumerate(probabilities_sick):
            optsize = group_sizes[np.argmin(e_time[i, :, k])]
            print('{} {} {}'.format(test_strategy, probability_sick, optsize))

    ######## poolsize / expected time ########
    labels = ['individual testing (IT)', 'two stage testing (2LT)',
              'binary splitting (BS)', 'recursive binary splitting (RBS)',
              'purim', 'sobel']
    for k, probability_sick in enumerate(probabilities_sick):
        # plt.figure()
        F = Figure(mode='thesis')
        plt.gcf().set_size_inches(6, 5)
        plt.title('$\mathrm{{infection}}\ \mathrm{{rate:}} {}\%$'.format(
            int(probability_sick*100)), fontsize=BIGGER_SIZE)
        for i, test_strategy in enumerate(test_strategies):
            plt.plot(group_sizes, e_time[i, :, k],
                     label=labels[i], marker=markers[i], color=colors[i])
            # plt.fill_between(group_sizes, e_time[i, :, k]-sd_time[i, :, k],
            #                  e_time[i, :, k] + sd_time[i, :, k], color=colors[i], alpha=0.4)
        if k == 0:
            plt.ylabel('$\mathrm{expected}\ \mathrm{time}\ \mathrm{to}\ \mathrm{test}\ \mathrm{pop.}\ \mathrm{(days)}$')
        else:
            plt.ylabel('expected time to test pop. (days)', color='white')
        plt.xlabel('$\mathrm{pool}\ \mathrm{size}$')
        if k == 5:
            plt.legend(loc='upper right')
        plt.ylim([0, 130])
        if saveFig:
            plt.savefig(figpath+'_{}.pdf'.format(probability_sick), bbox_inches='tight')
        plt.close()

    # fig = plt.figure()
    # plt.ylabel('expected time to test pop. (days)')
    # plt.plot([0], [0], color='white')
    # if saveFig:
    #     plt.savefig(figpath+'ylabel.pdf', bbox_inches='tight')


if __name__ == "__main__":
    # use precalculated data
    filename = getName()

    saveFig = 1
    plotting(filename, saveFig)
    if saveFig == 0:
        plt.show()
