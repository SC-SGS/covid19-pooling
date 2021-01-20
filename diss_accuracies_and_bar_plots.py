import multiprocessing
import sys
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from CoronaTestingSimulation import Corona_Simulation
from Statistics import Corona_Simulation_Statistics
import subprocess

'''
Accuracies and bar plot
Measure the sensitivities and false positives for the individual tests
'''

# default plot font sizes
SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def getName(success_rate_test=0.99):
    # name for the data dump and plots
    name = 'accuracies'
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
    success_rate_test = data['success_rate_test']
    test_strategies = data['test_strategies']
    e_ratio_of_sick_found = data['e_ratio_of_sick_found']
    e_false_positive_rate = data['e_false_positive_rate']
    e_num_confirmed_sick_individuals = data['e_num_confirmed_sick_individuals']
    e_num_sent_to_quarantine = data['e_num_sent_to_quarantine']
    sd_ratio_of_sick_found = data['sd_ratio_of_sick_found']
    sd_false_positive_rate = data['sd_false_positive_rate']
    # e_number_groupwise_tests = data['e_number_groupwise_tests']

    # plotting
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    ######## poolsize / expected time ########
    plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 2, 1)
    for i, (test_strategy, group_size) in enumerate(test_strategies):
        plt.errorbar([i], e_ratio_of_sick_found[i], sd_ratio_of_sick_found[i], label=test_strategy,
                     capsize=10, linestyle='None', linewidth=10, color=colors[i])
        plt.errorbar([i], e_ratio_of_sick_found[i], 0, capsize=5, linestyle='None', linewidth=10, color='k')
    plt.ylabel('sensitivity')
    plt.xticks(range(len(test_strategies)), ['ind.', '2l-p.', 'bin.', 'r.bin.', 'pu.', 's-r1'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlim([-0.5, len(test_strategies)])

    ax = plt.subplot(1, 2, 2)
    for i, (test_strategy, group_size) in enumerate(test_strategies):
        plt.errorbar([i], e_false_positive_rate[i], sd_false_positive_rate[i], label=test_strategy,
                     capsize=10, linestyle='None', linewidth=10, color=colors[i])
        plt.errorbar([i], e_false_positive_rate[i], 0, capsize=5, linestyle='None', linewidth=10, color='k')
    plt.ylabel('expected false positive rate')
    plt.xticks(range(len(test_strategies)), ['ind.', '2l-p.', 'bin.', 'r.bin.', 'pu.', 's-r1'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlim([-0.5, len(test_strategies)])

    plt.tight_layout(pad=1.0)
    if saveFig:
        plt.savefig(figpath+'.pdf', bbox_inches='tight')

    ######## test per 1M / bar plot - infected/identified/sent to quarantine ########
    labels = ['ind.', '2l-p.', 'bin.', 'r.bin.', 'pu.', 's-r1']

    if success_rate_test == 0.99:
        labelheight = 150
    elif success_rate_test == 0.75:
        labelheight = 1050
    if success_rate_test == 0.99:
        numberheight = 70
    elif success_rate_test == 0.75:
        numberheight = 20

    X = np.arange(len(labels))  # the label locations
    plt.figure(figsize=(6, 5))
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    for i, _ in enumerate(test_strategies):
        ax.bar(X[i] - width/2, e_num_confirmed_sick_individuals[i], width,
               label='correctly identified infected', color=colors[i], edgecolor='black')
        ax.annotate('{}'.format(int(e_num_confirmed_sick_individuals[i])),
                    xy=(X[i] - 6.5*width / 8, e_num_confirmed_sick_individuals[i]+numberheight),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", ha='center', va='bottom')

        ax.bar(X[i] + width/2, e_num_sent_to_quarantine[i], width,
               label='sent to quarantine', color=colors[i], alpha=0.6, edgecolor='black')
        ax.annotate('{}'.format(int(e_num_sent_to_quarantine[i])),
                    xy=(X[i] + 6.5*width / 8, e_num_sent_to_quarantine[i]+numberheight),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", ha='center', va='bottom')

        ax.text(X[i]-width/2-width/4, labelheight, 'correctly identified', rotation=90)
        ax.text(X[i]+width/2-width/4, labelheight, 'total quarantined', rotation=90)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('number of individuals')
    ax.set_xticks(X)
    ax.set_xticklabels(labels[:len(test_strategies)])
    plt.xlim([-1.75, len(test_strategies)])
    plt.plot([-1.75, len(test_strategies)], [1000, 1000], '--k')
    plt.text(-1.6, 1030, 'infected')
    plt.text(-1.6, 900, 'individuals')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if saveFig:
        plt.savefig(figpath+'_bar.pdf', bbox_inches='tight')


if __name__ == "__main__":
    # use precalculated data
    filename = getName()

    saveFig = 1
    plotting(filename, saveFig)
    if saveFig == 0:
        plt.show()
