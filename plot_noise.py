import ipdb
import pickle
import os
from sgpp_calculate_stochastic_noise import stochastic_noise
from setup import getSetup
import numpy as np
import matplotlib.pyplot as plt
from sgpp_create_response_surface import auxiliary

import sys
sys.path.append('/home/rehmemk/git/diss/gfx/py/helper')  # nopep8
from figure import Figure  # nopep8

# default plot font sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

gridType, dim, degree, _, _, name, _, _, \
    test_duration, num_simultaneous_tests,    _, lb, ub,\
    boundaryLevel = getSetup()

qoi = 'ppt'
# qoi = 'time'
# qoi = 'num_confirmed_sick_individuals'
# qoi = 'num_sent_to_quarantine'


test_strategies = [
    'individual-testing',
    'two-stage-testing',
    'binary-splitting',
    'RBS',
    'purim',
    'sobel'
]

legend_labels = ['Individual testing',
                 'two level pooling',
                 'Binary splitting',
                 'RBS',
                 'Purim',
                 'Sobel-R-1'
                 ]

markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
pop_rep = [[1000, 1], [1000, 10], [10000, 1], [10000, 5],  [10000, 10], [100000, 1], [100000, 10]]

# CALCULATE NOISE
X = range(len(pop_rep))
noises = np.zeros((len(test_strategies), len(pop_rep)))
for i, test_strategy in enumerate(test_strategies):
    for j, [pop, rep] in enumerate(pop_rep):
        numNoisePoints = 100
        number_outer_repetitions = 10
        noises[i, j] = stochastic_noise(test_strategy, qoi, pop, rep,
                                        numNoisePoints, number_outer_repetitions)
print(noises)

# CALCULATE BEST AVAILABLE SURROGATE
#  currently adaptive 800pts
gridType = 'nakBsplineBoundary'
reSurf_pop_rep = [[1000, 10],   [10000, 10], [100000, 10]]
adaptive_nrmses = np.zeros((len(test_strategies), len(reSurf_pop_rep), 1))
for j, [pop, rep] in enumerate(reSurf_pop_rep):
    if rep == 10:
        sample_size = pop
        num_daily_tests = int(pop/100)
        number_of_instances = rep
        print(f'calcualting error for {pop}/{rep}')
        _, adaptive_nrmses[:, j, :], _\
            = auxiliary('adaptive', test_strategies, [qoi], sample_size, num_daily_tests,
                        test_duration, dim, number_of_instances, gridType, degree, boundaryLevel, lb, ub,
                        'dummy', 800, 1, 10, verbose=False, calcError=True, numMCPoints=1000,
                        saveReSurf=False)

# PLOT
#plt.figure(figsize=[9, 18])
for i, test_strategy in enumerate(test_strategies):
    #plt.figure(figsize=[6, 4])
    F = Figure(mode='thesis')
    plt.gcf().set_size_inches(6, 4)
    #plt.subplot(3, 2, i+1)
    # hard coded X because I couldn't find a nice way of quickly getting what i want
    plt.plot([1, 4, 6], adaptive_nrmses[i, :, 0], '-', color=colors[i],
             marker=markers[i], label=legend_labels[i], linewidth=1.5)
    plt.plot(X, noises[i, :], '--', color=colors[i],  label='approx. noise', linewidth=1.5)

    # sqrt(N), Monte Carlo convergence h^(-1/2)
    if qoi == 'ppt':
        plt.plot([1, 4], [5e-2, 1e-2], 'grey', 'o', label=r'$h^{-1/2}$', linewidth=1.5)
        plt.ylim([1e-3, 1e-1])
    if qoi == 'time' and test_strategy != 'individual-testing':
        plt.plot([1, 4], [5e-0, 1e-0], 'grey', 'o', linewidth=1.5)
        plt.ylim([1e-1, 50])
        plt.ylabel('approx. noise in days')

    labels = [f'{int(pop/1000)}k/{rep}' for [pop, rep] in pop_rep]
    plt.xticks(range(len(pop_rep)), labels)
    plt.gca().set_yscale('log')
    # plt.title(f'{test_strategy}')

    # There is a weirde bug where the legend gets a strange extra entry. I hard code remove this here
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[:-1]
    labels = labels[:-1]
    plt.legend(handles, labels, loc='lower left')
    # if test_strategy == 'purim':
    #     plt.xlabel('population/repetitions')

    plt.tight_layout()
    # plt.savefig(f'plots/stochastic_noise_and_convergence_{test_strategy}.pdf')
    plt.savefig(f'/home/rehmemk/git/diss/gfx/pre/gfx_8_covid/stochastic_noise_and_convergence_{test_strategy}.pdf')
    plt.close()

# save data for diss plot
# data = {'test_strategies': test_strategies,
#         'adaptive_nrmses': adaptive_nrmses,
#         'noises': noises,
#         'pop_rep': pop_rep
#         }
# dissPath = '/home/rehmemk/git/diss/gfx/py/data/'
# fileName = f'stochastic_noise_and_convergence_data_{qoi}.pkl'
# savePath = os.path.join(dissPath, fileName)
# with open(savePath, 'wb+') as fp:
#     pickle.dump(data, fp)
# print(f'saved relevant plotting data to {savePath}')

# Legend
plt.figure()
plt.plot([0], [0], c=colors[0], marker=markers[0], label='$\mathrm{Individual}\ \mathrm{testing}$')
plt.plot([0], [0], c=colors[1], marker=markers[1], label='$2$-$\mathrm{level}\ \mathrm{pooling}$')
plt.plot([0], [0], c=colors[2], marker=markers[2], label='$\mathrm{Binary}\ \mathrm{splitting}$')
plt.plot([0], [0], c=colors[3], marker=markers[3], label='$\mathrm{Recursive}\ \mathrm{binary}\ \mathrm{splitting}$')
plt.plot([0], [0], c=colors[4], marker=markers[4], label='$\mathrm{Purim}')
plt.plot([0], [0], c=colors[5], marker=markers[5], label='$\mathrm{Sobel}$ $\mathrm{R}$-$1$')
plt.plot([0], [0], 'k--', label='$\mathrm{approx.}\ \mathrm{noise}$')
plt.plot([0], [0], c='grey', label=r'$h^{-1/2}$')
axe = plt.gca()
handles, labels = axe.get_legend_handles_labels()

originalHandles = handles[:]
originalLabels = labels[:]
plt.figure()
axe = plt.gca()
axe.axis('off')
labels[3] = originalLabels[1]
labels[1] = originalLabels[2]
labels[2] = originalLabels[4]
labels[4] = originalLabels[3]
handles[3] = originalHandles[1]
handles[1] = originalHandles[2]
handles[2] = originalHandles[4]
handles[4] = originalHandles[3]
axe.legend(handles, labels, loc='center', fontsize=SMALL_SIZE, ncol=3)
axe.xaxis.set_visible(False)
axe.yaxis.set_visible(False)
for v in axe.spines.values():
    v.set_visible(False)
# cut off whitespace
plt.subplots_adjust(left=0.0, right=1.0, top=0.6, bottom=0.4)
#plt.savefig('plots/stochastic_noise_and_convergence_legend.pdf', dpi=300,bbox_inches='tight', pad_inches=0.0, format='pdf')
plt.savefig('/home/rehmemk/git/diss/gfx/pre/gfx_8_covid/stochastic_noise_and_convergence_legend.pdf',
            dpi=300, bbox_inches='tight', pad_inches=0.0, format='pdf')

# plt.show()
