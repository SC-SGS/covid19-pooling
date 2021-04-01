import matplotlib
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

# whether to print plotdata
PRINTPLOTDATA = True

sys.path.append('/home/rehmemk/git/diss/gfx/py/helper')  # nopep8
from figure import Figure  # nopep8

# default plot font sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
legendfontsize = 14

# name for the data dump and plots


def getName(scale_factor_pop, scale_factor_test, success_rate_test=0.99):
    name = 'scenario1_scalepop{}_scaletest{}'.format(scale_factor_pop, scale_factor_test)
    if success_rate_test != 0.99:
        name += '_{}'.format(success_rate_test)
    return name


def plotting(filename, prob_sick_plot_index, saveFig=0):
    # load data
    datapath = 'data/{}.pkl'.format(filename)
    with open(datapath, 'rb') as fp:
        data = pickle.load(fp)
    figpath = '/home/rehmemk/git/diss/gfx/pre/gfx_8_covid/{}'.format(filename)

    # extract relevant parameters from data
    test_strategies = data['test_strategies']
    daily_tests_per_1m = data['daily_tests_per_1m']
    countries = data['countries']
    probabilities_sick = data['probabilities_sick']
    e_time = data['e_time']
    sd_time = data['sd_time']
    e_num_confirmed_per_test = data['e_num_confirmed_per_test']
    sd_num_confirmed_per_test = data['sd_num_confirmed_per_test']

    # plotting
    markers = ['o', '*', '^', '+', 's', 'd', 'v', '<', '>']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    linestyles = ['-', '-', '-', '-', '-', '--']
    labels = ['Individual testing', '2-level pooling',
              'Binary splitting', 'Recursive binary splitting', 'Purim', 'Sobel-R1']

    ######## prob sick / sick persons per test ########
    #plt.figure(figsize=(6, 5))
    F = Figure(mode='thesis')
    plt.gcf().set_size_inches(6, 5)
    for i, test_strategy in enumerate(test_strategies):
        for j in [0]:  # it's the same for all countries
            plt.plot(probabilities_sick, e_num_confirmed_per_test[i, j, :],
                     label=labels[i], marker=markers[i], color=colors[i], linestyle=linestyles[i])
            # plt.errorbar(probabilities_sick, e_num_confirmed_per_test[i, j, :],
            #              yerr=sd_num_confirmed_per_test[i, j, :], ecolor='k', linestyle='None', capsize=5)
    plt.xlabel('$\mathrm{infection}\ \mathrm{rate}$')
    plt.ylabel(
        '$\mathrm{exp.}\ \mathrm{number}\ \mathrm{of}\ \mathrm{identified}\ \mathrm{cases}\ \mathrm{per}\ \mathrm{test}$')
    plt.xticks([0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], [
               '.', '.', '5\%', '10\%', '15\%', '20\%', '25\%', '30\%', ])
    plt.text(-0.0252, -0.037, '0.1\%', fontsize=SMALL_SIZE)
    plt.text(0.006, -0.037, '1\%', fontsize=SMALL_SIZE)
    #plt.legend(loc='lower right', fontsize=11)
    if PRINTPLOTDATA:
        print(figpath+'psi{}_probsick_ppt.pdf'.format(prob_sick_plot_index))
        print("%20s" % "probabilities_sick", "".join(map(lambda x: "%7.4f " % x, probabilities_sick)))
        for i, test_strategy in enumerate(test_strategies):
            print("%20s" % test_strategy, "".join(map(lambda x: "%7.5f " % x, e_num_confirmed_per_test[i, 0, :])))
    plt.tight_layout()
    if saveFig:
        plt.savefig(figpath+'psi{}_probsick_ppt.pdf'.format(prob_sick_plot_index))  # , bbox_inches='tight')
        plt.close()
        # save legend seperately
        plt.figure()
        plt.plot([0], [0], c=colors[0], marker=markers[0], label='Individual testing')
        plt.plot([0], [0], c=colors[1], marker=markers[1], label='two level pooling')
        plt.plot([0], [0], c=colors[2], marker=markers[2], label='Binary splitting')
        plt.plot([0], [0], c=colors[3], marker=markers[3], label='RBS')
        plt.plot([0], [0], c=colors[4], marker=markers[4], label='Purim')
        plt.plot([0], [0], c=colors[5], marker=markers[5], label='Sobel-R-1')
        axe = plt.gca()
        handles, labels = axe.get_legend_handles_labels()

        originalHandles = handles[:]
        originalLabels = labels[:]
        plt.figure()
        axe = plt.gca()
        axe.axis('off')
        axe.legend(handles, labels, loc='center', fontsize=legendfontsize, ncol=6)
        axe.xaxis.set_visible(False)
        axe.yaxis.set_visible(False)
        for v in axe.spines.values():
            v.set_visible(False)
        # cut off whitespace
        plt.subplots_adjust(left=0.0, right=1.0, top=0.6, bottom=0.4)
        plt.savefig(figpath + 'legend.pdf', dpi=300,
                    bbox_inches='tight', pad_inches=0.0, format='pdf')
    plt.close()

    ######## test per 1M / expected time to test all ########
    plt.figure(figsize=(6, 5))
    #F = Figure(mode='thesis')
    #plt.gcf().set_size_inches(6, 5)
    for i, test_strategy in enumerate(test_strategies):
        plt.plot(daily_tests_per_1m, e_time[i, :, prob_sick_plot_index],
                 label=labels[i], marker=markers[i], color=colors[i], linestyle=linestyles[i])
        # plt.errorbar(daily_tests_per_1m, e_time[i, :, prob_sick_plot_index],
        #              yerr=sd_time[i, :, prob_sick_plot_index], ecolor='k',
        #              linestyle='None', capsize=5)
    plt.ylabel('$\mathrm{exp.}\ \mathrm{time}\ \mathrm{to}\ \mathrm{test}\ \mathrm{population}\ \mathrm{(days)}$')
    plt.xlabel('$\mathrm{daily}\ \mathrm{tests}\ / \mathrm{1M}\ \mathrm{population.}$')
    # plt.xticks(daily_tests_per_1m, ['{} {}'.format(country, int(daily_tests_per_1m[i]))
    #                                 for i, country in enumerate(countries)], rotation=55)
    #plt.xticks(daily_tests_per_1m, [' ', ' ', ' ', ' ', ' '])
    plt.xticks(daily_tests_per_1m, ['.', '.', '.', '.', '.'])
    plt.text(-18, 36, '$\mathrm{UK}(176)$', fontsize=SMALL_SIZE)
    plt.text(235, 36, '$\mathrm{US}(444)$', fontsize=SMALL_SIZE)
    plt.text(486, 36, '$\mathrm{SG}(514)$', fontsize=SMALL_SIZE)
    plt.text(753, 36, '$\mathrm{IT}(762)$', fontsize=SMALL_SIZE)
    plt.text(1311, 36, '$\mathrm{DE}(1479)$', fontsize=SMALL_SIZE)
    #plt.legend(loc='upper right')
    if PRINTPLOTDATA:
        print(figpath+'psi{}_testsper1M_time.pdf'.format(prob_sick_plot_index))
        print("infection rate: %7.4f" % probabilities_sick[prob_sick_plot_index])
        print(" "*20, "".join(map(lambda x: "%7s " % x, countries)))
        print("%20s" % "daily_test_per_1m", "".join(map(lambda x: "%7.2f " % x, daily_tests_per_1m)))
        for i, test_strategy in enumerate(test_strategies):
            print("%20s" % test_strategy, "".join(map(lambda x: "%7.2f " % x, e_time[i, :, prob_sick_plot_index])))
    #plt.yticks([1e+2, 1e+3, 1e+4])
    plt.ylim([50, 1e+4])
    plt.gca().set_yscale('log')
    plt.tight_layout()
    if saveFig:
        plt.savefig(figpath+'psi{}_testsper1M_time.pdf'.format(prob_sick_plot_index))  # , bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # use precalculated data
    scale_factor_pop = 10
    scale_factor_test = 100
    filename = getName(scale_factor_pop, scale_factor_test)

    saveFig = 1
    prob_sick_plot_index = 4  # 4 -> 0.01
    # out of [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    plotting(filename, prob_sick_plot_index, saveFig)
    if saveFig == 0:
        plt.show()
