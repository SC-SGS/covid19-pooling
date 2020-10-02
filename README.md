# covid19-pooling

This is the code used for the calculations in the paper

"EVALUATION OF POOL-BASED TESTING APPROACHES TO ENABLE POPULATION-WIDE SCREENING FOR COVID-19"

by Timo de Wolff, Dirk Pflüger, Michael Rehme,Janin Heuer, and Martin-Immanuel Bittner.

A preprint is available at https://arxiv.org/abs/2004.11851.

We provide a [website](https://ipvs.informatik.uni-stuttgart.de/sgs/cgi-bin/JA/covid19/) which performs this simulation in an intuitive  user-friendly interface. Unfortunately we cannot perform high-accuracy runs online, because the execution times would be unbearable. Instead we are using a precalculated surrogate For more accurate solutions you can clone this repository and execute the code on your computer using any desired parameter combination.

Execute scenario1.py to recreate the plots of the expected time to test a population in days and expected number of identified cases per test.
Execute accuracies_and_bar_plots to recreate the bar plots.
Execute data/eval_PLOS.py to reccreate the table of the 100,000 tests for a population of 10 million scenario.

For more information on the surrogate used for the website see branch 'sg_surrogate'

