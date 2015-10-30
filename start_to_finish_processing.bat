perl outputSummaryStats.pl
echo "Done generating summary statistics."
perl processSummaryStats.pl
echo "Done with post-processing of context-free and contextual summary statistics."
perl outputEnsembleProportionSummaryStats.pl
echo "Done with post-processing of ensemble summary statistics."

perl processEnsembleProportionSummaryStats.pl EnsembleRandom
perl processEnsembleProportionSummaryStats.pl EnsembleRandomUpdateAll
perl processEnsembleProportionSummaryStats.pl EnsembleEAnnealing
perl processEnsembleProportionSummaryStats.pl EnsembleSoftmax0.01
perl processEnsembleProportionSummaryStats.pl EnsembleTS
perl processEnsembleProportionSummaryStats.pl EnsembleBinomialUCI
perl processEnsembleProportionSummaryStats.pl EBUCIM1
perl processEnsembleProportionSummaryStats.pl EBUCIM2
echo "Done generating proportion statistics for ensembles."

python create_mean_aer_plot.py
python create_mean_aer_plot_contextual.py
python create_mean_aer_plot_ensemble.py
echo "Done generating mean AER plots."

python create_min_aer_plot.py
python create_min_aer_plot_contextual.py
python create_min_aer_plot_ensemble.py
echo "Done generating min AER plots."

python create_max_aer_plot.py
python create_max_aer_plot_contextual.py
python create_max_aer_plot_ensemble.py
echo "Done generating max AER plots."

python create_delta_aer_plot.py
python create_delta_aer_plot_contextual.py
python create_delta_aer_plot_ensemble.py
echo "Done generating delta AER plots."

python create_stdev_aer_plot.py
python create_stdev_aer_plot_contextual.py
python create_stdev_aer_plot_ensemble.py
echo "Done generating standard deviation AER plots."

python -W ignore create_runtime_plot.py
python -W ignore create_runtime_plot_contextual.py
python -W ignore create_runtime_plot_ensemble.py
echo "Done generating runtime plots."

python create_mean_aer_vs_time_plot.py
python create_mean_aer_vs_time_plot_contextual.py
python create_mean_aer_vs_time_plot_ensemble.py
python -W ignore create_mean_aer_plot_with_ranges.py
echo "Done generating mean AER vs. time plots."

python create_aer_proportion_of_optimal_ensemble_plot.py
python create_aer_proportion_of_optimal_vs_time_ensemble_plot.py
echo "Done generating AER proportion of optimality plots."

#python -W ignore create_mean_ensemble_bandit_proportions_plot.py EnsembleRandom
#python -W ignore create_mean_ensemble_bandit_proportions_plot.py EnsembleRandomUpdateAll
python -W ignore create_mean_ensemble_bandit_proportions_plot.py EnsembleEAnnealing
python -W ignore create_mean_ensemble_bandit_proportions_plot.py EnsembleSoftmax0.01
python -W ignore create_mean_ensemble_bandit_proportions_plot.py EnsembleTS
python -W ignore create_mean_ensemble_bandit_proportions_plot.py EnsembleBinomialUCI
python -W ignore create_mean_ensemble_bandit_proportions_plot.py EBUCIM1
python -W ignore create_mean_ensemble_bandit_proportions_plot.py EBUCIM2
echo "Done generating ensemble bandit proportion plots."

#python -W ignore create_mean_ensemble_bandit_proportions_vs_time_plot.py EnsembleRandom
#python -W ignore create_mean_ensemble_bandit_proportions_vs_time_plot.py EnsembleRandomUpdateAll
python -W ignore create_mean_ensemble_bandit_proportions_vs_time_plot.py EnsembleEAnnealing
python -W ignore create_mean_ensemble_bandit_proportions_vs_time_plot.py EnsembleSoftmax0.01
python -W ignore create_mean_ensemble_bandit_proportions_vs_time_plot.py EnsembleTS
python -W ignore create_mean_ensemble_bandit_proportions_vs_time_plot.py EnsembleBinomialUCI
python -W ignore create_mean_ensemble_bandit_proportions_vs_time_plot.py EBUCIM1
python -W ignore create_mean_ensemble_bandit_proportions_vs_time_plot.py EBUCIM2
echo "Done generating ensemble bandit proportion vs. time plots."
