__author__ = 'bixlermike'

#/usr/local/bin/python

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

majorFormatter = FormatStrFormatter('%d')

data = np.genfromtxt('banditMinAERSummary.csv', delimiter=',', names = True)
data2 = np.genfromtxt('banditMeanAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("Ensemble Bandit Algorithm Comparison - Relative Minimum AER")
ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Relative Minimum AER')

ax.plot(data['t'],data['EnsembleRandom']/data2['Random'], label='EnsRandom')
ax.plot(data['t'],data['EnsembleRandomUpdateAll']/data2['Random'], label='EnsRandomUpdateAll')
ax.plot(data['t'],data['EnsembleEAnnealingUpdateAll']/data2['Random'], label='EnsEAnnUpdateAll')
ax.plot(data['t'],data['EnsembleBayesianUpdateAll']/data2['Random'], label='EnsBayesianUpdateAll')
ax.plot(data['t'],data['EnsembleBinomialUCIUpdateAll']/data2['Random'], label='EnsBinomialUCIUpdateAll')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, max(data['t'])])
ax.set_ylim([1,2.5])
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')

plt.savefig("plots/minAEREnsemble.png", bbox_inches='tight')


