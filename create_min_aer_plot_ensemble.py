__author__ = 'bixlermike'

#/usr/local/bin/python

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

majorFormatter = FormatStrFormatter('%d')

data = np.genfromtxt('banditMinAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("Ensemble Bandit Algorithm Comparison - Minimum AER")
ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Minimum AER')

ax.plot(data['t'],data['EnsembleRandom'], label='EnsRandom')
ax.plot(data['t'],data['EnsembleRandomUpdateAll'], label='EnsRandomUpdateAll')
ax.plot(data['t'],data['EnsembleEAnnealingUpdateAll'], label='EnsEAnnUpdateAll')
ax.plot(data['t'],data['EnsembleBayesianUpdateAll'], label='EnsSoftmaxUpdateAll')
ax.plot(data['t'],data['EnsembleBinomialUCIUpdateAll'], label='EnsSoftmaxUpdateAll')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, max(data['t'])])
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')

plt.savefig("plots/minAEREnsemble.png", bbox_inches='tight')


