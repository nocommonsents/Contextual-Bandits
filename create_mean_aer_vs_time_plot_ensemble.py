__author__ = 'bixlermike'

#/usr/local/bin/python

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

majorFormatter = FormatStrFormatter('%d')

data = np.genfromtxt('banditMeanAERVsTimeSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("Ensemble Bandit Algorithm Comparison - Relative Mean AER vs. Time Bin")
#ax.set_title("Contextual Algorithm Comparison - Average Expected Reward vs. Runtime")
ax.set_xlabel('Time Bin')
ax.set_ylabel('Relative Mean AER')

ax.plot(data['TimeBin'],data['EnsembleRandom']/data['Random'], label='EnsRandom')
ax.plot(data['TimeBin'],data['EnsembleRandomUpdateAll']/data['Random'], label='EnsRandomUpdateAll')
ax.plot(data['TimeBin'],data['EnsembleEAnnealingUpdateAll']/data['Random'], label='EnsEAnnUpdateAll')
ax.plot(data['TimeBin'],data['EnsembleBayesianUpdateAll']/data['Random'], label='EnsSoftmaxUpdateAll')
ax.plot(data['TimeBin'],data['EnsembleBinomialUCIUpdateAll']/data['Random'], label='EnsSoftmaxUpdateAll')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
#ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)

ax.set_xlim([0, max(data['TimeBin'])])

ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')
#plt.tight_layout()
plt.savefig("plots/meanAERVsTimeEnsemble.png", bbox_inches='tight')


