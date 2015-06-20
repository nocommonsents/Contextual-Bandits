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

ax.set_title("Contextual Algorithm Comparison - Relative Mean AER vs. Time Bin")
#ax.set_title("Contextual Algorithm Comparison - Average Expected Reward vs. Runtime")
ax.set_xlabel('Time Bin')
ax.set_ylabel('Relative Mean AER')

#ax.plot(data['TimeBin'],data['Random'], label='Random')
ax.plot(data['TimeBin'],data['eGreedyContextual01']/data['Random'], label='e-Greedy(0.1)')
ax.plot(data['TimeBin'],data['eAnnealingContextual']/data['Random'], label='e-Annealing')
ax.plot(data['TimeBin'],data['SoftmaxContextual01']/data['Random'], label='Softmax(0.1)')
ax.plot(data['TimeBin'],data['LinUCB01']/data['Random'], label='LinUCB(0.1)')
ax.plot(data['TimeBin'],data['NaiveBayesContextual']/data['Random'], label='NaiveBayes', color='lawngreen')

#ax.plot(data['TimeBin'],data['Naive3'], label='NaiveBayes')
#ax.plot(data['TimeBin'],data['eGreedyContextual'], label='eGreedy')
#ax.plot(data['TimeBin'],data['eAnnealingContextual'], label='eAnnealingContextual')
#ax.plot(data['TimeBin'],data['SoftmaxContextual'], label='SoftmaxContextual')
#ax.plot(data['TimeBin'],data['LinUCB'], label='LinUCB')

#ax.plot(data['TimeBin'],data['EnsembleRandom'], label='EnsRandom')
#ax.plot(data['TimeBin'],data['EnsembleRandomUpdateAll'], label='EnsRandomUpdateAll')


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
plt.savefig("plots/meanAERVsTimeContextual.png", bbox_inches='tight')


