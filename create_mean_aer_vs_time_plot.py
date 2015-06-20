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

ax.set_title("Non-Contextual Algorithm Comparison - Relative Mean AER vs. Time Bin")
#ax.set_title("Contextual Algorithm Comparison - Average Expected Reward vs. Runtime")
ax.set_xlabel('Time Bin')
ax.set_ylabel('Relative Mean AER')

#ax.plot(data['TimeBin'],data['Random'], label='Random')
ax.plot(data['TimeBin'],data['MostClicked']/data['Random'], label='MostClicked')
ax.plot(data['TimeBin'],data['MostRecent']/data['Random'], label='MostRecent')
ax.plot(data['TimeBin'],data['MostCTR']/data['Random'], label='HighestCTR')
ax.plot(data['TimeBin'],data['eGreedy01']/data['Random'], label='e-Greedy(0.1)')
ax.plot(data['TimeBin'],data['eAnnealing']/data['Random'], label='e-Annealing')
ax.plot(data['TimeBin'],data['Softmax01']/data['Random'], label='Softmax(0.1)')
ax.plot(data['TimeBin'],data['EXP305']/data['Random'], label='EXP3(0.5)')
ax.plot(data['TimeBin'],data['UCB1']/data['Random'], label='UCB1', color='lawngreen')

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

#ax.set_xlim([0, max(data['TimeBin'])])

ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')
#plt.tight_layout()
plt.savefig("plots/meanAERVsTime.png", bbox_inches='tight')


