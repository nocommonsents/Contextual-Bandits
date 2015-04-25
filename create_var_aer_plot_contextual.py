__author__ = 'bixlermike'

#/usr/local/bin/python

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

majorFormatter = FormatStrFormatter('%d')

data = np.genfromtxt('banditVarAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("Contextual Algorithm Comparison - Variance of AER")
ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Variance of AER')

ax.plot(data['t'],data['Random'], label='Random')
ax.plot(data['t'],data['eGreedyContextual01'], label='eGreedy')
ax.plot(data['t'],data['eAnnealingContextual'], label='eAnnealing')
#ax.plot(data['t'],data['SoftmaxContextual'], label='Softmax')
ax.plot(data['t'],data['LinUCB01'], label='LinUCB')
ax.plot(data['t'],data['NaiveBayesContextual'], label='NaiveBayes')

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

plt.savefig("plots/varAERContextual.png", bbox_inches='tight')


