__author__ = 'bixlermike'

#/usr/local/bin/python

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

majorFormatter = FormatStrFormatter('%d')

data = np.genfromtxt('banditMaxAERSummary.csv', delimiter=',', names = True)
data2 = np.genfromtxt('banditMeanAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("Context-Free Algorithm Comparison - Relative Maximum AER")

ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Relative Maximum AER')

#ax.plot(data['t'],data['Random'], label='Random')
ax.plot(data['t'],data['MostClicked']/data2['Random'], label='MostClicked')
ax.plot(data['t'],data['MostRecent']/data2['Random'], label='MostRecent')
ax.plot(data['t'],data['MostCTR']/data2['Random'], label='HighestCTR')
ax.plot(data['t'],data['eGreedy01']/data2['Random'], label='e-Greedy(0.1)')
ax.plot(data['t'],data['eAnnealing']/data2['Random'], label='e-Annealing')
ax.plot(data['t'],data['Softmax01']/data2['Random'], label='Softmax(0.1)')
ax.plot(data['t'],data['EXP305']/data2['Random'], label='EXP3(0.5)', color='fuchsia')
ax.plot(data['t'],data['UCB1']/data2['Random'], label='UCB1', color='lawngreen')
ax.plot(data['t'],data['BinomialUCI']/data2['Random'], label='BinomialUCI', color='darkorange')


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, max(data['t'])])
ax.set_ylim([0, 2.5])
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')

plt.savefig("plots/maxAER.png", bbox_inches='tight')


