__author__ = 'bixlermike'

#/usr/local/bin/python

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

majorFormatter = FormatStrFormatter('%d')

data = np.genfromtxt('banditStDevAERSummary.csv', delimiter=',', names = True)
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("Algorithm Comparison - Standard Deviation of Expected Reward")
ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Standard Deviation of Expected Reward')

ax.plot(data['t'],data['Random'], label='Random')
ax.plot(data['t'],data['eGreedy01'], label='e-Greedy(0.1)')
ax.plot(data['t'],data['eAnnealing'], label='e-Annealing')
ax.plot(data['t'],data['Softmax01'], label='Softmax(0.1)')
ax.plot(data['t'],data['EXP305'], label='EXP3(0.5)')
ax.plot(data['t'],data['UCB1'], label='UCB1')
ax.plot(data['t'],data['Naive3'], label='Naive Bayes')
ax.plot(data['t'],data['EnsembleRandom'], label='EnsRandom')
ax.plot(data['t'],data['EnsembleSoftmax'], label='EnsSoftmax')
ax.plot(data['t'],data['EnsembleEAnnealing'], label='EnsEAnnealing')


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
plt.savefig("plots/stdevAER.png", bbox_inches='tight')


