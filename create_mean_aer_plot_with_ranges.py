__author__ = 'bixlermike'

#/usr/local/bin/python

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


majorFormatter = FormatStrFormatter('%d')

data = np.genfromtxt('banditMeanAERSummary.csv', delimiter=',', names = True)
data2 = np.genfromtxt('banditMinAERSummary.csv', delimiter=',', names = True)
data3 = np.genfromtxt('banditMaxAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("Mean AER Range vs. Evaluation Number - Highest CTR Bandit Algorithm")
ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('AER')

minimum = data2['MostCTR']
mean = data['MostCTR']
maximum = data3['MostCTR']
ax.plot(data['t'], maximum, label='Max')
ax.plot(data['t'], mean, label='Mean')
ax.plot(data['t'], minimum, label='Min')
ax.fill_between(data['t'], minimum, maximum, facecolor='blue', alpha=0.5)


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#fontP = FontProperties()
#fontP.set_size('small')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.legend(loc='lower right')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, min(max(data['t']),max(data2['t']),max(data3['t']))])
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')
plt.tight_layout()
#plt.show()
plt.savefig("plots/averageAERwithRangesHighestCTRAlgorithm.png", bbox_inches='tight')


