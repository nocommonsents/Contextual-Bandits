__author__ = 'bixlermike'

#/usr/local/bin/python

import matplotlib
from matplotlib import rc
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

majorFormatter = FormatStrFormatter('%d')

data = np.genfromtxt('banditMeanAERSummary.csv', delimiter=',', names = True)
data2 = np.genfromtxt('banditMinAERSummary.csv', delimiter=',', names = True)
data3 = np.genfromtxt('banditMaxAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title(r"$Relative\ Mean\ AER\ Range\ vs.\ Evaluation\ Number\ -\ BinomialUCI\ Algorithm$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Number\ of\ Evaluations$")
ax.set_ylabel(r"$Relative\ Mean\ AER$")

minimum = data2['BinomialUCI']/data['Random']
mean = data['BinomialUCI']/data['Random']
maximum = data3['BinomialUCI']/data['Random']
ax.plot(data['NumEvals'], maximum, label='Max', lw='1.25', color='green')
ax.plot(data['NumEvals'], mean, label='Mean', lw='1.25', color='blue')
ax.plot(data['NumEvals'], minimum, label='Min', lw='1.25', color='red')
ax.fill_between(data['NumEvals'], minimum, maximum, facecolor='blue', alpha=0.5)


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#fontP = FontProperties()
#fontP.set_size('small')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.legend(loc='lower right')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
#ax.set_xlim([0, min(max(data['t']),max(data2['t']),max(data3['t']))])
ax.set_xlim([0,340000])
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')
plt.tight_layout()
#plt.show()
plt.savefig("plots/averageAERwithRangesBinomialUCIAlgorithm.png", dpi=240, bbox_inches='tight')


