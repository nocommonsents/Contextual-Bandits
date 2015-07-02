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

data = np.genfromtxt('banditMinAERSummary.csv', delimiter=',', names = True)
data2 = np.genfromtxt('banditMeanAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title(r"$Contextual\ Algorithm\ Comparison\ -\ Relative\ Minimum\ AER$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Number\ of\ Evaluations$")
ax.set_ylabel(r"$Relative\ Minimum\ AER$")

#ax.plot(data['t'],data['Random'], label='Random')
ax.plot(data['t'],data['MostClicked']/data2['Random'], lw='1.25', label=r'$MostClicked$')
ax.plot(data['t'],data['MostRecent']/data2['Random'], lw='1.25', label=r'$MostRecent$')
ax.plot(data['t'],data['MostCTR']/data2['Random'], lw='1.25', label=r'$HighestCTR$')
ax.plot(data['t'],data['eGreedy01']/data2['Random'], lw='1.25', label=r'$e-Greedy(0.1)$')
ax.plot(data['t'],data['eAnnealing']/data2['Random'], lw='1.25', label=r'$e-Annealing$')
ax.plot(data['t'],data['Softmax01']/data2['Random'], lw='1.25', label=r'$Softmax(0.1)$')
ax.plot(data['t'],data['EXP305']/data2['Random'], lw='1.25', label=r'$EXP\ 3(0.5)$', color='fuchsia')
ax.plot(data['t'],data['UCB1']/data2['Random'], lw='1.25', label=r'$UCB1$', color='lawngreen')
ax.plot(data['t'],data['BinomialUCI']/data2['Random'], lw='1.25', label=r'$BinomialUCI$', color='darkorange')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, max(data['t'])])
ax.set_ylim([0,2.5])
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')

plt.savefig("plots/minAER.png", bbox_inches='tight')

