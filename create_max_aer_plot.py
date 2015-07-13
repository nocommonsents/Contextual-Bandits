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

data = np.genfromtxt('banditMaxAERSummary.csv', delimiter=',', names = True)
data2 = np.genfromtxt('banditMeanAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title(r"$Context-Free\ Algorithm\ Comparison\ -\ Relative\ Maximum\ AER$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Number\ of\ Evaluations$")
ax.set_ylabel(r"$Relative\ Maximum\ AER$")

#ax.plot(data['t'],data['Random'], label='Random')
ax.plot(data['t'],data['MostClicked']/data2['Random'], label=r'$MostClicked$', lw='1.25', marker='o', markevery=500, fillstyle='none')
ax.plot(data['t'],data['MostRecent']/data2['Random'], label=r'$MostRecent$', lw='1.25', marker='v', markevery=500, fillstyle='none')
ax.plot(data['t'],data['MostCTR']/data2['Random'], label=r'$HighestCTR$', lw='1.25', marker='^', markevery=500, fillstyle='none')
ax.plot(data['t'],data['eGreedy01']/data2['Random'], label=r'$e-Greedy(0.1)$', lw='1.25', marker='s', markevery=500, fillstyle='none')
ax.plot(data['t'],data['eAnnealing']/data2['Random'], label=r'$e-Annealing$', lw='1.25', marker='*', markevery=500, fillstyle='none')
ax.plot(data['t'],data['Softmax001']/data2['Random'], label=r'$Softmax(0.01)$', lw='1.25', color='deepskyblue', marker='>', markevery=500, fillstyle='none')
ax.plot(data['t'],data['Softmax01']/data2['Random'], label=r'$Softmax(0.1)$', lw='1.25', marker='+', markevery=500, fillstyle='none')
ax.plot(data['t'],data['EXP305']/data2['Random'], label=r'$EXP\ 3(0.5)$', color='fuchsia', lw='1.25', marker='D', markevery=500, fillstyle='none')
ax.plot(data['t'],data['UCB1']/data2['Random'], label=r'$UCB1$', color='lawngreen', lw='1.25', marker='<', markevery=500, fillstyle='none')
ax.plot(data['t'],data['BinomialUCI']/data2['Random'], label=r'$BinomialUCB$', color='darkorange', lw='1.25', marker='x', markevery=500, fillstyle='none')


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, max(data['t'])])
ax.set_ylim([0.8, 2.5])
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')

plt.savefig("plots/maxAER.png", dpi=240, bbox_inches='tight')


