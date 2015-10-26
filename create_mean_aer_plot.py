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

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title(r"$Context-Free\ Algorithm\ Comparison\ -\ Mean\ Relative\ AER$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Number\ of\ Evaluations$")
ax.set_ylabel(r"$Mean\ Relative\ AER$")

ax.plot(data['NumEvals'],data['MostClicked']/data['Random'], label=r'$MostClicked$', lw='1.25', marker='o', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['MostRecent']/data['Random'], label=r'$MostRecent$', lw='1.25', marker='v', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['MostCTR']/data['Random'], label=r'$HighestCTR$', lw='1.25', marker='^', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['eGreedy01']/data['Random'], label=r'$e-Greedy(0.1)$', lw='1.25', marker='s', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['eAnnealing']/data['Random'], label=r'$e-Annealing$', lw='1.25', marker='*', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['Softmax001']/data['Random'], label=r'$Softmax(0.01)$', lw='1.25', color='deepskyblue', marker='>', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['Softmax01']/data['Random'], label=r'$Softmax(0.1)$', lw='1.25', marker='+', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['EXP305']/data['Random'], label=r'$EXP\ 3(0.5)$', color='fuchsia', lw='1.25', marker='D', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['UCB1']/data['Random'], label=r'$UCB1$', color='lawngreen', lw='1.25', marker='<', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['BinomialUCI']/data['Random'], label=r'$BinomialUCI$', color='darkorange', lw='1.25', marker='x', markevery=500, fillstyle='none')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, max(data['NumEvals'])])
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')
#plt.tight_layout()

plt.savefig("plots/meanAER.png", dpi=120, bbox_inches='tight')


