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
data3 = np.genfromtxt('banditMinAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title(r"$Contextual\ Algorithm\ Comparison\ -\ Range\ of\ Relative\ AER$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Number\ of\ Evaluations$")
ax.set_ylabel(r"$Range\ of\ Relative\ AER$")

#ax.plot(data['NumEvals'],data['Random'], label='Random')
ax.plot(data['NumEvals'],(data['eGreedyContextual01']-data3['eGreedyContextual01'])/data2['Random'], lw='1.25', label=r'$e-Greedy(0.1)$', marker='s', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['eAnnealingContextual']-data3['eAnnealingContextual'])/data2['Random'], lw='1.25', label=r'$e-Annealing$', marker='v', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['SoftmaxContextual001']-data3['SoftmaxContextual001'])/data2['Random'], lw='1.25', label=r'$Softmax(0.01)$', marker='>', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['SoftmaxContextual01']-data3['SoftmaxContextual01'])/data2['Random'], lw='1.25', label=r'$Softmax(0.1)$', marker='^', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['LinUCB01']-data3['LinUCB01'])/data2['Random'], lw='1.25', label=r'$LinUCB(0.1)$', marker='s', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['NaiveBayesContextual']-data3['NaiveBayesContextual'])/data2['Random'], lw='1.25', label=r'$NaiveBayes$', color='darkorange', marker='*', markevery=500, fillstyle='none')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, max(data['NumEvals'])])
ax.set_axisbelow(True)
#ax.set_ylim([0.8,2.5])
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')

plt.savefig("plots/rangeAERContextual.png", dpi=120, bbox_inches='tight')


