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

data = np.genfromtxt('banditMeanAERVsTimeSummaryPostProcessed.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title(r"$Contextual\ Algorithm\ Comparison\ -\ Mean\ Relative\ AER\ vs.\ Time$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Time\ (Seconds)$")
ax.set_ylabel(r"$Mean\ Relative\ AER$")

#ax.plot(data['TimeBin'],data['Random'], label='Random')
ax.plot(data['TimeBin'],data['eGreedyContextual01']/data['Random'], lw='1.25', label=r'$eGreedy(0.1)$', marker='o', markevery=500, fillstyle='none')
ax.plot(data['TimeBin'],data['eAnnealingContextual']/data['Random'], lw='1.25', label=r'$eAnnealing$', marker='v', markevery=500, fillstyle='none')
ax.plot(data['TimeBin'],data['SoftmaxContextual001']/data['Random'], lw='1.25', label=r'$Softmax(0.01)$', marker='>', markevery=500, fillstyle='none')
ax.plot(data['TimeBin'],data['SoftmaxContextual01']/data['Random'], lw='1.25', label=r'$Softmax(0.1)$', marker='^', markevery=500, fillstyle='none')
ax.plot(data['TimeBin'],data['LinUCB01']/data['Random'], lw='1.25', label=r'$LinUCB$', marker='s', markevery=500, fillstyle='none')
ax.plot(data['TimeBin'],data['NaiveBayesContextual']/data['Random'], lw='1.25', label=r'$NaiveBayes$', color='darkorange', marker='*', markevery=500, fillstyle='none')

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
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)

ax.set_xlim([0, 27100])
#ax.set_xlim([0, max(data['TimeBin'])])

ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')
#plt.tight_layout()
plt.savefig("plots/meanAERVsTimeContextual.png", dpi=240, bbox_inches='tight')


