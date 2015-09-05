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

ax.set_title(r"$Ensemble\ Algorithm\ Comparison\ -\ Mean\ Relative\ AER\ vs.\ Time$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Time\ (Seconds)$")
ax.set_ylabel(r"$Mean\ Relative\ AER$")

ax.plot(data['TimeBin'],data['EnsembleRandom']/data['Random'], lw='1.25', label=r'$EnsRandom$', marker='o', markevery=50, fillstyle='none')
ax.plot(data['TimeBin'],data['EnsembleRandomUpdateAll']/data['Random'], lw='1.25', label=r'$EnsRandomUpdateAll$', marker='v', markevery=50, fillstyle='none')
ax.plot(data['TimeBin'],data['EnsembleEAnnealing']/data['Random'], lw='1.25', label=r'$EnsEAnnealing$', marker='^', markevery=50, fillstyle='none')
ax.plot(data['TimeBin'],data['EnsembleBayesian']/data['Random'], lw='1.25', label=r'$EnsTS$', marker='s', markevery=50, fillstyle='none')
ax.plot(data['TimeBin'],data['EnsembleBinomialUCI']/data['Random'], lw='1.25', label=r'$EnsBinomialUCI$', marker='*', markevery=50, fillstyle='none')
ax.plot(data['TimeBin'],data['EnsembleSoftmax001']/data['Random'], lw='1.25', label=r'$EnsSoftmax0.01$', color='deepskyblue', marker='>', markevery=50, fillstyle='none')
ax.plot(data['TimeBin'],data['EnsembleBinomialUCIMod1']/data['Random'], lw='1.25', label=r'$EBUCIM1$', marker='+', markevery=50, fillstyle='none')
ax.plot(data['TimeBin'],data['EnsembleBinomialUCIMod2']/data['Random'], lw='1.25', label=r'$EBUCIM2$', color='fuchsia', marker='D', markevery=50, fillstyle='none')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)

ax.set_xlim([0, 11000])
#ax.set_xlim([0, max(data['TimeBin'])])

ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')
#plt.tight_layout()
plt.savefig("plots/meanAERVsTimeEnsemble.png", dpi=240, bbox_inches='tight')


