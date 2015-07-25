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

ax.set_title(r"$Ensemble\ Algorithm\ Comparison\ -\ Relative\ Minimum\ AER$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Number\ of\ Evaluations$")
ax.set_ylabel(r"$Relative\ Minimum\ AER$")

ax.plot(data['t'],data['EnsembleRandom']/data2['Random'], lw='1.25', label=r'$EnsRandom$', marker='s', markevery=500, fillstyle='none')
ax.plot(data['t'],data['EnsembleRandomUpdateAll']/data2['Random'], lw='1.25', label=r'$EnsRandomUpdateAll$', marker='v', markevery=500, fillstyle='none')
ax.plot(data['t'],data['EnsembleEAnnealingUpdateAll']/data2['Random'], lw='1.25', label=r'$EnsEAnnUpdateAll$', marker='^', markevery=500, fillstyle='none')
#ax.plot(data['t'],data['EnsembleBayesianUpdateAll']/data2['Random'], lw='1.25', label=r'$EnsBayesianUpdateAll$', marker='s', markevery=500, fillstyle='none')
ax.plot(data['t'],data['EnsembleBinomialUCIUpdateAll']/data2['Random'], lw='1.25', label=r'$EnsBinomialUCIUpdateAll$', marker='*', markevery=500, fillstyle='none')
ax.plot(data['t'],data['EnsembleSoftmax001UpdateAll']/data2['Random'], lw='1.25', label=r'$EnsSoftmax0.01UpdateAll$', marker='+', markevery=500, fillstyle='none')
#ax.plot(data['t'],data['EnsembleMostCTRUpdateAll']/data2['Random'], lw='1.25', label=r'$EnsMostCTRUpdateAll$', color='deepskyblue', marker='>', markevery=500, fillstyle='none')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, max(data['t'])])
ax.set_axisbelow(True)
ax.set_ylim([0.25,2.5])
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')

plt.savefig("plots/minAEREnsemble.png", dpi=240, bbox_inches='tight')


