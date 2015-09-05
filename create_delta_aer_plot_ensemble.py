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

ax.set_title(r"$Ensemble\ Algorithm\ Comparison\ -\ Range\ of\ Relative\ AER$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Number\ of\ Evaluations$")
ax.set_ylabel(r"$Range\ of\ Relative\ AER$")

ax.plot(data['NumEvals'],(data['EnsembleRandom']-data3['EnsembleRandom'])/data2['Random'], lw='1.25', label=r'$EnsRandom$', marker='s', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['EnsembleRandomUpdateAll']-data3['EnsembleRandomUpdateAll'])/data2['Random'], lw='1.25', label=r'$EnsRandomUpdateAll$', marker='v', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['EnsembleEAnnealing']-data3['EnsembleEAnnealing'])/data2['Random'], lw='1.25', label=r'$EnsEAnnealing$', marker='^', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['EnsembleTS']-data3['EnsembleTS'])/data2['Random'], lw='1.25', label=r'$EnsTS$', marker='s', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['EnsembleBinomialUCI']-data3['EnsembleBinomialUCI'])/data2['Random'], lw='1.25', label=r'$EnsBinomialUCI$', marker='*', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['EnsembleSoftmax001']-data3['EnsembleSoftmax001'])/data2['Random'], lw='1.25', label=r'$EnsSoftmax0.01$', color='deepskyblue', marker='>', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['EnsembleBinomialUCIMod1']-data3['EnsembleBinomialUCIMod1'])/data2['Random'], lw='1.25', label=r'$EBUCIM1$', marker='+', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],(data['EnsembleBinomialUCIMod2']-data3['EnsembleBinomialUCIMod2'])/data2['Random'], lw='1.25', label=r'$EBUCIM2$', color='fuchsia', marker='D', markevery=500, fillstyle='none')

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

plt.savefig("plots/rangeAEREnsemble.png", dpi=240, bbox_inches='tight')


