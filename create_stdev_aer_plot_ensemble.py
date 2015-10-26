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

data = np.genfromtxt('banditStDevAERSummary.csv', delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title(r"$Ensemble\ Algorithm\ Comparison\ -\ Standard\ Deviation\ of\ AER$", fontsize='16', y=1.02)
ax.set_xlabel(r"$Number\ of\ Evaluations$")
ax.set_ylabel(r"$Standard\ Deviation$")

ax.plot(data['NumEvals'],data['EnsembleRandom'], lw='1.25', label=r'$EnsRandom$', marker='o', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['EnsembleRandomUpdateAll'], lw='1.25', label=r'$EnsRandomUpdateAll$', marker='v', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['EnsembleEAnnealing'], lw='1.25', label=r'$EnsEAnnealing$', marker='^', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['EnsembleTS'], lw='1.25', label=r'$EnsTS$', marker='<', color='darkorange', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['EnsembleBinomialUCI'], lw='1.25', label=r'$EnsBinomialUCI$', marker='*', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['EnsembleSoftmax001'], lw='1.25', label=r'$EnsSoftmax0.01$', color='greenyellow', marker='>', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['EBUCIM1'], lw='1.25', label=r'$EBUCIM1$', marker='+', markevery=500, fillstyle='none')
ax.plot(data['NumEvals'],data['EBUCIM2'], lw='1.25', label=r'$EBUCIM2$', color='fuchsia', marker='D', markevery=500, fillstyle='none')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

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

plt.savefig("plots/stDevAEREnsemble.png", dpi=120, bbox_inches='tight')


