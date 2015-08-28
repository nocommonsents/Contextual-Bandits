__author__ = 'bixlermike'

#/usr/local/bin/python

import matplotlib
from matplotlib import rc
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import sys

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

majorFormatter = FormatStrFormatter('%d')

ensemble_bandit = str(sys.argv[1])
input_file = "banditMeanPolicyProportionsVsEvalNumberSummary" + ensemble_bandit + ".csv"
data = np.genfromtxt(input_file, delimiter=',', names = True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title(r"$Base\ Bandit\ Algorithm\ Proportions\ vs.\ Time\ -\ %s$" % ensemble_bandit, fontsize='16', y=1.02)
ax.set_xlabel(r"$Time (sec)$")
ax.set_ylabel(r"$Proportion$")

ax.plot(data['Time'],data['MostCTR'], label=r'$HighestCTR$', lw='1.25', marker='o', markevery=500, fillstyle='none')
ax.plot(data['Time'],data['BinomialUCI'], label=r'$BinomialUCI$', lw='1.25', marker='v', markevery=500, fillstyle='none')
ax.plot(data['Time'],data['UCB1'], label=r'$UCB1$', lw='1.25', marker='^', markevery=500, fillstyle='none')
ax.plot(data['Time'],data['Softmax01'], label=r'$Softmax(0.1)$', lw='1.25', marker='s', markevery=500, fillstyle='none')
ax.plot(data['Time'],data['NaiveBayesContextual'], label=r'$NaiveBayes$', lw='1.25', marker='*', markevery=500, fillstyle='none')
ax.plot(data['Time'],data['LinUCB01'], label=r'$LinUCB(0.1)$', lw='1.25', color='deepskyblue', marker='>', markevery=500, fillstyle='none')
ax.plot(data['Time'],data['SoftmaxContextual01'], label=r'$SoftmaxContextual(0.1)$', lw='1.25', marker='+', markevery=500, fillstyle='none')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.set_xlim([0, 2897])
#ax.set_xlim([0, max(data['Time'])])
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(majorFormatter)
#for ymaj in ax1.yaxis.get_majorticklocs():
#    ax1.axhline(y=ymaj,ls='-')
plt.tight_layout()

output_file = "plots/banditMeanPolicyProportionsVsTime" + ensemble_bandit + ".png"

plt.savefig(output_file, dpi=240, bbox_inches='tight')


