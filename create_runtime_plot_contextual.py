__author__ = 'bixlermike'

#/usr/local/bin/python
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
rcParams.update({'figure.autolayout': True})

majorFormatter = FormatStrFormatter('%d')

data = np.genfromtxt('banditRuntimeOutputSummary.csv', delimiter=',', names = True, dtype=("|S30", int, float, float, float, float, float), usecols=(0,1,2,3,4,5,6))

N = 3
ind = np.arange(N)
width = 0.65
count = 0

x = np.arange(8)
ys = [i+x+(i*x)**2 for i in range(14)]

fig, ax = plt.subplots()
policies = []
means = []
mins = []
maxs = []
vars = []
stdevs = []
dict = {}

colors = iter(cm.rainbow(np.linspace(0.25, 2, len(ys))))

for line in data:
    policy, num_sims, mean, min, max, var, stdev = line
    dict[policy] = (mean, min, max, var, stdev)
    policies.append(str(policy))
    means.append(float(mean))
    mins.append(float(min))
    maxs.append(float(max))
    vars.append(float(var))
    stdevs.append(float(stdev))
    #ax.bar((N*count)+ind, dict[policy], width, color=next(colors))
    count+=1

# Contextual
rects1 = ax.bar((N+2) +ind, (mins[14], means[14], maxs[14]), width, color=next(colors))
rects2 = ax.bar((2*(N+2)) +ind, (mins[24], means[24], maxs[24]), width, color=next(colors))
rects3 = ax.bar((3*(N+2)) +ind, (mins[22], means[22], maxs[22]), width, color=next(colors))
rects4 = ax.bar((4*(N+2)) +ind, (mins[18], means[18], maxs[18]), width, color=next(colors))
rects5 = ax.bar((5*(N+2)) +ind, (mins[19], means[19], maxs[19]), width, color=next(colors))
rects6 = ax.bar((6*(N+2)) +ind, (mins[10], means[10], maxs[10]), width, color=next(colors))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.set_ylabel(r'$Runtime\ (sec)$')

ax.set_title(r'$Contextual\ Algorithm\ Comparison\ -\ Runtime\ (Min,\ Mean,\ Max)$', fontsize='16', y=1.02)
ax.xaxis.set_major_formatter(majorFormatter)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=False))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.xaxis.set_major_locator(MultipleLocator(5.2))
plt.tight_layout()

# Contextual
relevant_policies = [r'$$', r'$NaiveBayes$', r'$e-Greedy(0.1)$', r'$e-Annealing$', r'$Softmax(0.01)$', r'$Softmax(0.1)$', r'$LinUCB(0.1)$']
#relevant_policies = [policies[15],policies[13],policies[3],policies[10],policies[7]]
ax.set_xticklabels(relevant_policies)

ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.savefig("plots/runtimeComparisonContextual.png", dpi=240)
#plt.show()
