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
width = 0.7
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

colors = iter(cm.rainbow(np.linspace(0.2, 2, len(ys))))

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

rects1 = ax.bar((N+2) +ind, (mins[3], means[3], maxs[3]), width, color=next(colors))
rects2 = ax.bar((2*(N+2)) +ind, (mins[4], means[4], maxs[4]), width, color=next(colors))
rects3 = ax.bar((3*(N+2)) +ind, (mins[8], means[8], maxs[8]), width, color=next(colors))
rects4 = ax.bar((4*(N+2)) +ind, (mins[2], means[2], maxs[2]), width, color=next(colors))
rects5 = ax.bar((5*(N+2)) +ind, (mins[5], means[5], maxs[5]), width, color=next(colors))
rects6 = ax.bar((6*(N+2)) +ind, (mins[7], means[7], maxs[7]), width, color=next(colors))
rects7 = ax.bar((7*(N+2)) +ind, (mins[9], means[9], maxs[9]), width, color=next(colors))

#Mod1 and Mod2, respectively
#rects6 = ax.bar((6*(N+2)) +ind, (mins[4], means[4], maxs[4]), width, color=next(colors))
#rects7 = ax.bar((7*(N+2)) +ind, (mins[5], means[5], maxs[5]), width, color=next(colors))


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.set_ylabel(r'$Runtime\ (sec)$')

ax.set_title(r'$Ensemble\ Algorithm\ Comparison\ -\ Runtime\ (Min,\ Mean,\ Max)$', fontsize='16', y=1.02)
ax.xaxis.set_major_formatter(majorFormatter)
ax.get_xaxis().set_major_locator(MaxNLocator(integer=False))
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
ax.xaxis.set_major_locator(MultipleLocator(5.2))
plt.tight_layout()

# Ensemble
relevant_policies = [r'$$', r'$EBUCIM1$', r'$EBUCIM2$', r'$Softmax(0.01)$', r'$BinomialUCI$', r'$e-Annealing$', r'$RandomUpdateAll$', r'$TS$']
ax.set_xticklabels(relevant_policies)
ax.set_yscale('log')
ax.set_ylim([1000, 105000])


ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.savefig("plots/runtimeComparisonEnsemble.png", dpi=120)
#plt.show()
