__author__ = 'bixlermike'

#/usr/local/bin/python
from matplotlib.font_manager import FontProperties

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

data = np.genfromtxt('banditRuntimeOutputSummary.csv', delimiter=',', names = True, dtype=("|S30", int, float, float, float, float, float), usecols=(0,1,2,3,4,5,6))

N = 3
ind = np.arange(N)
width = 0.5
count = 0

x = np.arange(10)
ys = [i+x+(i*x)**2 for i in range(10)]

fig, ax = plt.subplots()
policies = []
means = []
mins = []
maxs = []
vars = []
stdevs = []
dict = {}

colors = iter(cm.rainbow(np.linspace(0, 1, len(ys))))

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

rects1 = ax.bar(ind, (mins[0], means[0], maxs[0]), width, color=next(colors))
rects2 = ax.bar((N+2) +ind, (mins[1], means[1], maxs[1]), width, color=next(colors))
rects3 = ax.bar((2*(N+2)) +ind, (mins[2], means[2], maxs[2]), width, color=next(colors))
rects4 = ax.bar((3*(N+2)) +ind, (mins[3], means[3], maxs[3]), width, color=next(colors))
rects5 = ax.bar((4*(N+2)) +ind, (mins[4], means[4], maxs[4]), width, color=next(colors))
rects6 = ax.bar((5*(N+2)) +ind, (mins[5], means[5], maxs[5]), width, color=next(colors))
rects7 = ax.bar((6*(N+2)) +ind, (mins[6], means[6], maxs[6]), width, color=next(colors))
rects8 = ax.bar((7*(N+2)) +ind, (mins[7], means[7], maxs[7]), width, color=next(colors))
rects9 = ax.bar((8*(N+2)) +ind, (mins[8], means[8], maxs[8]), width, color=next(colors))
#rects10 = ax.bar((9*(N+2)) +ind, (mins[9], means[9], maxs[9]), width, color=next(colors))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.set_ylabel('Runtime (sec)')
ax.set_title("Algorithm Comparison - Runtime (Min, Mean, Max)")
ax.set_xticklabels(ax.xaxis.get_majorticklocs(), rotation=45)
#ax.set_xticks(ind+ind*N)
ax.set_xticklabels( policies )
#ax.set_yscale('log')
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]), ('Min','Mean','Max'))
#ax.legend((rects1[0], rects2[0], rects3[0], rects1[1], rects2[1], rects3[1], rects1[2], rects2[2]), (policies))
plt.savefig("plots/runtimeComparison.png")
#plt.show()
