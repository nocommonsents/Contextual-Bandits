__author__ = 'bixlermike'

#/usr/local/bin/python
from matplotlib.font_manager import FontProperties

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('banditRuntimeOutputSummary.csv', delimiter=',', names = True, dtype=("|S30", int, float, float, float, float, float), usecols=(0,1,2,3,4,5,6))

N = 8
ind = np.arange(N)
width = 0.5

fig, ax = plt.subplots()
policies = []
means = []
mins = []
maxs = []
vars = []
stdevs = []

for line in data:
    policy, num_sims, mean, min, max, var, stdev = line
    policies.append(str(policy))
    means.append(float(mean))
    mins.append(float(min))
    maxs.append(float(max))
    vars.append(float(var))
    stdevs.append(float(stdev))

print policies
print means
rects1 = ax.bar(ind, mins, width, color='r')
rects2 = ax.bar((N+1) +ind, means, width, yerr = stdevs, color='b')
rects3 = ax.bar((2*N+2) +ind, maxs, width, color='g')
rects

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
ax.set_ylabel('Runtime (ms)')
ax.set_title("Algorithm Comparison - Runtime")
ax.set_xticks(ind+ind*8)
ax.set_xticklabels( ('Min', 'Mean', 'Max') )

ax.legend((rects1[0], rects2[0], rects3[0], rects1[1], rects2[1], rects3[1], rects1[6], rects1[7]), (policies))

plt.show()