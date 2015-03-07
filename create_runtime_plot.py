__author__ = 'bixlermike'

#/usr/local/bin/python

# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('banditRuntimeSummary.csv', delimiter=',')
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("Algorithm Comparison - Runtime")
mean = data['Mean Runtime']
std = data['Stdev Runtime']
min = data['Min Runtime']
max = data['Max Runtime']

ind = np.arange(3)
width = 0.5

fig, ax = plt.subplots()
rects1 = ax.bar(ind, min, width)
rects2 = ax.bar(ind+width, mean, width, yerr=std)
rects3 = ax.bar(ind+2*width, max, width)

# add some text for labels, title and axes ticks
ax.set_ylabel('Average Expected Reward')
ax.set_title('Runtime Comparison')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Min', 'Mean', 'Max') )

ax.legend( (rects1[0], rects2[0], rects3[0]), (data['Policy']))

plt.show()