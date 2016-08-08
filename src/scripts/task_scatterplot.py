import os
import sys
import sqlite3
import numpy as np
import pylab as pl
import time
import signal
import random
import csv

dictionary = {}
with open ('samples.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='\'')
    for row in spamreader:
        if not row[0].strip().startswith("#"):
            mytup = (float(row[0]),float(row[1]))
            if row[2] not in dictionary:
                dictionary[row[2]] = [mytup]
            else:
                dictionary[row[2]].append(mytup)

axes = pl.subplot()
axes.set_title("Title");
mymark = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
limit = 0
for key in sorted(dictionary, key=lambda key: len(dictionary[key]), reverse=True):
    timestamps = np.array([x[0] for x in dictionary[key]])
    values = np.array([x[1] for x in dictionary[key]])
    name = (key[:64] + '..') if len(key) > 64 else key
    pl.plot(timestamps, values, marker=mymark[limit], linestyle=' ', label=name)
    pl.draw()
    limit = limit + 1
    if limit > 10:
        break

axes.set_autoscale_on(True) # enable autoscale
axes.autoscale_view(True,True,True)
pl.legend(prop={'size':8})
pl.ylabel("usec")
pl.xlabel("seconds from program start")
pl.show()
