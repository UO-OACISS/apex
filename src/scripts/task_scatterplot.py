import os
import sys
import sqlite3
import numpy as np
import pylab as pl
import time
import signal
import random
import csv

def shorten_name(name):
    tmp = name
    # remove arguments in function name, if they exist
    leftparen = tmp.find('(')
    rightparen = tmp.rfind(')')
    if leftparen > 0 and rightparen > 0:
        tokens1 = tmp.split('(')
        tokens2 = tmp.split(')')
        tmp = tokens1[0] + '()' + tokens2[len(tokens2)-1]
    # otherwise, just take the first 64 characters.
    short = (tmp[:77] + '...') if len(tmp) > 77 else tmp
    return short.replace('_', '$\_$')

dictionary = {}
with open ('samples.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='\'')
    index = 0
    for row in spamreader:
        index = index + 1
        if len(row) == 3 and not row[0].strip().startswith("#"):
            try:
                mytup = (float(row[0]),float(row[1]))
            except ValueError as e:
                print(index, " Bad row: ", row)
                continue
            if row[2] not in dictionary:
                dictionary[row[2]] = [mytup]
            else:
                dictionary[row[2]].append(mytup)

#resize the figure
# Get current size
fig_size = pl.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
pl.rcParams["figure.figsize"] = fig_size

axes = pl.subplot()
axes.set_title("Title");
mymark = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
limit = 0
for key in sorted(dictionary, key=lambda key: len(dictionary[key]), reverse=True):
    timestamps = np.array([x[0] for x in dictionary[key]])
    values = np.array([x[1] for x in dictionary[key]])
    name = shorten_name(key)
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
