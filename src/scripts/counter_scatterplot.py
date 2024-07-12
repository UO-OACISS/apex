#!/usr/bin/env python3

import os
import glob
import string
import sys
import sqlite3
import numpy as np
import pylab as pl
import matplotlib as mpl
import time
import signal
import random
import csv

def shorten_name(name):
    tmp = name
    # otherwise, just take the first 64 characters.
    short = (tmp[:67] + '...') if len(tmp) > 67 else tmp
    return short.replace('_', '$\\_$')

dictionary = {}
max_timestamp = 0
total_index = 0
files = 0
for counter, infile in enumerate(glob.glob('apex_counter_samples.*.csv')):
    with open (infile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='\'')
        index = 0
        for row in spamreader:
            index = index + 1
            if len(row) == 4 and not row[1].strip().startswith("#"):
                try:
                    mytup = (float(row[1])/1000000000.0,float(row[2]),counter)
                except ValueError as e:
                    print(index, " Bad row: ", row)
                    continue
                if float(row[1]) > max_timestamp:
                    max_timestamp = float(row[1])
                if row[3] not in dictionary:
                    dictionary[row[3]] = [mytup]
                else:
                    dictionary[row[3]].append(mytup)
            if (index % 100000 == 0):
                print (index, 'rows parsed...', end='\r', flush=True)
        print ("Parsed", index, "samples")
        total_index = total_index + index
    files = files + 1

#resize the figure
# Get current size
fig_size = pl.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 16
fig_size[1] = 4
pl.rcParams["figure.figsize"] = fig_size

mymark = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
#mycolor = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'darkblue', 'darkgreen', 'darkred']
#mycolor = ('blue', 'green', 'red', 'royalblue', 'darkmagenta', 'darkslategrey', 'black', 'darkblue', 'darkgreen', 'darkred')
#mycolor = mpl.cm.Dark2.colors
mycolor = mpl.cm.tab10.colors


safechars = string.ascii_lowercase + string.ascii_uppercase + string.digits + '.-_:,'

for key in sorted(dictionary, key=lambda key: len(dictionary[key]), reverse=True):
    print ("Plotting", key)
    fig,axes = pl.subplots()
    marker = '.'
    extension = '.pdf'
    if len(dictionary[key]) > 1000000:
        marker = ','
        extension = '.png'
    name = shorten_name(key)
    pl.title(name, fontsize=28);
    #timestamps = np.array([x[0] for x in dictionary[key]])
    tvalues = np.array([x[1] for x in dictionary[key]])
    index = 0
    for f in range(files):
        timestamps = []
        values = []
        for x in dictionary[key]:
            if x[2] == index:
                timestamps.append(x[0])
                values.append(x[1])
        pl.plot(timestamps, values, marker=marker, color=mycolor[f], linestyle='-', label='locality '+str(f))
        index = index + 1
    y_mean = [np.mean(tvalues)]*2
    x_mean = [0.0, max_timestamp]
    pl.plot(x_mean, y_mean, label='Mean: '+str(round(y_mean[0],1)), linestyle='--', color='darkslategrey')
    pl.draw()
    axes.set_autoscale_on(True) # enable autoscale
    axes.autoscale_view(True,False,True)
    axes.set_xlim(left=0, right=(max_timestamp/1000000000))
    axes.ticklabel_format(useOffset=False, style='plain')
    pl.legend(prop={'size':16}, loc="upper right")
    pl.ylabel('value', fontsize=20)
    pl.xlabel("seconds from program start", fontsize=20)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    print ("Rendering...")
    pl.tight_layout()
    name = key.replace(" ", "_")
    name = ''.join([c for c in name if c in safechars])
    pl.savefig(name+extension)
    pl.close()
