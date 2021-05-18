#!/usr/bin/env python3

import os
import glob
import string
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
for counter, infile in enumerate(glob.glob('apex_counter_samples.*.csv')):
    with open (infile, 'r') as csvfile:
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
        print ("Parsed", index, "samples")

#resize the figure
# Get current size
fig_size = pl.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 8
fig_size[1] = 4
pl.rcParams["figure.figsize"] = fig_size

mymark = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
mycolor = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'darkblue', 'darkgreen', 'darkred']

safechars = string.ascii_lowercase + string.ascii_uppercase + string.digits + '.-_:,'

index = 0
for key in sorted(dictionary, key=lambda key: len(dictionary[key]), reverse=True):
    print ("Plotting", key)
    fig,axes = pl.subplots()
    timestamps = np.array([x[0] for x in dictionary[key]])
    values = np.array([x[1] for x in dictionary[key]])
    name = shorten_name(key)
    pl.title(name);
    #pl.semilogy(timestamps, values, color=mycolor[index-1], marker=mymark[index-1], linestyle=' ', label=name)
    #pl.plot(timestamps, values, color=mycolor[index-1], marker=mymark[index-1], linestyle=' ', label=name)
    pl.plot(timestamps, values, marker='o', color='blue')
    pl.draw()
    axes.set_autoscale_on(True) # enable autoscale
    axes.autoscale_view(True,True,True)
    pl.legend(prop={'size':8})
    pl.ylabel('value')
    pl.xlabel("seconds from program start")
    print ("Rendering...")
    pl.tight_layout()
    name = name.replace(" ", "_")
    name = ''.join([c for c in name if c in safechars])
    pl.savefig(name+".png")
    pl.close()
