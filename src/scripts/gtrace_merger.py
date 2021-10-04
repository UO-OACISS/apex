#!/usr/bin/env python3

import os
import glob
import json
import sys
import gzip
import argparse

parser = argparse.ArgumentParser(description='Merge APEX Google Trace Event trace files.')
parser.add_argument('--compressed', dest='compressed', action='store_true',
                    help='Merge trace_events.*.json.gz files (default: merge trace_events.*.json files)')
parser.add_argument('--strip', dest='strip', action='store_true',
                    help='Strip counters from the data (save timers only)')

args = parser.parse_args()

# Iterate over all other trace files
all_data = None

if args.compressed:
    myglob = 'trace_events.*.json.gz'
else:
    myglob = 'trace_events.*.json'

tracefiles = glob.glob(myglob);
if len(tracefiles) == 0:
    print("No files found!")
    if args.compressed:
        print("  Are you sure they're compressed?")
    else:
        print("  Are they compressed?  If so, please use the --compressed argument")
    parser.print_usage()
    sys.exit(1)

for counter, infile in enumerate(sorted(glob.glob(myglob))):
    print("Reading ", infile)
    #with open (infile, 'r') as jsonfile:
    if args.compressed:
        jsonfile = gzip.open(infile, 'r')
    else:
        jsonfile = open(infile, 'r')
    data = json.load(jsonfile)
    if args.strip:
        events = []
        for line in data['traceEvents']:
            if line['ph'] != 'C':
                events.append(line)
        data['traceEvents'] = events
    if (counter == 0):
        all_data = data
    else:
        if (all_data['displayTimeUnit'] != data['displayTimeUnit']):
            print('Error!  traces have different time units!')
            sys.exit(99)
        all_data['traceEvents'] = all_data['traceEvents'] + data['traceEvents']
    jsonfile.close()

#json_str = json.dumps(all_data) + '\n'
#json_bytes = json_str.encode('utf-8')

print('Writing and compressing trace_events.json.gz...')
with gzip.open('trace_events.json.gz', 'w') as fout:
    fout.write((json.dumps(all_data) + '\n').encode('utf-8'))
