#!/usr/bin/env python3

import os
import glob
import json
import sys
import gzip
import argparse
import re

parser = argparse.ArgumentParser(description='Merge and filter APEX Google Trace Event trace files.')
parser.add_argument('--compressed', dest='compressed', action='store_true',
                    help='Merge trace_events.*.json.gz files (default: merge trace_events.*.json files)')
parser.add_argument('--strip-counters', dest='strip_counters', action='store_true',
                    help='Strip counters from the data (save timers only)')
parser.add_argument('--strip-markers', dest='strip_markers', action='store_true',
                    help='Strip marker events from the data (save timers only)')
parser.add_argument('--strip-flow', dest='strip_flow', action='store_true',
                    help='Strip flow eventsfrom the data (save timers only)')
parser.add_argument('--nothread', dest='nothread', action='store_true',
                    help='Strip thread data (save main thread only)')
parser.add_argument("--filename", dest="filename", default=None, required=False, type=str,
                    help="The filename to parse (default is trace_events.*.json.gz)")
parser.add_argument("--outfile", dest="outfile", default="trace_events.filtered.json.gz", required=False, type=str,
                    help="The filename to parse (default is trace_events.*.json.gz)")
parser.add_argument("--strip-timers", dest="strip_timers", default=None, required=False, type=str,
                    help="A regluar expression of timers to strip")
parser.add_argument('--force', dest='force', action='store_true',
                    help='Overwrite output file (trace_events.filtered.json.gz) if it exists')

args = parser.parse_args()

# Iterate over all other trace files
all_data = None

if args.filename:
    myglob = args.filename
    if myglob[-3:] == ".gz" and not args.compressed:
        args.compressed = True
        print("Warning: filename ends with .gz, forcing compressed=True")
else:
    if args.compressed:
        myglob = 'trace_events.*.json.gz'
    else:
        myglob = 'trace_events.*.json'

if args.strip_timers:
    pattern = re.compile(args.strip_timers)

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
    events = []
    for line in data['traceEvents']:
        if args.nothread and 'tid' in line and int(line['tid']) != 0:
            continue;
        if args.strip_counters and line['ph'] == 'C':
            continue;
        if args.strip_markers and line['ph'] == 'R':
            continue;
        if args.strip_flow and (line['ph'] == 's' or line['ph'] == 'f'):
            continue;
        if args.strip_timers and (line['ph'] == 'X'):
            if pattern.match(line['name']):
                continue;
        events.append(line)
    data['traceEvents'] = events
    if (counter == 0):
        all_data = data
    else:
        all_data['traceEvents'] = all_data['traceEvents'] + data['traceEvents']
    jsonfile.close()

#json_str = json.dumps(all_data) + '\n'
#json_bytes = json_str.encode('utf-8')

print(f"Writing and compressing '{args.outfile}'...")
if os.path.exists(args.outfile) and not args.force:
    print(f"ERROR: File '{args.outfile}' already exists. Try the '--force' flag.")

with gzip.open(args.outfile, 'w') as fout:
    fout.write((json.dumps(all_data, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
