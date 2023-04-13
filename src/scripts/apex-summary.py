#!/usr/bin/env python3

#print('Importing modules...')
import pandas as pd
import numpy as np
import argparse
import os

# The output header looks like this:
#"rank","name","type","num samples/calls","minimum","mean","maximum","stddev","total","inclusive (ns)","num threads","total per thread"

def parseArgs():
    parser = argparse.ArgumentParser(description='Post-process APEX flat profiles.')
    parser.add_argument('--filename', type=str, required=False,
        help='The filename to parse (default: ./apex_profiles.csv)', default='./apex_profiles.csv')
    parser.add_argument('--counters', dest='counters', action='store_true',
        help='Print the counter data (default: false)', default=False)
    parser.add_argument('--timers', dest='timers', action='store_true',
        help='Print the timer data (default: false)', default=False)
    parser.add_argument('--tau', dest='tau', action='store_true',
        help='Convert to TAU profiles (default: false)', default=False)
    parser.add_argument('--other', dest='other', action='store_true',
        help='Aggregate all other timers and show value (default: false)', default=False)
    parser.add_argument('--limit', dest='timer_limit', type=int, default=30, required=False,
        metavar='N', help='Limit timers to top N timers (default: 30)')
    parser.add_argument('--agg', dest='timer_agg', type=str, default='mean', required=False,
        metavar='A', help='Aggregation operation for timers and counters (default: mean)')
    parser.add_argument('--sort', dest='sort_by', type=str, default='tot/thr', required=False,
        metavar='C', help='Column to sort timers (default: tot/thr)')
    args = parser.parse_args()
    if not os.path.isfile(args.filename):
        parser.print_usage()
        parser.exit()
    if (not args.tau) and (not args.timers) and (not args.counters):
        args.timers = True
        args.counters = True
    return args

def showCounters(counters, args):
    counters = counters.rename(columns={'name': 'Counter', 'num samples/calls': 'samples' })
    df = counters.groupby('Counter').agg(args.timer_agg, numeric_only=True)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print('-'*100)
    print('APEX Counters aggregated by', args.timer_agg)
    print('-'*100)
    print(df[['samples', 'minimum', 'mean', 'maximum', 'stddev']])
    print()

def showMeans(timers, args):
    timers = timers.rename(columns={'name': 'Timer', 'num samples/calls': 'calls', 'num threads': 'threads' })
    if 'yields' not in timers:
        timers['yields'] = 0
    timers['tot/call'] = timers['total'] / timers['calls']
    timers['tot/thr'] = timers['total'] / timers['threads']
    df = timers.groupby('Timer').agg(args.timer_agg, numeric_only=True)
    topN = df.nlargest(args.timer_limit,args.sort_by)
    top1 = df.nlargest(1,'tot/call')
    topN['%total'] = (topN['total'] / top1.iloc[0]['total']) * 100.0
    topN['%wall'] = (topN['tot/thr'] / top1.iloc[0]['total']) * 100.0
    # Aggregate all others?
    allTimers = df.agg('sum', numeric_only=True)
    allTopN = topN.agg('sum', numeric_only=True)
    if args.other:
        other = pd.Series({'calls':allTimers['calls']-allTopN['calls'],
            'threads':allTimers['calls']-allTopN['calls'],
            'tot/call':allTimers['tot/call']-allTopN['tot/call'],
            'total':allTimers['total']-allTopN['total'],
            'tot/thr':allTimers['tot/thr']-allTopN['tot/thr']
            }, name='other')
        topN = topN.append(other)
    # scale all values to seconds
    topN['total'] = topN['total'] * 1.0e-9
    topN['tot/call'] = topN['tot/call'] * 1.0e-9
    topN['tot/thr'] = topN['tot/thr'] * 1.0e-9
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print('-'*100)
    print('Top',args.timer_limit,'APEX Timers sorted by',args.sort_by, 'aggregated by', args.timer_agg)
    print('-'*100)
    print(topN[['total', 'calls', 'tot/call', 'yields', 'threads', 'tot/thr','%total','%wall']])
    print()

def writeTAUProfile(rank, timers, counters, args):
    # write the profile
    print('.', end='', flush=True)
    filename = 'profile.' + str(rank) + '.0.0'
    ntimers = len(timers.index)
    f = open(filename, "w")
    """
    78963 templated_functions_MULTI_TIME
    # Name Calls Subrs Excl Incl ProfileCalls #
    """
    f.write(str(ntimers) + ' templated_functions_TIME\n')
    f.write('# Name Calls Subrs Excl Incl ProfileCalls #\n')
    for index, row in timers.iterrows():
        f.write('"' + row['name'] + '" ')
        f.write(str(int(row['num samples/calls'])) + ' ')
        f.write('0 ')
        f.write(str(int(row['total'])*1e-3) + ' ')
        f.write(str(int(row['inclusive (ns)'])*1e-3) + ' ')
        f.write('0 ')
        f.write('"TAU_USER"')
        f.write('\n')
    f.write('0 aggregates\n')
    # revert to sumsquares
    compute_sumsqr = False
    if 'sumsqr' not in counters:
        compute_sumsqr = True
    ncounters = len(counters.index)
    f.write(str(ncounters) + ' userevents\n')
    f.write('# eventname numevents max min mean sumsqr\n')
    for index, row in counters.iterrows():
        f.write('"' + row['name'] + '" ')
        f.write(str(float(row['num samples/calls'])) + ' ')
        f.write(str(float(row['maximum'])) + ' ')
        f.write(str(float(row['minimum'])) + ' ')
        f.write(str(float(row['mean'])) + ' ')
        if compute_sumsqr:
            stddev = float(row['stddev'])
            mean = float(row['mean'])
            calls = float(row['num samples/calls'])
            sumsqr = ((stddev * stddev) + (mean * mean)) * calls
            f.write(str(float(sumsqr)) + ' ')
        else:
            f.write(str(float(row['sumsqr'])) + ' ')
        f.write('\n')
    f.close()

def convertToTAU(counters, timers, args):
    print('converting to TAU')
    # Get the max rank value
    maxrank = timers['rank'].max()
    # Get the number of timers
    for rank in range(maxrank+1):
        current_timers = timers[timers['rank'] == rank].copy()
        current_counters = counters[counters['rank'] == rank].copy()
        writeTAUProfile(rank, current_timers, current_counters, args)
    print('')

def main():
    args = parseArgs()
    #print('Reading profiles...')
    df = pd.read_csv(args.filename) #, index_col=[0,1])
    df = df.fillna(0)
    print()
    if (args.counters):
        # get the counters
        counters = df[df['type'] == 'counter']
        showCounters(counters, args)
    if (args.timers):
        timers = df[df['type'] == 'timer']
        # Get the means
        showMeans(timers, args)
    if (args.tau):
        counters = df[df['type'] == 'counter']
        timers = df[df['type'] == 'timer']
        convertToTAU(counters, timers, args)
    #print('done.')

if __name__ == '__main__':
    main()