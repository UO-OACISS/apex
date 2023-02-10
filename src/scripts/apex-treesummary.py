#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import math

# "process rank","node index","parent index","depth","name","calls","threads","total time(s)","inclusive time(s)","minimum time(s)","mean time(s)","maximum time(s)","stddev time(s)","total Recv Bytes","minimum Recv Bytes","mean Recv Bytes","maximum Recv Bytes","stddev Recv Bytes","median Recv Bytes","mode Recv Bytes","total Send Bytes","minimum Send Bytes","mean Send Bytes","maximum Send Bytes","stddev Send Bytes","median Send Bytes","mode Send Bytes"
endchar='\r'

agghelp = 'Aggregation operation for timers and counters (default: mean)'\
          'Accepted methods:\n'\
          '  count   Returns count for each group\n'\
          '  size    Returns size for each group\n'\
          '  sum     Returns total sum for each group\n'\
          '  mean    Returns mean for each group. Same as average()\n'\
          '  average Returns average for each group. Same as mean()\n'\
          '  std     Returns standard deviation for each group\n'\
          '  var     Return var for each group\n'\
          '  sem     Standard error of the mean of groups\n'\
          '  min     Returns minimum value for each group\n'\
          '  max     Returns maximum value for each group\n'\
          '  first   Returns first value for each group\n'\
          '  last    Returns last value for each group\n'\
          '  nth     Returns nth value for each group\n'

def parseArgs():
    parser = argparse.ArgumentParser(description='Post-process APEX flat profiles.', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--filename', type=str, required=False,
        help='The filename to parse (default: ./apex_tasktree.csv)', default='./apex_tasktree.csv')
    parser.add_argument('--tau', dest='tau', action='store_true',
        help='Convert to TAU profiles (default: false)', default=False)
    parser.add_argument('--dot', dest='dot', action='store_true',
        help='Generate DOT file for graphviz (default: false)', default=False)
    parser.add_argument('--ascii', dest='ascii', action='store_true',
        help='Output ASCII tree output (default: true)', default=True)
    parser.add_argument('--limit', dest='limit', type=float, default=0.0, required=False,
        metavar='N', help='Limit timers to those with value over N (default: 0)')
    parser.add_argument('--qlimit', dest='qlimit', type=float, default=0.9, required=False,
        metavar='Q', help='Limit timers to those in the Q quantile (default: 0.9)')
    parser.add_argument('--rlimit', dest='rlimit', type=float, default=0, required=False,
        metavar='R', help='Limit data to those in the first R ranks (default: all ranks)')
    parser.add_argument('--agg', dest='timer_agg', type=str, default='mean', required=False,
        metavar='A', help=agghelp)
    parser.add_argument('--sort', dest='sort_by', type=str, default='tot/thr', required=False,
        metavar='C', help='Column to sort timers (default: tot/thr)')
    args = parser.parse_args()
    return args

nodeIndex = 0
class TreeNode:
    def __init__(self, name, df):
        global nodeIndex
        self.name = name
        self.index = nodeIndex
        nodeIndex = nodeIndex + 1
        self.children = {}
        self.df = df
    def get(self, name):
        if name in self.children.keys():
            return self.children[name]
        else:
            return None

    # Add or get child if exists
    def addChild(self, name, df):
        child = self.get(name)
        if child == None:
            child = TreeNode(name, df)
            self.children[child.name] = child
            df['node index'] = child.index
            df['parent index'] = self.index
        return child
    def print(self, depth, total):
        tmpstr = str()
        acc_mean = 0.0
        if not self.df.empty:
            metric = 'total time(s)'
            acc_mean = self.df[metric].mean() # get min
            if total == None:
                total = acc_mean
            acc_percent = (acc_mean / total) * 100.0
            acc_minimum = self.df[metric].min() # get min
            acc_maximum = self.df[metric].max() # get max
            acc_threads = self.df['threads'].sum() # get sum
            acc_calls = self.df['calls'].mean() # get sum
            acc_mean_per_call = acc_mean / acc_calls
            tmpstr = ' |'*depth
            tmpstr = tmpstr + '-> ' + '%.3f' % acc_mean
            tmpstr = tmpstr + ' - ' + '%.3f' % acc_percent
            tmpstr = tmpstr + '% [' + str(int(acc_calls))
            tmpstr = tmpstr + '] {min=' + '%.3f' % acc_minimum
            tmpstr = tmpstr + ', max=' + '%.3f' % acc_maximum
            tmpstr = tmpstr + ', mean=' + '%.3f' % acc_mean_per_call
            tmpstr = tmpstr + ', threads=' + str(int(acc_threads))
            tmpstr = tmpstr + '} ' + self.name + '\n'
        totals = {}
        strings = {}
        for key in self.children:
            value, childstr = self.children[key].print(depth+1, total)
            totals[key] = value
            strings[key] = childstr
        sorted_by_value = dict(sorted(totals.items(), key=lambda x:x[1], reverse=True))
        for key in sorted_by_value:
            tmpstr = tmpstr + strings[key]
        if depth == 0:
            tmpstr = tmpstr + str(nodeIndex) + 'total graph nodes'
        return acc_mean, tmpstr
    def getMergedDF(self):
        # The root node has NO dataframe, so only concatentate all children.
        dfs = []
        if not self.df.empty:
            dfs.append(self.df)
        for key in self.children:
            dfs.append(self.children[key].getMergedDF())
        return pd.concat(dfs)

def get_node_color_visible(v, vmin, vmax):
    if v > vmax:
        v = vmax
    dv = vmax - vmin
    if dv == 0:
        dv = 1
    frac = 1.0 - ((v-vmin)/dv)
    # red is full on
    red = 255
    # blue should grow proportionally
    blue = int(frac * 255)
    green = int(frac * 255)
    return red, blue, green

def get_node_color_visible_one(v, vmin, vmax):
    if math.isnan(v) or  math.isnan(vmin) or  math.isnan(vmax):
        return 255
    if v > vmax:
        v = vmax
    dv = vmax - vmin
    if dv == 0:
        dv = 1
    frac = 1.0 - ((v-vmin)/dv)
    intensity = int(frac * 255)
    return intensity

def drawDOT(df):
    # computing new stats
    print('Computing new stats...')
    if 'total Send Bytes' not in df:
        df['total Send Bytes'] = 0
    if 'total Recv Bytes' not in df:
        df['total Recv Bytes'] = 0
    df['total bytes'] = df['total Send Bytes'] + df['total Recv Bytes']
    df['bytes per call'] = df['total bytes'] / df['calls']
    metric = 'bytes per call'
    # Make a new dataframe from rank 0
    f = open('tasktree.0.dot', 'w')
    f.write('digraph prof {\n')
    f.write(' label = "(get this from metadata file output!)";\n')
    f.write(' labelloc = "t";\n')
    f.write(' labeljust = "l";\n')
    f.write(' overlap = false;\n')
    f.write(' splines = true;\n')
    f.write(' rankdir = "LR";\n')
    f.write(' node [shape=box];\n')
    ignored=set()
    metric = 'total time(s)'
    acc_minimum = df[metric].min() # get min
    acc_maximum = df[metric].max() # get max
    bpc_minimum = 0
    bpc_maximum = 0
    bpc_minimum = min(filter(lambda x: x > 0, df['bytes per call']), default=1)
    bpc_maximum = df['bytes per call'].max() # get max
    print('Building dot file')
    for ind in df.index:
        name = df['name'][ind]
        node_index = df['node index'][ind]
        parent_index = df['parent index'][ind]
        if 'INIT' in name:
            ignored.add(node_index)
            continue
        if parent_index in ignored:
            ignored.add(node_index)
            continue
        """
        if df[metric][ind] < limit:
            ignored.add(node_index)
            continue
        """
        # Remember, the root node is bogus. so skip it.
        if node_index > 0 and parent_index > 0:
            f.write('  "' + str(parent_index) + '" -> "' + str(node_index) + '";\n')
        f.write('  "' + str(node_index) + '" [shape=box; ')
        f.write('style=filled; ')
        acc = df['total time(s)'][ind]
        bpc = df['bytes per call'][ind]
        if "MPI" in name and bpc > 0:
            red = int(255)
            green = get_node_color_visible_one(bpc, bpc_minimum, bpc_maximum)
            blue = get_node_color_visible_one(acc, acc_minimum, acc_maximum)
            if blue > red:
                red = green
                blue = int(255)
            else:
                blue = green
        else:
            red = get_node_color_visible_one(acc, acc_minimum, acc_maximum)
            green = red
            blue = int(255)
        f.write('color=black; ')
        if (acc > 0.5 * acc_maximum):
            f.write('fontcolor=white; ')
        else:
            f.write('fontcolor=black; ')
        f.write('fillcolor="#')
        f.write(f'{red:02x}' + f'{green:02x}' + f'{blue:02x}' + '"; ')
        f.write('depth=' + str(df['depth'][ind])+ '; ')
        f.write('time=' + str(f'{acc:.20f}')+ '; ')
        f.write('label="' + str(name)+ '\\l')
        f.write('calls: ' + str(df['calls'][ind]) + '\\l')
        f.write('threads: ' + str(df['threads'][ind]) + '\\l')
        if (df['total Send Bytes'][ind] > 0):
            f.write('total send bytes: ' + str(df['total Send Bytes'][ind]) + '\\l')
            f.write('mean send bytes: ' + str(df['mean Send Bytes'][ind]) + '\\l')
            f.write('mode send bytes: ' + str(df['mode Send Bytes'][ind]) + '\\l')
        if (df['total Recv Bytes'][ind] > 0):
            f.write('total recv bytes: ' + str(df['total Recv Bytes'][ind]) + '\\l')
            f.write('mean recv bytes: ' + str(df['mean Recv Bytes'][ind]) + '\\l')
            f.write('mode recv bytes: ' + str(df['mode Recv Bytes'][ind]) + '\\l')
        if (df['bytes per call'][ind] > 0):
            f.write('bytes per call: ' + str(int(df['bytes per call'][ind])) + '\\l')
        f.write('time: ' + str(acc) + '\\l"; ')

        f.write('];\n')
    f.write('}')
    f.close()
    print('done.')

def graphRank(index, df, parentNode):
    # get the name of this node
    childDF = df[df['node index'] == index].copy()#.reset_index()
    name = childDF['name'].iloc[0]
    #name = df.loc[df['node index'] == index, 'name'].iloc[0]
    childNode = parentNode.addChild(name, childDF)

    # slice out the children from the dataframe
    children = df[df['parent index'] == index]
    # Iterate over the children indexes and add to our node
    for child in children['node index'].unique():
        if child == index:
            continue
        graphRank(child, df, childNode)

def main():
    args = parseArgs()
    if (args.tau):
        print('TAU conversion coming soon.')
        quit()

    print('Reading tasktree...')
    df = pd.read_csv(args.filename) #, index_col=[0,1])
    df = df.fillna(0)

    # ONLY merge the top 10%
    pd.set_option('display.expand_frame_repr', False)
    # Get the max rank value
    # Only keep the first 4
    if args.rlimit > 0:
        print('Ignoring any ranks over ', args.rlimit)
        df = df[~(df['process rank'] >= args.rlimit)]

    maxrank = df['process rank'].max()
    # maxindex = df[metric].idxmax()
    maxindex = df['node index'].max()
    maxdepth = df['depth'].max()
    print('Found', maxrank, 'ranks, with max graph node index of', maxindex, 'and depth of', maxdepth)

    metric = 'total time(s)'
    threshold = df[metric].quantile(args.qlimit) # get 90th percentile
    if args.limit > 0.0:
        threshold = args.limit
    print('Ignoring any tree nodes with less than', threshold, 'accumulated time...')
    df = df[~(df[metric] <= threshold)].reset_index()

    pd.set_option('display.max_rows', None)
    #print(df[['process rank','node index','parent index','name']])
    # FIRST, build a master graph with all nodes from all ranks.
    print('building common tree...')
    root = TreeNode('apex tree base', pd.DataFrame())
    for x in range(maxrank+1):
        print('Rank', x, '...', end=endchar, flush=True)
        # slice out this rank's data
        rank = df[df['process rank'] == x]
        # build a tree of this rank's data
        graphRank(0, rank, root)
    if args.ascii:
        value, treestr = root.print(0, None)
        print(treestr)

    merged = root.getMergedDF().reset_index()
    # remove the bogus root node
    merged = merged[~(merged['name'] == 'apex tree base')]

    if args.dot:
        mean = merged.groupby(['node index','parent index','name']).agg(args.timer_agg, numeric_only=False).reset_index()
        drawDOT(mean)

if __name__ == '__main__':
    main()