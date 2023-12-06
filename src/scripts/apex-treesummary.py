#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import math
import os
import re

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
    parser.add_argument('--dot_show', dest='dot_show', action='store_true',
        help='Show DOT file for graphviz (default: false)', default=False)
    parser.add_argument('--ascii', dest='ascii', action='store_true',
        help='Output ASCII tree output (default: false)', default=False)
    parser.add_argument('--shorten', dest='shorten', action='store_true',
        help='Shorten timer names (default: false)', default=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true',
        help='Verbose output (default: false)', default=False)
    parser.add_argument('--tlimit', dest='tlimit', type=float, default=0.0, required=False,
        metavar='N', help='Limit timers to those with value over N (default: 0)')
    parser.add_argument('--dlimit', dest='dlimit', type=float, default=0, required=False,
        metavar='d', help='Limit tree to depth of d (default: none)')
    parser.add_argument('--qlimit', dest='qlimit', type=float, default=0.0, required=False,
        metavar='Q', help='Limit timers to those in the Q quantile (default: None)')
    parser.add_argument('--rlimit', dest='rlimit', type=float, default=0, required=False,
        metavar='R', help='Limit data to those in the first R ranks (default: all ranks)')
    parser.add_argument('--keep', dest='keep', type=str, default='APEX MAIN', required=False,
        metavar='K', help='Keep only subtree starting at K (default: "APEX MAIN")')
    parser.add_argument('--drop', dest='drop', type=str, default='', required=False,
        metavar='D', help='Drop subtree starting at D (default: drop nothing)')
    parser.add_argument('--agg', dest='timer_agg', type=str, default='mean', required=False,
        metavar='A', help=agghelp)
    parser.add_argument('--sort', dest='sort_by', type=str, default='tot/thr', required=False,
        metavar='C', help='Column to sort timers (default: tot/thr)')
    args = parser.parse_args()
    if not os.path.isfile(args.filename):
        parser.print_usage()
        parser.exit()
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
        else:
            tmpdf = df.copy()
            tmpdf['node index'] = child.index
            tmpdf['parent index'] = self.index
            dfs = [child.df, tmpdf]
            child.df = pd.concat(dfs)
        return child
    def print(self, depth, total, maxranks):
        tmpstr = str()
        acc_mean = 0.0
        if not self.df.empty:
            metric = 'total time(s)'
            rows = str(len(self.df.index))
            tmpstr = tmpstr + rows.rjust(len(str(maxranks)), ' ')
            acc_mean = self.df[metric].mean() # get min
            if total == None:
                total = acc_mean
            acc_percent = (acc_mean / total) * 100.0
            acc_minimum = self.df[metric].min() # get min
            acc_maximum = self.df[metric].max() # get max
            acc_threads = self.df['threads'].sum() # get sum
            acc_calls = self.df['calls'].mean() # get sum
            acc_mean_per_call = acc_mean / acc_calls
            tmpstr += ' |'*depth
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
            value, childstr = self.children[key].print(depth+1, total, maxranks)
            totals[key] = value
            strings[key] = childstr
        sorted_by_value = dict(sorted(totals.items(), key=lambda x:x[1], reverse=True))
        for key in sorted_by_value:
            tmpstr = tmpstr + strings[key]
        if depth == 0:
            tmpstr = tmpstr + str(nodeIndex) + ' total graph nodes\n'
        return acc_mean, tmpstr
    def getMergedDF(self):
        # The root node has NO dataframe, so only concatentate all children.
        dfs = []
        if not self.df.empty:
            dfs.append(self.df)
        for key in self.children:
            dfs.append(self.children[key].getMergedDF())
        return pd.concat(dfs)
    def findKeepers(self, keeplist, rootlist, args):
        if self.name in keeplist:
            rootlist.append(self)
            self.df['parent index'] = self.index
            if args.verbose:
                print('Keeping: \'', self.name, '\'', sep='')
        else:
            for keeper in keeplist:
                p = re.compile(keeper)
                if p.match(self.name):
                    rootlist.append(self)
                    self.df['parent index'] = self.index
                    if args.verbose:
                        print('Keeping: \'', self.name, '\'', sep='')
        for key in self.children:
            self.children[key].findKeepers(keeplist, rootlist, args)

def shorten_name(name):
    tmp = name
    # remove arguments in function name, if they exist
    leftparen = tmp.find('(')
    rightparen = tmp.rfind(')')
    if leftparen > 0 and rightparen > 0:
        tokens1 = tmp.split('(')
        tokens2 = tmp.split(')')
        tmp = tokens1[0] + '(...)' + tokens2[len(tokens2)-1]
    # remove template arguments in function name, if they exist
    leftparen = tmp.find('<')
    rightparen = tmp.rfind('>')
    if leftparen > 0 and rightparen > 0:
        tokens1 = tmp.split('<')
        tokens2 = tmp.split('>')
        tmp = tokens1[0] + '<...>' + tokens2[len(tokens2)-1]
    # otherwise, just take the first 64 characters.
    short = (tmp[:67] + '...') if len(tmp) > 67 else tmp
    return short

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
    if math.isnan(v) or  math.isnan(vmin) or  math.isnan(vmax) or vmax == vmin or \
       math.isinf(v) or  math.isinf(vmin) or  math.isinf(vmax) or vmax == vmin:
        return 255
    if v > vmax:
        v = vmax
    dv = vmax - vmin
    if dv <= 0.0:
        dv = 1.0
    frac = 1.0 - ((v-vmin)/dv)
    if math.isnan(frac):
        frac = 1.0
    intensity = int(frac * 255)
    return intensity

def drawDOT(df, args, name):
    # computing new stats
    if args.verbose:
        print('Computing new stats...')
    if 'total Send Bytes' not in df:
        df['total Send Bytes'] = 0
    if 'total Recv Bytes' not in df:
        df['total Recv Bytes'] = 0
    df['total bytes'] = df['total Send Bytes'] + df['total Recv Bytes']
    df['bytes per call'] = df['total bytes'] / df['calls']
    df.loc[df['calls'] == 0, 'bytes per call'] = df['total bytes']
    metric = 'bytes per call'
    # Make a new dataframe from rank 0
    filename = name + 'tasktree.dot';
    f = open(filename, 'w')
    f.write('digraph prof {\n')
    #f.write(' label = "(get this from metadata file output - or, generate it from apex-treesummary.py!)";\n')
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
    if args.verbose:
        print('Building dot file')
    for ind in df.index:
        name = df['name'][ind]
        node_index = df['node index'][ind]
        parent_index = df['parent index'][ind]
        if args.shorten:
            name = shorten_name(name)
        # Remember, the root node is bogus. so skip it.
        if node_index != parent_index:
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
            f.write('bytes per call: ' + str(df['bytes per call'][ind]) + '\\l')
        f.write('time: ' + str(acc) + '\\l"; ')

        f.write('];\n')
    f.write('}')
    f.close()
    if args.dot_show:
        from graphviz import Source
        s = Source.from_file(filename)
        s.view()
        os.wait()
    if args.verbose:
        print('done.')

def graphRank(index, df, parentNode, droplist, args):
    # get the name of this node
    childDF = df[df['node index'] == index].copy()#.reset_index()
    name = childDF['name'].iloc[0]
    # should we skip this subtree?
    if name in droplist:
        if args.verbose:
            print('Dropping: \'', name, '\'', sep='')
        return
    for dropped in droplist:
        p = re.compile(dropped)
        if p.match(name):
            if args.verbose:
                print('Dropping: \'', name, '\'', sep='')
            return

    #name = df.loc[df['node index'] == index, 'name'].iloc[0]
    childNode = parentNode.addChild(name, childDF)

    # slice out the children from the dataframe
    children = df[df['parent index'] == index]
    # Iterate over the children indexes and add to our node
    for child in children['node index'].unique():
        if child == index:
            continue
        graphRank(child, df, childNode, droplist, args)

def graphRank2(index, df, parentNode, droplist, args):
    # get the name of this node
    childDF = df[df['node index'] == index].copy()#.reset_index()
    name = childDF['name'].iloc[0]
    # should we skip this subtree?
    if name in droplist:
        if args.verbose:
            print('Dropping: \'', name, '\'', sep='')
        return
    for dropped in droplist:
        p = re.compile(dropped)
        if p.match(name):
            if args.verbose:
                print('Dropping: \'', name, '\'', sep='')
            return

    #name = df.loc[df['node index'] == index, 'name'].iloc[0]
    childNode = parentNode.addChild(name, childDF)

    # slice out the children from the dataframe
    children = df[df['parent index'] == index]
    # Iterate over the children indexes and add to our node
    for child in children['node index'].unique():
        if child == index:
            continue
        graphRank2(child, df, childNode, droplist, args)

def main():
    args = parseArgs()
    if (args.tau):
        print('TAU conversion coming soon.')
        quit()

    if args.verbose:
        print('Reading tasktree...')
    df = pd.read_csv(args.filename) #, index_col=[0,1])
    df = df.fillna(0)
    if args.verbose:
        print('Read', len(df.index), 'rows')

    # ONLY merge the top 10%
    pd.set_option('display.expand_frame_repr', False)
    # Only keep the first N
    if args.rlimit > 0:
        if args.verbose:
            print('Ignoring any ranks over ', args.rlimit)
        df = df[~(df['process rank'] >= args.rlimit)]
        if args.verbose:
            print('Kept', len(df.index), 'rows')

    if args.dlimit > 0:
        if args.verbose:
            print('Dropping all tree nodes with depth > ', args.dlimit)
        df = df[~(df['depth'] > args.dlimit)]
        if args.verbose:
            print('Kept', len(df.index), 'rows')

    # Get the max rank value
    maxrank = df['process rank'].max()
    maxindex = df['node index'].max()
    maxdepth = df['depth'].max()
    if args.verbose:
        print('Found', maxrank, 'ranks, with max graph node index of', maxindex, 'and depth of', maxdepth)

    metric = 'total time(s)'
    threshold = 0.0
    if args.dlimit > 0:
        threshold = df[metric].quantile(args.qlimit) # get 90th percentile
    if args.tlimit > 0.0:
        threshold = args.tlimit

    if threshold > 0.0:
        if args.verbose:
            print('Ignoring any tree nodes with less than', threshold, 'accumulated time...')
        df = df[~(df[metric] <= threshold)].reset_index()
        if args.verbose:
            print('Kept', len(df.index), 'rows')

    # are there nodes to drop?
    droplist = []
    if len(args.drop) > 0:
        droplist = args.drop.split(',')

    pd.set_option('display.max_rows', None)
    #print(df[['process rank','node index','parent index','name']])
    # FIRST, build a master graph with all nodes from all ranks.
    print('building common tree...')
    root = TreeNode('apex tree base', pd.DataFrame())
    """
    for x in range(maxrank+1):
        print('Rank', x, '...', end=endchar, flush=True)
        # slice out this rank's data
        rank = df[df['process rank'] == x]
        # build a tree of this rank's data
        graphRank(0, rank, root, droplist, args)
    print() # write a newline
    """
    #unique = df.drop_duplicates(subset=["node index", "parent index", "name"], keep='first')
    graphRank2(0, df, root, droplist, args)

    roots = [root]
    if len(args.keep) > 0:
        roots = []
        keeplist = args.keep.split(',')
        root.findKeepers(keeplist, roots, args)

    if args.ascii:
        for root in roots:
            value, treestr = root.print(0, None, maxrank)
            f = open('tasktree.txt', 'w')
            f.write(treestr)
            f.close()
            print(treestr)
            print('Task tree also written to tasktree.txt.')

    if args.dot or args.dot_show:
        for root in roots:
            merged = root.getMergedDF().reset_index()
            # remove the bogus root node
            mean = merged.groupby(['node index','parent index','name']).agg(args.timer_agg, numeric_only=False).reset_index()
            drawDOT(mean, args, root.name)

if __name__ == '__main__':
    main()