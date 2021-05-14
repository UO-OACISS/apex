#!/usr/bin/python3

from __future__ import print_function
import os.path
import sys
import glob
import csv

colnames = []
columns = dict()
rows = 0
rowdata = {}

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    #print('\r%s |%s| %s%% %s') % (prefix, bar, percent, suffix),
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total: 
        print()

def get_col_names():
    global colnames

    files = glob.glob('concurrency.[0-9]*.dat')
    for f in files:
        with open(f, "r") as f_in:
            headers = f_in.readline()
            for col in headers.split('\t'):
                col = col.rstrip()
                col = col.strip('"')
                if col not in colnames:
                    colnames.append(col)
    print (colnames)

def merger():
    global colnames
    global rowdata

    files = glob.glob('concurrency.[0-9]*.dat')
    numfiles = len(files)
    index = 0
    for f in files:
        printProgressBar (index, numfiles)
        with open(f, "r") as f_in:
            #reader = csv.DictReader(f_in, delimiter='\t', fieldnames=colnames, restval='0')
            #headers = next(reader) # skip the headers
            reader = csv.DictReader(f_in, delimiter='\t')
            for line in reader:
                period = int(line['period'])
                if period not in rowdata:
                    rowdata[period] = {}
                total = 0
                for c in colnames:
                    if c not in line.keys():
                        rowdata[period][c] = 0
                        continue
                    if c not in rowdata[period]:
                        if line[c] == '':
                            rowdata[period][c] = 0
                        else:
                            rowdata[period][c] = int(line[c])
                    else:
                        if c != 'period' and line[c] != '':
                            rowdata[period][c] = rowdata[period][c] + int(line[c])
                    if c != 'period' and c != 'thread cap' and c != 'power' and line[c] != '':
                        total = total + int(line[c])
                if '_total' not in rowdata[period]:
                    rowdata[period]['_total'] = total
                else:
                    rowdata[period]['_total'] = rowdata[period]['_total'] + total
        index = index + 1
    printProgressBar (index, numfiles)

def write_gnuplot_file(numcolumns,numrows):
    divisor = 1
    while numrows / divisor > 20:
        divisor = divisor * 10
    print (divisor)
    f = open('concurrency.all.gnuplot', 'w')
    f.write("# palette\n")
    f.write("set palette maxcolors 16\n")
    f.write("set palette defined ( 0 '#E41A1C', ")
    f.write("1 '#377EB8', ")
    f.write("2 '#4DAF4A', ")
    f.write("3 '#984EA3', ")
    f.write("4 '#FF7F00', ")
    f.write("5 '#FFFF33', ")
    f.write("6 '#A65628', ")
    f.write("7 '#F781BF', ")
    f.write("8 '#66C2A5', ")
    f.write("9 '#FC8D62', ")
    f.write("10 '#8DA0CB', ")
    f.write("11 '#E78AC3', ")
    f.write("12 '#A6D854', ")
    f.write("13 '#FFD92F', ")
    f.write("14 '#E5C494', ")
    f.write("15 '#B3B3B3' )\n")
    f.write("set terminal postscript eps size 16,9 enhanced color font 'Helvetica,16'\n")
    f.write("set output 'concurrency.eps'\n")
    f.write("everyhundredth(col) = (int(column(col))%")
    f.write(str(divisor))
    f.write("==0)?stringcolumn(1):\"\"\n")
    f.write("set key outside bottom center invert box\n")
    f.write("set xtics auto nomirror\n")
    f.write("set ytics auto nomirror\n")
    f.write("set y2tics auto nomirror\n")
    f.write("# Set the y ranges explicitly, so we can see the lines.\n")
    f.write("stats 'concurrency.all.dat' using ")
    f.write(str(numcolumns))
    f.write(" name \"A\"\n")
    f.write("stats 'concurrency.all.dat' using 3 name \"B\"\n")
    f.write("set yrange[0:A_max]\n")
    f.write("set y2range[0:(B_max*1.1)]\n")
    f.write("set xlabel \"Time\"\n")
    f.write("set ylabel \"Concurrency\"\n")
    f.write("set y2label \"Power\"\n")
    f.write("# Select histogram data\n")
    f.write("set style data histogram\n")
    f.write("# Give the bars a plain fill pattern, and draw a solid line around them.\n")
    f.write("set style fill solid border\n")
    f.write("set style histogram rowstacked\n")
    f.write("set boxwidth 1.0 relative\n")
    f.write("unset colorbox\n")
    f.write("set key noenhanced\n")
    f.write("plot for [COL=4:")
    f.write(str(numcolumns-1))
    f.write("] 'concurrency.all.dat' using COL:xticlabel(everyhundredth(1)) palette frac (COL-4)/")
    f.write(str(numcolumns-4))
    f.write(". title columnheader axes x1y1")
    #f.write(", 'concurrency.all.dat' using 2 with lines dashtype 2 linecolor rgb \"black\" linewidth 2 axes x1y1 title columnheader")
    f.write(", 'concurrency.all.dat' using 3 with lines dashtype 2 linecolor rgb \"black\" linewidth 2 axes x1y2 title columnheader\n")
    f.close()

def main():
    global columns
    global rows
    global colnames
    print ("--------------- Python test script start ------------")

    get_col_names()
    merger()

    f = open('concurrency.all.dat', 'w')
    for key in colnames:
        f.write('"')
        f.write(key)
        f.write('"')
        f.write('\t')
    f.write('"_total"\n')
    for r in range(0,len(rowdata)):
        for c in colnames:
            f.write(str(rowdata[r][c]))
            f.write('\t')
        f.write(str(rowdata[r]['_total']))
        f.write('\n')
    f.close()
    print ("found", len(colnames), "columns")
    write_gnuplot_file(len(colnames)+1,len(rowdata))
    
    print ("---------------- Python test script end -------------")

if __name__ == "__main__":
    main()
