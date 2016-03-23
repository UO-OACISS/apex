#!/usr/bin/python

import os.path

True = 1
False = 0

colnames = []
columns = dict()
rows = 0

def loadfile(i):
    global columns
    global rows
    fname = "concurrency." + str(i) + ".dat"
    if not os.path.isfile(fname):
        return
    f = open(fname, 'r')
    headers = f.readline()
    localcols = []
    index = 0
    for col in headers.split('\t'):
        col = col.rstrip()
        col = col.strip('"')
        localcols.append(col)
        columns[col] = columns.get(col, [])
        if col not in colnames:
            colnames.append(col)
    row = 0
    for line in f.readlines():
        index = 0
        for col in line.split('\t'):
            col = col.rstrip()
            if col != '':
                if row < len(columns[localcols[index]]):
                    if index == 0:
                        columns[localcols[index]][row] = int(col)
                    else:
                        columns[localcols[index]][row] += int(col)
                else:
                    columns[localcols[index]].append(int(col))
            index = index + 1
        row = row + 1
    rows = row

def write_gnuplot_file(numcolumns):
    f = open('concurrency.all.gnuplot', 'w')
    f.write("everyhundredth(col) = (int(column(col))%100 ==0)?stringcolumn(1):\"\"\n")
    f.write("set key outside bottom center invert box\n")
    f.write("set xtics auto nomirror\n")
    f.write("set ytics auto nomirror\n")
    f.write("set y2tics auto nomirror\n")
    f.write("# Set the y ranges explicitly, so we can see the lines.\n")
    f.write("stats 'concurrency.all.dat' using 2 name \"A\"\n")
    f.write("stats 'concurrency.all.dat' using 3 name \"B\"\n")
    f.write("set yrange[0:(A_max*1.1)]\n")
    f.write("set y2range[0:B_max]\n")
    f.write("set xlabel \"Time\"\n")
    f.write("set ylabel \"Concurrency\"\n")
    f.write("set y2label \"Power\"\n")
    f.write("# Select histogram data\n")
    f.write("set style data histogram\n")
    f.write("# Give the bars a plain fill pattern, and draw a solid line around them.\n")
    f.write("set style fill solid border\n")
    f.write("set style histogram rowstacked\n")
    f.write("set boxwidth 1.0 relative\n")
    f.write("set palette rgb 33,13,10\n")
    f.write("unset colorbox\n")
    f.write("set key noenhanced\n")
    f.write("plot for [COL=4:")
    f.write(str(numcolumns-1))
    f.write("] 'concurrency.all.dat' using COL:xticlabel(everyhundredth(1)) palette frac (COL-3)/")
    f.write(str(numcolumns-4))
    f.write(". title columnheader axes x1y1, 'concurrency.all.dat' using 2 with lines linecolor rgb \"red\" linewidth 2 axes x1y1 title columnheader, 'concurrency.all.dat' using 3 with lines linecolor rgb \"black\" linewidth 2 axes x1y2 title columnheader\n")
    f.close()

def main():
    global columns
    global rows
    print "--------------- Python test script start ------------"

    for i in range(0,334):
        loadfile(i)

    f = open('concurrency.all.dat', 'w')
    for key in colnames:
        f.write('"')
        #f.write(key.replace('_',''))
        f.write(key)
        f.write('"')
        f.write('\t')
    f.write('\n')
    for r in range(0,rows):
        for c in colnames:
            if r < len(columns[c]):
                value = (columns[c][r])
                f.write(str(value))
            else:
                f.write(str(0))
            f.write('\t')
        f.write('\n')
    f.close()
    print "found", len(colnames), "columns"
    write_gnuplot_file(len(colnames))
    
    print "---------------- Python test script end -------------"

if __name__ == "__main__":
    main()
