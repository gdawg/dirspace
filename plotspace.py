#!/usr/bin/env python
import os, sys, argparse
from sys import stdout, stderr

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import csv
from os.path import relpath
from heapq import heappush, heappop, nlargest

def heappop_depth(heap, depth):
    found = []
    while len(heap) > 0 and heap[0][0] == depth:
        _,rec = heappop(heap)
        found.append(rec)
    return found


def nlargest_paths(csvpath, n=10):
    """Parse csv with directory sizes and return top n by size.
       Expected csv format is [path,size,ignored...]"""

    if csvpath == '-':
        csvpath = '/dev/stdin'

    sep = os.path.sep

    # load into sorted heap by path depth
    heap = []
    with open(csvpath) as fp:
        reader = csv.reader(fp)

        for path,sizestr,_ in reader:
            size = int(sizestr)
            depth = len(path.split(sep))
            rec = (size, path)
            heappush(heap, (depth, rec))

    # pop the root path - all children will be deeper in the tree
    depth,root = heappop(heap)
    top = [root]

    # successively go deeper until we find enough paths to return
    while len(top) < n:
        unsorted_sp = heappop_depth(heap, depth + 1)
        subpaths = nlargest(n - len(top), unsorted_sp)
        top += subpaths

    return top


def dumpdata(root, names, sizes):
    head = 'Largest dirs under {}'.format(root)
    print(head)
    print('-' * len(head))
    data = [names, [str(v) for v in sizes]]
    widths = [max([len(v) for v in a]) for a in data]
    just = ['ljust', 'rjust']
    for row in zip(*data):
        print(' | '.join([getattr(v, j)(w) for v,w,j in zip(row,widths,just)]))

def main():
    args = parse_args()

    # find top paths by size
    toppaths = nlargest_paths(args.csvpath, n=args.number)

    # pop the root so we don't graph 100%
    root = toppaths[0]
    rootpath = root[1]
    paths = toppaths[1:]
    sizes = np.array([p[0] for p in paths])
    names = [p[1][len(rootpath) + 1:] for p in paths]

    dumpdata(rootpath, names, sizes)

    plt.xkcd()

    fig = plt.figure()
    ax = fig.gca()

    graymap = [mpl.cm.gray(v) for v in np.linspace(0.5,1,len(names))]
    plt.pie(sizes, labels=names, colors=graymap)
    plt.title('space used\n{}'.format(root))
    ax.set_aspect('equal')
    plt.show()
    
def parse_args():
    parser = argparse.ArgumentParser(description='Reads csv data and plots space used')
    parser.add_argument('csvpath')
    parser.add_argument('--number', '-n', type=int, default=10)
    return parser.parse_args()

if __name__ == '__main__':
    main()
