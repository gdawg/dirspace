#!/usr/bin/env python
import os, sys, argparse
from sys import stdout, stderr

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import csv
from os.path import relpath
   
def read_data(path, n=10):
    """Reads top n entries, ignoring subdirs"""
    if path == '-':
        path = '/dev/stdin'

    with open(path) as fp:
        reader = csv.reader(fp)
        row = reader.next()
        root = row[0]

        names = []
        sizes = []
        parents = []

        for row in reader:
            p = row[0]
            keep = True
            for parent in parents:
                if p.startswith(parent):
                    keep = False
                    break

            if not keep:
                continue

            names.append(relpath(row[0], root))
            sizes.append(int(row[1]))
            parents.append(p)

            if len(names) > n:
                other = 0
                for r in reader:
                    other += int(row[1])
                names.append('other')
                sizes.append(other)

    return (root,names,np.array(sizes))

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
    root,names,sizes = read_data(args.csvpath, n=args.number)
    dumpdata(root, names, sizes)

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
