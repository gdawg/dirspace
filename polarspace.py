#!/usr/bin/env python

__doc__ = """
Polar Chart Generator
=====================

Requires Matplotlib (pip intall matplotlib).
"""

import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
import numpy as np
import re
import argparse
import sys
import fnmatch
from itertools import count, izip

def genwidth(theta, left, right):
    """Generate widths for tightly packed bars ranging from left to right"""
    assert(theta[0] == left)
    width = np.zeros(theta.size, dtype=np.float)
    width[:-1] = theta[1:] - theta[:-1]
    width[-1] = right - theta[-1]
    return width

def rescale(arr, low, hi):
    """Normalise the range of a numpy array from low to hi"""
    return np.normalize(arr) * hi + low

class Node(object):
    """Stores size information for a node and subnodes inclusive"""
    def __init__(self, name, n=None):
        self.name = name
        self.leaf = name.split('/')[-1]
        self.children = {}
        self.n = n
        self.parent = None

    def addchild(self, child):
        assert(not child.name in self.children)

        self.children[child.name] = child
        child.parent = self

    def update(self, n):
        self.n = n

    def remove(self, child):
        """Remove child node. This adjusts sizes recorded for parent
        nodes automatically."""
        assert(child in self.children.values())

        node = self
        while node != None:
            node.n -= child.n
            node = node.parent

        del(self.children[child.name])

    def describesize(self):
        sz = self.n
        for unit in 'bkmgt':
            if sz >= 1024:
                sz = sz / 1024
            else:
                break
        return '{}{}'.format(int(sz), unit)

    def __repr__(self):
        return '{}[n={}, children={}]'.format(
                   self.name, self.n, len(self.children))

    def __len__(self):
        return ( 1 if len(self.children) == 0 
                   else sum([len(c) for c in self.children.values()]) )

def subnodes(name):
    """Returns each full nodename required for a node to exist
    except the final full name itself.
        e.g. for com.co.xxx this would return com, com.co"""
    for i,c in enumerate(name):
        if c == '/':
            yield name[:i]
    raise StopIteration()

def read_data(it):
    """Read sizes from csv into a Node graph"""
    root = Node('', n=0) # everything stems from here

    for lineno,line in izip(count(), it):
        try:
            fields = line[:-1].split(',')
            nodepath,nstr = fields[:2]
        except ValueError, e:
            sys.stderr.write('line {}: {}\n'.format(lineno, e))
        n = int(nstr)

        # walk the tree to find or create any intermediate nodes
        # needed to properly place this node
        parent = root
        for node in subnodes(nodepath):
            if not node in parent.children:
                parent.addchild(Node(node, 0))
            parent = parent.children[node]

        if nodepath in parent.children:
            # when provided with unsorted data we may get parent nodes
            # after their children - if that happens just update their data
            child = parent.children[nodepath]
            child.update(n)
        else:
            # then add it in place
            parent.addchild(Node(nodepath, n))

    return root

def dumptree(node, depth=0):
    """Dump the tree (or a portion of it) as text"""
    print('{}{}'.format('  ' * depth, node))
    for c in node.children.values():
        dumptree(c, depth=depth+1)

def recalculate(node, minsize):
    """Recalculate totals after filtering"""
    children = node.children.values()
    if len(children) > 0:
        node.n = 0
        newchildren = {}
        for c in children:
            recalculate(c, minsize)
            node.n += c.n
            if c.n >= minsize:
                newchildren[c.name] = c
        node.children = newchildren
    if node.n < minsize:
        node.n = 0

class PolarSizeChart(object):
    """Polar chart displaying relative sizes"""
    def __init__(self, *args, **kwargs):
        title = kwargs['title'] if 'title' in kwargs else 'Size by directory'

        self.ax = plt.subplot(projection='polar', *args, **kwargs)
        self.ax.set_axis_off()
        self.ax.set_title(title)

        self.base = 1.0
        self.fontsize = 8.0
        self.min_plotted = np.pi * 0.1
        self.cmap = mplcm.gray
        self.color_by_size = True

    def getbase(self, level):
        maxheight = 10.0
        b = self.base
        h = maxheight
        for x in xrange(level):
            b += h
            h = max(3.0, maxheight - 2.0 * x)
        return b,h

    def should_plot(self, node, width):
        return width > self.min_plotted

    def plot_node(self, node, level, left, right):
        segments = node.children.values() if len(node.children) > 0 else [node]
        slices = len(segments)

        # sort 
        segments.sort(key=lambda n: n.n, reverse=True)
        
        sizes = np.array([c.n for c in segments], dtype=np.float)
        sizesum = sizes.sum()
        if sizesum == 0:
            return [[], [], []]

        # scale the child sizes so the range matches left <-> right
        span = right - left
        widths = span * sizes / sizesum

        # work low to high with the widths to get the starting offsets
        tmp = [left]
        for w in widths[:-1]:
            tmp.append(tmp[-1] + w)
        theta = np.array(tmp, dtype=np.float)

        # calculate the distance from the center (bottom and top)
        base,height = self.getbase(level)

        bottoms = np.ones(slices, dtype=np.float) * base
        tops = np.ones(slices, dtype=np.float) * height

        bars = self.ax.bar(left=theta, 
                           height=tops, 
                           width=widths, 
                           bottom=bottoms, 
                           align='edge')


        # post-process the bars now they've been made
        for child,bar,w,t in zip(segments, bars, widths, theta):
            # it's easier to just plot thin slices then remove than filter
            # up front because we want them to acsize for their space
            if not self.should_plot(child, w):
                bar.remove()
                continue

            if self.color_by_size:
                # calculate range from 0-1
                pos = 1.0 - w / np.pi
                # shift it up to avoid black
                hpos = 0.5 * pos + 0.5
                bar.set_color(self.cmap(hpos))

            fcolor = np.array(bar.get_facecolor())
            ecolor = fcolor * [0.5, 0.5, 0.5, 1.0]
            bar.set_edgecolor(ecolor)

            barpos = bar.get_xy() # get the center of this chunk
            cx = barpos[0] + 0.5 * w
            cy = barpos[1] + 0.5 * height

            # add a text nodename label
            bartext = '{}\n{}'.format(child.leaf, child.describesize())
            cy -= 0.2 * self.fontsize # acsize for newline
            self.ax.text(cx, cy, bartext, 
                horizontalalignment='center',
                fontsize=self.fontsize)

        # return some info on what we drew - useful for linking deeper
        return [segments, theta, widths]

    def show(self):
        self.ax.figure.tight_layout()
        plt.show()


def add_node_to_chart(chart, node, left, right, depth=0):
    print('{}:{}'.format(node.name, node.n))
    result = chart.plot_node(node, depth, left, right)

    for child,theta,width in zip(*result):
        if not chart.should_plot(child, width):
            continue

        if len(child.children) < 2:
            continue

        assert(theta >= left)
        assert(theta + width <= right + 0.1)
        add_node_to_chart(
            chart, child,
            left=theta,
            right=theta + width,
            depth=depth+1)

def polarchart(root, *args, **kwargs):
    chart = PolarSizeChart(*args, **kwargs)

    left,right = (0.0, 2.0 * np.pi)
    add_node_to_chart(chart, root, left, right)

    # This can be fairly trivially extended 
    # to save a pdf, svg, png... etc.
    chart.show()

def findtop(root):
    """Search down through the graph until we find the first node with
    children so we can avoid drawing many full rings"""
    node = root
    while len(node.children) == 1:
        node = node.children.values()[0]
    return node

class ParseSizeAction(argparse.Action):
    """Parser for size arguments which takes suffixes for common byte sizes"""

    smap = {
        'tb': 1024 * 1024 * 1024 * 1024,
        't': 1024 * 1024 * 1024 * 1024,
        'gb': 1024 * 1024 * 1024,
        'g': 1024 * 1024 * 1024,
        'mb': 1024 * 1024,
        'm': 1024 * 1024,
        'kb': 1024,
        'k': 1024,
    }

    def __call__(self, parser, namespace, values, option_string=None):
        numstr = values
        suffix = ''
        for x in xrange(len(values)):
            if values[x] not in '0123456789.':
                numstr,suffix = values[:x],values[x:]
                break

        num = int(float(numstr) * self.multiplier(suffix.lower()))
        setattr(namespace, self.dest, num)

    def multiplier(self, suffix):
        if suffix in self.smap:
            return self.smap[suffix]
        return 1

def filter_tree(node, minsize=0, pattern=None):
    """Recursively filter a node graph to prune unwanted elements"""

    # Apply filtering depth first so that 
    # minsize constraints are properly applied.
    for child in node.children.values():
        filter_tree(child, minsize, pattern)

    if pattern != None and len(node.children) == 0:
        # remove this node if it has a non-matching name and
        # no children. if it has children it must be kept 
        # or they'd be orphaned and lost
        if not fnmatch.fnmatch(node.name, pattern):
            node.parent.remove(node)

    if minsize != 0:
        # The minsize constraint is applied without respect to
        # if the node has children. If it's too small - they're too small.
        if node.n < minsize:
            node.parent.remove(node)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--title', default='Size by Directory')
    parser.add_argument('--minsize', 
        help='draw directories larger than the specified value. accepts units T,G,M,K',
        action=ParseSizeAction)
    parser.add_argument('--fnmatch', 
        help='draw only directories with a name matching given value')
    args = parser.parse_args()

    if args.csv == '-':
        root = read_data(sys.stdin)
    else:
        with open(args.csv) as fp:
            root = read_data(fp)

    filter_tree(root, minsize=args.minsize, pattern=args.fnmatch)

    node = findtop(root)
    kwargs = vars(args)
    for k in ['csv', 'minsize', 'fnmatch']:
        del(kwargs[k])

    polarchart(node, **kwargs)
    # dumptree(root)

if __name__ == '__main__':
    main()
