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

    for line in it:
        fields = line[:-1].split(',')
        nodepath,nstr = fields[:2]
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
        if 'title' in kwargs:
            title = kwargs['title']
            del(kwargs['title'])
        else:
            title = 'Size by node'

        if 'minsize' in kwargs:
            minsize = kwargs['minsize']
            del(kwargs['minsize'])
        else:
            minsize = -1
        self.minsize = minsize

        self.ax = plt.subplot(projection='polar', 
                              *args, **kwargs)
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
        if self.minsize > 0:
            return node.n > self.minsize
        return width > self.min_plotted

    def plot_node(self, node, level, left, right):
        segments = node.children.values() if len(node.children) > 0 else [node]
        slices = len(segments)

        # sort 
        segments.sort(key=lambda n: n.n, reverse=True)
        
        sizes = np.array([c.n for c in segments], dtype=np.float)
        sizesum = sizes.sum()

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

def polarchart(root, minsize=-1, *args, **kwargs):
    chart = PolarSizeChart(minsize=minsize, *args, **kwargs)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    args = parser.parse_args()

    if args.csv == '-':
        root = read_data(sys.stdin)
    else:
        with open(args.csv) as fp:
            root = read_data(fp)

    node = findtop(root)
    polarchart(node, title='Size by Directory')
    # dumptree(root)

if __name__ == '__main__':
    main()
