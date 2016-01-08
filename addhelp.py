#!/usr/bin/env python

path = 'dirspace.c'
temp = path + '.tmp'
with open(path) as src, open(temp, 'w') as dest:
    for line in src:
        if line.find('BEGIN HELP') != -1:
            indent,_,_ = line.partition('//')
            dest.write(line)
            for line in src:
                if line.find('END HELP') != -1:
                    end = line
                    break
            with open('info.txt') as info:
                for line in info:
                    dest.write(indent)
                    dest.write('"')
                    dest.write(line[:-1])
                    dest.write('\\n"\n')
            dest.write(end)
        else:
            dest.write(line)

import os
os.rename(temp, path)
