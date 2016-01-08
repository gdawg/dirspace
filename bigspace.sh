#!/bin/bash

## Sort the output from dirspace by size
## usage: dirspace /home/* |bigspace.sh |head -10
##

case $1 in
	-h|--help )
		grep '^##' $0 |cut -b4-
		exit 1
		;;
esac

cat "$@" |sort -t, -k2 -g -r

