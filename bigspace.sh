#!/bin/bash

## Uses the "dirspace" command in conjunction with sort.
##

usage(){
	grep '^##' $0 |cut -b4-
	echo "usage: $(basename $0) DIR"
	exit 1
}

onError() {
	echo "$@" >&2
	exit 1
}
[[ -z $1 ]] && usage

dirspace "$@" |sort -t, -k2 -g -r

