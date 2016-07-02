#!/usr/bin/env python

"""
Defines a command line interface for user-scripted operations on neural networks and graph neural networks
"""
import os
import sys
import argparse
import gnnlib as gnn


def nncompute_function(args):

    nn = gnn.NeuralNet()
    res1 = nn.read(args.nnfile)

    if res1["status"] == "error":
        sys.stderr.write(res1["message"] + '\n')
        os.sys.exit(1)

    res2 = nn.compute_datafile()
    if res2["status"] == "error":
        sys.stderr.write(res2["message"] + '\n')
        os.sys.exit(1)

    os.sys.exit(0)


## Define functions that become active when user invokes a command
def nnshow_function(args):
    print "this command is not available yet"


def nntrain_function(args):
    print "this command is not available yet"

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

nncompute_parser = subparsers.add_parser('nncompute')
nncompute_parser.add_argument('-n', '--nnfile', help="neural network file")
nncompute_parser.set_defaults(func=nncompute_function)

nnshow_parser = subparsers.add_parser('nnshow')
nnshow_parser.add_argument('-n', '--nnfile', help="neural network file")
nnshow_parser.set_defaults(func=nnshow_function)


nntrain_parser = subparsers.add_parser('nntrain')
nntrain_parser.set_defaults(func=nntrain_function)

if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)
