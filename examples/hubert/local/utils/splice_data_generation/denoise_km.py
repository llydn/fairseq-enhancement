#!/usr/bin/env python3

import os, sys
import numpy as np
import tqdm
from scipy.ndimage import generic_filter
from scipy import stats
from tqdm import tqdm
import argparse

def str2ints(value):
    return [int(_) for _ in value.split("_")]

def modal(P):
    mode = stats.mode(P)
    return mode.mode[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input ali file")
    parser.add_argument("--output", type=str, help="output ali file")
    parser.add_argument("--kernels", type=str2ints, help="kernels to apply. e.g. 3_5_5_5", required=True)
    args = parser.parse_args()

    num_lines = sum(1 for _ in open(args.input))

    with open(args.input) as ifp, open(args.output, 'w', 1) as ofp:
        for line in tqdm(ifp, total=num_lines):
            sequence = np.array([int(_) for _ in line.strip().split()])
            for kernel in args.kernels:
                sequence = generic_filter(sequence, modal, (kernel, ))
            ofp.write(" ".join([str(_) for _ in sequence]) + "\n")
 
