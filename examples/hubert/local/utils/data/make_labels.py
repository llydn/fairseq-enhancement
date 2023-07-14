#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a flashlight (previously called wav2letter++) dataset
"""

import argparse
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    num_lines = sum(1 for _ in open(args.text))

    with open(args.text, "r") as text, open(
        os.path.join(args.dir, args.split + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.dir, args.split + ".wrd"), "w"
    ) as wrd_out:
        for line in tqdm(text, total=num_lines):
            uttid, words = line.strip().split(maxsplit=1)
            print(words, file=wrd_out)
            print(
                " ".join(list(words.replace(" ", "|"))) + " |",
                file=ltr_out,
            )

if __name__ == "__main__":
    main()
