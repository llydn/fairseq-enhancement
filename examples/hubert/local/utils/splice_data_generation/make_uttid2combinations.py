import sys, os
import random
import argparse
import pickle
import shelve
import numpy as np
import time
from tools import get_range_for_rank, GetBestSegmentsCombinations
from tqdm import tqdm
import pdb

def load_tokendb(tokendb_dir, min_n_token, max_n_token):
    tokendb = {}
    for i in range(min_n_token, max_n_token + 1):
        with open(os.path.join(tokendb_dir, f"{i}.pkl"), "rb") as fp:
            tokendb[i] = dict(pickle.load(fp))
    return tokendb

def oracle_wavlm_unit_handler(args):
    tokendb = load_tokendb(args.tokendb_dir, args.min_n_token, args.max_n_token)
    get_best_segments_combinations = GetBestSegmentsCombinations(tokendb, args.min_n_token, args.max_n_token, max_combinations_per_iter=args.max_combinations_per_iter, max_time_consumption=args.max_time_consumption, forbid_self=args.forbid_self)

    def item_handler(data):
        uttid, wavlm_unit_seqs = data
        combinations = []

        tokens = tuple(wavlm_unit_seqs.split())
        combinations = get_best_segments_combinations(tokens)

        if len(combinations) > args.max_combinations:
            combinations = random.sample(combinations, args.max_combinations)

        return uttid, combinations

    with open(args.text_unit) as fp:
        lines = fp.readlines()

    start, end = get_range_for_rank(len(lines), args.nj, args.rank - 1)
    lines = lines[start:end]


    # make uttid2combinations
    # filter token2segments
    with shelve.open(f'{args.uttid2combinations}') as uttid2combinations, \
         shelve.open(f'{args.token2segments}') as token2segments, \
        open(f'{args.utt2num_samples}', 'w') as utt2num_samples:

        for line in tqdm(lines):
            data = line.strip().split(maxsplit=1)
            uttid, combinations = item_handler(data)
            if len(combinations) == 0:
                continue

            uttid2combinations[uttid] = tuple([tuple([tuple([int(unit) for unit in token]) for token in combination]) for combination in combinations])

            max_duration = 0
            for combination in combinations:
                duration=0
                for token in combination:
                    segments = tokendb[len(token)][token]
                    if str(token) not in token2segments:
                        if len(segments) > args.max_segments_per_token:
                            segments = random.sample(segments, args.max_segments_per_token)
                        token2segments[str(tuple(int(unit) for unit in token))] = segments
                    duration += max([(segment[2] - segment[1]) for segment in segments])
                max_duration = max(duration, max_duration)
            utt2num_samples.write(f"{uttid} {int(max_duration * 16000)}\n")


def oracle_phoneme_handler(args):
    tokendb = load_tokendb(args.tokendb_dir, args.min_n_token, args.max_n_token)
    get_best_segments_combinations = GetBestSegmentsCombinations(tokendb, args.min_n_token, args.max_n_token, max_combinations_per_iter=args.max_combinations_per_iter, max_time_consumption=args.max_time_consumption)

    def item_handler(data):
        uttid, phn_seqs = data
        combinations = []

        tokens = tuple(phn_seqs.split())
        combinations = get_best_segments_combinations(tokens)

        if len(combinations) > args.max_combinations:
            combinations = random.sample(combinations, args.max_combinations)

        return uttid, combinations

    with open(args.text_unit) as fp:
        lines = fp.readlines()

    start, end = get_range_for_rank(len(lines), args.nj, args.rank - 1)
    lines = lines[start:end]


    # make uttid2combinations
    # filter token2segments
    with shelve.open(f'{args.uttid2combinations}') as uttid2combinations, \
         shelve.open(f'{args.token2segments}') as token2segments, \
        open(f'{args.utt2num_samples}', 'w') as utt2num_samples:

        for line in tqdm(lines):
            data = line.strip().split(maxsplit=1)
            uttid, combinations = item_handler(data)
            if len(combinations) == 0:
                continue

            uttid2combinations[uttid] = tuple([tuple(combination) for combination in combinations])

            max_duration=0
            for combination in combinations:
                duration=0
                for token in combination:
                    segments = tokendb[len(token)][token]
                    if str(token) not in token2segments:
                        if len(segments) > args.max_segments_per_token:
                            segments = random.sample(segments, args.max_segments_per_token)
                        token2segments[str(token)] = segments
                    duration += max([(segment[2] - segment[1]) for segment in segments])
                max_duration = max(duration, max_duration)
            utt2num_samples.write(f"{uttid} {int(max_duration * 16000)}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--text_unit", type=str, default="")
    parser.add_argument("--handler", type=str, default="oracle_wavlm_unit", choices=["oracle_wavlm_unit", "oracle_phn"])
    parser.add_argument("--max_combinations", type=int, default=100)
    parser.add_argument("--max_combinations_per_iter", type=int, default=3)
    parser.add_argument("--max_segments_per_token", type=int, default=30)
    parser.add_argument("--max_time_consumption", type=float, default=10)
    parser.add_argument("--max_n_token", type=int, default=10)
    parser.add_argument("--min_n_token", type=int, default=3)
    parser.add_argument("--nj", type=int, default=4)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--tokendb_dir", type=str)
    parser.add_argument("--forbid_self", type=lambda v: v.lower() == False, default=False)

    # output files
    parser.add_argument("--utt2num_samples", type=str)
    parser.add_argument("--token2segments", type=str)
    parser.add_argument("--uttid2combinations", type=str)
    args = parser.parse_args()

    if args.handler == "oracle_wavlm_unit":
        oracle_wavlm_unit_handler(args)
    elif args.handler == "oracle_phn":
        oracle_phoneme_handler(args)

