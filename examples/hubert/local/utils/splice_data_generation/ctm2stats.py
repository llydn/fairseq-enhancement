## python local/concat_tools/ctm2stats.py --ctm exp/make_ali_e2e/news/ctm --n-token 3 --output exp/make_ali_e2e/news/bpe3.stats.pkl

import argparse
from tqdm import tqdm
import pickle
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctm", type=str, help="path to ctm file")
    parser.add_argument("--n-token", type=int, help="calculate n-gram tokens stats")
    parser.add_argument("--output", type=str, help="path to output")
    parser.add_argument("--token-variety-threshold", type=int, help="tokens with less than n locations will be removed", default=3)
    args = parser.parse_args()

    prev_uttid = None
    num_lines = sum(1 for line in open(args.ctm))
    token2segments = {}
    tokens, starts = [], []
    with open(args.ctm) as fp_ctm: 
        for line in tqdm(fp_ctm, total=num_lines):
            uttid, *info = line.strip().split()
            if uttid != prev_uttid and prev_uttid is not None:
                starts.append(float(end))
                for i in range(0, len(tokens) - args.n_token + 1):
                    token = tuple(tokens[i:i+args.n_token])
                    start, end = starts[i], starts[i+args.n_token]
                    if token not in token2segments:
                        token2segments[token] = []
                    token2segments[token].append((prev_uttid, start, end))

                tokens, starts = [], []

            start, end, token = info
            tokens.append(token)
            starts.append(float(start))
            prev_uttid = uttid

        starts.append(float(end))
        for i in range(0, len(tokens) - args.n_token + 1):
            token = tuple(tokens[i:i+args.n_token])
            start, end = starts[i], starts[i+args.n_token]
            if token not in token2segments:
                token2segments[token] = []
            token2segments[token].append((uttid, start, end))

        tokens_to_delete = [token for token in token2segments if len(token2segments[token]) < args.token_variety_threshold]
        for token in tokens_to_delete:
            del token2segments[token]

    with open(args.output, "wb") as fp:
        pickle.dump(token2segments, fp)

