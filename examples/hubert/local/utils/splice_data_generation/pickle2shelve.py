import pickle
import shelve
import argparse
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--shelve", type=str, required=True)
    args = parser.parse_args()

    d = {}

    with shelve.open(args.shelve) as db:
        with open(args.pickle, "rb") as fp:
            d = pickle.load(fp)
            for key in tqdm(d):
                db[str(key)] = d[key]

