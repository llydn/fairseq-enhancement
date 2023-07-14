import pickle
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    d = {}
    for fname in args.input:
        with open(fname, "rb") as fp:
            d.update(pickle.load(fp))

    with open(args.output, "wb") as fp:
        pickle.dump(d, fp)

