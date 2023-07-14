import shelve
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with shelve.open(args.output) as db:
        for fname in args.input:
            with shelve.open(fname) as db_partial:
                db.update(db_partial)


