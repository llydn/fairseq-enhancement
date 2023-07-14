import sys, os
from tqdm import tqdm

if __name__ == "__main__":
    uttid2label = {}
    with open(sys.argv[2]) as fp:
        for line in fp:
            uttid, label = line.strip().split(maxsplit=1)
            uttid2label[uttid] = label

    num_lines = sum(1 for _ in open(sys.argv[1]))
    with open(sys.argv[1]) as ifp, open(sys.argv[3], 'w') as ofp:
        for line in tqdm(ifp, total=num_lines):
            uttid, _ = line.strip().split(maxsplit=1)
            uttid_ = uttid.rsplit("-", maxsplit=1)[0].split("-", maxsplit=1)[1]
            label = uttid2label[uttid_]
            label = " ".join(list("|".join(label.split()))) + " |"
            ofp.write(f"{uttid} {label}\n")


