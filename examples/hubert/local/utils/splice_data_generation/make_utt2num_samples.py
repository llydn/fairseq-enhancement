import os
import shelve
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()

    with shelve.open(os.path.join(args.dir, "token2segments.shelve")) as token2segments, \
            shelve.open(os.path.join(args.dir, "uttid2combinations.shelve")) as uttid2combinations, \
                open(os.path.join(args.dir, "utt2num_samples"), 'w') as utt2num_samples:
            for uttid, combinations in tqdm(uttid2combinations.items()):
                max_duration = 0
                for combination in combinations:
                    duration=0
                    for token in combination:
                        segments = token2segments[str(token)]
                        duration += max([(segment[2] - segment[1]) for segment in segments])
                    max_duration = max(duration, max_duration)
                utt2num_samples.write(f"{uttid} {int(max_duration * 16000)}\n")


