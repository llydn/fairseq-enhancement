import sys, os
import random
import numpy as np
import argparse
import json
import shelve
from tqdm import tqdm
import soundfile as sf
import kaldiio
import pdb

class GenerateAudio:
    def __init__(self, uttid2audio, token2segments, uttid2combinations, load_audio=None):
        self.uttid2audio = uttid2audio
        self.token2segments = token2segments
        self.load_audio = kaldiio.load_mat if load_audio is None else load_audio
        self.uttid2combinations = uttid2combinations

    def __call__(self, uttid):

        synthesis_segments = []
        if uttid not in self.uttid2combinations:
            raise ValueError(f"combination for uttid {uttid} has not been computed")

        segment_combination = random.choice(self.uttid2combinations[uttid])
        for token in segment_combination:
            segment = random.choice(self.token2segments[str(token)])
            uttid, start, end = segment
            rate, wave = self.load_audio(self.uttid2audio[uttid])
            synthesis_segment = wave[int(start * rate) : int(end * rate)]
            if synthesis_segment.max() > 1e-4:
                synthesis_segment = synthesis_segment / synthesis_segment.max() * 0.9
            synthesis_segments.append(synthesis_segment)

        synthesis_wave = np.concatenate(synthesis_segments)
        return synthesis_wave


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--dir", type=str)
    parser.add_argument("--num_samples", "-n", type=int, default=10)
    parser.add_argument("--dst", type=str)

    args = parser.parse_args()
    with open(os.path.join(args.dir, "db_wav.scp")) as fp:
        sdg_db_uttid2audio = {}
        for line in fp:
            uttid, audio = line.strip().split(maxsplit=1)
            sdg_db_uttid2audio[uttid] = audio
    sdg_uttid2combinations = shelve.open(os.path.join(args.dir, "uttid2combinations.shelve"))
    sdg_token2segments = shelve.open(os.path.join(args.dir, "token2segments.shelve"))

    sdg_generate_audio = GenerateAudio(sdg_db_uttid2audio, sdg_token2segments, sdg_uttid2combinations)

    with open(os.path.join(args.dir, "utt2num_samples")) as fp:
        cnt = 0
        for line in fp:
            cnt += 1
            if cnt > args.num_samples:
                break
            uttid, _ = line.strip().split(maxsplit=1)
            wav = sdg_generate_audio(uttid)
            sf.write(os.path.join(args.dst, f"{uttid}.wav"), wav, 16000)


