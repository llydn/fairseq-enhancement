import sys, os
import argparse
import json
from tqdm import tqdm
import pdb


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--text", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--wavlm_unit", type=str)
    parser.add_argument("--origin", type=str, default="common_voice")

    args = parser.parse_args()

    data = {}
    num_lines = sum(1 for _ in open(args.text))
    with open(args.text) as fp_text, open(args.wavlm_unit) as fp_wavlm_unit:
        for line_text, line_wavlm_unit in tqdm(zip(fp_text, fp_wavlm_unit), total=num_lines):
            uttid, text = line_text.strip().split(maxsplit=1)
            uttid1, wavlm_unit = line_wavlm_unit.strip().split(maxsplit=1)
            assert uttid == uttid1, f"{uttid} != {uttid1}"
            wavlm_unit = wavlm_unit.split()
            sample = {
                    'sample_id': uttid,
                    'origin': args.origin,
                    'char': " " + text + " ",
                    'phn': wavlm_unit
                    }
            data[uttid] = sample

    with open(args.output, 'w') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)






