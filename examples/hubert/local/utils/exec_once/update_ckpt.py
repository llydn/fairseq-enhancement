# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from fairseq.models.wavlm import WavLM, WavLMConfig

# load the pre-trained checkpoints

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=str)
    parser.add_argument("--ref", type=str, default="/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/wavlm/models/wavlm_base.ref.pt")
    parser.add_argument("--new", type=str)
    args = parser.parse_args()


    model = torch.load(args.old)["model"]
    checkpoint = torch.load(args.ref)
    checkpoint['model'] = model
    torch.save(checkpoint, args.new)
