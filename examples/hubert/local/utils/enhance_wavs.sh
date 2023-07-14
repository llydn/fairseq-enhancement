#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=data/
batch_size=3

. ./path.sh
. utils/parse_options.sh

for dir in $data_dir/ls_test_*_wham*; do
    python local/utils/enhance_wavs.py \
        --noisy_scp $dir/test.scp \
        --model_dict models/enh_models.json \
        --models DPTNet DCCRNet DCUNet \
        --output_dir $dir/enhanced_wavs \
        --batch_size $batch_size
done