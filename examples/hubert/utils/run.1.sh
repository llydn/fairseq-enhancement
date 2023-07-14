#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k

train_set="train_far"
valid_set="dev_far"
test_sets="test_far"
feats_normalize=utterance_mvn

# ./enh_asr.nospace.debug.sh \
./enh_asr.nospace.sh \
    --lang "en" \
    --ngpu 1 \
    --num_nodes 8 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --max_wav_duration 30 \
    --token_type char \
    --feats_normalize ${feats_normalize} \
    --lm_config conf/tuning/train_lm.yaml \
    --joint_config conf/tuning/train_asr_transformer_no_enh.yaml \
    --decode_config conf/decode_asr.yaml \
    --use_lm false \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --use_signal_ref true \
    --fs "${sample_rate}" \
    --lm_train_text "data/${train_set}/text" "$@"
