#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail
expdir=exp/finetune/base_100h_finetune_ls_clean_100_from_enhfromlibrimix_42950
#lang=frisian
data=data
checkpoint=checkpoint_best.pt
test_sets="ls_test_clean ls_test_other"
device=0
config_name=infer_viterbi_pad_audio

. ./path.sh
. utils/parse_options.sh

expdir=`realpath $expdir`
checkpoint=`realpath $expdir/checkpoints/$checkpoint`
python_path="/mnt/lustre/sjtu/home/dny20/fairseqs/fairseq-enhancement:$PYTHONPATH"
data=`realpath $data`
checkpoint_base_name=${checkpoint%.*}
checkpoint_base_name=${checkpoint_base_name##*/}

for test_set in $test_sets; do
    PYTHONPATH=$python_path CUDA_VISIBLE_DEVICES=$device python ../speech_recognition/new/infer.py \
        --config-dir config/decode \
        --config-name $config_name \
        task.data=`realpath $data/$test_set` \
        task.normalize=false \
        task.already_enhanced=false \
        task.enhanced_data_dir='' \
        common_eval.results_path=$expdir/decode/$test_set/NoEnh/${checkpoint_base_name} \
        common_eval.path=$checkpoint \
        dataset.gen_subset=test
done

