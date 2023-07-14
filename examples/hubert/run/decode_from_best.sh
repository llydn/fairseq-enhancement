#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail
expdir=exp/finetune/
#lang=frisian
data=data
device=0
config_name=infer_viterbi_pad_audio
best_decode_list=RESULTS/result_07_09_16_36.tmp

. ./path.sh
. utils/parse_options.sh

expdir=`realpath $expdir`
python_path="/mnt/lustre/sjtu/home/dny20/fairseqs/fairseq-enhancement:$PYTHONPATH"
data=`realpath $data`


while IFS= read -r line
do
    expname=$(echo $line | cut -d" " -f1)
    test_set=$(echo $line | cut -d" " -f2)
    enh_model=$(echo $line | cut -d" " -f3)
    checkpoint_base_name=$(echo $line | cut -d" " -f4)

    if [ $enh_model == "NoEnh" ]; then
        already_enhanced=false
    else
        already_enhanced=true
    fi

    PYTHONPATH=$python_path CUDA_VISIBLE_DEVICES=$device python ../speech_recognition/new/infer.py \
        --config-dir config/decode \
        --config-name $config_name \
        task.data=`realpath $data/$test_set` \
        task.normalize=false \
        task.already_enhanced=$already_enhanced \
        task.enhanced_data_dir=$data/$test_set/enhanced_wavs/$enh_model \
        common_eval.results_path=$expdir/$expname/decode/$test_set/$enh_model/${checkpoint_base_name} \
        common_eval.path=$expdir/$expname/checkpoints/$checkpoint_base_name.pt \
        dataset.gen_subset=test    

done < $best_decode_list
