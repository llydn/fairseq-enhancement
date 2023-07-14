#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

expdirs= # "exp/finetune/base_100h_finetune_ls_clean_100_from_fusion_interpolate_50000 exp/finetune/base_100h_finetune_ls_clean_100_from_fusion_interpolate_40000"
log_dir=RESULTS
decode_subdir=viterbi_pad_audio
new_wer_file=wer.filtered
replace_exist=false

# decode_path fusion enhance_model finetune_set test_set checkpoint wer

. ./path.sh
. utils/parse_options.sh

log_file=$log_dir/reevaluate.$(date +'%m%d%H%M').log

for expdir in $expdirs; do
    echo "Processing $expdir ..."
    for decode_wer_dir in ${expdir}/decode/*/*/*/${decode_subdir}; do
        decode_wer_path=${decode_wer_dir}/wer.[0-9]*
        decode_infer_path=${decode_wer_dir}/infer.log
        if [ ! -e $decode_wer_path ] || [ ! -e $decode_infer_path ]; then
            continue
        fi
        if [ -e ${decode_wer_dir}/${new_wer_file} ] && [ $replace_exist == false ]; then
            continue
        fi
        python local/utils/reevaluate_wer.py \
        --infer_log $decode_infer_path \
        --new_wer_file_name ${new_wer_file} >> ${log_file} 2>&1
    done
done


echo "Log saved to $log_file"